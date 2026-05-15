import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event # 用于多进程同步的事件

        # 1. 初始化分布式环境：实现多显卡张量并行（NCCL 协议）
        # 每个 GPU 进程都会运行这个 init，通过 tcp 握手建立通信组
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank) # 将当前进程绑定到对应的 GPU ID

        # 2. 加载模型
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype) # 设置为模型所需的精度（如 BF16/FP16）
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config) # 初始化 Qwen3 模型结构
        load_model(self.model, config.model) # 从磁盘加载真实权重到显存
        self.sampler = Sampler() # 采样器：负责将 Logits 转化为 Token ID

        # 3. 显存管理：PagedAttention 预分配
        self.warmup_model() # 热身：模拟运行一次模型，触发 PyTorch 的显存管理器分配中间变量
        self.allocate_kv_cache() # 分配剩余的几乎所有显存作为 KV Cache 池（PagedAttention 核心）

        # 4. 性能优化：捕获 CUDA Graph
        # 目的：消除 Decode 阶段（Token-by-token）由于频繁下发小内核导致的 CPU 调度开销
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu") # 恢复默认设置
        torch.set_default_dtype(default_dtype)

        # 5. 多进程控制逻辑：Rank 0 作为主控，其他 Rank 作为从属
        if self.world_size > 1:
            if rank == 0:
                # Rank 0 创建共享内存，用于存放发给其他 GPU 的指令
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier() # 同步，确保所有进程都准备好
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm") # 从属进程连接到共享内存
                self.loop() # 其他 Rank 进入死循环，等待 Rank 0 发令

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        跨进程方法调用分发器。
        
        设计意图：
        在张量并行模式下，所有 Rank 必须同步执行相同的推理步骤（如模型前向传播、KV Cache 更新）。
        为了避免网络通信的复杂性和延迟，采用“主从调度”模式：
        1. Rank 0 (Master) 决定执行什么方法，并通过共享内存(SHM)向 Worker 广播。
        2. 其他 Rank (Worker) 通过循环监听 SHM 获取指令并调用本地对应方法。
        
        Args:
            method_name: 要执行的方法名字符串 (如 'run', 'prepare_prefill')。
            *args: 传递给该方法的参数。
        """
        # [分发逻辑] 只有 Rank 0 拥有“指挥权”。
        # 当 Rank 0 调用某个方法时，它首先将指令序列化写入共享内存。
        # 此时所有 Worker 进程由于正在调用 call()，也会触发自身的逻辑同步。
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)

        # [反射执行] 通过 getattr 获取类中的函数引用。
        # 这里实现了“逻辑一致性”：所有 Rank 的 GPU 进程都会执行相同的模型计算逻辑。
        # 例如：当 Rank 0 调用 self.run()，所有 Rank 的 GPU 都会同步进行 NCCL AllReduce 计算。

        method = getattr(self, method_name, None) # 从 self 上按名字查找 "run" 这个属性
        return method(*args) # 调用它 等价于：return self.run(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # --- 显存管理逻辑 ---
    def allocate_kv_cache(self):
        """
        计算 GPU 剩余显存，并将其全部划分为 PagedAttention 的 Block 块。
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info() # 获取当前显存状态
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # 计算每个 Block（块）占用多少字节
        num_kv_heads = hf_config.num_key_value_heads // self.world_size # 考虑张量并行后的 Head 数
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 2 (K和V) * 层数 * 块大小 * Head数 * 每个Head维度 * 字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

        # 计算可以容纳多少个 Block
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # 预分配巨大的张量作为 KV Cache 池
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 将分配好的显存地址绑定到模型的各个层中
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1


    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # --- 推理准备逻辑 ---
    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 Prefill 阶段送入 GPU 的数据。

        flash_attn_varlen_func 需要 4 类输入:
          1. input_ids / positions — 要计算的 token 及它们的位置编码
          2. cu_seqlens — 变长 batch 中每个序列的起始/结束偏移（用于 flash attention）
          3. slot_mapping — 每个 token 的 KV cache 应该写入显存池的哪个位置（PagedAttention 核心）
          4. block_tables — prefix cache 场景下，attention 需要读取历史的 KV cache，
             通过 block_table 找到对应的物理块位置
        """
        # 为每类元数据分别构造列表，最后统一转 Tensor
        input_ids = []       # 本轮所有 seq 要算的 token id，flatten 成一维
        positions = []       # 每个 token 在原始 prompt 中的位置索引（用于 RoPE）
        cu_seqlens_q = [0]   # cumsum of query lengths，flash_attn 用它切分变长 batch
        cu_seqlens_k = [0]   # cumsum of key   lengths，同上
        max_seqlen_q = 0     # batch 中最长的 query（用于 flash_attn 的内部调度）
        max_seqlen_k = 0     # batch 中最长的 key（同上）
        slot_mapping = []    # 每个 token 对应写入 KV cache 池的绝对位置
        block_tables = None  # 物理块表（只在 prefix cache 场景下传给 attention）

        for seq in seqs:
            # ---- 确定该 seq 本轮要算哪一段 token ----
            start = seq.num_cached_tokens              # 从哪开始（跳过已缓存的）
            seqlen_q = seq.num_scheduled_tokens        # 本轮要算多少个 token
            end = start + seqlen_q                     # 结束位置（开区间）
            seqlen_k = end                             # key 也是从 0 到 end

            # 1. token 输入和位置编码
            input_ids.extend(seq[start:end])           # flatten: 所有 seq 的 token 拼一起
            positions.extend(range(start, end))        # 每个 token 的绝对位置

            # 2. cu_seqlens — 变长 batch 的分段标记
            #    flash_attn 不接收 [seq1_tokens, seq2_tokens, ...] 这种 list of list，
            #    而是把全部 token flatten 成一维，然后用 cu_seqlens 记录每段的起止。
            #    例如 cu_seqlens_q = [0, 5, 8] 表示: seq0 占 [0,5), seq1 占 [5,8)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)  # query: 只到本轮算的 token
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)  # key:   到所有已见的 token

            max_seqlen_q = max(seqlen_q, max_seqlen_q)        # 本 batch 最长的 query
            max_seqlen_k = max(seqlen_k, max_seqlen_k)        # 本 batch 最长的 key

            # ---- Slot Mapping —— PagedAttention 的核心映射 ----
            # 每个 token 算出来的 K、V 要写到显存池的具体位置。
            # 位置不是连续的，而是由 slot_mapping 指定（每个 token 一个 slot）。
            # slot = block_table[block_idx] * block_size + offset_in_block
            if not seq.block_table:    # warmup 阶段没有真正的块，跳过
                continue

            # 这个 seq 本轮涉及的物理块范围
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size  # 向上取整

            for i in range(start_block, end_block):
                # 这个物理块在 KV cache 池中的起始地址
                slot_start = seq.block_table[i] * self.block_size

                # 如果是起始块，可能不是从块的开头开始的（因为 chunked prefill 跳过了一些已缓存的 token）
                if i == start_block:
                    slot_start += start % self.block_size

                # 结束地址：
                #   - 中间块直接+ block_size（填满整个块）
                #   - 最后一块只到 end 对应的位置
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size

                # 这块内每个 token 对应一个 slot，取 range 生成
                slot_mapping.extend(range(slot_start, slot_end))

        # ---- Prefix Cache 场景：attention 需要读取历史的 KV cache ----
        # 只要还有已缓存的 key（cu_seqlens_k[-1] > cu_seqlens_q[-1]），
        # 就需要把 block_tables 传给 flash_attn_varlen_func，
        # 让它在计算 attention 时能去历史块里取 KV。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # ---- 转成 GPU Tensor ----
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 写入全局上下文，底层 attention kernel 从这里读取
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 Decode（解码）阶段的数据。这是逐个生成 Token 的阶段。
        """
        # ... 构建针对单个 Token 推理的元数据 ...
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        根据情况决定是用普通的 Eager 模式还是 CUDA Graph 模式。
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Prefill 阶段通常不用 CUDA Graph，因为 Sequence 长度变化太大
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Decode 阶段使用预先捕获的 CUDA Graph 来加速
            bs = input_ids.size(0)
            context = get_context()
            # 找到合适的 Batch Size 档位
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 将输入数据填充到 CUDA Graph 预留的静态内存中
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay() # 执行录制好的指令
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # --- 核心运行逻辑 ---
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        执行推理。
        """
        # 1. 准备数据
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 2. 模型前向计算
        logits = self.run_model(input_ids, positions, is_prefill)

        # 3. 采样获取 Token（仅在 Rank 0 进行采样决策，然后分发或同步）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context() # 清理当前推理请求的上下文
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
