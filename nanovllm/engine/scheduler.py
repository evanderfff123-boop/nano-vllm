from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs  # 一次 Batch 最多处理多少个请求
        self.max_num_batched_tokens = config.max_num_batched_tokens # 一次 Batch 最多处理多少个 Token
        self.eos = config.eos # 结束符 ID
        self.block_size = config.kvcache_block_size # 每个物理块的大小

        # 初始化显存管理器，管理所有的 KV Cache 物理块
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

        # 两个核心队列
        self.waiting: deque[Sequence] = deque() # 等待进行首词填充 (Prefill) 的请求
        self.running: deque[Sequence] = deque() # 正在进行增量生成 (Decode) 的请求

    def is_finished(self):
        """所有队列都为空时，代表任务全部完成"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """添加新请求到等待队列"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        核心调度算法：决定这一步跑哪些请求

        vLLM 中一个 "step" 可以是 Prefill 或者 Decode，但不会同时混跑。
        每次调用 schedule() 只产出一种类型的工作。

        max_num_batched_tokens: 一个 step 能处理的最大 token 总数（硬上限）。
        max_num_seqs:          一个 step 能处理的最大序列数（硬上限）。

        返回: (选中的序列列表, 是否是 Prefill 阶段)
        """
        scheduled_seqs: list[Sequence] = []   # 本轮选中要执行的序列
        num_batched_tokens: int = 0           # 本轮已经累计了多少 token

        # ========== 阶段 1: Prefill（首词填充） ==========
        # 只要有 waiting 序列，优先做 prefill，不做 decode。
        # 因为 decode 依赖完整的 KV cache，不先算完 prompt 就无法进行。
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]  # peek 队首，不弹出（可能只算了 chunk 还没算完）

            # remaining = 本轮 batch 还能塞多少 token
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break  # batch 已经满了，不能再加更多 token

            # ---------- 计算这个 seq 还剩多少 token 要算 ----------
            if not seq.block_table:
                # 这个 seq 还没分配过任何物理块（首次处理）
                # 先去问 block_manager 能不能分配，同时获取 prefix cache 命中的块数
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # block_manager 说显存彻底不够了，停
                    break
                # 该 seq 总共还剩多少 token 要算 =
                #   总 prompt 长度 - 已缓存的物理块所覆盖的 token 数
                # Q: 如果这个seq没有分配过物理块，为什么还要计算该seq还剩下多少token要算呢，不是代表没算过吗 ?
                # A: can_allocate(seq)做过了prefix cache匹配
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 这个 seq 已经分配过物理块（之前算过 chunk，没算完）
                # 还剩的 token = 总 prompt 长度 - 累计已算完的 token 数
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # ---------- Chunked Prefill 判断 ----------
            # 场景: remaining < num_tokens，即 batch 剩余额度装不下整个 prompt。
            # 策略: 如果 scheduled_seqs 为空，说明这个 seq 是 batch 里的第一个，
            #       允许切分（chunk），算一部分，剩下的等下一轮。
            #       如果 scheduled_seqs 已经有别的序列了，不切分，而是 break，
            #       让这个 seq 等下一轮再做，避免一个大 prompt 占满 batch 饿死其他请求。
            if remaining < num_tokens and scheduled_seqs:
                break

            # ---------- 分配物理块（只有首次需要） ----------
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            # ---------- 确定本轮实际算多少 token ----------
            # 能全算完就全算完，否则只算 remaining 这么多（chunked prefill）
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens  # 累加到 batch 计数

            # ---------- 判断这个 seq 的 prefill 是否全部完成 ----------
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 所有 prompt token 都算完了，从 waiting → running
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()   # 从 waiting 弹出
                self.running.append(seq) # 加入 running，下轮开始 decode
            # 没算完的话，seq 仍然留在 waiting 队首，下次 schedule 继续算
            # Q: scheduled_seqs代表了什么，seq又代表了什么，为什么不管这个seq的prefill完成与否，都会加到scheduled_seqs中
            # A: seq = 当前正在处理的这一个请求（序列）。它代表 waiting 队列里的某一个具体请求。
            # scheduled_seqs = 本轮要交给 engine 实际执行的序列列表。engine 拿到它之后会调用 model_runner.run(scheduled_seqs, is_prefill) 做真正的计算。
            #两种情况都要算：
            #   - prefill 没完成（chunked）：这轮算了 num_scheduled_tokens 个 token，当然要交给 engine 去前向传播，生成 KV cache。不然 chunk 切出来干嘛？
            #   - prefill 完成了：还是一样要交给 engine 去算这轮的 token。区别只在算完之后把它从 waiting 挪到 running，下轮开始 decode。
            scheduled_seqs.append(seq)

        # 本轮只要调度到了任何 prefill 工作，就只做 prefill，不做 decode
        if scheduled_seqs:
            return scheduled_seqs, True

        # ========== 阶段 2: Decode（逐个生成） ==========
        # 只有 waiting 队列空了，或者显存不够分配新的 prefill 时，才会走到这里
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()  # 从队首取一个 decode 序列

            # 检查：这个 seq 再生成 1 个 token 的话，显存够不够存它对应的 KV cache？
            while not self.block_manager.can_append(seq):
                # 不够 → 触发抢占（preemption）：踢掉一个序列，腾出显存块
                if self.running:
                    # 优先踢运行队列末尾的序列（最晚被调度的，牺牲最小）
                    self.preempt(self.running.pop())
                else:
                    # running 队列只剩当前这个序列了，连自己也得被踢掉
                    self.preempt(seq)
                    break  # 跳出内层 while，也跳过了 else 分支
            else:
                # while 条件为 False（can_append 返回 True），说明显存够
                # decode 阶段每个序列每轮只生成 1 个 token
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                # 预标记：这个序列即将用到的新物理块（实际还没写入，先占个位）
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        # 如果所有序列都被抢占踢掉了，scheduled_seqs 为空 → 断言失败，说明显存爆了
        assert scheduled_seqs, "所有请求都被抢占，没有可执行的序列"

        # 把本轮选中的 decode 序列放回 running 队列队首（保持原顺序）
        # running 队列的序列会反复进出：每次 popleft 取一个，算完再放回
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占机制：当显存不足时，强行中断一个请求"""
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq) # 释放该请求占用的所有显存块
        self.waiting.appendleft(seq) # 插回等待队列的最前面，下次优先处理

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """推理完后的账务更新"""
        for seq, token_id in zip(seqs, token_ids):
            # 1. 尝试对完成计算的块进行 Hash（用于 Prefix Caching 优化）
            self.block_manager.hash_blocks(seq)

            # 2. 更新已处理 Token 计数
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # 如果是 Prefill 且还没把 Prompt 跑完（Chunked Prefill 情况），先不生成新词
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            # 3. 将新生成的词加入序列
            seq.append_token(token_id)
            # 4. 判断是否结束（遇到 EOS 或达到最大长度）
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq) # 运行结束，释放显存
                self.running.remove(seq)
