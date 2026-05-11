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
        返回: (选中的序列列表, 是否是 Prefill 阶段)
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        # --- 阶段 1: 优先处理 Prefill (首词填充/计算 Prompt) ---
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            # 如果请求还没分配过显存块
            if not seq.block_table:
                # 检查显存是否足够分配给这个新的 Prompt
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1: # 显存完全不够了
                    break
                # 计算还需要计算多少 Token (Prompt 长度 - 已经缓存的块对应的 Token)
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # Chunked Prefill 逻辑：如果剩下的 Token 额度塞不下整个 Prompt
            # 只有当 scheduled_seqs 为空（即当前 batch 第一个请求）时才允许切分 Prompt
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break

            # 真正分配显存物理块
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            # 计算本次迭代能处理多少个 Token (如果 Prompt 太长，就只处理一部分)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # 如果整个 Prompt 都处理完了，将其从等待队列移到运行队列
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        # 如果有 Prefill 任务，这一步就只做 Prefill
        if scheduled_seqs:
            return scheduled_seqs, True

        # --- 阶段 2: 如果没有 Prefill，则处理 Decode (增量生成) ---
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()

            # 检查显存是否足够再多存 1 个生成的 Token
            while not self.block_manager.can_append(seq):
                # 显存不足！执行【抢占】机制 (Preemption)
                # 踢掉运行队列末尾的一个请求，腾出显存给当前的 seq
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq) # 连自己都跑不动了，直接踢掉自己 
                    break
            else:
                # 显存足够，安排生成 1 个 Token
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq) # 预分配/标记要增加的块
                scheduled_seqs.append(seq)
        # 极端情况：所有请求都被抢占了
        assert scheduled_seqs
        # 将本次处理的 Decode 请求重新放回 running 队列开头，准备下一轮
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
