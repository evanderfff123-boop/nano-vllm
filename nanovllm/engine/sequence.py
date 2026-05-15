from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列的生命周期状态"""
    WAITING = auto()  # 在等待队列中，还没开始处理或已被抢占
    RUNNING = auto()  # 正在运行队列中，正在进行生成
    FINISHED = auto() # 已完成生成（遇到结束符或达到最大长度）


class Sequence:
    # 静态变量：每个物理块存多少 Token
    block_size = 256
    # 静态变量：全局自增计数器，为每个序列分配唯一 ID
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter) # 序列唯一 ID
        self.status = SequenceStatus.WAITING # 初始状态为等待

        # Token 相关
        self.token_ids = copy(token_ids) # 存储所有的 Token ID（含 Prompt 和生成的）
        self.last_token = token_ids[-1] # 最近的一个 Token ID
        self.num_tokens = len(self.token_ids) # 当前总 Token 数量
        self.num_prompt_tokens = len(token_ids) # 原始输入的 Prompt 长度

        # 进度相关
        self.num_cached_tokens = 0 # 已经完成 KV Cache 计算并存入 Block 的 Token 数量
        self.num_scheduled_tokens = 0 # 当前这一步(step) 被调度器安排要计算的 Token 数量

        # 状态标志
        self.is_prefill = True # 是否处于首词填充阶段
        self.block_table = [] # 物理块表：记录了对应的物理块 ID 列表 (例如 [5, 12, 8])

        # 采样与限制参数
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        """是否已生成结束"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已经生成了多少个新 Token (不含 Prompt)"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取原始输入的 Prompt 部分"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取生成出来的部分"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """根据当前总 Token 数，计算需要多少个物理块 (向上取整)"""
        # / : 除法； // ：整除
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个物理块中实际装了多少个 Token"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """获取第 i 个逻辑块包含的 Token ID 列表"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """新增一个生成的 Token，并更新计数"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        定义当进程间传输 (Pickle) 该对象时，需要传递哪些数据。
        优化：如果是 Decode 阶段，不需要传输完整的 token_ids 列表，只传最后一个词，节省带宽。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """子进程接收到数据后，反序列化还原对象"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list): # Prefill 阶段传的是整个列表
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else: # Decode 阶段传的是单个 Token ID
            self.token_ids = [] # 子进程在 Decode 时不维护完整的 token_ids，减少内存开销
            self.last_token = last_state
