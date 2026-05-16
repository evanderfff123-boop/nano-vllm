from dataclasses import dataclass
import torch

# 使用 dataclass 定义一个上下文类，用于存储推理过程中的各种张量和状态
# slots=True 可以提升属性访问速度并减少内存占用（在 Python 3.10+ 中支持）
@dataclass(slots=True)
class Context:
    """
    存储当前推理步（Forward Pass）中所需的各种元数据。
    这些参数通常由推理引擎的调度器（Scheduler）生成，并传递给注意力算子。
    """
    # 是否处于预填充（Prefill）阶段。
    # True: 处理用户输入的 Prompt；False: 处于逐 Token 生成的 Decode 阶段。
    is_prefill: bool = False
    # Q 的累积序列长度序列（Cumulative Sequence Lengths）。
    # 用于 FlashAttention 的 varlen 算子，标识 Batch 中每个不同长度序列的起始位置。
    # 形状通常为 [batch_size + 1]
    cu_seqlens_q: torch.Tensor | None = None
    # K/V 的累积序列长度序列。同上，主要针对非对称（如 Cross-Attention）或特殊优化场景。
    cu_seqlens_k: torch.Tensor | None = None
    # 当前 Batch 中 Q 的最大序列长度。用于 Kernel 计算时的维度参考。
    max_seqlen_q: int = 0
    # 当前 Batch 中 K/V 的最大序列长度（通常等于最大历史缓存长度 + 1）。
    max_seqlen_k: int = 0
    # 槽位映射（Slot Mapping）。
    # 将逻辑上的 Token 索引映射到物理 KV Cache 显存池中的具体位置。
    # 形状为 [num_tokens]，每个值指向 KV 缓存池的一个具体 Offset。
    slot_mapping: torch.Tensor | None = None
    # 序列当前的总长度（Context Length）。
    # 记录 Batch 中每个请求已经处理了多少个 Token。
    context_lens: torch.Tensor | None = None
    # 分块表（Block Tables）。
    # PagedAttention 的核心数据结构，记录每个请求占用了哪些非连续的物理显存块。
    # 形状为 [batch_size, max_num_blocks_per_seq]
    block_tables: torch.Tensor | None = None

# 初始化一个全局单例对象，作为当前线程/进程的推理上下文
_CONTEXT = Context()

def get_context():
    """
    获取当前的推理上下文对象。
    在模型层级（如 Attention 层）中调用，以获取算子所需的元数据。
    """
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """
    更新全局上下文。
    在每一轮 Forward 开始前，由调度器根据当前 Batch 的状态进行填充。
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """
    重置上下文，恢复为默认状态。
    通常在一轮推理结束或发生异常时调用，防止数据污染。
    """
    global _CONTEXT
    _CONTEXT = Context()
