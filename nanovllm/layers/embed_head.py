import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词表并行嵌入层。
    将整个词表（Vocab）按行切分到不同的 GPU 上。
    """
    def __init__(
        self,
        num_embeddings: int, # 总词表大小
        embedding_dim: int,  # 隐藏层维度 (hidden_size)
    ):
        super().__init__()
        # 获取当前的张量并行状态
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # 确保词表大小能被并行度整除
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings

        # 计算当前 GPU 负责的部分（分片）
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size

        # 计算当前 GPU 负责的词表索引范围 [start, end)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 初始化当前 GPU 上的权重分片
        # 注意：这里只申请了总权重的 1/tp_size 大小的内存
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 绑定一个自定义的权重加载函数
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        从完整权重中加载属于当前分片的权重。
        param: 当前 GPU 上的参数
        loaded_weight: 从检查点文件读取的完整权重张量
        """
        param_data = param.data
        shard_size = param_data.size(0)
        # 根据当前 Rank 计算切片起始位置
        start_idx = self.tp_rank * shard_size
        # 使用 narrow 获取切片并拷贝
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播。
        核心逻辑：每张卡查自己的词，查不到的置零，最后通过 all-reduce 汇总。
        """
        if self.tp_size > 1:
            # 1. 创建掩码：判断输入的 Token ID 是否在当前 GPU 负责的范围内
            # mask 是布尔类型，1 表示该词在当前卡，0 表示不在
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 2. 将全局 ID 转换为本地 ID
            # 比如当前卡负责索引 [1000, 2000)，输入的 ID 是 1050，
            # 则本地对应的索引是 1050 - 1000 = 50。
            # 对于不在范围内的词，这里会计算出无效值，但后续会被 mask 过滤掉。
            x = mask * (x - self.vocab_start_idx)
        # 3. 查表：在本地权重分片中查找
        # 如果 ID 不在范围内，查出来的是本地分片的第 0 行，这是无效数据。
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 4. 掩码过滤：将那些不在当前卡范围内的词查出来的“无效数据”全部置为 0
            y = mask.unsqueeze(1) * y
            # 5. 通信聚合 (All-Reduce Sum)：
            # 假设输入 Token 5000 在 GPU 1 上。
            # GPU 0: 查出来是 0 向量
            # GPU 1: 查出来是对应的 Embedding 向量
            # GPU 2: 查出来是 0 向量
            # 所有 GPU 做一次 All-Reduce Sum，结果就是每张卡都得到了 Token 5000 的正确向量。
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    张量并行的语言模型输出头。
    继承自 VocabParallelEmbedding，因为 LM Head 和 Embedding 
    在权重结构上（即词表切分）是非常相似的。
    """
    def __init__(
        self,
        num_embeddings: int, # 词表大小 (vocab_size)
        embedding_dim: int,  # 隐藏层维度 
        bias: bool = False,
    ):
        assert not bias  # 大多数现代模型（如 Llama, Qwen）的 LM Head 不使用偏置
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        参数 x: 输入的隐藏状态，形状通常为 [total_tokens, hidden_size]
        """
        # 1. 获取推理上下文
        context = get_context()
        # 2. 【关键优化】Prefill 阶段的选择性计算
        # 在 Prefill（处理 Prompt）阶段，虽然输入了许多 Token，
        # 但我们通常只需要预测“下一个”Token。
        # 因此，只需要提取每个序列最后一个 Token 的隐藏状态进行线性变换。
        if context.is_prefill:
            # cu_seqlens_q 记录了每个请求在 Batch 中的终止索引
            # 例如 cu_seqlens_q 是 [0, 10, 25]，说明有两个请求，分别结束于索引 9 和 24
            last_indices = context.cu_seqlens_q[1:] - 1
            # 只取每个 Prompt 的最后一个 Token，将计算量从 [Total_Tokens, V] 降至 [Batch_Size, V]
            x = x[last_indices].contiguous()
        # 3. 线性变换：计算词表得分 (Logits)
        # 注意：由于使用了张量并行，self.weight 只包含词表的一部分。
        # 这里的 logits 是“局部 Logits”，即当前 GPU 负责的那部分词的分数。
        logits = F.linear(x, self.weight)
        # 4. 【张量并行】通信聚合
        # 如果开启了张量并行 (tp_size > 1)，需要将各张显卡计算的部分词表结果合并
        if self.tp_size > 1:
            # 仅在主卡（Rank 0）上准备接收所有 Logits 的列表
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 使用 gather 通信，将所有 GPU 上的局部 Logits 收集到主卡
            dist.gather(logits, all_logits, 0)
            # 在主卡上，沿着词表维度（最后一个维度）拼接结果，得到完整的词表概率
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        # 返回最终 Logits（非主卡返回 None）
        return logits
