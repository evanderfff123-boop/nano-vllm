import os
from dataclasses import dataclass
from transformers import AutoConfig

# 使用 slots=True 优化内存占用，避免为每个实例创建 __dict__
@dataclass(slots=True)
class Config:
    # --- 模型基础设置 ---
    model: str # 模型路径，必须是本地文件夹路径

    # --- 推理性能与显存管理参数 ---
    max_num_batched_tokens: int = 16384  # 批处理中一次性处理的最大 Token 总数
    max_num_seqs: int = 512              # 同时支持的最大并发请求序列数
    max_model_len: int = 4096            # 推理时支持的最大上下文长度
    gpu_memory_utilization: float = 0.9  # GPU 显存占用上限 (90%)

    # --- 并行与执行模式设置 ---
    tensor_parallel_size: int = 1        # 使用多少张显卡进行张量并行 (TP)
    enforce_eager: bool = False          # 是否强制使用即时执行模式 (True 时关闭性能编译)

    # --- 运行时状态参数 (初始化时通常不需要手动设置) ---
    hf_config: AutoConfig | None = None  # 存放 huggingface 的模型配置文件
    eos: int = -1                        # 结束符 ID
    kvcache_block_size: int = 256        # KV Cache 的块大小 (PagedAttention 相关)
    num_kvcache_blocks: int = -1         # 自动计算出的 KV Cache 块数量

    def __post_init__(self):
        """
        初始化后的校验逻辑 (Hook)
        在创建 Config 实例后自动运行，确保所有参数逻辑正确
        """
        # 1. 校验路径是否存在
        assert os.path.isdir(self.model), f"模型路径不存在: {self.model}"
        # 2. 校验 KV Cache 块大小是否为 256 的倍数 (PagedAttention 内存对齐需求)
        assert self.kvcache_block_size % 256 == 0, "KV Cache block size 必须是 256 的倍数"
        # 3. 校验并行数是否在 1-8 之间 (这是 GPU 并行处理的合理范围)
        assert 1 <= self.tensor_parallel_size <= 8, "TP size 必须在 1 到 8 之间"

        # 4. 从磁盘加载 HuggingFace 原生配置文件
        # 这会自动加载 config.json 中的模型元数据
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 5. 自动修正上下文长度
        # 确保设置的 max_model_len 不会超过模型本身能处理的物理长度限制
        # max_position_embeddings 是模型结构能处理的最大 Token 数
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
