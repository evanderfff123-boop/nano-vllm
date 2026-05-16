import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    带分布式支持和 GQA 机制的注意力层
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        # 初始化 TP (Tensor Parallel) 大小--- 张量并行设置 ---
        tp_size = dist.get_world_size()  # 获取并行 GPU 数量
        self.total_num_heads = num_heads # 每张显卡分到的 Query 头数
        assert self.total_num_heads % tp_size == 0 
        self.num_heads = self.total_num_heads // tp_size # 每张显卡分到的 KV 头数 (GQA)
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        # --- 线性层（并行版） ---
        # QKVParallelLinear: 将 Q, K, V 合并计算，减少通信开销
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # RowParallelLinear: 汇总所有头的结果并做最后的线性变换，内部包含一次 All-Reduce 通信
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        # --- 旋转位置编码 (RoPE) ---
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # --- 注意力算子 (如 PagedAttention 或 FlashAttention) ---
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # QK Norm 优化：某些大模型为了稳定训练，会在 Attention 前对 Q 和 K 进行归一化
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 1. 一次性投影得到 QKV 混合张量
        qkv = self.qkv_proj(hidden_states)
        # 2. 拆分 Q, K, V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1) # 拆分 QKV
        # 3. 变形以进行多头计算 [tokens, heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 4. 可选的 Q/K Normalization
        if not self.qkv_bias: # 如果启用 QK Norm，在此处标准化
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 5. 应用旋转位置编码 (将位置信息注入 Q 和 K)
        q, k = self.rotary_emb(positions, q, k) # 应用 RoPE
        # 6. 计算 Attention (内部处理 KV Cache 和 PagedAttention)
        o = self.attn(q, k, v) # 计算 Attention
        # 7. 合并所有头并进行输出映射 [tokens, hidden_size]
        output = self.o_proj(o.flatten(1, -1)) # 输出映射
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # 1. 融合列并行线性层：同时计算 Gate 门控路径和 Up 升维路径
        # 权重形状为 [hidden_size, intermediate_size * 2]
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # 2. 行并行线性层：将维度降回 hidden_size，内部含 All-Reduce
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        # 3. 融合激活函数：实现 x = silu(gate) * up
        # 这是一个自定义的高效 Kernel，避免在 Python 中多次切片和拷贝张量
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # 1. 投影到中间高维空间
        gate_up = self.gate_up_proj(x)
        # 2. 激活并逐元素相乘
        x = self.act_fn(gate_up)
        # 3. 投影回低维空间
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    解码器层：Transformer 的一个核心 Block。
    包含：1. 自注意力机制 (Self-Attention) 
          2. 多层感知机 (MLP / Feed-Forward)
          3. 两次 RMSNorm (归一化)
    """
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 1. 自注意力模块 (Attention)
        # 包含了 QKV 映射、RoPE 位置编码处理、以及 PagedAttention 逻辑
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            # 支持 GQA (Grouped Query Attention)
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),  # 旋转位置编码的底数，大底数有助于长文本
            rope_scaling=getattr(config, "rope_scaling", None), # 长文本缩放策略
        )
        # 2. MLP 模块 (通常采用 SwiGLU 结构)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 3. 归一化层 (RMSNorm)
        # input_layernorm: 用于 Attention 之前
        # post_attention_layernorm: 用于 MLP 之前
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,  # 位置索引（RoPE 用）
        hidden_states: torch.Tensor,  # 当前层的输入数据
        residual: torch.Tensor | None,  # 残差项
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：
        采用 Pre-Norm 结构：Norm -> Attention -> Add -> Norm -> MLP -> Add
        """

        # --- 步骤 1: Attention 及其前面的 Norm ---
        # 这里使用了推理框架常见的“融合算子”逻辑：
        # 在做 Norm 的同时，把之前的残差 residual 加进来。
        # 这样可以将 [加法] 和 [Norm] 在一个 CUDA Kernel 里完成，减少显存读写频率。
        if residual is None:
            # 第一层时 residual 为 None
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # hidden_states 是上一层的输出，residual 是累积的残差
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # 执行自注意力计算
        hidden_states = self.self_attn(positions, hidden_states)
        # --- 步骤 2: MLP 及其前面的 Norm ---
        # 同样是 Fused Add-Norm：将 Attention 的输出与之前的残差合并并归一化
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # 执行 MLP（全连接层）计算
        hidden_states = self.mlp(hidden_states)
        # 返回当前处理后的状态和最新的残差
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    模型主体：嵌入层 + N 个 Decoder 层 + 最终 Norm。
    不包含最后的 lm_head 部分。
    """
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 1. 词向量嵌入层 (Embedding)
        # 使用 VocabParallelEmbedding，支持张量并行。
        # 如果词表很大，它会将词表权重切分到多个 GPU 上。
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 2. 堆叠的 Decoder 层 (Transformer Blocks)
        # 根据配置中的层数（num_hidden_layers）创建 N 个 Qwen3DecoderLayer
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 3. 最终层归一化 (Final Layer Norm)
        # Qwen/Llama 系列通常使用 RMSNorm 而不是 LayerNorm
        # 它只做缩放不加偏置，计算效率更高
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            input_ids: 输入的 Token 序列 ID
            positions: 位置编码信息（用于 RoPE 旋转位置嵌入）
        """
        # 1. 将 Token ID 映射为稠密向量 [batch_size * seq_len, hidden_size]
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        # 2. 核心循环：遍历所有 Transformer 层
        # 这里使用了 residual 变量来显式传递残差
        for layer in self.layers:
            # 每一层不仅返回新的 hidden_states，还可能返回更新后的 residual
            # 这种写法方便底层实现 Fused Add-Norm 算子（将残差相加和 LayerNorm 合并为一个 Kernel）
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 3. 最后一步：应用最终的归一化
        # 将最后一层的输出与残差进行最终融合并归一化
        hidden_states, _ = self.norm(hidden_states, residual)
        # 返回最终的隐层状态，准备交给 lm_head 计算 Logits
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    # 1. 权重打包/合并映射表 (Packed Modules Mapping)
    # 这是为了推理加速。在模型加载时，会将独立的线性层合并成一个大的矩阵。
    # 例如：q_proj, k_proj, v_proj 三个矩阵合并成一个 qkv_proj 矩阵。
    # 这样在推理时只需运行一次矩阵乘法（GEMM），显著减少 Kernel Launch 次数。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),       # q_proj 映射到 qkv_proj 的 q 部分
        "k_proj": ("qkv_proj", "k"),       # k_proj 映射到 qkv_proj 的 k 部分
        "v_proj": ("qkv_proj", "v"),       # v_proj 映射到 qkv_proj 的 v 部分
        "gate_proj": ("gate_up_proj", 0),  # MLP 的 gate_proj 映射到 gate_up_proj 的第 0 部分
        "up_proj": ("gate_up_proj", 1),    # MLP 的 up_proj 映射到 gate_up_proj 的第 1 部分
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        # 2. 初始化 Qwen3 模型主体（Transformer 堆栈，不含输出层）
        self.model = Qwen3Model(config)
        # 3. 初始化语言模型头（LM Head）
        # 这里使用 ParallelLMHead，说明模型支持张量并行（Tensor Parallelism）。
        # 它负责将 hidden_states 映射回词表大小（vocab_size）。
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 4. 权重共享逻辑（Tie Word Embeddings）
        # 某些模型会让输入 Embedding 和输出 LM Head 共享同一份权重，以节省显存。
        if config.tie_word_embeddings:
            # 将 lm_head 的权重指向输入 Embedding 的权重地址
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播：仅获取模型主体的输出（Hidden States）。
        结合你之前的代码，compute_logits 会在之后单独调用。
        """
        # 调用 Qwen3Model 获得最后一层 Transformer 的输出
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Logits（词表原始得分）。
        
        该函数将模型最后一层 Transformer Block 输出的稠密向量（Hidden States），
        转换为对应词典中每个词的预测得分。
        
        参数:
            hidden_states: 形状为 [total_tokens, hidden_size] 的张量。
                        这是模型对每个输入 Token 提取的高维特征。
                        
        返回:
            logits: 形状为 [total_tokens, vocab_size] 的张量。
                每一行代表对应 Token 在整个词表（Vocabulary）上的原始得分。
        """
        
        # lm_head 通常是一个 nn.Linear 层（线性变换层）。
        # 它的权重矩阵形状通常是 [vocab_size, hidden_size]。
        # 计算过程本质上是矩阵乘法：Hidden_States @ LM_Head_Weights.T
        # 结果是一个维度极大的向量（例如对于 Llama 3，vocab_size 为 128,256）。
        return self.lm_head(hidden_states)
