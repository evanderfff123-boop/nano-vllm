import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    '''
    LLM推理引擎的主控制类，负责管理模型运行、调度和请求处理。
    '''

    def __init__(self, model, **kwargs):
        # 1. 提取配置：从传入参数中过滤出 Config 类需要的字段
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 2. 设置全局块大小：PagedAttention 的核心逻辑，每个 Block 存储多少个 Token
        Sequence.block_size = config.kvcache_block_size

        self.ps = []  # 存储子进程对象（用于张量并行Tensor Parallel）
        self.events = []  # 存储进程间同步事件

        # 3. 初始化多进程环境：使用 "spawn" 模式启动子进程
        ctx = mp.get_context("spawn")

        # 启动 TP (Tensor Parallel) 的子进程
        # rank 0 在主进程运行，rank 1 到 tensor_parallel_size - 1 在子进程运行 
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # 为每个额外的 GPU 启动一个 ModelRunner
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # 4. 初始化主进程的 ModelRunner (Rank 0)
        self.model_runner = ModelRunner(config, 0, self.events)

        # 5. 加载分词器并设置结束符
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # 6. 初始化调度器：负责管理请求队列和显存分配
        self.scheduler = Scheduler(config)

        # 注册退出钩子，确保程序崩溃或正常关闭时清理 GPU 进程
        atexit.register(self.exit)

    def exit(self):
        """清理资源，停止所有子进程"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """将一个新的用户请求添加到引擎中"""
        # 如果输入是字符串，先进行分词
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        # 将请求封装成 Sequence 对象（包含状态、Token、采样参数等）
        seq = Sequence(prompt, sampling_params)

        # 将序列加入调度器的等待队列
        self.scheduler.add(seq)

    def step(self):
        """
        引擎的核心迭代步：执行一次推理迭代
        包含：调度 -> 模型前向计算 -> 后处理
        """
        # 1. 调度：决定当前这一步哪些请求可以进入 GPU 进行计算
        # is_prefill 表示当前这批请求是在做“首词填充”（计算 Prompt）还是“解码”（生成新词）
        seqs, is_prefill = self.scheduler.schedule()

        # 统计本轮处理的 Token 数量（用于计算吞吐量）
        # 如果是 prefill，返回 token 总数；如果是 decode，返回请求数的负值（表示处理了 n 个请求，每个请求 1 token）
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

        # 2. 执行推理：调用 ModelRunner 运行模型，获取生成的新 Token ID
        # 内部会通过 multiprocessing 触发所有 GPU rank 同步运行
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 3. 后处理：更新 Sequence 状态，根据新 Token 更新 KV Cache 引用，判断请求是否完成
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # 4. 收集已完成的请求
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        """判断所有请求是否都已处理完毕"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        高层 API：批量处理一组 Prompt 并返回最终生成的文本。
        """
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        # 规格化采样参数
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 批量添加请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.

        # 核心循环：只要调度器里还有任务，就不断执行 step()
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            dt = perf_counter() - t
            
            # 计算并实时更新吞吐量显示
            if num_tokens > 0: # Prefill 阶段
                prefill_throughput = num_tokens / dt
            else: # Decode 阶段
                decode_throughput = -num_tokens / dt
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            # 记录已完成请求的结果
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()

        # 按请求 ID 排序并解码回字符串
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
