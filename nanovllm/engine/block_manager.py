from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """代表显存中的一个物理块 (Physical Block)"""
    def __init__(self, block_id):
        self.block_id = block_id # 物理块的索引 ID
        self.ref_count = 0 # 引用计数：有多少个 Sequence 正在共享这个块
        self.hash = -1 # 该块内容的哈希值（用于 Prefix Caching 匹配）
        self.token_ids = []  # 存储在该块中的具体 Token ID 列表

    def update(self, hash: int, token_ids: list[int]):
        """更新块的内容信息"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态，准备分配给新的请求"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """显存管理器：负责物理显存块的分配、回收和共享"""
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # 初始化所有物理块
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希查找表：通过 Token 内容的哈希值快速找到对应的物理块 ID
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块队列：存放在显存中还没被使用的块 ID
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 正在使用的块 ID 集合
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算一块 Token 的哈希值。
        prefix: 前一个块的哈希值。这种级联哈希保证了只有前缀完全一致时，哈希才相同。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """底层分配：从空闲池取出一个块"""
        # 使用 popleft() 说明 free_block_ids 是一个 collections.deque，这比列表 list 的 pop(0) 操作效率高得多（O(1) 时间复杂度 vs O(n)）。
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        # 确保拿到的是干净的
        assert block.ref_count == 0
        # 如果这个旧块还留在哈希表里，先把它删掉（因为它要被重新改写了）
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            # 删除映射，防止后续错误路由
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """底层释放：将块归还到空闲池"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """
        检查是否能为新请求分配显存。
        返回：命中的缓存块数量。如果显存不足返回 -1。
        """
        h = -1                          # 初始哈希值
        num_cached_blocks = 0           # 统计命中了多少个已存在的缓存块
        num_new_blocks = seq.num_blocks # 这个新请求一共需要的物理块总数

        # 遍历请求的所有逻辑块（除了最后一个块，因为最后一个块通常包含未生成的 token，还没完全填满）
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)                     # 获取当前逻辑块里的 token
            h = self.compute_hash(token_ids, h)          # 计算哈希（这是为了快速匹配内存内容）
            block_id = self.hash_to_block_id.get(h, -1)  # 查找是否已经存在这个内容的块

            # 缓存未命中逻辑：
            # 1. block_id == -1: 完全没见过这个内容
            # 2. 内容不匹配: 哈希冲突或内容变动
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break # 只要有一个块不匹配，后续的块就都不用查了，直接停止

            num_cached_blocks += 1

            # 如果这个块已经在被其他请求使用，说明这是“共享内存”
            # 那么该请求就不需要再申请新的物理块空间，节省了显存开销
            # 这里指的是物理块
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        
        # 检查剩余的“空闲显存块”是否足够容纳那些没命中的部分
        if len(self.free_block_ids) < num_new_blocks:
            return -1 # 显存不足，无法接纳该请求
        return num_cached_blocks  # 返回成功命中的数量

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """正式为序列分配物理块"""
        # 确保当前序列还没分配任何块，防止重复分配
        assert not seq.block_table
        h = -1

        # 1. 处理“已命中缓存”的块 (Prefix Caching 阶段)
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h] # 根据刚才计算的哈希找到物理块ID
            block = self.blocks[block_id]

            # 如果该块已经在被他人使用 (共享内存)
            if block_id in self.used_block_ids:
                block.ref_count += 1 # 引用计数加1，这是为了防止内存被错误释放
            else:
                # 如果该块虽然存着内容，但当前没人用 (可能是缓存命中，但还未被激活)
                block.ref_count = 1 
                self.free_block_ids.remove(block_id) # 从空闲池中移出
                self.used_block_ids.add(block_id) # 加入已用池

            # 关键操作：将物理块ID加入序列的 block_table
            # 以后模型推理读取 KV Cache 时，就是通过这个表去内存里找数据的
            seq.block_table.append(block_id)

        # 2. 为剩下的部分分配“全新的”物理块
        for i in range(num_cached_blocks, seq.num_blocks):
            # _allocate_block() 会从 free_block_ids 中弹出一个物理块
            seq.block_table.append(self._allocate_block())

        # 记录该序列到底缓存了多少 Token (用于后续性能统计或续写)
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有块"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                # 只有当没有任何序列使用这个块时，才真正回收到空闲池
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Decode 阶段检查：如果当前块满了，是否有空闲块开辟新块"""
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """如果当前块已满，在生成下一个词前分配一个新物理块"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """
        重要：计算并登记已完成计算的块的哈希值。
        这样下次其他请求进来时，就能通过 can_allocate 命中这些块。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        # 级联哈希计算
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)  # 记录块的内容和哈希
            self.hash_to_block_id[h] = block.block_id # 登记到全局查找表
