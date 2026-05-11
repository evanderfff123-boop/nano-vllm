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
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 如果这个旧块还留在哈希表里，先把它删掉（因为它要被重新改写了）
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
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
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks # 请求总共需要的块数

        # 遍历请求的所有逻辑块（除了最后一个可能没满的块）
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)

            # 如果命中缓存，且内容完全一致
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 如果这个块已经在被别人用了，新请求就不需要额外占用新的物理块空间
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        
        # 检查剩余空闲块是否够用
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """正式为序列分配物理块"""
        assert not seq.block_table
        h = -1
        # 处理命中的缓存块
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1 # 增加引用计数，实现共享
            else:
                # 这种情况是块在空闲池但哈希还匹配（LRU 命中）
                block.ref_count = 1 
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 为剩下的部分分配全新的块
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
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
