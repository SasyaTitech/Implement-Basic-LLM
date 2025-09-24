import heapq
from functools import total_ordering


class PreTokens:
    all_docs: list[tuple[bytes, ...]]
    docs_count: list[int]

    def __init__(self, pre_tokens_dict: dict[bytes, int]) -> None:
        self.all_docs: list[tuple[bytes, ...]] = []
        self.docs_count: list[int] = []
        for word, tokens_count in pre_tokens_dict.items():
            tokens = tuple(word[i : i + 1] for i in range(len(word)))
            self.all_docs.append(tokens)
            self.docs_count.append(tokens_count)

    def __len__(self) -> int:
        return len(self.all_docs)

    def __getitem__(self, idx: int) -> tuple[bytes, ...]:
        return self.all_docs[idx]

    def __setitem__(self, idx: int, value: tuple[bytes, ...]) -> None:
        self.all_docs[idx] = value

    def get_item_count(self, idx: int) -> int:
        return self.docs_count[idx]


Pair = tuple[bytes, bytes]


@total_ordering
class MaxKey:
    data: tuple[int, Pair]

    def __init__(self, count: int, pair: Pair):
        self.data = (count, pair)

    def __lt__(self, other: "MaxKey") -> bool:
        # heapq 需要“更小”的元素在堆顶
        # 我们把“更大 count / 更大 pair”定义为“更小”的顺序，达到最大堆+字典序大的优先
        return self.data > other.data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MaxKey):
            return NotImplemented
        return self.data == other.data

    def __repr__(self) -> str:
        return f"MaxKey(count={self.data[0]}, pair={self.data[1]})"


class TokenPairCounter:
    pair_counts: dict[Pair, int]
    pair_to_pretokens: dict[Pair, set[int]]
    _max_heap: list[MaxKey]

    def __init__(self) -> None:
        self.pair_counts = {}
        self.pair_to_pretokens = {}
        self._max_heap = []

    def _rebuild_heap(self) -> None:
        # 当堆里陈旧元素太多时，可重建以控制内存/弹出成本
        self._max_heap = [MaxKey(cnt, pair) for pair, cnt in self.pair_counts.items()]
        heapq.heapify(self._max_heap)

    def push_heap(self, pair: Pair, count: int) -> None:
        heapq.heappush(self._max_heap, MaxKey(count, pair))

    def _peek_heap(self) -> tuple[Pair, int]:
        data = self._max_heap[0].data
        return (data[1], data[0])  # (pair, count)

    def init_from(self, pre_tokens: PreTokens):
        self.clear()
        for idx, tokens in enumerate(pre_tokens.all_docs):
            tokens_count = pre_tokens.get_item_count(idx)
            for i in range(len(tokens) - 1):
                pair: Pair = (tokens[i], tokens[i + 1])
                self.add_pair(pair, idx, tokens_count, update_heap=False)
        self._rebuild_heap()

    def add_pair(
        self, pair: Pair, doc_idx: int, count: int, update_heap: bool = True
    ) -> int:
        if pair not in self.pair_counts:
            self.pair_counts[pair] = 0
            self.pair_to_pretokens[pair] = set()
        new_count = self.pair_counts[pair] + count
        self.pair_counts[pair] = new_count
        self.pair_to_pretokens[pair].add(doc_idx)
        if update_heap:
            self.push_heap(pair, new_count)
        return new_count

    def remove_pair(self, pair: Pair, doc_idx: int, count: int, update_heap: bool = True) -> int:
        assert pair in self.pair_counts, f"Trying to remove non-existing pair {pair}"
        old_count = self.pair_counts[pair]
        new_count = old_count - count
        assert new_count >= 0, f"Trying to reduce count of pair {pair}"

        if new_count <= 0:
            del self.pair_counts[pair]
            del self.pair_to_pretokens[pair]
        else:
            self.pair_counts[pair] = new_count
            self.pair_to_pretokens[pair].discard(doc_idx)
            if update_heap:
                self.push_heap(pair, new_count)
        return new_count

    def _peek_valid_max(self) -> tuple[Pair, int]:
        # 懒删除：弹出堆顶直到与当前计数一致
        while self._max_heap:
            (pair, count) = self._peek_heap()
            cur = self.pair_counts.get(pair, 0)
            if cur == 0 or count != cur:
                heapq.heappop(self._max_heap)
                continue
            return pair, cur
        return (b"", b""), 0

    def get_max_pair(self) -> tuple[Pair, int]:
        if not self.pair_counts:
            return ((b"", b""), 0)
        if len(self._max_heap) >= 3 * len(self.pair_counts):
            print(
                "Rebuilding heap... {} pairs, {} heap size".format(
                    len(self.pair_counts), len(self._max_heap)
                )
            )
            self._rebuild_heap()
        return self._peek_valid_max()
        # max_count = 0
        # max_key = (b"", b"")
        # for key in self.pair_counts:
        #     count = self.pair_counts[key]
        #     if count > max_count:
        #         max_count = count
        #         max_key = key
        #     elif count == max_count and key > max_key:
        #         max_key = key
        # return (max_key, max_count)

    def get_pretokens(self, pair: Pair) -> set[int]:
        return self.pair_to_pretokens.get(pair, set())

    def get_token_count(self, pair: Pair) -> int:
        return self.pair_counts.get(pair, 0)

    def clear(self):
        self.pair_counts.clear()
        self.pair_to_pretokens.clear()
        self._max_heap.clear()
