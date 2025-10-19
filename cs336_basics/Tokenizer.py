import heapq, cProfile, time, regex
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path
from tqdm.auto import tqdm
from pretokenization_example import find_chunk_boundaries
from dataclasses import dataclass


@dataclass #帮忙自动生成类的其他要素，比如__init__,__repr__之类的
class RevBytes:
    s: bytes
    def __lt__(self, other: "RevBytes") -> bool:
        return self.s > other.s
# heapq在比较两个元素的时候，只调用他们的小于运算，运算(a < b, a.__lt__(b))

class Tokenizer():
    def __init__(
            self, vocab : dict[int, bytes], 
            merges : list[tuple[bytes, bytes]] | None = None, 
            special_tokens : list[str] | None = None, 
            input_path: str = None, 
            vocab_size : int = None, 
            num_processes: int = 1
            ):
        self.input_path = Path(input_path)
        self.vocab_size = vocab_size if vocab is None else len(vocab)
        self.special_tokens = sorted(special_tokens, key = len, reverse = True) # 先匹配长再匹配短
        self.num_processes = num_processes
        self.vocab = {i: bytes([i]) for i in range(256)} if vocab is None else vocab
        self.merges = [] if merges is None else merges
        self.word_counts = None
        self.bytes2id : dict[bytes, int] = {bytes([i]) : i for i in range(256)} if vocab is None else {v : k for k, v in vocab.items()}

    @staticmethod #解除和class的绑定让他成为实例方法
    def pre_tok_chunk(chunk, special_tokens, include_special_tokens : bool = False) -> list[str]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
        
        escaped = [regex.escape(s) for s in special_tokens]

        if include_special_tokens :
            sp_PAT = "(" + "|".join(escaped) + ")"
        else :
            sp_PAT = "|".join(escaped)

        parts = regex.split(sp_PAT, chunk)
        tokens : list[str] = []

        for part in parts:
            if not part:
                continue
            if include_special_tokens and part in special_tokens:
                tokens.append(part) # 大坑，如果+str的话会逐字符加进list里
                continue
            tokens += regex.findall(PAT, part)
        return tokens
    
    def pre_tokenize(self, file_path = None):
        self.word_counts = Counter()
        if file_path is None:
            file_path = self.input_path
        chunk_list = []
        with open(file_path, "rb") as f:
            boundaries = find_chunk_boundaries(f,self.num_processes, b"<|endoftext|>")

            for start, end in zip(boundaries[:-1], boundaries[1:]): # 拉链起来 [0, 5, 12] -> [[0, 5], [5, 12]]
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # print(chunk)
                chunk_list.append(chunk)

        with mp.Pool(self.num_processes) as pool:
            result = pool.starmap(self.pre_tok_chunk, [(chunk, self.special_tokens) for chunk in chunk_list])
            for r in result:
                self.word_counts += Counter(r)
        
    
    def train(self):
        start_id = 256 # 下一个token id 是256
        self.pre_tokenize()
         # 初始词表，对0~255的byte编码ids
        self.bytes2id: dict[bytes, int] = {bytes([i]) : i for i in range(256)}
        for sp_token in self.special_tokens:
            self.vocab[start_id] = sp_token.encode("utf-8")
            self.bytes2id[sp_token.encode("utf-8")] = start_id
            start_id += 1

        words: dict[str, list[int]] = {w: list(w.encode("utf-8")) for w in self.word_counts}
        # 这里encode之后变成bytes，迭代成list已经是整数的list
        # 因为python是没有byte这种东西的，所以迭代拿出来就是int了

        pair_count: dict[tuple[int, int], int] = defaultdict(int) # 用来记录token id pair的数量 (ids1, ids2) : count
        pair2words: dict[tuple[int, int], set[str]] = defaultdict(set) # token id pair所出现的词
        # defaultdict可以直接对不存在的值做自加自减

        # 第一次先全部扫描以初始化
        for w, ids in words.items(): 
            if(len(ids) < 2):
                continue
            for a, b in zip(ids[:-1], ids[1:]):
                p = (a, b)
                pair_count[p] += self.word_counts[w]
                pair2words[p].add(w) # set用add
            
            # words : [ids1, ids2, ids3, ... , idsn]

        # 懒更新堆：频次最大优先，其次字典序最大（bytes更大）
        # 按token id做字典序就是(-count, -a, -b, a, b)
        # 按bytes 字典序，排序应该为(-count, invert_bytes(vocab[a]), invert_bytes(vocab[b]), a, b)
        heap = []
        def push_pair(p):
            cnt = pair_count.get(p, 0) # 如果有就返回p的val，没有就返回0(default)
            if cnt <= 0:
                return
            a, b = p # a, b 是这个pair的两个ids
            heapq.heappush(heap, (-cnt, RevBytes(self.vocab[a]), RevBytes(self.vocab[b]), a, b)) 
            # 这里只有cnt和pair_count里数量对上的才是真的最新的pair count
            # 这个排序用最小堆做最大排序，权重 cnt > a > b
        
        for p in pair_count.keys(): # 把所有的pair都先进堆
            push_pair(p)

        for next_id in tqdm(range(start_id, self.vocab_size + 1)):
            # 懒更新弹堆
            while heap:
                negc, _na, _nb, a, b = heapq.heappop(heap) # 出一个
                c = -negc
                if pair_count.get((a, b), 0) == c: # 只有和cnt对上才是最新的pair
                    best_pair = (a, b)
                    break
            else:
                break #堆空出循环

            self.vocab[next_id] = self.vocab[a] + self.vocab[b]
            self.bytes2id[self.vocab[a] + self.vocab[b]] = next_id
            self.merges.append((self.vocab[a], self.vocab[b]))

            affected_words = list(pair2words.get(best_pair, ())) # 需要被更新的word set
            for w in affected_words:
                ids = words[w]
                c_w = self.word_counts[w]
                if len(ids) < 2:
                    continue
                
                # 从右往左合并
                i = len(ids) - 2 # 倒数第一个开始
                while i >= 0:
                    if ids[i] == a and ids[i + 1] == b:
                        left = ids[i - 1] if i > 0 else None # 用类似？表达式简写
                        right = ids[i + 2] if i + 2 < len(ids) else None 

                        if left is not None:
                            old_left = (left, a)

                            pair_count[old_left] -= c_w
                            if pair_count[old_left] <= 0:
                                pair2words.pop(old_left, None)
                                pair_count.pop(old_left, None) # 如果已减干净了，就扔掉
                            else:
                                push_pair(old_left) # 把old_left重新入堆？
                            
                            new_left = (left, next_id)
                            pair_count[new_left] += c_w # defaultdic提供的特性
                            pair2words[new_left].add(w)
                            push_pair(new_left) # 新对入队列
                        
                        if right is not None:
                            old_right = (b, right)
                            
                            pair_count[old_right] -= c_w
                            if pair_count[old_right] <= 0:
                                pair_count.pop(old_right, None)
                                pair2words.pop(old_right, None)
                            else:
                                push_pair(old_right)
                            
                            new_right = (next_id, right)
                            pair_count[new_right] += c_w
                            pair2words[new_right].add(w)
                            push_pair(new_right)

                        ids[i : i + 2] = [next_id]

                        i -= 1
                    else:
                        i -= 1
                
                words[w] = ids # 两边都得改
            
            pair_count.pop(best_pair, None)
            pair2words.pop(best_pair, None)

            next_id += 1
        
        return self.vocab

    # 使用naive方法实现
    def word2ids(self, word : str) -> list[int]:
        if word in self.special_tokens:
            return [self.bytes2id[word.encode("utf-8")]]
        word = [bytes([b]) for b in word.encode("utf-8")] # 把word变成bytes的list
        # 要注意bytes(n)本身创建的是每个元都是0的bytes，长为n。如果要把int变成byte得给列表
        c = len(word) # 这是bytes的个数 
        if c == 1 :
            return [self.bytes2id[word[0]]]

        for merge_pair in self.merges:
            for i in reversed(range(c - 1)):
                if tuple(word[i: i+2]) == merge_pair:
                    # print(word[i: i+2])
                    # print(merge_pair)
                    word[i: i + 2] = [word[i] + word[i + 1]]

        ids = [self.bytes2id[b] for b in word]
        # print(f"ids : {ids} , word : {word}")

        return ids

    def encode(self, text : str) -> list[int]:
        if self.vocab is None:
            raise RuntimeError("Run tokenize.train() first.")
        
        word_list = self.pre_tok_chunk(text, self.special_tokens, include_special_tokens = True)

        ids = []
        for w in word_list:
            ids += self.word2ids(w)

        return ids

    def decode(self, ids : list[int]) -> list[str]:
        str_list : list[str] = []
        for id in ids:
            str_list.append(self.vocab[id].decode("utf-8"))
        return str_list


        
            
