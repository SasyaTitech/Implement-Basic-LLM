from pathlib import Path
from pre_tokenizer import pre_tokenize
import heapq
from collections import Counter, defaultdict
import cProfile
from tqdm.auto import tqdm

profiler = cProfile.Profile()
profiler.enable()

def tokenize(input_path: str, vocab_size : int , special_tokens : list[str] | None):
    file_path = Path(input_path)
    words_counts: dict[str, int] = pre_tokenize(file_path) #预分词{word, cnt}

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)} # 初始词表，对0~255的byte编码ids
    words: dict[str, list[int]] = {w: list(w.encode("utf-8")) for w in words_counts}
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
            pair_count[p] += words_counts[w]
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
        heapq.heappush(heap, (-cnt, -a, -b, a, b)) 
        # 这里只有cnt和pair_count里数量对上的才是真的最新的pair count
        # 这个排序用最小堆做最大排序，权重 cnt > a > b
    
    for p in pair_count.keys(): # 把所有的pair都先进堆
        push_pair(p)

    start_id = 256 # 下一个token id 是256
    merges: list[tuple[bytes, bytes]] = [] # 这是作业要求的记录bytes合并的历史

    for next_id in tqdm(range(start_id, vocab_size + 1)):
        # 懒更新弹堆
        while heap:
            negc, _na, _nb, a, b = heapq.heappop(heap) # 出一个
            c = -negc
            if pair_count.get((a, b), 0) == c: # 只有和cnt对上才是最新的pair
                best_pair = (a, b)
                break
        else:
            break #堆空出循环

        vocab[next_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        affected_words = list(pair2words.get(best_pair, ())) # 需要被更新的word set
        for w in affected_words:
            ids = words[w]
            c_w = words_counts[w]
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
    
    return vocab, merges



    
    # best_pair, _ = max(ids_pair_dic.items(), key = lambda kv : (kv[1], vocab[kv[0][0]], vocab[kv[0][1]])) # 字节序排序
    # # bytes本身是可以比较的所以没问题
    # vocab[next_sym_id_start] = vocab[best_pair[0]] + vocab[best_pair[1]]

    # for w in ids_dic_tmp_word[best_pair]:
    #     ids_list = words[w]
    #     word_counts = pre_words_count[w]
    #     for i in reversed(range(len(ids_list) - 1)):
    #         if(ids_list[i], ids_list[i + 1]) == best_pair:
    #             if i + 2 < len(ids_list) :
    #                 pair_now = (next_sym_id_start, ids_list[i + 2])
    #                 if ids_pair_dic.get(pair_now) is None:
    #                     ids_pair_dic[pair_now] = word_counts
    #                     ids_dic_tmp_word[pair_now] = []
    #                 else:
    #                     ids_pair_dic[pair_now] += word_counts
    #                 ids_dic_tmp_word[pair_now].append(w)
    #                 ids_pair_dic[(ids_list[i+1], ids_list[i+2])] -= word_counts
    #                 if ids_pair_dic[(ids_list[i+1], ids_list[i+2])] <= 0:
    #                     del ids_pair_dic[(ids_list[i+1], ids_list[i+2])]
    #             if i - 1 >= 0:
    #                 pair_now = (ids_list[i - 1], next_sym_id_start)
    #                 if ids_pair_dic.get(pair_now) is None:
    #                     ids_pair_dic[pair_now] = word_counts
    #                     ids_dic_tmp_word[pair_now] = []
    #                 else:
    #                     ids_pair_dic[pair_now] += word_counts
    #                 ids_dic_tmp_word[pair_now].append(w)
    #                 ids_pair_dic[(ids_list[i - 1], ids_list[i])] -= word_counts
    #                 if ids_pair_dic[(ids_list[i - 1], ids_list[i])] <= 0:
    #                     del ids_pair_dic[(ids_list[i - 1], ids_list[i])]
    #             ids_list[i:i+2] = [next_sym_id_start]
    #             words[w] = ids_list
    # del ids_pair_dic[best_pair]
    # del ids_dic_tmp_word[best_pair]

    # merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

    # for next_sym_id in tqdm(range(next_sym_id_start + 1, vocab_size + 1)):
    #     best_pair, _ = max(ids_pair_dic.items(), key = lambda kv : (kv[1], kv[0][0], kv[0][1]))
    #     vocab[next_sym_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

    #     for w in ids_dic_tmp_word[best_pair]:
    #         ids_list = words[w]
    #         word_counts = pre_words_count[w]
    #         for i in reversed(range(len(ids_list) - 1)):
    #             if(ids_list[i], ids_list[i + 1]) == best_pair:
    #                 if i + 2 < len(ids_list) :
    #                     pair_now = (next_sym_id, ids_list[i + 2])
    #                     if ids_pair_dic.get(pair_now) is None:
    #                         ids_pair_dic[pair_now] = word_counts
    #                         ids_dic_tmp_word[pair_now] = []
    #                     else:
    #                         ids_pair_dic[pair_now] += word_counts
    #                     ids_dic_tmp_word[pair_now].append(w)
    #                     ids_pair_dic[(ids_list[i+1], ids_list[i+2])] -= word_counts
    #                     if ids_pair_dic[(ids_list[i+1], ids_list[i+2])] <= 0:
    #                         del ids_pair_dic[(ids_list[i+1], ids_list[i+2])]
    #                 if i - 1 >= 0:
    #                     pair_now = (ids_list[i - 1], next_sym_id)
    #                     if ids_pair_dic.get(pair_now) is None:
    #                         ids_pair_dic[pair_now] = word_counts
    #                         ids_dic_tmp_word[pair_now] = []
    #                     else:
    #                         ids_pair_dic[pair_now] += word_counts
    #                     ids_dic_tmp_word[pair_now].append(w)
    #                     ids_pair_dic[(ids_list[i - 1], ids_list[i])] -= word_counts
    #                     if ids_pair_dic[(ids_list[i - 1], ids_list[i])] <= 0:
    #                         del ids_pair_dic[(ids_list[i - 1], ids_list[i])]
    #                 ids_list[i:i+2] = [next_sym_id]
    #                 words[w] = ids_list

    #     del ids_pair_dic[best_pair]
    #     del ids_dic_tmp_word[best_pair]
        
    #     merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
    
    return vocab, merges

    
file_path = Path("../data") / "TinyStoriesV2-GPT4-valid.txt"
vocab, merges = tokenize("data/TinyStoriesV2-GPT4-train.txt", 10000, None)

for i in range(6666, 6696, 1):
   print(f"{i} : {vocab[i]} \n")


profiler.disable()
profiler.dump_stats('./output.prof')