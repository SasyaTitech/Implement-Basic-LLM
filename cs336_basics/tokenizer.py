from pathlib import Path
from pre_tokenizer import pre_tokenize
from collections import Counter
import cProfile
from tqdm.auto import tqdm

profiler = cProfile.Profile()
profiler.enable()

def tokenize(input_path: str, vocab_size : int , special_tokens : list[str]):
    file_path = Path(input_path)
    pre_tokenize_counts = pre_tokenize(file_path)

    vocab = {i: bytes([i]) for i in range(256)} # ids : bytes
    words = {w: [int(b) for b in list(w.encode("utf-8"))] for w in pre_tokenize_counts}
    # words : [ids1, ids2, ids3, ... , idsn]
    next_sym_id_start = 256
    merges = []

    # 第一遍是全部扫描
    ids_pair_dic = {} # (ids1, ids2) : count
    ids_dic_tmp_word = {}
    for w, ids_list in words.items(): 
        if(len(ids_list) < 2):
            continue
        for i in reversed(range(len(ids_list) - 1)):
            pair_now = (ids_list[i], ids_list[i + 1]) # (ids1, ids2)
            if ids_pair_dic .get(pair_now) is None:
                ids_pair_dic[pair_now] = pre_tokenize_counts[w]
                ids_dic_tmp_word[pair_now] = []
                ids_dic_tmp_word[pair_now].append(w)
            else:
                ids_pair_dic[pair_now] += pre_tokenize_counts[w]
                ids_dic_tmp_word[pair_now].append(w)
    
    best_pair, _ = max(ids_pair_dic.items(), key = lambda kv : (kv[1], kv[0][0], kv[0][1]))
    vocab[next_sym_id_start] = vocab[best_pair[0]] + vocab[best_pair[1]]

    for w in ids_pair_dic[best_pair]:
        ids_list = words[w]
        word_counts = pre_tokenize_counts[w]
        for i in reversed(range(len(ids_list) - 1)):
            if(ids_list[i], ids_list[i + 1]) == best_pair:
                if i + 2 < len(ids_list) :
                    pair_now = (next_sym_id_start, ids_list[i + 2])
                    if ids_pair_dic.get(pair_now) is None:
                        ids_pair_dic[pair_now] = word_counts
                        ids_dic_tmp_word[pair_now] = []
                    else:
                        ids_pair_dic[pair_now] += word_counts
                    ids_dic_tmp_word[pair_now].append(w)
                    ids_pair_dic[(ids_list[i+1], ids_list[i+2])] -= word_counts
                    if ids_pair_dic[(ids_list[i+1], ids_list[i+2])] <= 0:
                        del ids_pair_dic[(ids_list[i+1], ids_list[i+2])]
                if i - 1 >= 0:
                    pair_now = (ids_list[i - 1], next_sym_id_start)
                    if ids_pair_dic.get(pair_now) is None:
                        ids_pair_dic[pair_now] = word_counts
                        ids_dic_tmp_word[pair_now] = []
                    else:
                        ids_pair_dic[pair_now] += word_counts
                    ids_dic_tmp_word[pair_now].append(w)
                    ids_pair_dic[(ids_list[i - 1], ids_list[i])] -= word_counts
                    if ids_pair_dic[(ids_list[i - 1], ids_list[i])] <= 0:
                        del ids_pair_dic[(ids_list[i - 1], ids_list[i])]
                ids_list[i:i+2] = [next_sym_id_start]
                words[w] = ids_list
    del ids_pair_dic[best_pair]
    del ids_dic_tmp_word[best_pair]

    merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

    for next_sym_id in tqdm(range(next_sym_id_start + 1, vocab_size + 1)):
        best_pair, _ = max(ids_pair_dic.items(), key = lambda kv : (kv[1], kv[0][0], kv[0][1]))
        vocab[next_sym_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        for w in ids_pair_dic[best_pair]:
            ids_list = words[w]
            word_counts = pre_tokenize_counts[w]
            for i in reversed(range(len(ids_list) - 1)):
                if(ids_list[i], ids_list[i + 1]) == best_pair:
                    if i + 2 < len(ids_list) :
                        pair_now = (next_sym_id, ids_list[i + 2])
                        if ids_pair_dic.get(pair_now) is None:
                            ids_pair_dic[pair_now] = word_counts
                            ids_dic_tmp_word[pair_now] = []
                        else:
                            ids_pair_dic[pair_now] += word_counts
                        ids_dic_tmp_word[pair_now].append(w)
                        ids_pair_dic[(ids_list[i+1], ids_list[i+2])] -= word_counts
                        if ids_pair_dic[(ids_list[i+1], ids_list[i+2])] <= 0:
                            del ids_pair_dic[(ids_list[i+1], ids_list[i+2])]
                    if i - 1 >= 0:
                        pair_now = (ids_list[i - 1], next_sym_id)
                        if ids_pair_dic.get(pair_now) is None:
                            ids_pair_dic[pair_now] = word_counts
                            ids_dic_tmp_word[pair_now] = []
                        else:
                            ids_pair_dic[pair_now] += word_counts
                        ids_dic_tmp_word[pair_now].append(w)
                        ids_pair_dic[(ids_list[i - 1], ids_list[i])] -= word_counts
                        if ids_pair_dic[(ids_list[i - 1], ids_list[i])] <= 0:
                            del ids_pair_dic[(ids_list[i - 1], ids_list[i])]
                    ids_list[i:i+2] = [next_sym_id]
                    words[w] = ids_list

        del ids_pair_dic[best_pair]
        del ids_dic_tmp_word[best_pair]
        
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
    
    return vocab, merges

    
file_path = Path("../data") / "TinyStoriesV2-GPT4-valid.txt"
vocab, merges = tokenize("data/TinyStoriesV2-GPT4-train.txt", 10000, None)

for i in range(230, 350, 1):
   print(f"{i} : {vocab[i].decode("utf-8")} \n")


profiler.disable()
profiler.dump_stats('./output.prof')