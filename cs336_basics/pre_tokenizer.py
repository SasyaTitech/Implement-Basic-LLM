from pretokenization_example import find_chunk_boundaries
from pathlib import Path
import regex
from collections import Counter
import cProfile
import multiprocessing as mp

# profiler = cProfile.Profile()
# profiler.enable()

def pre_tok_chunk(chunk):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    chunk = chunk.replace("<|endoftext|>", "\n")
    word_list = regex.findall(PAT, chunk)
    word_one_chunk = Counter(word_list)
    return word_one_chunk


def pre_tokenize(file_path : Path, num_processes : int = 24):
    word_list = Counter()
    chunk_list = []
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]): # 拉链起来 [0, 5, 12] -> [[0, 5], [5, 12]]
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # print(chunk)
            chunk_list.append([chunk])

    with mp.Pool(num_processes) as pool:
        result = pool.starmap(pre_tok_chunk, chunk_list)
        for r in result:
            word_list += r
    
    return word_list



# file_path = Path("../data") / "TinyStoriesV2-GPT4-valid.txt"
# word_list = pre_tokenize(file_path)

# # print(len(word_list))
# print(word_list.most_common(100))
# print(len(word_list))

# profiler.disable()
# profiler.dump_stats('./output.prof')