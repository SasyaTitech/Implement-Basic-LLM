import pickle
import difflib
from pretokenization import pretokenize_pattern_compiled

file_01 = "chunk-1-pre-tokens_1.pkl"
file_02 = "chunk-1-pre-tokens_2.pkl"

with open(file_01, "rb") as f:
    data_01: list[str] = pickle.load(f)

with open(file_02, "rb") as f:
    data_02: list[str] = pickle.load(f)

print(f"Data from {file_01}: {len(data_01)}")
print(f"Data from {file_02}: {len(data_02)}")
print(f"delta: {len(data_01) - len(data_02)}")
max_count = 50000

diff = difflib.unified_diff(
    data_01[:max_count],
    data_02[:max_count+37],
    fromfile="data_01",
    tofile="data_02",
    lineterm=''
)

# minus_count = 0
# plus_count = 0
# for line in diff:
#     if line.startswith('+'):
#         plus_count += 1
#         print(line.encode('utf-8'))
#     elif line.startswith('-'):
#         minus_count += 1
#         print(line.encode('utf-8'))

# print(f"minus_count: {minus_count}, plus_count: {plus_count}")

ret = pretokenize_pattern_compiled.findall("     \nHello,   world!  \n")
print(ret)
        
