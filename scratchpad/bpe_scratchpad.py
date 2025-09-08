from collections import Counter
import re

input_text = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""

# Steps:
# 1. Pre-tokenization -> split on whitespace (check the implementation in the .py file for more details)
# 2. 


if __name__ == "__main__":
    # print(Counter(input_text.split(" ", "\n")))
    pretokenized_dict = Counter(re.split(r'[ \n]+',input_text))
    print(pretokenized_dict.keys())

    
    