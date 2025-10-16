from pathlib import Path
from pre_tokenizer import pre_tokenize

def tokenize(input_path: str, vocab_size : int, special_tokens : list[str]):
    file_path = Path(str)
    word_list = pre_tokenize(file_path)