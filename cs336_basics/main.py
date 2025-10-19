from Tokenizer import Tokenizer

def test_tokenizer():
    file_path = "data/TinyStoriesV2-GPT4-train.txt"

    bpe_tokenizer = Tokenizer(file_path, 10000, ["<|endoftext|>"], 24)
    vocab = bpe_tokenizer.train()

    key, longest_token = max(vocab.items(), key = lambda kv : len(kv[1]))
    print(f"The longest word in Vocab is \"{longest_token.decode("utf-8")}\"")

    ids = bpe_tokenizer.encode("hello, finish building your encoder is a great accomplishment~ Congrat!")
    print(ids)
    print(bpe_tokenizer.decode(ids))


if __name__ == "__main__": # 启动哨兵,只有在作为脚本启动的那次执行，被子进程之类导入的时候不执行

    test_tokenizer()