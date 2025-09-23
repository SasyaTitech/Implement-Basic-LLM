from dataclasses import dataclass
import pickle
from rich import print
from rich.progress import track
from cs336_basics.region_timer import RegionTimer
from cs336_basics.pretokenization import get_pre_token_counts

is_main_file: bool = __name__ == "__main__"


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""

    vocab: dict[int, bytes]  # index -> bytes
    merges: dict[tuple[bytes, bytes], int]  # (bytes1, bytes2) -> new_index
    merges_list: list[tuple[bytes, bytes]]  # (bytes1, bytes2) in order of creation


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


class TokenPairCounter:
    pair_counts: dict[tuple[bytes, bytes], int]
    pair_to_pretokens: dict[tuple[bytes, bytes], set[int]]

    def __init__(self) -> None:
        self.pair_counts = {}
        self.pair_to_pretokens = {}

    def init_from(self, pre_tokens: PreTokens):
        self.pair_counts.clear()
        self.pair_to_pretokens.clear()
        for idx, tokens in enumerate(pre_tokens.all_docs):
            tokens_count = pre_tokens.get_item_count(idx)
            for i in range(len(tokens) - 1):
                pair: tuple[bytes, bytes] = (tokens[i], tokens[i + 1])
                self.add_pair(pair, idx, tokens_count)

    def add_pair(self, pair: tuple[bytes, bytes], doc_idx: int, count: int) -> None:
        if pair not in self.pair_counts:
            self.pair_counts[pair] = 0
            self.pair_to_pretokens[pair] = set()
        self.pair_counts[pair] += count
        self.pair_to_pretokens[pair].add(doc_idx)

    def remove_pair(self, pair: tuple[bytes, bytes], doc_idx: int, count: int) -> None:
        assert pair in self.pair_counts, f"Trying to remove non-existing pair {pair}"

        self.pair_counts[pair] -= count
        if self.pair_counts[pair] <= 0:
            del self.pair_counts[pair]
            del self.pair_to_pretokens[pair]
        else:
            self.pair_to_pretokens[pair].discard(doc_idx)

    def get_max_pair(self) -> tuple[tuple[bytes, bytes], int]:
        if not self.pair_counts:
            return ((b"", b""), 0)
        return max(self.pair_counts.items(), key=lambda x: (x[1], x[0]))

    def get_pretokens(self, pair: tuple[bytes, bytes]) -> set[int]:
        return self.pair_to_pretokens.get(pair, set())

    def get_token_count(self, pair: tuple[bytes, bytes]) -> int:
        return self.pair_counts.get(pair, 0)

    def clear(self):
        self.pair_counts.clear()
        self.pair_to_pretokens.clear()


def update_pair_counts_opt(
    new_tokens_list: list[tuple[bytes, ...]],
    affected_docs_list: list[int],
    pre_tokens: PreTokens,
    token_pair_counter: TokenPairCounter,
):
    for idx, new_tokens in zip(affected_docs_list, new_tokens_list):
        tokens_count = pre_tokens.get_item_count(idx)
        tokens = pre_tokens[idx]

        # remove old pairs
        for i in range(len(tokens) - 1):
            pair: tuple[bytes, bytes] = (tokens[i], tokens[i + 1])
            token_pair_counter.remove_pair(pair, idx, tokens_count)

        # add new pairs
        for i in range(len(new_tokens) - 1):
            pair: tuple[bytes, bytes] = (new_tokens[i], new_tokens[i + 1])
            token_pair_counter.add_pair(pair, idx, tokens_count)


def train(vocab_size: int, pre_tokens_dict: dict[bytes, int], timer: RegionTimer, fast: bool = True) -> BPETokenizerParams:
    timer.start("initialize")
    if not pre_tokens_dict:
        pre_tokens_dict = dict[bytes, int]()
        with open("../data/pretokenization_output.dat", "rb") as in_f:
            pre_tokens_dict = pickle.load(in_f)
    pre_tokens = PreTokens(pre_tokens_dict)

    pre_tokens_count = len(pre_tokens)
    print(f"Loaded {pre_tokens_count} unique pre-tokens")

    bpe_params = BPETokenizerParams(vocab=dict(), merges=dict(), merges_list=[])
    for i in range(256):
        bpe_params.vocab[i] = bytes([i])
    
    assert len(bpe_params.vocab) == 256

    token_pair_counter: TokenPairCounter = TokenPairCounter()
    token_pair_counter.init_from(pre_tokens)
    timer.stop("initialize")

    for iter in track(range(vocab_size)):
        timer.start("find max")
        item: tuple[tuple[bytes, bytes], int] = token_pair_counter.get_max_pair()
        if item[1] == 0:
            print("No more pairs to merge.")
            break
        timer.stop("find max")

        timer.start("add new token")
        merged_tokens: tuple[bytes, bytes] = item[0]
        new_token = merged_tokens[0] + merged_tokens[1]
        new_token_count: int = item[1]

        new_token_idx: int = len(bpe_params.vocab)
        bpe_params.vocab[new_token_idx] = new_token
        
        # add message with this assertion failure
        assert merged_tokens not in bpe_params.merges, \
            f"Merge {merged_tokens} already exists in merges. merge_tokens={merged_tokens} new_token={new_token}"
        bpe_params.merges[merged_tokens] = new_token_idx
        bpe_params.merges_list.append(merged_tokens)
        if bpe_params.vocab and len(bpe_params.vocab) >= vocab_size:
            break

        if iter < 5 or iter % 500 == 99:
            print(
                f"New token: {new_token} with count {new_token_count}.\tnew_token_idx={new_token_idx}"
            )
        timer.stop("add new token")

        # update current_tokens_dict and token_pair_counts
        timer.start("update tokens")
        new_pre_tokens_list: list[tuple[bytes, ...]] = []
        affected_docs_list: list[int] = []
        for idx in token_pair_counter.get_pretokens(merged_tokens):
            old_pre_tokens: tuple[bytes, ...] = pre_tokens[idx]
            new_pre_tokens: list[bytes] = []
            i = 0
            while i < len(old_pre_tokens):
                if i < len(old_pre_tokens) - 1 and (old_pre_tokens[i], old_pre_tokens[i + 1]) == merged_tokens:
                    new_pre_tokens.append(new_token)
                    i += 2
                else:
                    new_pre_tokens.append(old_pre_tokens[i])
                    i += 1

            assert len(new_pre_tokens) != len(old_pre_tokens), f"Token {new_token} not found in pre_tokens"

            new_pre_tokens_list.append(tuple(new_pre_tokens))
            affected_docs_list.append(idx)
        timer.stop("update tokens")

        if fast:
            timer.start("update pairs")
            update_pair_counts_opt(
                new_pre_tokens_list, affected_docs_list, pre_tokens, token_pair_counter
            )
            assert token_pair_counter.get_token_count(new_token) == 0
            timer.stop("update pairs")

        timer.start("update pre_tokens")
        for idx, new_tokens_tuple in zip(affected_docs_list, new_pre_tokens_list):
            pre_tokens[idx] = new_tokens_tuple
        timer.stop("update pre_tokens")

        if not fast:
            timer.start("rebuild pairs")
            token_pair_counter.init_from(pre_tokens)
            count: int = token_pair_counter.get_token_count(merged_tokens)
            assert count == 0, f"After rebuilding, {merged_tokens} has count {count}, expected 0"
            timer.stop("rebuild pairs")
    
    assert len(bpe_params.vocab) <= vocab_size
    return bpe_params


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> BPETokenizerParams:
    timer = RegionTimer()
    timer.start("get pre-token counts")
    pre_tokens_dict: dict[bytes, int] = get_pre_token_counts(input_path)
    timer.stop("get pre-token counts")

    bpe_params: BPETokenizerParams = train(
        vocab_size - len(special_tokens), pre_tokens_dict, timer, fast=True
    )
    # Add special tokens at the end of the vocab
    for token in special_tokens:
        token = token.encode("utf-8")
        bpe_params.vocab[len(bpe_params.vocab)] = token
    print(f"Final vocab size: {len(bpe_params.vocab)}")
    if is_main_file:
        timer.report()
    return bpe_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["native", "fast"],
        help="Mode to run the BPE training",
    )
    args = parser.parse_args()
    train_bpe("data/TinyStoriesV2-GPT4-train.txt", 500, ["<|endoftext|>"])
    # vocab_size = 10000
    # special_tokens = ["<|endoftext|>"]
    # vocab, merges = train_bpe("../data/TinyStoriesV2-GPT4-train.txt", vocab_size, special_tokens)
    # print(f"Trained BPE with vocab size {len(vocab)} and {len(merges)} merges")
