from dataclasses import dataclass
import pickle
from typing import Iterable, Iterator, Self
from cs336_basics.token_pair_counter import Pair, PairIndex


is_main_file: bool = __name__ == "__main__"


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""

    vocab: dict[int, bytes]
    merges: dict[Pair, int]
    merges_list: list[Pair]


class Tokenizer:
    """Abstract base class for tokenizers."""

    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""

    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""

    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")
        indices = list(map(int, string_bytes))
        return indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)
        string = string_bytes.decode("utf-8")
        return string


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""

    vocab: dict[int, bytes]
    merges: list[Pair]
    merges_index_list: list[tuple[PairIndex, int]]

    def __init__(self, vocab: dict[int, bytes], merges: list[Pair], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        assert len(merges) == len(vocab) - 256 - 1, f"Expected {len(vocab)-256-1} merges, got {len(merges)}"

        for token_str in special_tokens or []:
            token: bytes = token_str.encode("utf-8")
            if token in self.vocab:
                continue
            self.vocab[len(self.vocab)] = token

        token_to_index = {v: k for k, v in vocab.items()}
        self.merges_index_list = []
        for t, merge in enumerate(merges):
            index = t + 256
            pair_left_index = token_to_index[merge[0]]
            pari_right_index = token_to_index[merge[1]]
            merge_index = (pair_left_index, pari_right_index)
            self.merges_index_list.append((merge_index, index))

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Self:
        vocab: dict[int, bytes] = dict()
        with open(vocab_filepath, "rb") as vocab_f:
            vocab = pickle.load(vocab_f)

        merges: list[Pair] = []
        with open(merges_filepath, "rb") as merges_f:
            merges = pickle.load(merges_f)

        return cls(vocab, merges, special_tokens)

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))
        if len(indices) < 2:
            return indices
        # Note: this is a very slow implementation
        for pair, new_index in self.merges_index_list:
            indices = self.merge(indices, pair, new_index)
            if len(indices) < 2:
                return indices
        return indices

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)

    def decode(self, indices: list[int]) -> str:
        bytes_list: list[bytes] = []
        for i in indices:
            if i not in self.vocab:
                raise ValueError(f"Index {i} not in vocabulary")
            bytes_list.append(self.vocab[i])
        string = b"".join(bytes_list).decode("utf-8", errors="replace")
        return string

    def merge(self, indices: list[int], pair: PairIndex, new_index: int) -> list[int]:
        """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
        new_indices = []
        i = 0
        while i < len(indices):
            if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
                new_indices.append(new_index)
                i += 2
            else:
                new_indices.append(indices[i])
                i += 1
        return new_indices


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`, ."""
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens


if is_main_file:
    string = "Hello, 🌍! 你好!"
    tokenizer = ByteTokenizer()
    indices = tokenizer.encode(string)
    reconstructed_string = tokenizer.decode(indices)
    assert string == reconstructed_string
    compression_ratio = get_compression_ratio(string, indices)
    print(f"Byte tokenizer Compression ratio: {compression_ratio:.2f}")

    demo_vocab_file: str = "data/TinyStoriesV2-GPT4-valid-vocab.dat"
    demo_merges_file: str = "data/TinyStoriesV2-GPT4-valid-merges.dat"
    tokenizer = BPETokenizer.from_files(demo_vocab_file, demo_merges_file, special_tokens=["<|endoftext|>"])
    demo = "s"
    indices = tokenizer.encode(demo)
    print(f"BPE tokenizer encoded '{demo}' to indices: {indices}")
    reconstructed_demo = tokenizer.decode(indices)  # should not raise
    assert demo == reconstructed_demo, f"Expected '{demo}', got '{reconstructed_demo}'"
