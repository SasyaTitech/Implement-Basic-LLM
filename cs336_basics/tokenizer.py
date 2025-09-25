from collections import defaultdict
from dataclasses import dataclass
import os
import pickle
import regex as re
from typing import Iterable, Iterator, Self
from cs336_basics.pretokenization import convert_special_token_to_regex, word_pattern_compiled, global_special_tokens
from cs336_basics.region_timer import ContextTimer, RegionTimer
from cs336_basics.token_pair import Pair, PairIndex, PairIndexState
from rich.live import Live
from rich.text import Text
import heapq

from tests.common import FIXTURES_PATH

is_main_file: bool = __name__ == "__main__"
logging_enabled: bool = False and is_main_file


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
    merges_index_dict: dict[PairIndex, int]
    token_to_index: dict[bytes, int]
    special_tokens: list[str] | None
    pattern_compiled: re.Pattern[str]
    timer: RegionTimer

    def __init__(self, vocab: dict[int, bytes], merges: list[Pair], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or global_special_tokens
        self.special_tokens = sorted(self.special_tokens, reverse=True)
        sp_tokens_count: int = len(self.special_tokens)
        assert (
            len(merges) == len(vocab) - 256 - sp_tokens_count
        ), f"Expected {len(vocab)-256-sp_tokens_count} merges, got {len(merges)}"

        self.token_to_index = dict()
        for token_index, token in self.vocab.items():
            self.token_to_index[token] = token_index

        self.pattern_compiled = convert_special_token_to_regex(self.special_tokens or [])
        for token_str in special_tokens or []:
            token: bytes = token_str.encode("utf-8")
            if token in self.token_to_index:
                continue
            print(f"Adding special token {token} to vocabulary at index {len(self.vocab)}")
            index = len(self.vocab)
            self.vocab[index] = token
            self.token_to_index[token] = index

        token_to_index = {v: k for k, v in vocab.items()}
        self.merges_index_list = []
        self.merges_index_dict = {}
        for t, merge in enumerate(merges):
            index = t + 256
            pair_left_index = token_to_index[merge[0]]
            pari_right_index = token_to_index[merge[1]]
            merge_index = (pair_left_index, pari_right_index)
            self.merges_index_list.append((merge_index, index))
            self.merges_index_dict[merge_index] = index

        self.timer = RegionTimer()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Self:
        vocab: dict[int, bytes] = dict()
        with open(vocab_filepath, "rb") as vocab_f:
            vocab = pickle.load(vocab_f)

        merges: list[Pair] = []
        with open(merges_filepath, "rb") as merges_f:
            merges = pickle.load(merges_f)

        return cls(vocab, merges, special_tokens)

    def start_timer(self, region_name: str) -> None:
        if not is_main_file:
            return
        self.timer.start(region_name)

    def stop_timer(self, region_name: str) -> None:
        if not is_main_file:
            return
        self.timer.stop(region_name)

    def _convert_simple_string_to_indices(self, string: str) -> Iterable[list[int]]:
        if logging_enabled:
            print(f"Converting simple string to indices: '{string}'")
        indices = []
        for m in word_pattern_compiled.finditer(string):
            with ContextTimer(self.timer, "Get byte token indices", is_main_file):
                indices.clear()
                word = m.group()
                pre_tokens: bytes = word.encode("utf-8")
                for token_b in pre_tokens:
                    token: bytes = bytes([token_b])
                    index: int = self.token_to_index[token]
                    indices.append(index)
            yield indices

    def _convert_to_indices(self, string: str) -> Iterable[list[int]]:
        start: int = 0
        end: int = 0
        if is_main_file:
            self.timer.start("Convert to indices")
        for ma in self.pattern_compiled.finditer(string):
            start = ma.start()
            special_token: str = ma.group()
            if logging_enabled:
                print(f"Found special token '{special_token}' at {start}-{ma.end()}")
            if start > end:
                segment = string[end:start]
                yield from self._convert_simple_string_to_indices(segment)

            with ContextTimer(self.timer, "Get special token index", is_main_file):
                # Add the special token itself
                token: bytes = special_token.encode("utf-8")
                index: int = self.token_to_index[token]
            yield [index]
            end = ma.end()
        if is_main_file:
            self.timer.stop("Convert to indices")

        # Process any remaining text after the last special token
        remaining = string[end:]
        if not remaining:
            return
        yield from self._convert_simple_string_to_indices(remaining)

    def encode(self, string: str) -> list[int]:
        all_indices: list[int] = []
        for indices in self._convert_to_indices(string):
            with ContextTimer(self.timer, "Merging indices", is_main_file):
                merged_indices = self._merge_indices(indices)
                all_indices.extend(merged_indices)
        return all_indices

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        with Live() as live:
            for i, string in enumerate(iterable):
                live.update(Text(f"{i}: {string}"))
                for indices in self._convert_to_indices(string):
                    with ContextTimer(self.timer, "Merging indices", is_main_file):
                        yield from self._merge_indices(indices)

    def decode(self, indices: list[int]) -> str:
        bytes_list: list[bytes] = []
        for i in indices:
            assert i in self.vocab, f"Index {i} not in vocabulary"
            bytes_list.append(self.vocab[i])
        byte_string = b"".join(bytes_list)
        string = byte_string.decode("utf-8", errors="replace")
        return string

    def _merge_indices(self, indices: list[int]) -> list[int]:
        if len(indices) < 2:
            return indices
        
        pair_state: PairIndexState = PairIndexState(self.merges_index_dict, self.timer, is_main_file)
        pair_state.rebuild(indices)

        while True:
            pair, new_index = pair_state.get_next()
            if new_index < 0:
                break
            indices = self._merge(indices, pair, new_index, pair_state)
            if len(indices) < 2:
                return indices
        return indices

    def _merge(
        self, indices: list[int], pair: PairIndex, new_index: int, pair_state: PairIndexState
    ) -> list[int]:
        """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
        count: int = pair_state.count(pair)
        if count == 0:
            return indices
        i: int = 0
        last_found_i: int = -1
        new_indices: list[int] = []
        new_pair_indices: list[int] = []
        # print(f"pair = {pair} indices {indices} pair_state {dict(pair_state)}")
        while i < len(indices):
            if i + 1 >= len(indices) or len(new_pair_indices) >= count:
                new_indices.append(indices[i])
                i += 1
                continue
            if indices[i] != pair[0] or indices[i + 1] != pair[1]:
                new_indices.append(indices[i])
                i += 1
                continue

            pair_state.remove(pair)
            if i > 0 and last_found_i < 0 or i - 1 > last_found_i + 1:
                prev_pair = (indices[i - 1], indices[i])
                pair_state.remove(prev_pair)

            if i + 2 < len(indices):
                next_pair = (indices[i + 1], indices[i + 2])
                pair_state.remove(next_pair)

            last_found_i = i
            new_pair_indices.append(len(new_indices))
            new_indices.append(new_index)
            i += 2

        if is_main_file and len(new_pair_indices) == 0:
            # print(f"Warning: no pairs found for {pair} in {indices}. indices_pair_state: {dict(pair_state)}")
            return indices

        for idx in new_pair_indices:
            if idx > 0:
                pair_index = new_indices[idx - 1], new_indices[idx]
                pair_state.add(pair_index)
            if idx + 1 < len(new_indices):
                pair_index = new_indices[idx], new_indices[idx + 1]
                pair_state.add(pair_index)
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
    # demo = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    # indices = tokenizer.encode(demo)
    # print(f"BPE tokenizer encoded '{demo}' to indices: {indices}")
    # reconstructed_demo = tokenizer.decode(indices)  # should not raise
    # assert demo == reconstructed_demo, f"Expected '{demo}', got '{reconstructed_demo}'"

    # from tests.test_tokenizer import MERGES_PATH, VOCAB_PATH, get_tokenizer_from_vocab_merges_path
    # tokenizer = get_tokenizer_from_vocab_merges_path(
    #     vocab_path=VOCAB_PATH,
    #     merges_path=MERGES_PATH,
    # )
    # test_string = "s"
    # encoded_ids = tokenizer.encode(test_string)
    # print(f"Encoded IDs for '{test_string}': {encoded_ids}")
    # decoded_string = tokenizer.decode(encoded_ids)
    # assert test_string == decoded_string, f"Expected '{test_string}', got '{decoded_string}'"

    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        ids = []
        for _id in tokenizer.encode_iterable(f):
            ids.append(_id)
    tokenizer.timer.report()
