from dataclasses import dataclass
import gc
import multiprocessing
import os
import pickle
import regex as re
from typing import Iterable, Iterator, Self
from cs336_basics.pretokenization import convert_special_token_to_regex, find_chunk_boundaries, word_pattern_compiled, global_special_tokens, split_pattern_compiled
from cs336_basics.region_timer import ContextTimer, RegionTimer
from cs336_basics.token_pair import Pair, PairIndex, PairIndexState
from rich.live import Live
from rich.text import Text
import numpy as np

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
        for i, string in enumerate(iterable):
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


def generate_docs(idx: int, start: int, end: int, file_path: str) -> Iterator[str]:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk: str = f.read(end - start).decode("utf-8")
    for doc in split_pattern_compiled.splititer(chunk):
        yield doc

def process_chunk(idx: int, file_path: str, start: int, end: int, tokenizer: BPETokenizer) -> int:
    # check np file exists
    output_file = file_path.replace(".txt", f"-bpe-{idx}.npy")
    if os.path.exists(output_file):
        print(f"Chunk {idx+1} already processed, skipping")
        return 0
    now = time.time()
    token_list: list[np.uint16] = []
    chunk_size = end - start
    token_count: int = 0
    for doc in generate_docs(idx, start, end, file_path):
        if len(doc) == 0:
            continue
        indices = tokenizer.encode(doc)
        indices.append(tokenizer.token_to_index[b"<|endoftext|>"])
        token_count += len(indices)
        token_list.extend(np.array(indices, dtype=np.uint16))
    time_taken = time.time() - now
    compress_ratio = chunk_size / token_count if token_count > 0 else 0
    bytes_per_sec = chunk_size / time_taken / 1024 / 1024
    print(f"Processed chunk {idx+1}, time_taken {time_taken:.2f} seconds, {bytes_per_sec:.2f} MB/s compression ratio {compress_ratio:.2f}")
    np_data = np.array(token_list, dtype=np.uint16)
    np.save(output_file, np_data)
    print(f"Saved {len(np_data) / 1000 / 1000:.2f}M tokens to {output_file}")
    return np_data.shape[0]

def process_file_multi_process(file_path: str, tokenizer: BPETokenizer) -> None:
    from rich.progress import track
    now = time.time()
    total_bytes: int = 0
    with open(file_path, "rb") as f:
        # get file size
        f.seek(0, os.SEEK_END)
        total_bytes = f.tell()
        file_size = total_bytes / 1024 / 1024  # in MB
        f.seek(0)
        print(f"File size: {file_size:.2f} MB")
        num_processes = max(10, int(file_size // 20) + 1)
        print(f"Using {num_processes} processes")
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        print(f"Found {len(boundaries)-1} chunks")

    token_count: int = 0
    with multiprocessing.Pool(processes=10) as pool:
        all_token_lists = pool.starmap(
            process_chunk,
            [(t, file_path, start, end, tokenizer) for t, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))],
        )
        for c in all_token_lists:
            token_count += c
    time_taken = time.time() - now
    compress_ratio: float = total_bytes / token_count
    bytes_per_sec = total_bytes / time_taken / 1024 / 1024
    print("-" * 80)
    print(f"Generate {token_count / 1000 / 1000:.2f}M tokens, time_taken {time_taken:.2f} seconds, {bytes_per_sec:.2f} MB/s, compression ratio {compress_ratio:.2f}")
    return

def process_file(file_path: str, tokenizer: BPETokenizer) -> None:
    from rich.progress import track
    now = time.time()
    total_bytes: int = 0
    sample_size = 250_000
    with open(file_path, "rb") as f:
        # get file size
        f.seek(0, os.SEEK_END)
        total_bytes = f.tell()
        file_size = total_bytes / 1024 / 1024  # in MB
        f.seek(0)
        print(f"File size: {file_size:.2f} MB")
        sample_bytes = f.read(sample_size)
    sample_str = sample_bytes.decode("utf-8", errors="replace")
    sample_tokens = tokenizer.encode(sample_str)
    sample_compress_ratio = get_compression_ratio(sample_str, sample_tokens)
    print(f"Sampled {sample_size} bytes, got {len(sample_tokens)} tokens, compression ratio {sample_compress_ratio:.2f}")

    approx_total_tokens = int(total_bytes / sample_compress_ratio)
    print(f"Estimated total tokens: {approx_total_tokens / 1_000_000:.2f}M")
    print('-' * 80)
    output_file = file_path.replace(".txt", f"-bpe.npy")
    mm = np.memmap(output_file, dtype=np.uint16, mode="w+", shape=(int(approx_total_tokens * 1.15),))
    idx = 0
    with open(file_path, "r") as f:
        for token_id in track(tokenizer.encode_iterable(f), total=approx_total_tokens, description="Tokenizing"):
            mm[idx] = token_id
            idx += 1
    time_taken = time.time() - now
    bytes_per_sec = total_bytes / time_taken / 1024 / 1024
    compress_ratio = total_bytes / idx
    print(f"Time taken: {time_taken:.2f} seconds, {bytes_per_sec:.2f} MB/s, compression ratio {compress_ratio:.2f}")
    print(f"Generated {idx / 1000 / 1000:.2f}M tokens, saved to {output_file}")
    mm.flush()
    del mm
    return


if is_main_file:
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to the input text file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not actually run the tokenizer, just print the first 10 documents.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="",
        help="If set, run a test on the given string.",
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        default=True,
        help="If set, run in single-process mode.",
    )

    arg = parser.parse_args()

    now = time.time()
    file_path = arg.file

    # Load the tokenizer
    file_prefix = file_path.replace("train.txt", "train")
    file_prefix = file_prefix.replace("valid.txt", "train")
    vocab_file: str = f"{file_prefix}-vocab.dat"
    merges_file: str = f"{file_prefix}-merges.dat"
    print(f"Loading tokenizer from {vocab_file} and {merges_file}")
    tokenizer = BPETokenizer.from_files(vocab_file, merges_file, special_tokens=["<|endoftext|>"])

    if arg.dry_run:
        sum_compress_ratio: float = 0.0
        sample_file: str = f"{file_prefix}-sample.pkl"
        with open(sample_file, "rb") as f:
            all_docs: list[str] = pickle.load(f)
        print(f"Loaded {len(all_docs)} documents from {sample_file}")
        total_bytes: int = 0
        token_count: int = 0
        for doc in all_docs[:10]:
            indices = tokenizer.encode(doc)
            doc_bytes = doc.encode("utf-8")
            total_bytes += len(doc_bytes)
            token_count += len(indices)
            decoded = tokenizer.decode(indices)
            assert doc == decoded, f"Decoded document does not match original. Original: {doc}, Decoded: {decoded}"
        print(f"Total time: {time.time() - now:.2f} seconds compression ratio {total_bytes / token_count:.2f}")
        tokenizer.timer.report()
    elif arg.single_process:
        file_path = arg.test if arg.test else arg.file
        process_file(file_path, tokenizer)
    else:
        file_path = arg.test if arg.test else arg.file
        process_file_multi_process(file_path, tokenizer)

