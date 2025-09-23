import regex as re
import os
from typing import BinaryIO
from rich.progress import track
from rich import print
import pickle

word_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
special_tokens = ["<|endoftext|>"]
special_tokens = list(map(lambda x: re.escape(x), special_tokens))
split_pattern = "|".join(special_tokens)
word_count_dict = dict[bytes, int]()


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_doc(doc: str) -> int:
    count = 0
    for m in re.finditer(word_pattern, doc):
        word = m.group()
        key = word.encode("utf-8")
        count += 1
        word_count_dict[key] = word_count_dict.get(key, 0) + 1
    return count


def process_chunk(chunk: str) -> int:
    count = 0
    documents = re.split(split_pattern, chunk)
    print(f"Processing {len(documents)} documents in chunk")
    for doc in track(documents):
        count += process_doc(doc)
    return count


if __name__ == "__main__":
    with open("../data/TinyStoriesV2-GPT4-train.txt", "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        print(f"Found {len(boundaries)-1} chunks")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            print(f"Processing bytes {start} to {end}, chunk size {len(chunk)}")
            process_chunk(chunk)
            print(f"Total unique pre-tokens so far: {len(word_count_dict)}")
    
    with open("../data/pretokenization_output.dat", "wb") as out_f:
        # Save the pre-token counts to a file, use simple binary serialization
        pickle.dump(word_count_dict, out_f)

    print(f"Total unique pre-tokens: {len(word_count_dict)}")