import os
import pickle
from pretokenization import find_chunk_boundaries, split_pattern_compiled


def process_chunk(idx: int, start: int, end: int, file_path: str) -> list[str]:
    all_docs = []
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk: str = f.read(end - start).decode("utf-8")
        print(f"Processing chunk from {start} to {end}, size {len(chunk)} bytes")
        for doc in split_pattern_compiled.splititer(chunk):
            all_docs.append(doc)
            if len(all_docs) >= 10:
                break
    return all_docs


def sample(file_path: str):
    num_processes: int = 100
    with open(file_path, "rb") as f:
        # get file size
        f.seek(0, os.SEEK_END)
        file_size = f.tell() / 1024 / 1024  # in MB
        f.seek(0)
        print(f"File size: {file_size} MB")
        num_processes = max(num_processes, int(file_size // 400) + 1)
        print(f"Using {num_processes} processes")
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        print(f"Found {len(boundaries)-1} chunks")

    all_docs = []
    for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        docs = process_chunk(idx, start, end, file_path)
        all_docs.extend(docs)

    data_dir = os.path.dirname(args.file)
    file_name = os.path.basename(args.file).replace(".txt", "")
    print(f"Total documents sampled from {file_name}: {len(all_docs)}")
    with open(os.path.join(data_dir, f"{file_name}-sample.pkl"), "wb") as f:
        pickle.dump(all_docs, f)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to the input text file.",
    )
    args = parser.parse_args()
    now = time.time()
    sample(args.file)
    print(f"Time taken: {time.time() - now} seconds")
