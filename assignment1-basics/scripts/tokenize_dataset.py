"""Encode a text corpus into a flat binary token-ID file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cs336_basics.bpe import Tokenizer


DTYPES = {
    "uint16": np.uint16,
    "int32": np.int32,
    "int64": np.int64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 input text file")
    parser.add_argument("--output", type=Path, required=True, help="Output .bin file")
    parser.add_argument("--vocab", type=Path, required=True, help="Base64 JSON vocabulary")
    parser.add_argument("--merges", type=Path, required=True, help="Base64 JSON merge rules")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="uint16")
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=[],
        help="Special token to preserve; repeat this option to add more than one",
    )
    parser.add_argument("--write-chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.write_chunk_size <= 0:
        parser.error("--write-chunk-size must be positive")
    return args


def tokenize_file(args: argparse.Namespace) -> None:
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)
    dtype = np.dtype(DTYPES[args.dtype])
    dtype_max = np.iinfo(dtype).max
    max_token_id = max(tokenizer.vocab, default=-1)
    if max_token_id > dtype_max:
        raise ValueError(
            f"Tokenizer ID {max_token_id} does not fit in {args.dtype}; choose a wider --dtype"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    buffer: list[int] = []
    with open(args.input, encoding="utf-8") as input_file, open(args.output, "wb") as output_file:
        for token_id in tokenizer.encode_iterable(input_file):
            buffer.append(token_id)
            if len(buffer) >= args.write_chunk_size:
                np.asarray(buffer, dtype=dtype).tofile(output_file)
                token_count += len(buffer)
                buffer.clear()
                print(f"encoded {token_count:,} tokens", flush=True)

        if buffer:
            np.asarray(buffer, dtype=dtype).tofile(output_file)
            token_count += len(buffer)

    print(f"encoded {token_count:,} tokens from {args.input}")
    print(f"saved {args.output} as {args.dtype}")


if __name__ == "__main__":
    tokenize_file(parse_args())
