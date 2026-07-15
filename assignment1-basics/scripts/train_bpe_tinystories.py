from __future__ import annotations

import argparse
import base64
import json
import platform
import resource
import time
from pathlib import Path

from cs336_basics.bpe import tokenizer


def bytes_to_b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def save_vocab(vocab: dict[int, bytes], out_path: Path) -> None:
    payload = {str(token_id): bytes_to_b64(token_bytes) for token_id, token_bytes in vocab.items()}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_merges(merges: list[tuple[bytes, bytes]], out_path: Path) -> None:
    payload = [[bytes_to_b64(left), bytes_to_b64(right)] for left, right in merges]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_peak_rss_mb() -> float:
    # ru_maxrss is KiB on Linux and bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on TinyStories.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
        help="Path to the TinyStories training corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tinystories_bpe"),
        help="Directory used to store the trained vocabulary, merges, and profiling stats.",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Maximum vocabulary size.")
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=["<|endoftext|>"],
        help="Special token to add to the vocabulary. Repeat to add more.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    vocab, merges = tokenizer(str(args.input), args.vocab_size, args.special_tokens)
    elapsed = time.perf_counter() - start_time
    peak_rss_mb = get_peak_rss_mb()

    save_vocab(vocab, args.output_dir / "vocab.json")
    save_merges(merges, args.output_dir / "merges.json")

    longest_token_id, longest_token = max(vocab.items(), key=lambda item: (len(item[1]), item[0]))
    longest_token_preview = longest_token.decode("utf-8", errors="replace")

    stats = {
        "input": str(args.input),
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "elapsed_seconds": elapsed,
        "peak_rss_mb": peak_rss_mb,
        "longest_token_id": longest_token_id,
        "longest_token_len": len(longest_token),
        "longest_token_utf8_preview": longest_token_preview,
        "special_tokens": args.special_tokens,
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"trained vocab size: {len(vocab)}")
    print(f"merge count: {len(merges)}")
    print(f"elapsed: {elapsed:.2f}s")
    print(f"peak rss: {peak_rss_mb:.1f} MB")
    print(f"longest token id: {longest_token_id}")
    print(f"longest token len: {len(longest_token)}")
    print(f"longest token preview: {longest_token_preview!r}")
    print(f"saved to: {args.output_dir}")


if __name__ == "__main__":
    main()