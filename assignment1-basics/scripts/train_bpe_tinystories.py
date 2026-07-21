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
    """把任意字节串无损转换为可写入 JSON 的 ASCII 字符串。"""
    # BPE token 不保证是合法 UTF-8，不能直接 decode；Base64 可以保留每一个原始字节。
    return base64.b64encode(value).decode("ascii")


def save_vocab(vocab: dict[int, bytes], out_path: Path) -> None:
    """序列化 token_id -> token bytes 词表。"""
    # JSON 对象的键必须是字符串；读取时可用 int(key) 恢复 token ID。
    payload = {str(token_id): bytes_to_b64(token_bytes) for token_id, token_bytes in vocab.items()}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_merges(merges: list[tuple[bytes, bytes]], out_path: Path) -> None:
    """按学习顺序序列化 BPE merge 规则。"""
    # merges 的先后顺序就是规则优先级，因此使用列表而不是无序映射。
    payload = [[bytes_to_b64(left), bytes_to_b64(right)] for left, right in merges]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_peak_rss_mb() -> float:
    """返回当前 Python 主进程的历史峰值常驻内存，单位为 MiB。"""
    # ru_maxrss 在 Linux 上以 KiB 计数，在 macOS 上却以 bytes 计数，需要分别换算。
    # RUSAGE_SELF 不包含并行预分词的子进程；测整个进程树可在命令外使用 /usr/bin/time -v。
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def main() -> None:
    """读取命令行参数，训练 tokenizer，并保存词表、merge 规则和统计信息。"""
    # argparse 让同一训练流程可以复用于不同语料、词表大小和输出目录。
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

    # parents=True 会递归创建缺失的父目录；exist_ok=True 允许复用已有目录。
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # perf_counter 是单调的高精度计时器，适合测量一段程序的实际耗时。
    start_time = time.perf_counter()
    # tokenizer 返回最终词表，以及严格按照训练顺序排列的 merge 规则。
    vocab, merges = tokenizer(str(args.input), args.vocab_size, args.special_tokens)
    elapsed = time.perf_counter() - start_time
    peak_rss_mb = get_peak_rss_mb()

    # 训练成功后再统一写文件，避免中途失败留下看似完整但实际残缺的产物。
    save_vocab(vocab, args.output_dir / "vocab.json")
    save_merges(merges, args.output_dir / "merges.json")

    # 长度相同时选择 token ID 更大的项，使结果确定且可复现。
    longest_token_id, longest_token = max(vocab.items(), key=lambda item: (len(item[1]), item[0]))
    # 这里只生成便于人阅读的预览；errors="replace" 不会改变 vocab 中保存的原始 bytes。
    longest_token_preview = longest_token.decode("utf-8", errors="replace")

    # stats.json 汇总实验配置和结果，便于之后填写作业报告或比较不同实现。
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
    (args.output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # 同时把关键结果打印到终端，用户无需先打开 stats.json 就能确认训练是否成功。
    print(f"trained vocab size: {len(vocab)}")
    print(f"merge count: {len(merges)}")
    print(f"elapsed: {elapsed:.2f}s")
    print(f"peak rss: {peak_rss_mb:.1f} MB")
    print(f"longest token id: {longest_token_id}")
    print(f"longest token len: {len(longest_token)}")
    print(f"longest token preview: {longest_token_preview!r}")
    print(f"saved to: {args.output_dir}")


if __name__ == "__main__":
    # 只有直接执行本文件时才启动训练；被其他模块 import 时不会产生副作用。
    main()
