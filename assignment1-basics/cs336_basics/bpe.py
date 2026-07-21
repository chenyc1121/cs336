from __future__ import annotations

import heapq
import base64
import json
import multiprocessing
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

import regex as regex_re


# BPE 的基本操作是把一对相邻 token 合并成一个新 token。
# token 用 bytes 表示，因此 Pair 也直接由两个 bytes 组成。
Pair = tuple[bytes, bytes]

# GPT-2 的预分词规则：BPE 只允许在每个正则匹配结果内部发生，不跨越 pre-token 边界。
_GPT2_PATTERN = regex_re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
# 小文件启动多进程反而更慢，因此仅对较大的语料并行，并限制进程数以控制峰值内存。
_MAX_PRETOKENIZATION_WORKERS = 8
_MIN_PARALLEL_FILE_SIZE = 16 * 1024 * 1024
_BPE_MONITOR = os.environ.get("BPE_MONITOR", "0") not in ("0", "", "false", "False")
_BPE_MONITOR_INTERVAL = int(os.environ.get("BPE_MONITOR_INTERVAL", "100"))


class _ReversePair:
    """反转 Pair 的比较方向，使最小堆能弹出字节序最大的 Pair。"""

    __slots__ = ("pair",)

    def __init__(self, pair: Pair) -> None:
        self.pair = pair

    def __lt__(self, other: _ReversePair) -> bool:
        # 作业规定：出现次数相同时，选择字节序更大的 pair。
        return self.pair > other.pair


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_token: bytes,
) -> list[int]:
    """在 special token 的起点寻找分块边界。

    special token 本来就是文档边界，BPE 不允许跨过它合并。因此在这里切分不会
    改变预分词结果，可以让不同进程安全地独立统计各自的语料块。
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # 先按文件大小均匀估计边界，再从每个估计位置向后寻找真正安全的切分点。
    chunk_size = file_size // desired_num_chunks
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    read_size = 4096
    # 相邻读取块保留少量重叠，避免 special token 恰好跨越 4 KiB 边界而漏检。
    overlap_size = max(len(split_token) - 1, 0)
    for index in range(1, len(boundaries) - 1):
        position = boundaries[index]
        file.seek(position)
        overlap = b""

        while True:
            block = file.read(read_size)
            if not block:
                boundaries[index] = file_size
                break

            searchable = overlap + block
            found_at = searchable.find(split_token)
            if found_at != -1:
                boundaries[index] = position - len(overlap) + found_at
                break

            position += len(block)
            overlap = searchable[-overlap_size:] if overlap_size else b""

    return sorted(set(boundaries))


def _count_text_pretokens(text: str, special_tokens: tuple[str, ...]) -> Counter[bytes]:
    """统计一段文本中每种 pre-token 的出现次数。"""
    counts: Counter[bytes] = Counter()

    def count_segment(start: int, end: int) -> None:
        counts.update(match.group().encode("utf-8") for match in _GPT2_PATTERN.finditer(text, start, end))

    if not special_tokens:
        count_segment(0, len(text))
        return counts

    # special token 只充当硬边界：不参与统计，也不能与两侧文本发生 merge。
    # 长 token 放在正则前面，避免多个 special token 互为前缀时先匹配到短 token。
    special_pattern = re.compile("|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))
    segment_start = 0
    for match in special_pattern.finditer(text):
        count_segment(segment_start, match.start())
        segment_start = match.end()
    count_segment(segment_start, len(text))
    return counts


def _count_file_chunk(args: tuple[str, int, int, tuple[str, ...]]) -> Counter[bytes]:
    """多进程 worker：读取一个字节区间并返回局部 pre-token 计数。"""
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8")
    return _count_text_pretokens(text, special_tokens)


def _count_pretokens(input_path: str, special_tokens: list[str]) -> Counter[bytes]:
    """完成整份语料的预分词，并在安全时使用多进程并行。"""
    file_size = os.path.getsize(input_path)
    special_tuple = tuple(special_tokens)
    num_workers = min(_MAX_PRETOKENIZATION_WORKERS, os.cpu_count() or 1)

    # 没有非空 special token 时无法保证任意文件边界也是 pre-token 边界，必须串行处理。
    can_parallelize = (
        file_size >= _MIN_PARALLEL_FILE_SIZE and num_workers > 1 and bool(special_tokens) and bool(special_tokens[0])
    )
    if not can_parallelize:
        return _count_file_chunk((input_path, 0, file_size, special_tuple))

    with open(input_path, "rb") as file:
        boundaries = _find_chunk_boundaries(file, num_workers, special_tokens[0].encode("utf-8"))

    chunks = [
        (input_path, start, end, special_tuple) for start, end in zip(boundaries[:-1], boundaries[1:]) if start < end
    ]
    if len(chunks) == 1:
        return _count_file_chunk(chunks[0])

    # Counter.update 会把各进程对同一 pre-token 的计数直接相加。
    counts: Counter[bytes] = Counter()
    with multiprocessing.Pool(processes=len(chunks)) as pool:
        for chunk_counts in pool.imap_unordered(_count_file_chunk, chunks):
            counts.update(chunk_counts)
    return counts


def _merge_tokens(tokens: tuple[bytes, ...], pair: Pair, merged_token: bytes) -> tuple[bytes, ...]:
    """从左到右合并一个词中的目标 pair；重叠位置只合并一次。"""
    merged: list[bytes] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == pair[0] and tokens[index + 1] == pair[1]:
            merged.append(merged_token)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return tuple(merged)


def tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    """训练 byte-level BPE tokenizer。

    核心思路：先把重复的 pre-token 压缩成“唯一词 + 频次”，之后只更新包含
    当前最佳 pair 的词。这样内存取决于不同 pre-token 的数量，而不是 2GB 语料
    中的几十亿个字节。
    """
    # byte-level BPE 的基础词表固定包含全部 256 种单字节值。
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    if len(vocab) >= vocab_size:
        return vocab, []

    # 例如 b" the" 在语料中出现一百万次，也只保存一份 token 序列和频次 1_000_000。
    pretoken_counts = _count_pretokens(input_path, special_tokens)
    byte_tokens = tuple(bytes([value]) for value in range(256))

    # words[word_id]：该唯一 pre-token 当前被切成的 token 序列。
    # frequencies[word_id]：它在原语料中的出现次数，训练过程中保持不变。
    # pair_counts[pair]：pair 在完整语料中的加权出现次数。
    # pair_to_words[pair]：包含该 pair 的唯一词 ID，用来避免每轮扫描所有词。
    words: list[tuple[bytes, ...]] = []
    frequencies: list[int] = []
    pair_counts: dict[Pair, int] = {}
    pair_to_words: defaultdict[Pair, set[int]] = defaultdict(set)

    for pretoken, frequency in pretoken_counts.items():
        tokens = tuple(byte_tokens[value] for value in pretoken)
        word_id = len(words)
        words.append(tokens)
        frequencies.append(frequency)

        # 同一词内 pair 可以出现多次，每次都要乘以该词的语料频次。
        # 倒排索引只需记录“这个词是否包含 pair”，所以使用 set 去重。
        word_pairs: set[Pair] = set()
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + frequency
            word_pairs.add(pair)
        for pair in word_pairs:
            pair_to_words[pair].add(word_id)

    # heap 保存 (-出现次数, 反向字节序)，从而优先弹出次数最多、字节序最大的 pair。
    # 后续采用惰性更新：计数变化时压入新条目，弹出时丢弃已经过期的旧条目。
    heap = [(-count, _ReversePair(pair)) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    if _BPE_MONITOR:
        print(
            f"[BPE] pretokens={sum(pretoken_counts.values())} unique_pretokens={len(words)} pairs={len(pair_counts)}",
            flush=True,
        )

    def pop_best_pair() -> Pair | None:
        while heap:
            negative_count, reverse_pair = heapq.heappop(heap)
            pair = reverse_pair.pair
            # heap 中可能有同一 pair 的历史计数；只有与当前字典一致的条目才有效。
            if pair_counts.get(pair, 0) == -negative_count:
                return pair
        return None

    merges: list[Pair] = []
    while len(vocab) < vocab_size:
        best_pair = pop_best_pair()
        if best_pair is None:
            break

        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = merged_token
        if len(vocab) >= vocab_size:
            break

        # 只有包含 best_pair 的词会发生变化。先复制 ID，避免更新倒排索引时修改迭代中的 set。
        affected_word_ids = tuple(pair_to_words.get(best_pair, ()))
        # 多个词可能同时改变同一个 pair，先汇总 delta，最后只更新一次全局计数和 heap。
        count_deltas: defaultdict[Pair, int] = defaultdict(int)

        for word_id in affected_word_ids:
            old_tokens = words[word_id]
            frequency = frequencies[word_id]
            # 先减去该词旧 token 序列贡献的全部相邻 pair。
            old_pairs: set[Pair] = set()
            for pair in zip(old_tokens, old_tokens[1:]):
                count_deltas[pair] -= frequency
                old_pairs.add(pair)

            # 执行本轮 merge，再加回新 token 序列贡献的全部相邻 pair。
            new_tokens = _merge_tokens(old_tokens, best_pair, merged_token)
            words[word_id] = new_tokens
            new_pairs: set[Pair] = set()
            for pair in zip(new_tokens, new_tokens[1:]):
                count_deltas[pair] += frequency
                new_pairs.add(pair)

            # 同步维护 pair -> word_id 倒排索引，供下一轮快速定位受影响的词。
            for pair in old_pairs:
                word_ids = pair_to_words.get(pair)
                if word_ids is not None:
                    word_ids.discard(word_id)
                    if not word_ids:
                        pair_to_words.pop(pair, None)
            for pair in new_pairs:
                pair_to_words[pair].add(word_id)

        # 将本轮所有局部变化一次性应用到全局 pair count。
        for pair, delta in count_deltas.items():
            if delta == 0:
                continue
            new_count = pair_counts.get(pair, 0) + delta
            if new_count > 0:
                pair_counts[pair] = new_count
                heapq.heappush(heap, (-new_count, _ReversePair(pair)))
            else:
                pair_counts.pop(pair, None)

        # 惰性更新会留下过期 heap 条目；过多时整体重建，限制长期内存增长。
        if len(heap) > 4 * len(pair_counts) + 1024:
            heap = [(-count, _ReversePair(pair)) for pair, count in pair_counts.items()]
            heapq.heapify(heap)

        if _BPE_MONITOR and len(merges) % _BPE_MONITOR_INTERVAL == 0:
            print(
                f"[BPE] merges={len(merges)} pairs={len(pair_counts)} heap={len(heap)}",
                flush=True,
            )

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        # Add special tokens that are not already present, preserving existing IDs.
        next_id = max(self.vocab, default=-1) + 1
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[next_id] = token_bytes
                next_id += 1

        self._token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._special_token_ids = {
            special: self._token_to_id[special.encode("utf-8")] for special in self.special_tokens
        }
        self._special_pattern = (
            re.compile("|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)))
            if self.special_tokens
            else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            raw_vocab = json.load(vocab_file)

        vocab: dict[int, bytes] = {}
        for raw_id, raw_token in raw_vocab.items():
            if not isinstance(raw_token, str):
                raise TypeError("vocabulary values must be strings")
            try:
                token_bytes = base64.b64decode(raw_token, validate=True)
            except ValueError:
                token_bytes = raw_token.encode("utf-8")
            vocab[int(raw_id)] = token_bytes

        merges: list[Pair] = []
        merge_path = Path(merges_filepath)
        if merge_path.suffix.lower() == ".json":
            with open(merge_path, encoding="utf-8") as merge_file:
                raw_merges = json.load(merge_file)
            for raw_left, raw_right in raw_merges:
                try:
                    left = base64.b64decode(raw_left, validate=True)
                    right = base64.b64decode(raw_right, validate=True)
                except ValueError:
                    left = raw_left.encode("utf-8")
                    right = raw_right.encode("utf-8")
                merges.append((left, right))
        else:
            with open(merge_path, encoding="utf-8") as merge_file:
                for line in merge_file:
                    fields = line.strip().split()
                    if len(fields) == 2:
                        merges.append((fields[0].encode("utf-8"), fields[1].encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        pieces = [bytes([value]) for value in pretoken.encode("utf-8")]
        while len(pieces) > 1:
            best_pair = None
            best_rank = None
            for pair in zip(pieces, pieces[1:]):
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break

            merged: list[bytes] = []
            index = 0
            while index < len(pieces):
                if index + 1 < len(pieces) and (pieces[index], pieces[index + 1]) == best_pair:
                    merged.append(pieces[index] + pieces[index + 1])
                    index += 2
                else:
                    merged.append(pieces[index])
                    index += 1
            pieces = merged

        try:
            return [self._token_to_id[piece] for piece in pieces]
        except KeyError as error:
            raise ValueError(f"BPE produced a token missing from the vocabulary: {error.args[0]!r}") from error

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        cursor = 0
        matches = self._special_pattern.finditer(text) if self._special_pattern else ()
        for match in matches:
            ids.extend(self._encode_non_special(text[cursor : match.start()]))
            ids.append(self._special_token_ids[match.group(0)])
            cursor = match.end()
        ids.extend(self._encode_non_special(text[cursor:]))
        return ids

    def _encode_non_special(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in _GPT2_PATTERN.finditer(text):
            ids.extend(self._encode_pretoken(match.group(0)))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Processing one chunk at a time keeps memory bounded for large files.
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
