import heapq
import re
import os
import weakref
from collections import defaultdict
from collections.abc import Iterable

import regex as regex_re

# Monitoring enabled via env var `BPE_MONITOR=1` (optional)
_BPE_MONITOR = os.environ.get("BPE_MONITOR", "0") not in ("0", "", "false", "False")
_BPE_MONITOR_INTERVAL = int(os.environ.get("BPE_MONITOR_INTERVAL", "100"))

def _count_nodes(corpus):
    total = 0
    for head in corpus:
        node = head
        while node is not None:
            total += 1
            node = node.next
    return total

def _sum_positions(pair_positions):
    return sum(len(s) for s in pair_positions.values())


class LinkNode:
    __slots__ = ("value", "prev", "next", "alive", "seq_id", "rank", "corpus_idx", "__weakref__")

    def __init__(self, value, prev=None, next=None, seq_id=-1, rank=0, corpus_idx=-1):
        # value: bytes 或已合并的 token bytes
        # prev/next: 链表指针
        self.value = value
        self.prev = prev
        self.next = next
        self.alive = True
        # seq_id/rank: 用于稳定排序
        self.seq_id = seq_id
        self.rank = rank
        # corpus_idx: 该节点所在链的头在 corpus 列表中的索引
        self.corpus_idx = corpus_idx

    def merge_with_next(self, merged_value):
        # 将当前节点与下一个节点合并，返回新节点
        if self.next is None:
            return None

        right = self.next
        new_node = LinkNode(merged_value, self.prev, right.next, self.seq_id, self.rank, self.corpus_idx)

        if self.prev is not None:
            self.prev.next = new_node
        if right.next is not None:
            right.next.prev = new_node

        # 标记旧节点死亡（WeakSet 会自动从 pair_positions 中清理死弱引用）
        self.alive = False
        right.alive = False
        return new_node


class ReversePair:
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

def tokenizer(input_path, vocab_size, special_tokens):
    """
    Train a byte-level BPE tokenizer.

    Returns:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
    """
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    # 1) 初始词表：256 个单字节 token
    vocab = {i: bytes([i]) for i in range(256)}

    # 2) 加入 special tokens
    for special in special_tokens:
        vocab[len(vocab)] = special.encode("utf-8")

    # 如果目标词表大小已经到达，就直接返回
    if len(vocab) >= vocab_size:
        return vocab, []


    gpt2_pattern = regex_re.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )

    # special tokens 作为硬边界：先切开，再对普通片段做 GPT-2 pre-tokenization
    if special_tokens:
        special_pattern = re.compile(
            "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
        )
        text_parts = special_pattern.split(text)
    else:
        text_parts = [text]

    corpus = []
    pair_counts = {}
    pair_positions = defaultdict(weakref.WeakSet)

    for seq_id, part in enumerate(text_parts):
        if not part:
            continue
        for pre_token in gpt2_pattern.findall(part):
            token_bytes = pre_token.encode("utf-8")
            head = None
            prev = None
            rank = 0
            for byte_value in token_bytes:
                node = LinkNode(bytes([byte_value]), prev=prev, seq_id=seq_id, rank=rank)
                if head is None:
                    head = node
                else:
                    prev.next = node

                if prev is not None:
                    pair = (prev.value, node.value)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                    pair_positions[pair].add(prev)

                prev = node
                rank += 1

            head.corpus_idx = len(corpus)
            corpus.append(head)

    heap = [(-count, ReversePair(pair)) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(heap)

    if _BPE_MONITOR:
        node_count = _count_nodes(corpus)
        total_pair_counts = len(pair_counts)
        total_pair_positions = _sum_positions(pair_positions)
        heap_len = len(heap)
        print(f"[BPE_MON] initial nodes={node_count} pairs={total_pair_counts} pos_sum={total_pair_positions} heap_len={heap_len}")

        # Check for duplicate seq_id usage in corpus (possible bug described by user)
        seq_counts = defaultdict(int)
        for head in corpus:
            seq_counts[head.seq_id] += 1
        dup = {k: v for k, v in seq_counts.items() if v > 1}
        if dup:
            print(f"[BPE_MON] WARNING: duplicate seq_id counts in corpus: {dup}")

    def push_pair(pair):
        count = pair_counts.get(pair, 0)
        if count > 0:
            heapq.heappush(heap, (-count, ReversePair(pair)))

    def add_occurrence(pair, left_node):
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        pair_positions[pair].add(left_node)
        pairs_to_push.add(pair)

    def remove_occurrence(pair, left_node):
        positions = pair_positions.get(pair)
        if positions is None or left_node not in positions:
            return

        positions.remove(left_node)
        if not positions:
            pair_positions.pop(pair, None)

        count = pair_counts.get(pair, 0)
        if count <= 1:
            pair_counts.pop(pair, None)
        else:
            pair_counts[pair] = count - 1

    def pop_best_pair():
        while heap:
            neg_count, reverse_pair = heapq.heappop(heap)
            pair = reverse_pair.pair
            count = -neg_count
            current_count = pair_counts.get(pair, 0)
            if count == current_count and current_count > 0:
                return pair
            # 过期条目：用当前正确的 count 重新入堆
            if current_count > 0:
                heapq.heappush(heap, (-current_count, ReversePair(pair)))
        return None

    merges = []
    merge_check = 0

    while len(vocab) < vocab_size:
        best_pair = pop_best_pair()
        if best_pair is None:
            break

        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        merge_check += 1
        if _BPE_MONITOR and (merge_check % _BPE_MONITOR_INTERVAL == 0):
            node_count = _count_nodes(corpus)
            total_pair_counts = len(pair_counts)
            total_pair_positions = _sum_positions(pair_positions)
            heap_len = len(heap)
            print(f"[BPE_MON] after merges={len(merges)} nodes={node_count} pairs={total_pair_counts} pos_sum={total_pair_positions} heap_len={heap_len}")
        vocab[len(vocab)] = merged_token

        if len(vocab) >= vocab_size:
            break

        if _BPE_MONITOR:
            occ_before = len(pair_positions.get(best_pair, set()))
            print(f"[BPE_MON] merging {best_pair} occ_before={occ_before}")

        occurrences = sorted(
            pair_positions.get(best_pair, ()), key=lambda node: (node.seq_id, node.rank)
        )
        pairs_to_push = set()

        if _BPE_MONITOR:
            occ_after = len(occurrences)
            print(f"[BPE_MON] merging {best_pair} occ_after_sorted={occ_after}")

        for node in occurrences:
            live_positions = pair_positions.get(best_pair)
            if live_positions is None or node not in live_positions:
                continue

            right = node.next
            if right is None or not node.alive or not right.alive:
                remove_occurrence(best_pair, node)
                continue

            if node.value != best_pair[0] or right.value != best_pair[1]:
                remove_occurrence(best_pair, node)
                continue

            left = node.prev
            right_right = right.next

            if left is not None:
                remove_occurrence((left.value, node.value), left)

            remove_occurrence(best_pair, node)

            if right_right is not None:
                remove_occurrence((right.value, right_right.value), right)

            new_node = node.merge_with_next(merged_token)

            if left is None:
                corpus[node.corpus_idx] = new_node

            if left is not None:
                add_occurrence((left.value, merged_token), left)
            if right_right is not None:
                add_occurrence((merged_token, right_right.value), new_node)

        for pair in pairs_to_push:
            push_pair(pair)

    return vocab, merges
