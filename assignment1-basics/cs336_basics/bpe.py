import heapq
import re
from collections import defaultdict
from collections.abc import Iterable

import regex as regex_re


class LinkNode:
    __slots__ = ("value", "prev", "next", "alive", "seq_id", "rank")

    def __init__(self, value, prev=None, next=None, seq_id=-1, rank=0):
        self.value = value
        self.prev = prev
        self.next = next
        self.alive = True
        self.seq_id = seq_id
        self.rank = rank

    def merge_with_next(self, merged_value):
        if self.next is None:
            return None

        right = self.next
        new_node = LinkNode(merged_value, self.prev, right.next, self.seq_id, self.rank)

        if self.prev is not None:
            self.prev.next = new_node
        if right.next is not None:
            right.next.prev = new_node

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
    pair_positions = defaultdict(set)

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

            corpus.append(head)

    heap = [(-count, ReversePair(pair)) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(heap)

    def push_pair(pair):
        count = pair_counts.get(pair, 0)
        if count > 0:
            heapq.heappush(heap, (-count, ReversePair(pair)))

    def add_occurrence(pair, left_node):
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        pair_positions[pair].add(left_node)
        push_pair(pair)

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
            push_pair(pair)

    def pop_best_pair():
        while heap:
            neg_count, reverse_pair = heapq.heappop(heap)
            pair = reverse_pair.pair
            count = -neg_count
            if pair_counts.get(pair) == count and count > 0:
                return pair
        return None

    merges = []

    while len(vocab) < vocab_size:
        best_pair = pop_best_pair()
        if best_pair is None:
            break

        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = merged_token

        if len(vocab) >= vocab_size:
            break

        occurrences = sorted(
            pair_positions.get(best_pair, ()), key=lambda node: (node.seq_id, node.rank)
        )

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
                corpus[node.seq_id] = new_node

            if left is not None:
                add_occurrence((left.value, merged_token), left)
            if right_right is not None:
                add_occurrence((merged_token, right_right.value), new_node)

    return vocab, merges