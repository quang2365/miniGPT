from collections import Counter, defaultdict
import json
import time
import heapq
def merge_pair(sequences, pair, new_id):
    result = []
    i = 0 
    while i < len(sequences):
        if i < len(sequences) - 1 and sequences[i] == pair[0] and sequences[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(sequences[i])
            i += 1
    return result
def count_pairs(sequences):
    counter = Counter()
    for sequence in sequences:
        for i in range(len(sequence) - 1):
            pair = (sequence[i],sequence[i+1])
            counter[pair] += 1

    return counter
class BPE():
    def __init__(self,num_merge):
        self.num_merge = num_merge

        self.vocab = {i:bytes([i]) for i in range(256)}

        self.merge = []


    def train(self,texts, log_every=100):
        if isinstance(texts,str):
            texts = texts.splitlines()
        sequences = [list(line.encode('utf-8')) for line in texts if line.strip()]

        if log_every < 1:
            raise ValueError("log_every must be greater than 0")

        started_at = time.perf_counter()
        total_bytes = sum(len(sequence) for sequence in sequences)
        print(
            f"[BPE] train start: texts={len(sequences):,}, "
            f"bytes={total_bytes:,}, target_merges={self.num_merge:,}",
            flush=True,
        )

        # Linked lists let us update only pairs touching a merge.  The heap is
        # lazy: stale entries are discarded when they reach the top.
        tokens, prev, next_node, alive = [], [], [], []
        heads = []
        for sequence in sequences:
            sid_nodes = []
            for token in sequence:
                sid_nodes.append(len(tokens))
                tokens.append(token)
                prev.append(len(tokens) - 2)
                next_node.append(-1)
                alive.append(True)
                if len(sid_nodes) > 1:
                    next_node[sid_nodes[-2]] = sid_nodes[-1]
            heads.append(sid_nodes[0] if sid_nodes else -1)

        pair_counts = Counter()
        pair_positions = defaultdict(set)
        heap = []

        active_touched = None

        def add_pair(left):
            right = next_node[left] if left != -1 and alive[left] else -1
            if right == -1 or not alive[right]:
                return
            pair = (tokens[left], tokens[right])
            pair_counts[pair] += 1
            pair_positions[pair].add(left)
            if active_touched is not None:
                active_touched.add(pair)

        def remove_pair(left):
            right = next_node[left] if left != -1 and alive[left] else -1
            if right == -1 or not alive[right]:
                return
            pair = (tokens[left], tokens[right])
            pair_counts[pair] -= 1
            pair_positions[pair].discard(left)
            if active_touched is not None:
                active_touched.add(pair)

        for sid, head in enumerate(heads):
            node = head
            while node != -1:
                add_pair(node)
                node = next_node[node]
        for pair, count in pair_counts.items():
            heapq.heappush(heap, (-count, pair))

        for merge_index in range(1, self.num_merge + 1):
            while heap:
                neg_count, best_pair = heapq.heappop(heap)
                best_count = pair_counts.get(best_pair, 0)
                if -neg_count == best_count:
                    break
            else:
                print("[BPE] stop: no token pairs remain", flush=True)
                break

            if best_count < 2:
                print(
                    f"[BPE] stop: most frequent pair count={best_count}",
                    flush=True,
                )
                break
            new_id = len(self.vocab)

            new_bytes = (self.vocab[best_pair[0]] + self.vocab[best_pair[1]])

            self.vocab[new_id] = new_bytes

            self.merge.append((best_pair[0],best_pair[1], new_id))

            # Snapshot occurrences because merging changes the index sets.
            active_touched = set()
            for left in list(pair_positions[best_pair]):
                right = next_node[left]
                if (not alive[left] or right == -1 or not alive[right] or
                        (tokens[left], tokens[right]) != best_pair):
                    continue
                left_left = prev[left]
                right_right = next_node[right]
                for node in (left_left, left, right):
                    if node != -1:
                        remove_pair(node)
                tokens[left] = new_id
                next_node[left] = right_right
                if right_right != -1:
                    prev[right_right] = left
                alive[right] = False
                for node in (left_left, left):
                    if node != -1:
                        add_pair(node)

            pair_positions[best_pair].clear()
            pair_counts[best_pair] = 0
            for pair in active_touched:
                if pair_counts.get(pair, 0) > 0:
                    heapq.heappush(heap, (-pair_counts[pair], pair))
            active_touched = None

            if merge_index == 1 or merge_index % log_every == 0 or merge_index == self.num_merge:
                elapsed = time.perf_counter() - started_at
                print(
                    f"[BPE] merge {merge_index:,}/{self.num_merge:,} | "
                    f"pair=({best_pair[0]}, {best_pair[1]}) | "
                    f"count={best_count:,} | vocab={len(self.vocab):,} | "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        elapsed = time.perf_counter() - started_at
        print(
            f"[BPE] train done: merges={len(self.merge):,}, "
            f"vocab={self.vocab_size:,}, elapsed={elapsed:.1f}s",
            flush=True,
        )

    def encode(self, texts):
        ids = list(texts.encode('utf-8'))

        for token_a, token_b, new_id in self.merge:
            ids = merge_pair(ids,(token_a,token_b),new_id)

        return ids
    def decode(self, ids):
        byte_sequence = b"".join(self.vocab[token_id] for token_id in ids)

        return byte_sequence.decode('utf-8',errors='replace')      
        
    @property
    def vocab_size(self):
        return len(self.vocab)
    def save(self,path):
        data= {
            'num_merge':self.num_merge,

            'vocab':{
                str(token_id): list(byte_value)
                for token_id, byte_value in self.vocab.items()
            },

            'merge':[[a,b,new_id] for a,b,new_id in self.merge]
        }
        with open(path,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
    @classmethod
    def load(cls, path):
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        tokenizer = cls(
            num_merge=data["num_merge"]
        )

        tokenizer.vocab = {
            int(token_id): bytes(byte_values)
            for token_id, byte_values
            in data["vocab"].items()
        }

        tokenizer.merge = [
            tuple(item)
            for item in data["merge"]
        ]

        return tokenizer
