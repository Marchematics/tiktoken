"""Compare the educational BPE trainer with the pre-optimisation implementation."""

import collections
import random
import time
import tracemalloc

import regex

from tiktoken._educational import bpe_train


def reference_bpe_train(data: str, vocab_size: int, pat_str: str) -> dict[bytes, int]:
    ranks = {bytes([i]): i for i in range(256)}
    words = [
        [bytes([b]) for b in word.encode("utf-8")]
        for word in regex.findall(pat_str, data)
    ]
    while len(ranks) < vocab_size:
        stats = collections.Counter(
            pair for word in words for pair in zip(word[:-1], word[1:])
        )
        most_common_pair = max(stats, key=lambda pair: stats[pair])
        token_bytes = most_common_pair[0] + most_common_pair[1]
        ranks[token_bytes] = len(ranks)
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_common_pair:
                    new_word.append(token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[i])
            new_words.append(new_word)
        words = new_words
    return ranks


def measure(trainer, data: str, vocab_size: int, pat_str: str) -> tuple[float, int]:
    tracemalloc.start()
    start = time.perf_counter()
    trainer(data, vocab_size, pat_str, **({"visualise": None} if trainer is bpe_train else {}))
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak


def main() -> None:
    rng = random.Random(0)
    data = " ".join(
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(40))
        for _ in range(500)
    )
    vocab_size = 512
    pat_str = r"[^ ]+"
    for name, trainer in (("reference", reference_bpe_train), ("optimized", bpe_train)):
        elapsed, peak = measure(trainer, data, vocab_size, pat_str)
        print(f"{name}: {elapsed:.3f}s, peak={peak / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
