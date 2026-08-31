from tiktoken._educational import bpe_train


def test_bpe_train_preserves_merge_order():
    ranks = bpe_train("abab baba", vocab_size=260, pat_str=r"[^ ]+", visualise=None)

    assert [(token, ranks[token]) for token in (b"ab", b"abab", b"bab", b"baba")] == [
        (b"ab", 256),
        (b"abab", 257),
        (b"bab", 258),
        (b"baba", 259),
    ]


def test_bpe_train_merges_overlapping_pairs_left_to_right():
    ranks = bpe_train("aaa aaa", vocab_size=258, pat_str=r"[^ ]+", visualise=None)

    assert ranks[b"aa"] == 256
    assert ranks[b"aaa"] == 257
