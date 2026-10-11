import random

import pytest

WORDS = [
    *[
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "was",
        "for",
        "on",
        "that",
        "with",
        "as",
        "by",
        "at",
        "from",
        "it",
        "an",
        "be",
        "this",
        "which",
        "or",
        "are",
    ],
    *[
        "have",
        "search",
        "engine",
        "ranking",
        "document",
        "query",
        "term",
        "frequency",
        "inverse",
        "model",
    ],
    *[
        "retrieval",
        "index",
        "score",
        "london",
        "windy",
        "weather",
        "today",
        "hello",
        "there",
        "good",
        "man",
        "quite",
        "how",
    ],
]


def make_corpus(rng, docs, max_len, vocab=WORDS, empty_rate=0.05):
    corpus = []
    for _ in range(docs):
        if rng.random() < empty_rate:
            corpus.append([])
            continue
        n = rng.randint(1, max_len)
        corpus.append(
            [vocab[min(int(rng.paretovariate(1.2)) - 1, len(vocab) - 1)] for _ in range(n)]
        )
    return corpus


def make_queries(rng, count, vocab=WORDS):
    pool = vocab + ["unseen", "", "Search"]
    return [[rng.choice(pool) for _ in range(rng.randint(0, 6))] for _ in range(count)]


@pytest.fixture(scope="session")
def corpora():
    rng = random.Random(42)
    out = []
    for docs, max_len in [(1, 3), (2, 5), (5, 20), (50, 40), (400, 80), (2000, 30)]:
        out.append((make_corpus(rng, docs, max_len), make_queries(rng, 25)))
    return out
