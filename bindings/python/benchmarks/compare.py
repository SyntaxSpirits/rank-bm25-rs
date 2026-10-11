"""Time rank_bm25 0.2.2 and rank_bm25_rs.

The corpus is the documents of the three pinned test splits (29,482 documents, 32.7M tokens),
and for larger sizes the same documents repeated. Queries are 6 tokens sampled from documents.
Index building and get_scores are timed separately; get_scores_many uses all cores.
"""

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from functools import partial

import numpy as np
import rank_bm25
from data import SOURCES, load

import rank_bm25_rs


def timed(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="10000,29482,100000")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--output", default="benchmark-results")
    args = parser.parse_args()

    docs = []
    for name in SOURCES:
        docs += load(name)[0]
    rng = random.Random(1729)
    pool = [d for d in docs if len(d) >= 6]
    queries = [rng.sample(d, 6) for d in rng.sample(pool, args.queries)]
    results = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "rank_bm25_rs": rank_bm25_rs.__version__,
        "query_tokens": 6,
        "rows": [],
    }
    for size in map(int, args.sizes.split(",")):
        corpus = [docs[i % len(docs)] for i in range(size)]
        tokens = sum(map(len, corpus))
        build_py = timed(partial(rank_bm25.BM25Okapi, corpus), 1)
        build_rs = timed(partial(rank_bm25_rs.BM25Okapi, corpus), 3)
        reference = rank_bm25.BM25Okapi(corpus)
        index = rank_bm25_rs.BM25Okapi(corpus)
        for q in queries:
            expected = reference.get_scores(q).view(np.uint64)
            assert np.array_equal(expected, index.get_scores(q).view(np.uint64))
        query_py = statistics.median(
            timed(partial(reference.get_scores, q), 1) for q in queries[:8]
        )
        query_rs = statistics.median(timed(partial(index.get_scores, q), 5) for q in queries)
        many_rs = timed(partial(index.get_scores_many, queries), 5) / len(queries)
        del reference
        row = {
            "documents": size,
            "tokens": tokens,
            "build_s": {"rank_bm25": round(build_py, 3), "rank_bm25_rs": round(build_rs, 3)},
            "get_scores_ms": {
                "rank_bm25": round(query_py * 1000, 3),
                "rank_bm25_rs": round(query_rs * 1000, 4),
                "rank_bm25_rs_many_per_query": round(many_rs * 1000, 4),
            },
            "build_speedup": round(build_py / build_rs, 1),
            "query_speedup": round(query_py / query_rs, 1),
        }
        results["rows"].append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
