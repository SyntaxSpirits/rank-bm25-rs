"""Compare rank_bm25_rs with rank_bm25 0.2.2 on the full test splits.

For each split and BM25 variant, every document is indexed and the reference summaries of the
first --queries documents are used as queries. Scores must be identical bit for bit.
"""

import argparse
import json
import sys
import time

import numpy as np
import rank_bm25
from data import SOURCES, load

import rank_bm25_rs

VARIANTS = ["BM25Okapi", "BM25L", "BM25Plus"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = {"python": sys.version.split()[0], "results": []}
    start = time.perf_counter()
    for name in SOURCES:
        docs, queries = load(name)
        queries = queries[: args.queries]
        for variant in VARIANTS:
            a = getattr(rank_bm25, variant)(docs)
            b = getattr(rank_bm25_rs, variant)(docs)
            same_index = a.avgdl == b.avgdl and list(a.idf.items()) == list(b.idf.items())
            differ = sum(
                not np.array_equal(a.get_scores(q).view(np.uint64), b.get_scores(q).view(np.uint64))
                for q in queries
            )
            report["results"].append(
                {
                    "dataset": name,
                    "variant": variant,
                    "documents": len(docs),
                    "queries": len(queries),
                    "same_idf_and_avgdl": same_index,
                    "queries_with_differences": differ,
                }
            )
            print(report["results"][-1], flush=True)
    report["seconds"] = round(time.perf_counter() - start, 1)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
    failed = any(
        not r["same_idf_and_avgdl"] or r["queries_with_differences"] for r in report["results"]
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
