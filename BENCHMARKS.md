# Benchmarks

`bindings/python/benchmarks/compare.py` times `rank_bm25` 0.2.2 and `rank-bm25-rs` with
`BM25Okapi`. The corpus is the documents of three pinned test splits, lowercased and split on
whitespace: [CNN/DailyMail 3.0.0](https://huggingface.co/datasets/abisee/cnn_dailymail),
[XSum](https://huggingface.co/datasets/EdinburghNLP/xsum) and
[PubMed](https://huggingface.co/datasets/ccdv/pubmed-summarization), 29,482 documents and 32.7
million tokens in total. The 10,000-document corpus is the first 10,000 of them; the
100,000-document corpus repeats them. Queries are 20 sets of 6 tokens sampled from documents
(seed 1729). Every timed query's scores were checked to be identical to `rank_bm25`'s.

Index building is one run for `rank_bm25` and the median of three for `rank-bm25-rs`. Query
times are medians over queries; `get_scores_many` passes all 20 queries in one call.

## Apple M1 Pro, 8 cores, macOS 14.5, CPython 3.13.9, rank-bm25-rs 0.1.0

| Documents | Tokens | Build, rank_bm25 | Build, rank-bm25-rs | Speed-up |
|---|---|---|---|---|
| 10,000 | 6.7M | 1.84 s | 0.32 s | 5.9× |
| 29,482 | 32.7M | 9.49 s | 2.04 s | 4.7× |
| 100,000 | 106.0M | 20.79 s | 4.86 s | 4.3× |

| Documents | get_scores, rank_bm25 | get_scores, rank-bm25-rs | Speed-up | get_scores_many, per query |
|---|---|---|---|---|
| 10,000 | 16.05 ms | 0.025 ms | 635× | 0.017 ms |
| 29,482 | 58.27 ms | 0.065 ms | 904× | 0.045 ms |
| 100,000 | 187.58 ms | 0.197 ms | 952× | 0.111 ms |

`rank_bm25` looks up every query term in a dictionary of every document, in Python.
`rank-bm25-rs` visits only the documents that contain the term, so the gain grows with the
corpus. Building the index is bounded by reading the Python lists of tokens.

Raw results: [`bindings/python/benchmarks/results/m1-pro.json`](bindings/python/benchmarks/results/m1-pro.json).

## Parity on the full test splits

`bindings/python/benchmarks/parity.py` indexes every document of each split with each variant,
compares `idf` and `avgdl`, and compares the scores of the first 200 reference summaries used as
queries. On the machine above, with CPython 3.10.21 and 3.13.9:

| Split | Documents | BM25Okapi | BM25L | BM25Plus |
|---|---|---|---|---|
| CNN/DailyMail | 11,490 | 0 / 200 differ | 0 / 200 differ | 0 / 200 differ |
| XSum | 11,334 | 0 / 200 differ | 0 / 200 differ | 0 / 200 differ |
| PubMed | 6,658 | 0 / 200 differ | 0 / 200 differ | 0 / 200 differ |

The "Full parity" GitHub workflow runs the same script on Linux, macOS and Windows.

## Reproducing

```sh
cd bindings/python
uv sync --group dev --group bench
cd benchmarks
uv run python compare.py --output results-local
uv run python parity.py
```

The datasets (105 MB) are downloaded from Hugging Face on the first run, checked against their
SHA-256 and cached in `~/.cache/rank-bm25-rs-bench` (`RANK_BM25_BENCH_CACHE` overrides it).
