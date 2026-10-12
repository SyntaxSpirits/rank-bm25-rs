# Changelog

## 0.1.0 - 2026-10-12

First release.

- Rust port of `rank_bm25` 0.2.2: `BM25Okapi`, `BM25L` and `BM25Plus` with bit-identical scores,
  idf values and attributes.
- Python package `rank-bm25-rs`: the `rank_bm25` API, `get_scores_many` for many queries on all
  cores without the GIL, pickling, and `install()` to replace `import rank_bm25`.
- Inputs the Rust index does not handle exactly (generators, tokens that are not `str`, NumPy
  scalar or non-finite parameters, changed parameters, non-default `np.errstate`, queries with NaN
  scores) run `rank_bm25`'s own code, bundled in the package.
