# rank-bm25-rs

[![CI](https://github.com/SyntaxSpirits/rank-bm25-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/SyntaxSpirits/rank-bm25-rs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rank-bm25-rs.svg)](https://pypi.org/project/rank-bm25-rs/)
[![Crates.io](https://img.shields.io/crates/v/rank-bm25.svg)](https://crates.io/crates/rank-bm25)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[`rank_bm25`](https://pypi.org/project/rank-bm25/) with a Rust index: the same BM25 scores, down
to the last bit, 600–950 times faster per query on 10,000 to 100,000 documents, and indexing
4–6 times faster ([benchmarks](https://github.com/SyntaxSpirits/rank-bm25-rs/blob/main/BENCHMARKS.md)).

`rank_bm25` is downloaded about nine million times a month, for example through LangChain's
`BM25Retriever` and crawl4ai. It scores a query by looking up each term in a dictionary of every
document, in Python, so a query over 100,000 documents takes about 190 ms. `rank-bm25-rs` keeps an
inverted index and visits only the documents that contain a query term, while performing every
remaining floating-point operation in NumPy's order, so the scores, idf values and rankings are
exactly those of `rank_bm25`.

The Python package is `rank-bm25-rs`; the Rust crate is `rank-bm25`. This is an independent
project, not affiliated with the author of `rank_bm25`.

## Python

```sh
pip install rank-bm25-rs
```

Change the import and keep the rest of your code:

```python
from rank_bm25_rs import BM25Okapi  # was: from rank_bm25 import BM25Okapi

corpus = ["Hello there good man!", "It is quite windy in London", "How is the weather today?"]
bm25 = BM25Okapi([doc.split(" ") for doc in corpus])

query = "windy London".split(" ")
bm25.get_scores(query)
# array([0.        , 0.93729472, 0.        ])
bm25.get_top_n(query, corpus, n=1)
# ['It is quite windy in London']
```

`BM25Okapi`, `BM25L` and `BM25Plus` take the same arguments as in `rank_bm25` 0.2.2, including
`tokenizer`, and have the same methods (`get_scores`, `get_batch_scores`, `get_top_n`) and
attributes (`idf`, `doc_freqs`, `doc_len`, `avgdl`, `corpus_size`, `average_idf`). Indexes can be
pickled.

To score many queries at once, `get_scores_many` runs them on all CPU cores without the GIL and
returns one row per query:

```python
bm25.get_scores_many([["windy", "London"], ["is"]])
# array([[0.        , 0.93729472, 0.        ],
#        [0.        , 0.10042443, 0.10946263]])
```

Set `RAYON_NUM_THREADS` to limit the threads.

### Code you do not control

When a library imports `rank_bm25` itself, call `install()` before importing it:

```python
import rank_bm25_rs

rank_bm25_rs.install()  # `import rank_bm25` now returns rank_bm25_rs

from langchain_community.retrievers import BM25Retriever
```

Modules that imported `rank_bm25` before the call keep the original.

## Rust

```toml
[dependencies]
rank-bm25 = "0.1"
```

```rust
use rank_bm25::{Builder, Params};

let mut builder = Builder::new(Params::okapi());
for doc in ["Hello there good man!", "It is quite windy in London", "How is the weather today?"] {
    builder.add_document(doc.split(' '));
}
let index = builder.build()?;
let scores = index.get_scores(&["windy", "London"]);
```

`Params::l()` and `Params::plus()` select the other variants; the fields hold `k1`, `b` and the
variant's `epsilon` or `delta`. `Index` is `Send + Sync`.

## Compatibility

The tests compare `rank-bm25-rs` with `rank_bm25` 0.2.2 and fail on any difference in any score:
corpora from one to thousands of documents, empty documents, repeated and unknown query terms,
property-based tests, and every variant with parameters that make `rank_bm25` return NaN or
negative scores (`k1 = 0`, `b > 1`, negative `delta` and `epsilon`). A separate script compares
the full CNN/DailyMail, XSum and PubMed test splits; CI runs it on Linux, macOS and Windows.

`rank_bm25`'s own code, bundled in the package, runs whenever the Rust index could behave
differently, so results, NumPy warnings and errors do not change; only the speed-up is lost:

- the corpus is not an exact list or tuple of exact lists or tuples of `str` (for example a
  generator, or integer tokens);
- a parameter is not a finite `int` or `float` (for example a NumPy scalar or NaN), or an integer
  parameter could overflow NumPy's integer arithmetic;
- a subclass overrides `_initialize` or `_calc_idf`, or `k1`, `b`, `delta` or `avgdl` were changed
  after construction;
- a query is not a list or tuple of `str`, or `doc_ids` is not a list, tuple or array of
  integers;
- a query produces NaN or infinite scores, or NumPy's error handling (`np.errstate`) is not the
  default.

Known differences:

- The Rust index does not see changes made to the `idf`, `doc_freqs` or `doc_len` attributes
  after construction; `rank_bm25` would use a modified `idf` dictionary.
- Python 3.10 or newer is required.

## Development

```sh
cargo test -p rank-bm25
cd bindings/python
uv sync --group dev
uv run pytest
cd benchmarks && uv run --group bench python parity.py   # full test splits
```

## License

Apache-2.0, the same licence as `rank_bm25`, whose scoring this project ports. See
[NOTICE](NOTICE).
