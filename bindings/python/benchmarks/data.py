"""Pinned test splits used as BM25 corpora: documents and their reference summaries as queries."""

import hashlib
import io
import os
import urllib.request
from pathlib import Path

SOURCES = {
    "cnn_dailymail": {
        "repo": "abisee/cnn_dailymail",
        "revision": "96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
        "file": "3.0.0/test-00000-of-00001.parquet",
        "sha256": "04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e",
        "bytes": 30000471,
        "document": "article",
        "query": "highlights",
    },
    "xsum": {
        "repo": "EdinburghNLP/xsum",
        "revision": "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9",
        "file": "data/test-00000-of-00001.parquet",
        "sha256": "224e9dbc6fed987759c1954603b43cb280b8d475d78893779130aa707d967ed7",
        "bytes": 16731226,
        "document": "document",
        "query": "summary",
    },
    "pubmed": {
        "repo": "ccdv/pubmed-summarization",
        "revision": "6b30a2cae59b11ed77cb19959bffccbbd18e1106",
        "file": "document/test-00000-of-00001.parquet",
        "sha256": "d315d092a3b2d9eeec9cc5869f5526ade3c5b4495693f0f8c7f1de1feeb8e6c3",
        "bytes": 58464984,
        "document": "article",
        "query": "abstract",
    },
}
CACHE = Path(os.environ.get("RANK_BM25_BENCH_CACHE", Path.home() / ".cache/rank-bm25-rs-bench"))


def _download(source):
    path = CACHE / f"{source['sha256']}.parquet"
    if path.exists():
        content = path.read_bytes()
    else:
        url = (
            f"https://huggingface.co/datasets/{source['repo']}/resolve/"
            f"{source['revision']}/{source['file']}"
        )
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read(source["bytes"] + 1)
    if len(content) != source["bytes"] or hashlib.sha256(content).hexdigest() != source["sha256"]:
        raise ValueError(f"size or SHA-256 mismatch for {source['repo']}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_bytes(content)
        partial.replace(path)
    return content


def tokenize(text):
    return text.lower().split()


def load(name):
    """Tokenized documents and tokenized queries of one test split."""
    import pyarrow.parquet as pq

    source = SOURCES[name]
    table = pq.read_table(
        io.BytesIO(_download(source)), columns=[source["document"], source["query"]]
    )
    docs = [tokenize(t) for t in table.column(source["document"]).to_pylist()]
    queries = [tokenize(t) for t in table.column(source["query"]).to_pylist()]
    return docs, queries
