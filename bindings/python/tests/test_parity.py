"""Bit-for-bit comparison with rank_bm25 0.2.2."""

import itertools
import random

import numpy as np
import pytest
import rank_bm25
from conftest import make_corpus, make_queries
from hypothesis import given, settings
from hypothesis import strategies as st

import rank_bm25_rs

VARIANTS = ["BM25Okapi", "BM25L", "BM25Plus"]


def bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def build(name, module, corpus, **kw):
    return getattr(module, name)(corpus, **kw)


def assert_same_index(a, b):
    assert b.corpus_size == a.corpus_size
    assert b.avgdl == a.avgdl
    assert list(b.idf.items()) == list(a.idf.items())
    assert b.doc_len == a.doc_len
    assert b.doc_freqs == a.doc_freqs
    assert [list(d) for d in b.doc_freqs] == [list(d) for d in a.doc_freqs]
    if hasattr(a, "average_idf"):
        assert b.average_idf == a.average_idf


@pytest.mark.parametrize("name", VARIANTS)
def test_scores_match(corpora, name):
    for corpus, queries in corpora:
        a = build(name, rank_bm25, corpus)
        b = build(name, rank_bm25_rs, corpus)
        assert b._core is not None
        assert_same_index(a, b)
        for q in queries:
            assert np.array_equal(bits(a.get_scores(q)), bits(b.get_scores(q))), q
            ids = list(range(0, a.corpus_size, 3)) + [-1]
            assert (
                bits(a.get_batch_scores(q, ids)).tolist()
                == bits(b.get_batch_scores(q, ids)).tolist()
            )
            assert a.get_top_n(q, corpus, n=7) == b.get_top_n(q, corpus, n=7)


PARAMS = list(
    itertools.product(
        [1.5, 0.0, 1.2, -0.5, 2, 1e-300], [0.75, 0.0, 1.0, 1, 2.0], [0.25, 0.0, -1.0, 0.5, 1, -1.5]
    )
)


@pytest.mark.parametrize("name", VARIANTS)
def test_parameters_match(name):
    rng = random.Random(name)
    for k1, b, extra in PARAMS:
        corpus = make_corpus(rng, rng.choice([1, 3, 20]), rng.choice([1, 5, 30]), empty_rate=0.3)
        kw = {"k1": k1, "b": b, ("epsilon" if name == "BM25Okapi" else "delta"): extra}
        try:
            a = build(name, rank_bm25, corpus, **kw)
        except ZeroDivisionError as e:
            with pytest.raises(ZeroDivisionError, match=str(e)):
                build(name, rank_bm25_rs, corpus, **kw)
            continue
        b_ = build(name, rank_bm25_rs, corpus, **kw)
        assert_same_index(a, b_)
        for q in make_queries(rng, 6):
            assert np.array_equal(bits(a.get_scores(q)), bits(b_.get_scores(q))), (kw, q)


@settings(max_examples=300, deadline=None)
@given(
    corpus=st.lists(st.lists(st.text(max_size=3), max_size=12), min_size=1, max_size=12),
    query=st.lists(st.text(max_size=3), max_size=6),
    name=st.sampled_from(VARIANTS),
)
def test_random_corpora(corpus, query, name):
    try:
        a = build(name, rank_bm25, corpus)
    except ZeroDivisionError:
        with pytest.raises(ZeroDivisionError):
            build(name, rank_bm25_rs, corpus)
        return
    b = build(name, rank_bm25_rs, corpus)
    assert_same_index(a, b)
    assert np.array_equal(bits(a.get_scores(query)), bits(b.get_scores(query)))


@pytest.mark.parametrize("name", VARIANTS)
def test_get_scores_many(corpora, name):
    corpus, queries = corpora[4]
    b = build(name, rank_bm25_rs, corpus)
    many = b.get_scores_many(queries)
    assert many.shape == (len(queries), len(corpus))
    for row, q in zip(many, queries, strict=False):
        assert np.array_equal(bits(row), bits(b.get_scores(q)))
    assert b.get_scores_many([]).shape == (0, len(corpus))
