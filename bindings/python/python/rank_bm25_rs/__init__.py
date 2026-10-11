"""Fast drop-in replacement for rank_bm25 with bit-identical scores."""

import math
import operator
import sys
from functools import cached_property

import numpy as np

from . import _core, _reference
from ._core import __version__
from ._reference import BM25

__all__ = ["BM25", "BM25L", "BM25Okapi", "BM25Plus", "__version__", "install"]

# NumPy's default floating-point error handling, under which rank_bm25 only warns.
_DEFAULT_ERRSTATE = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
_EXACT = 2**53


def _usable(value):
    """Parameters the Rust index computes exactly like NumPy: finite floats of moderate size,
    and ints that stay exact as float64."""
    if type(value) is float:
        return math.isfinite(value) and abs(value) <= 1e100
    return type(value) is int and abs(value) < _EXACT


_NUMPY_INTEGERS = frozenset(type(np.dtype(c).type(0)) for c in "bBhHiIlLqQ")


def _plain_query(query):
    """Queries scored in Rust: an exact list or tuple of exact str. rank_bm25 looks each term up
    while scoring, so any other iterable, or terms of other types, take its own path."""
    return type(query) in (list, tuple) and all(type(q) is str for q in query)


def _plain_doc_ids(doc_ids):
    """doc_ids handled in Python with rank_bm25's NumPy formulas: a replayable list, tuple or
    array of built-in or NumPy integers."""
    if type(doc_ids) is np.ndarray:
        return doc_ids.dtype.kind in "iu"
    return type(doc_ids) in (list, tuple) and all(
        type(di) in (int, bool) or type(di) in _NUMPY_INTEGERS for di in doc_ids
    )


class _RustBacked:
    """The Rust index behind BM25Okapi, BM25L and BM25Plus.

    Rust builds the index for an exact list or tuple of exact lists or tuples of exact `str`
    with plain parameters. For anything else, for subclasses that change how the index is built,
    and whenever the result could involve NaN, infinities or non-default NumPy error handling,
    rank_bm25's own code runs, so results, warnings and errors never differ.
    """

    # Unpickled rank_bm25 objects (for example after install()) have no Rust index.
    _core = None
    _built_with = None
    _variant = None
    _reference_class = None
    _extra = None

    def _setup(self, corpus, tokenizer):
        for name in ("idf", "doc_freqs", "doc_len"):
            self.__dict__.pop(name, None)
        self.tokenizer = tokenizer
        if tokenizer:
            corpus = self._tokenize_corpus(corpus)
        core = None
        own_build = (
            type(self)._initialize is BM25._initialize
            and type(self)._calc_idf is self._reference_class._calc_idf
        )
        params = (self.k1, self.b, getattr(self, self._extra))
        if own_build and type(corpus) in (list, tuple) and all(map(_usable, params)):
            core = _core.build(corpus, self._variant, *params)
        self._core = core
        if core is None:
            self._built_with = None
            self.corpus_size = 0
            self.avgdl = 0
            self.doc_freqs = []
            self.idf = {}
            self.doc_len = []
            nd = self._initialize(corpus)
            self._calc_idf(nd)
        else:
            self.corpus_size = core.corpus_size
            self.avgdl = core.avgdl
            self._built_with = self._scoring_params()

    def _scoring_params(self):
        """What get_scores reads besides the index, with types, to detect later changes."""
        values = (self.k1, self.b, getattr(self, "delta", None), self.avgdl, self.corpus_size)
        return tuple((type(v), v) for v in values)

    def _fast(self):
        core = self._core
        if core is None or self._scoring_params() != self._built_with:
            return False
        if np.geterr() != _DEFAULT_ERRSTATE:
            return False
        # NumPy multiplies integer parameters with integer frequencies and lengths in int64.
        if type(self.k1) is int and (abs(self.k1) + 1) * core.max_tf >= _EXACT:
            return False
        return not (type(self.b) is int and abs(self.b) * core.max_doc_len >= _EXACT)

    @cached_property
    def idf(self):
        return self._core.idf()

    @cached_property
    def doc_len(self):
        return self._core.doc_len()

    @cached_property
    def doc_freqs(self):
        return self._core.doc_freqs()

    def get_scores(self, query):
        if not (self._fast() and _plain_query(query)):
            return self._reference_class.get_scores(self, query)
        buf, finite = self._core.get_scores(query)
        if not finite:
            return self._reference_class.get_scores(self, query)
        return np.frombuffer(buf, dtype=np.float64)

    def get_scores_many(self, queries):
        """`get_scores` for each query, on all CPU cores without the GIL.

        Returns a float64 array with one row per query.
        """
        fast = (
            type(queries) in (list, tuple)
            and type(self).get_scores is _RustBacked.get_scores
            and self._fast()
            and all(map(_plain_query, queries))
        )
        if not fast:
            rows = [self.get_scores(q) for q in queries]
            return np.array(rows, dtype=np.float64).reshape(len(rows), self.corpus_size)
        buf, finite = self._core.get_scores_many(queries)
        scores = np.frombuffer(buf, dtype=np.float64).reshape(len(queries), self.corpus_size)
        for i, ok in enumerate(finite):
            if not ok:
                scores[i] = self._reference_class.get_scores(self, queries[i])
        return scores

    def _batch_terms(self, query, doc_ids):
        """Per query term, the idf and term frequencies rank_bm25's get_batch_scores reads."""
        ids = None
        n = self.corpus_size
        for q in query:
            if ids is None:
                ids = [operator.index(di) for di in doc_ids]
                ids = [i + n if i < 0 else i for i in ids]
                if not all(0 <= i < n for i in ids):
                    raise IndexError("list index out of range")
            yield self._core.idf_of(q), np.array(self._core.tfs(q, ids))

    def _batch_fast(self, query, doc_ids):
        return self._fast() and _plain_query(query) and _plain_doc_ids(doc_ids)


class BM25Okapi(_RustBacked, _reference.BM25Okapi):
    _variant = "okapi"
    _reference_class = _reference.BM25Okapi
    _extra = "epsilon"

    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._setup(corpus, tokenizer)
        if self._core is not None:
            self.average_idf = self._core.average_idf

    def get_batch_scores(self, query, doc_ids):
        if not self._batch_fast(query, doc_ids):
            return _reference.BM25Okapi.get_batch_scores(self, query, doc_ids)
        assert all(di < self.corpus_size for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for idf, q_freq in self._batch_terms(query, doc_ids):
            score += (idf or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score.tolist()


class BM25L(_RustBacked, _reference.BM25L):
    _variant = "l"
    _reference_class = _reference.BM25L
    _extra = "delta"

    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self._setup(corpus, tokenizer)

    def get_batch_scores(self, query, doc_ids):
        if not self._batch_fast(query, doc_ids):
            return _reference.BM25L.get_batch_scores(self, query, doc_ids)
        assert all(di < self.corpus_size for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for idf, q_freq in self._batch_terms(query, doc_ids):
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (
                (idf or 0)
                * q_freq
                * (self.k1 + 1)
                * (ctd + self.delta)
                / (self.k1 + ctd + self.delta)
            )
        return score.tolist()


class BM25Plus(_RustBacked, _reference.BM25Plus):
    _variant = "plus"
    _reference_class = _reference.BM25Plus
    _extra = "delta"

    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self._setup(corpus, tokenizer)

    def get_batch_scores(self, query, doc_ids):
        if not self._batch_fast(query, doc_ids):
            return _reference.BM25Plus.get_batch_scores(self, query, doc_ids)
        assert all(di < self.corpus_size for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for idf, q_freq in self._batch_terms(query, doc_ids):
            score += (idf or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score.tolist()


def install():
    """Make `import rank_bm25` return rank_bm25_rs.

    Call it before anything imports rank_bm25; modules that already imported it keep the
    original.
    """
    sys.modules["rank_bm25"] = sys.modules[__name__]
