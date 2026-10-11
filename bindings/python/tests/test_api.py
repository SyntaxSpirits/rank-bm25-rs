import os
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import rank_bm25

import rank_bm25_rs

CORPUS = [
    ["hello", "there", "good", "man"],
    ["it", "is", "quite", "windy", "in", "london"],
    ["how", "is", "the", "weather", "today"],
]
VARIANTS = ["BM25Okapi", "BM25L", "BM25Plus"]


def normalize(value):
    """Comparable form: arrays and floats by their bits, so NaN equals NaN."""
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.view(np.uint8).tobytes())
    if isinstance(value, float):
        return ("float", np.float64(value).view(np.uint64).item())
    if isinstance(value, dict):
        return ("dict", [(k, normalize(v)) for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [normalize(v) for v in value])
    return value


def run(scenario):
    out = []
    for module in (rank_bm25, rank_bm25_rs):
        try:
            out.append(("ok", normalize(scenario(module))))
        except Exception as e:
            out.append((type(e).__name__, str(e)))
    assert out[1] == out[0]
    return out[0]


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize(
    "corpus",
    [
        [[1, 2, 3], [2, 3], [3]],
        [("a", "b"), ("b",)],
        (["a", "b"], ["b"]),
        [["a", 1], ["b"]],
        [np.str_("a").__str__(), "b"],
        [["\ud800", "a"], ["a"]],
        [],
        [[]],
        [[], []],
    ],
)
def test_corpus_types(name, corpus):
    def scenario(m):
        idx = getattr(m, name)(corpus)
        q = ["a", "b", 2, 3]
        return idx.get_scores(q), idx.idf, idx.doc_freqs, idx.get_batch_scores(q, [0])

    run(scenario)


def test_generator_corpus_and_fallback_params():
    for kw in ({"k1": np.float32(1.2)}, {"b": True}, {"k1": 2**60}):

        def scenario(m, kw=kw):
            idx = m.BM25Okapi((doc for doc in CORPUS), **kw)
            return idx.get_scores(["windy", "is"])

        run(scenario)
    assert rank_bm25_rs.BM25Okapi(CORPUS, k1=np.float32(1.2))._core is None
    assert rank_bm25_rs.BM25Okapi(CORPUS, k1=2)._core is not None


@pytest.mark.parametrize(
    "query",
    [["windy"], "windy is", ("is" for _ in range(2)), [b"windy", 1.5, None], [], ["is", "is"]],
)
def test_query_types(query):
    items = list(query) if not isinstance(query, (list, str)) else query
    run(
        lambda m: m.BM25Okapi(CORPUS).get_scores(
            iter(items) if not isinstance(query, (list, str)) else items
        )
    )


def test_unhashable_query_term():
    run(lambda m: m.BM25L(CORPUS).get_scores([["windy"]]))


@pytest.mark.parametrize("ids", [[0, 2], [-1], [-4], [3], [True, False], [1.0], [], [0, 0, 1]])
@pytest.mark.parametrize("name", VARIANTS)
def test_batch_scores_indexing(name, ids):
    run(lambda m: getattr(m, name)(CORPUS).get_batch_scores(["is", "windy"], ids))
    run(lambda m: getattr(m, name)(CORPUS).get_batch_scores([], ids))


def test_get_top_n():
    run(lambda m: m.BM25Plus(CORPUS).get_top_n(["is"], CORPUS, n=2))
    run(lambda m: m.BM25Plus(CORPUS).get_top_n(["is"], CORPUS[:2]))


def tokenize(text):
    return text.lower().split()


def test_tokenizer():
    texts = [" ".join(d) for d in CORPUS]
    run(lambda m: m.BM25Okapi(texts, tokenizer=tokenize).get_scores(["windy"]))


class CustomIdf(rank_bm25_rs.BM25Okapi):
    def _calc_idf(self, nd):
        for word in nd:
            self.idf[word] = 1.0


def test_subclass_overriding_idf_uses_python_path():
    idx = CustomIdf(CORPUS)
    assert idx._core is None
    assert idx.idf == {w: 1.0 for d in CORPUS for w in d}


@pytest.mark.parametrize("name", VARIANTS)
def test_pickle(name):
    idx = getattr(rank_bm25_rs, name)(CORPUS)
    _ = idx.idf  # cached attributes travel too
    back = pickle.loads(pickle.dumps(idx))
    q = ["is", "windy"]
    assert np.array_equal(back.get_scores(q), idx.get_scores(q))
    assert back.doc_freqs == idx.doc_freqs


def test_isinstance_and_install():
    assert isinstance(rank_bm25_rs.BM25Okapi(CORPUS), rank_bm25_rs.BM25)
    code = (
        "import rank_bm25_rs; rank_bm25_rs.install()\n"
        "from rank_bm25 import BM25Okapi\n"
        "assert BM25Okapi is rank_bm25_rs.BM25Okapi\n"
        "print(BM25Okapi([['a', 'b'], ['b']]).get_scores(['a']).tolist())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(
        rank_bm25.BM25Okapi([["a", "b"], ["b"]]).get_scores(["a"]).tolist()
    )


def test_scores_are_writable_float64():
    s = rank_bm25_rs.BM25Okapi(CORPUS).get_scores(["is"])
    assert s.dtype == np.float64 and s.flags.writeable
    s += 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_get_scores_many_after_fork():
    code = textwrap.dedent(
        """
        import os, signal, rank_bm25_rs
        idx = rank_bm25_rs.BM25Okapi([["a", "b"], ["b"], ["c"]] * 50)
        idx.get_scores_many([["a"]] * 8)
        pid = os.fork()
        if pid == 0:
            signal.alarm(10)
            ok = idx.get_scores_many([["a"]] * 8).shape == (8, 150)
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        print("ok")
        """
    )
    env = dict(os.environ, RAYON_NUM_THREADS="2")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert out.stdout.strip() == "ok", out.stderr


# Corner cases found in review; each runs on both packages.


def test_integer_parameters_that_overflow_int64():
    corpus = [["a"] * 2048, ["b"], ["c"]]
    for kw in ({"k1": 2**52}, {"b": 2**50}, {"k1": 3, "b": 2}):
        for name in VARIANTS:
            run(lambda m, kw=kw, name=name: getattr(m, name)(corpus, **kw).get_scores(["a", "b"]))


def test_generator_corpus_is_consumed_lazily():
    def corpus():
        d = ["a"]
        yield d
        d[:] = ["b"]
        yield d

    run(lambda m: m.BM25L(corpus()).doc_freqs)

    def failing():
        yield [set()]
        raise RuntimeError("boom")

    run(lambda m: m.BM25L(failing()))


def test_query_terms_compare_like_dict_keys():
    import collections

    class Unhashable(str):
        __hash__ = None

    corpus = [["a"], ["b"], ["c"]]
    for name in VARIANTS:
        run(lambda m, name=name: getattr(m, name)(corpus).get_scores([collections.UserString("a")]))
        run(lambda m, name=name: getattr(m, name)(corpus).get_scores([Unhashable("a")]))
        run(
            lambda m, name=name: getattr(m, name)(corpus).get_batch_scores(
                [collections.UserString("a")], [0]
            )
        )


@pytest.mark.parametrize(
    "kw",
    [
        {"k1": float("nan")},
        {"k1": float("nan"), "epsilon": -float("nan")},
        {"b": float("nan"), "delta": -float("nan")},
        {"k1": float("inf")},
        {"k1": 1e308},
    ],
)
def test_non_finite_parameters(kw):
    for name in VARIANTS:
        kw2 = {
            k: v
            for k, v in kw.items()
            if k in ("k1", "b") or (k == "epsilon") == (name == "BM25Okapi")
        }
        run(
            lambda m, name=name, kw2=kw2: getattr(m, name)([["a"], ["b"], ["b"]], **kw2).get_scores(
                ["a", "b"]
            )
        )


def test_numpy_errstate_is_honoured():
    for name in VARIANTS:

        def scenario(m, name=name):
            idx = getattr(m, name)(
                [["a"], ["b"], ["c"]], k1=0, **({"delta": 0} if name != "BM25Okapi" else {})
            )
            with np.errstate(all="raise"):
                return idx.get_scores(["absent"])

        run(scenario)


def test_numpy_bool_doc_ids():
    for name in VARIANTS:
        run(
            lambda m, name=name: getattr(m, name)([["a", "a"], ["b"], ["c"]]).get_batch_scores(
                ["a"], np.array([True, False, True])
            )
        )
        run(
            lambda m, name=name: getattr(m, name)([["a", "a"], ["b"], ["c"]]).get_batch_scores(
                ["a"], np.array([0, 2])
            )
        )


def test_changed_parameters_are_used():
    for attr, value in (("k1", 3), ("b", 0.1), ("avgdl", 5.0), ("k1", float("nan"))):
        for name in VARIANTS:

            def scenario(m, name=name, attr=attr, value=value):
                idx = getattr(m, name)([["a", "a"], ["b"], ["c"]])
                setattr(idx, attr, value)
                return idx.get_scores(["a"]), idx.get_batch_scores(["a"], [0, 1, 2])

            run(scenario)
    idx = rank_bm25_rs.BM25L([["a", "a"], ["b"], ["c"]])
    idx.delta = 2
    assert np.array_equal(idx.get_scores(["a"]), np.array(idx.get_batch_scores(["a"], [0, 1, 2])))
    assert np.array_equal(idx.get_scores_many([["a"]])[0], idx.get_scores(["a"]))


def test_original_pickles_load_after_install():
    payload = pickle.dumps(rank_bm25.BM25Okapi([["a"], ["b"], ["c"]]))
    code = (
        "import pickle, sys, rank_bm25_rs; rank_bm25_rs.install()\n"
        f"idx = pickle.loads({payload!r})\n"
        "print(type(idx).__module__, idx.get_scores(['a']).tolist())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    expected = rank_bm25.BM25Okapi([["a"], ["b"], ["c"]]).get_scores(["a"]).tolist()
    assert out.stdout.strip() == f"rank_bm25_rs {expected}"


def test_reinitialising_clears_cached_attributes():
    for name in VARIANTS:

        def scenario(m, name=name):
            idx = getattr(m, name)([["a"]])
            _ = idx.idf, idx.doc_freqs, idx.doc_len
            idx.__init__([["b", "b"], ["c"]])
            return idx.idf, idx.doc_freqs, idx.doc_len, idx.get_scores(["b"])

        run(scenario)


def test_get_scores_many_uses_overridden_get_scores():
    class Shift(rank_bm25_rs.BM25L):
        def get_scores(self, query):
            return super().get_scores(query) + 1

    idx = Shift([["a"], ["b"], ["c"]])
    assert np.array_equal(idx.get_scores_many([["a"]])[0], idx.get_scores(["a"]))


def test_query_generators_keep_reference_order():
    import collections

    def mutating():
        t = collections.UserString("a")
        yield t
        t.data = "b"
        yield t

    run(lambda m: m.BM25L([["a"], ["b"], ["c"]]).get_scores(mutating()))

    def failing():
        yield []
        raise RuntimeError("boom")

    run(lambda m: m.BM25L([["a"], ["b"], ["c"]]).get_scores(failing()))


def test_get_scores_many_passes_queries_unchanged_to_overrides():
    class Lower(rank_bm25_rs.BM25L):
        def get_scores(self, q):
            return super().get_scores(q.lower().split())

    idx = Lower([["a"], ["b"], ["c"]])
    assert np.array_equal(idx.get_scores_many(["A"])[0], idx.get_scores("A"))


def test_integer_subclass_and_iterator_doc_ids():
    class OddInt(np.int64):
        def __int__(self):
            return 1

    run(lambda m: m.BM25L([["a", "a"], ["b"], ["c"]]).get_batch_scores(["a"], [OddInt(0)]))
    run(lambda m: m.BM25L([["a"], ["b"], ["c"]]).get_batch_scores(["a"], iter([99])))


def test_get_scores_many_scores_generators_as_they_come():
    idx = rank_bm25_rs.BM25L([["a"], ["b"], ["c"]])

    def queries():
        q = ["a"]
        yield q
        q[:] = ["b"]
        yield q

    expected = [idx.get_scores(q).tolist() for q in queries()]
    assert idx.get_scores_many(queries()).tolist() == expected
