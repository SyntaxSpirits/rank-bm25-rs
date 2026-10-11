use std::sync::OnceLock;

use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyString, PyTuple};
use rank_bm25::{Builder, Error, Index as CoreIndex, Params, Variant};
use rayon::prelude::*;

fn to_py_err(e: Error) -> PyErr {
    match e {
        Error::ZeroDivision => PyZeroDivisionError::new_err("division by zero"),
        Error::InvalidData | Error::TooLarge => PyValueError::new_err(e.to_string()),
    }
}

fn as_bytes(v: &[f64]) -> &[u8] {
    // SAFETY: f64 has no padding or invalid bit patterns, and u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// The process that started Rayon's global thread pool. Its threads do not exist in a forked
/// child, so a child scores queries on the calling thread.
static POOL_OWNER: OnceLock<u32> = OnceLock::new();

fn thread_pool_usable() -> bool {
    *POOL_OWNER.get_or_init(std::process::id) == std::process::id()
}

/// The items of an exact list or tuple, or None for any other type.
fn sequence<'py>(obj: &Bound<'py, PyAny>) -> Option<Vec<Bound<'py, PyAny>>> {
    if let Ok(list) = obj.cast_exact::<PyList>() {
        Some(list.iter().collect())
    } else if let Ok(tuple) = obj.cast_exact::<PyTuple>() {
        Some(tuple.iter().collect())
    } else {
        None
    }
}

/// Adds the tokens of an exact list or tuple of exact `str`; false for anything else.
fn push_document(builder: &mut Builder, doc: &Bound<'_, PyAny>) -> bool {
    let ptr = doc.as_ptr();
    // SAFETY: `doc` keeps the sequence alive; PyList_GetItem and PyTuple_GetItem return borrowed
    // references to items the sequence owns, valid while it is not modified, and no Python
    // code runs in between. PyUnicode_AsUTF8AndSize returns a buffer cached in the str object.
    unsafe {
        let (len, get): (ffi::Py_ssize_t, unsafe extern "C" fn(_, _) -> _) =
            if ffi::PyList_CheckExact(ptr) != 0 {
                (ffi::PyList_Size(ptr), ffi::PyList_GetItem)
            } else if ffi::PyTuple_CheckExact(ptr) != 0 {
                (ffi::PyTuple_Size(ptr), ffi::PyTuple_GetItem)
            } else {
                return false;
            };
        for i in 0..len {
            let item = get(ptr, i);
            if item.is_null() || ffi::PyUnicode_CheckExact(item) == 0 {
                ffi::PyErr_Clear();
                return false;
            }
            let mut size: ffi::Py_ssize_t = 0;
            let data = ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
            if data.is_null() {
                // Lone surrogates have no UTF-8 form.
                ffi::PyErr_Clear();
                return false;
            }
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), size as usize);
            if builder
                .push_token(std::str::from_utf8_unchecked(bytes))
                .is_err()
            {
                return false;
            }
        }
    }
    true
}

/// Builds the index for an exact list or tuple of documents that are exact lists or tuples of
/// exact `str`. Returns None for any other corpus, which the Python layer hands to rank_bm25's
/// own code.
#[pyfunction]
fn build(
    corpus: &Bound<'_, PyAny>,
    variant: &str,
    k1: f64,
    b: f64,
    extra: f64,
) -> PyResult<Option<Index>> {
    let variant = match variant {
        "okapi" => Variant::Okapi { epsilon: extra },
        "l" => Variant::L { delta: extra },
        "plus" => Variant::Plus { delta: extra },
        _ => return Err(PyValueError::new_err("unknown variant")),
    };
    let Some(docs) = sequence(corpus) else {
        return Ok(None);
    };
    let mut builder = Builder::new(Params { k1, b, variant });
    for doc in &docs {
        if !push_document(&mut builder, doc) || builder.end_document().is_err() {
            return Ok(None);
        }
    }
    builder
        .build()
        .map(|inner| Some(Index { inner }))
        .map_err(to_py_err)
}

#[pyclass(frozen, module = "rank_bm25_rs._core")]
struct Index {
    inner: CoreIndex,
}

impl Index {
    /// The term id of an exact `str` (the Python layer passes nothing else).
    fn term(&self, item: &Bound<'_, PyAny>) -> Option<u32> {
        let s = item.cast_exact::<PyString>().ok()?;
        self.inner.term_id(s.to_str().ok()?)
    }

    fn lookup(&self, query: &Bound<'_, PyAny>) -> PyResult<Vec<Option<u32>>> {
        query
            .try_iter()?
            .map(|item| Ok(self.term(&item?)))
            .collect()
    }

    /// Scores into `out`; false when a score is NaN or infinite.
    fn score(&self, query: &[Option<u32>], out: &mut [f64]) -> bool {
        self.inner.scores_into(query, out);
        self.inner.always_finite() || out.iter().all(|x| x.is_finite())
    }

    /// A bytearray of `rows * corpus_size` float64 scores filled by `fill`, which returns
    /// whether each row is finite.
    fn scores_bytearray<'py>(
        &self,
        py: Python<'py>,
        rows: usize,
        fill: impl Fn(&mut [f64]) -> Vec<bool> + Send + Sync,
    ) -> PyResult<(Bound<'py, PyByteArray>, Vec<bool>)> {
        let len = rows * self.inner.corpus_size();
        let mut finite = Vec::new();
        let buf = PyByteArray::new_with(py, len * 8, |bytes| {
            // SAFETY: f64 accepts every bit pattern; align_to_mut only yields the aligned middle.
            let (head, scores, _) = unsafe { bytes.align_to_mut::<f64>() };
            if head.is_empty() && scores.len() == len {
                finite = py.detach(|| fill(scores));
            } else {
                let mut tmp = vec![0.0f64; len];
                finite = py.detach(|| fill(&mut tmp));
                bytes.copy_from_slice(as_bytes(&tmp));
            }
            Ok(())
        })?;
        Ok((buf, finite))
    }
}

#[pymethods]
impl Index {
    #[staticmethod]
    fn _from_bytes(data: &[u8]) -> PyResult<Self> {
        CoreIndex::from_bytes(data)
            .map(|inner| Index { inner })
            .map_err(to_py_err)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let bytes = PyBytes::new(py, &slf.get().inner.to_bytes());
        let constructor = slf.get_type().getattr("_from_bytes")?;
        PyTuple::new(
            py,
            [
                constructor.into_any(),
                PyTuple::new(py, [bytes])?.into_any(),
            ],
        )
    }

    #[getter]
    fn corpus_size(&self) -> usize {
        self.inner.corpus_size()
    }

    #[getter]
    fn avgdl(&self) -> f64 {
        self.inner.avgdl()
    }

    #[getter]
    fn average_idf(&self) -> Option<f64> {
        self.inner.average_idf()
    }

    #[getter]
    fn max_tf(&self) -> u32 {
        self.inner.max_tf()
    }

    #[getter]
    fn max_doc_len(&self) -> u64 {
        self.inner.doc_len().iter().copied().max().unwrap_or(0)
    }

    fn doc_len(&self) -> Vec<u64> {
        self.inner.doc_len().to_vec()
    }

    fn idf(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        for (term, idf) in self.inner.idf() {
            d.set_item(term, idf)?;
        }
        Ok(d.unbind())
    }

    /// The idf of a term, or None if the corpus does not contain it.
    fn idf_of(&self, term: &Bound<'_, PyAny>) -> Option<f64> {
        self.term(term).map(|t| self.inner.term_idf(t))
    }

    fn doc_freqs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for d in 0..self.inner.corpus_size() {
            let freqs = PyDict::new(py);
            for (term, tf) in self.inner.document(d) {
                freqs.set_item(term, tf)?;
            }
            out.append(freqs)?;
        }
        Ok(out)
    }

    /// Frequencies of one query term in the given documents (indices already normalised).
    fn tfs(&self, term: &Bound<'_, PyAny>, doc_ids: Vec<u32>) -> Vec<u32> {
        match self.term(term) {
            Some(t) => doc_ids.iter().map(|&d| self.inner.tf(t, d)).collect(),
            None => vec![0; doc_ids.len()],
        }
    }

    /// `get_scores` as raw float64 bytes, and whether every score is finite.
    fn get_scores<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyByteArray>, bool)> {
        let ids = self.lookup(query)?;
        let (buf, finite) = self.scores_bytearray(py, 1, |out| vec![self.score(&ids, out)])?;
        Ok((buf, finite[0]))
    }

    /// `get_scores` for several queries on all cores, as one row-major float64 buffer, and
    /// whether each row is finite.
    fn get_scores_many<'py>(
        &self,
        py: Python<'py>,
        queries: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyByteArray>, Vec<bool>)> {
        let ids = queries
            .iter()
            .map(|q| self.lookup(q))
            .collect::<PyResult<Vec<_>>>()?;
        let n = self.inner.corpus_size();
        if n == 0 || ids.is_empty() {
            return Ok((PyByteArray::new(py, &[]), vec![true; ids.len()]));
        }
        self.scores_bytearray(py, ids.len(), |out| {
            let work = |(row, q): (&mut [f64], &Vec<Option<u32>>)| self.score(q, row);
            if thread_pool_usable() {
                out.par_chunks_mut(n)
                    .zip(ids.par_iter())
                    .map(work)
                    .collect()
            } else {
                out.chunks_mut(n).zip(ids.iter()).map(work).collect()
            }
        })
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Index>()?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    Ok(())
}
