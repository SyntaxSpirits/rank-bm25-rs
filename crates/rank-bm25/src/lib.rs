//! BM25Okapi, BM25L and BM25Plus with scores identical to Python's
//! [rank_bm25](https://github.com/dorianbrown/rank_bm25) 0.2.2, down to the last bit.
//!
//! rank_bm25 computes each term's contribution for every document with NumPy. This crate keeps
//! an inverted index, visits only the documents that contain a term where the other documents'
//! contribution is exactly zero, and performs the remaining floating-point operations in the
//! same order as NumPy, so every score equals rank_bm25's.
//!
//! ```
//! use rank_bm25::{Builder, Params};
//!
//! let mut builder = Builder::new(Params::okapi());
//! builder.add_document(["hello", "there", "good", "man"])?;
//! builder.add_document(["it", "is", "quite", "windy", "in", "london"])?;
//! builder.add_document(["how", "is", "the", "weather", "today"])?;
//! let index = builder.build()?;
//! let scores = index.get_scores(&["windy", "london"]);
//! assert!(scores[1] > 0.0 && scores[0] == 0.0);
//! # Ok::<(), rank_bm25::Error>(())
//! ```

use std::fmt;

use rustc_hash::FxHashMap;

/// The BM25 variant and its own parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Variant {
    /// `BM25Okapi`: idf floored at `epsilon * average idf` for terms in more than half of the
    /// documents.
    Okapi { epsilon: f64 },
    /// `BM25L`.
    L { delta: f64 },
    /// `BM25Plus`.
    Plus { delta: f64 },
}

/// `k1`, `b` and the variant, with rank_bm25's defaults in the constructors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub k1: f64,
    pub b: f64,
    pub variant: Variant,
}

impl Params {
    pub fn okapi() -> Self {
        Params {
            k1: 1.5,
            b: 0.75,
            variant: Variant::Okapi { epsilon: 0.25 },
        }
    }

    pub fn l() -> Self {
        Params {
            k1: 1.5,
            b: 0.75,
            variant: Variant::L { delta: 0.5 },
        }
    }

    pub fn plus() -> Self {
        Params {
            k1: 1.5,
            b: 0.75,
            variant: Variant::Plus { delta: 1.0 },
        }
    }
}

/// rank_bm25 divides by the number of documents, and BM25Okapi by the number of distinct terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// An empty corpus, or for BM25Okapi a corpus without any term (Python's `ZeroDivisionError`).
    ZeroDivision,
    /// A serialised index that cannot be read.
    InvalidData,
    /// More than `u32::MAX` documents, distinct terms or occurrences of a term in a document.
    TooLarge,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ZeroDivision => f.write_str("division by zero"),
            Error::InvalidData => f.write_str("invalid serialised index"),
            Error::TooLarge => f.write_str("corpus too large for 32-bit identifiers"),
        }
    }
}

impl std::error::Error for Error {}

/// Documents and frequencies of every term, stored contiguously term after term.
#[derive(Clone, Debug, Default)]
struct Postings {
    off: Vec<usize>,
    docs: Vec<u32>,
    tfs: Vec<u32>,
}

impl Postings {
    #[inline]
    fn get(&self, term: u32) -> (&[u32], &[u32]) {
        let (a, b) = (self.off[term as usize], self.off[term as usize + 1]);
        (&self.docs[a..b], &self.tfs[a..b])
    }

    fn terms(&self) -> usize {
        self.off.len() - 1
    }

    fn df(&self, term: usize) -> usize {
        self.off[term + 1] - self.off[term]
    }
}

/// Collects documents, then builds an [`Index`].
#[derive(Debug)]
pub struct Builder {
    params: Params,
    vocab: FxHashMap<Box<str>, u32>,
    terms: Vec<Box<str>>,
    df: Vec<u32>,
    doc_len: Vec<u64>,
    forward: Vec<u32>,
    forward_tf: Vec<u32>,
    forward_off: Vec<usize>,
    counts: Vec<u32>,
    touched: Vec<u32>,
    current_len: u64,
    /// Vocabulary size when the current document started.
    terms_at_start: usize,
}

impl Builder {
    pub fn new(params: Params) -> Self {
        Builder {
            params,
            vocab: FxHashMap::default(),
            terms: Vec::new(),
            df: Vec::new(),
            doc_len: Vec::new(),
            forward: Vec::new(),
            forward_tf: Vec::new(),
            forward_off: vec![0],
            counts: Vec::new(),
            touched: Vec::new(),
            current_len: 0,
            terms_at_start: 0,
        }
    }

    /// Adds one token to the current document.
    #[inline]
    pub fn push_token(&mut self, token: &str) -> Result<(), Error> {
        let id = match self.vocab.get(token) {
            Some(&id) => id,
            None => {
                let id = u32::try_from(self.terms.len()).map_err(|_| Error::TooLarge)?;
                self.vocab.insert(token.into(), id);
                self.terms.push(token.into());
                self.df.push(0);
                self.counts.push(0);
                id
            }
        };
        let count = &mut self.counts[id as usize];
        if *count == 0 {
            self.touched.push(id);
        }
        *count = count.checked_add(1).ok_or(Error::TooLarge)?;
        self.current_len += 1;
        Ok(())
    }

    /// Ends the current document; the next token starts a new one.
    pub fn end_document(&mut self) -> Result<(), Error> {
        if self.doc_len.len() >= u32::MAX as usize {
            return Err(Error::TooLarge);
        }
        for &t in &self.touched {
            let count = std::mem::take(&mut self.counts[t as usize]);
            self.df[t as usize] += 1;
            self.forward.push(t);
            self.forward_tf.push(count);
        }
        self.touched.clear();
        self.doc_len.push(self.current_len);
        self.current_len = 0;
        self.forward_off.push(self.forward.len());
        self.terms_at_start = self.terms.len();
        Ok(())
    }

    /// Discards the tokens of the current, unfinished document, including the terms it
    /// introduced.
    pub fn abandon_document(&mut self) {
        for &t in &self.touched {
            self.counts[t as usize] = 0;
        }
        self.touched.clear();
        self.current_len = 0;
        for term in self.terms.drain(self.terms_at_start..) {
            self.vocab.remove(&term);
        }
        self.df.truncate(self.terms_at_start);
        self.counts.truncate(self.terms_at_start);
    }

    pub fn add_document<'a, I: IntoIterator<Item = &'a str>>(
        &mut self,
        tokens: I,
    ) -> Result<(), Error> {
        for token in tokens {
            if let Err(e) = self.push_token(token) {
                self.abandon_document();
                return Err(e);
            }
        }
        self.end_document()
    }

    pub fn build(mut self) -> Result<Index, Error> {
        self.abandon_document();
        let mut off = Vec::with_capacity(self.df.len() + 1);
        off.push(0usize);
        for &df in &self.df {
            off.push(off.last().unwrap() + df as usize);
        }
        let total = *off.last().unwrap();
        let mut docs = vec![0u32; total];
        let mut tfs = vec![0u32; total];
        let mut cursor: Vec<usize> = off[..off.len() - 1].to_vec();
        for d in 0..self.doc_len.len() {
            let (a, b) = (self.forward_off[d], self.forward_off[d + 1]);
            for (&t, &tf) in self.forward[a..b].iter().zip(&self.forward_tf[a..b]) {
                let c = &mut cursor[t as usize];
                docs[*c] = d as u32;
                tfs[*c] = tf;
                *c += 1;
            }
        }
        drop(self.forward_tf);
        let postings = Postings { off, docs, tfs };
        Index::new(
            self.params,
            self.vocab,
            self.terms,
            postings,
            self.doc_len,
            self.forward,
            self.forward_off,
        )
    }
}

/// A BM25 index over a fixed corpus.
#[derive(Clone, Debug)]
pub struct Index {
    params: Params,
    avgdl: f64,
    vocab: FxHashMap<Box<str>, u32>,
    terms: Vec<Box<str>>,
    idf: Vec<f64>,
    average_idf: Option<f64>,
    postings: Postings,
    doc_len: Vec<u64>,
    forward: Vec<u32>,
    forward_off: Vec<usize>,
    /// Okapi and Plus: `k1 * ((1 - b) + (b * dl) / avgdl)`; L: `(1 - b) + (b * dl) / avgdl`.
    norm: Vec<f64>,
    kp1: f64,
    /// Documents without a term contribute exactly zero, so only its postings need visiting.
    sparse: bool,
    /// Every contribution is finite whatever the query, so scores need no NaN check.
    always_finite: bool,
    max_tf: u32,
}

impl Index {
    fn new(
        params: Params,
        vocab: FxHashMap<Box<str>, u32>,
        terms: Vec<Box<str>>,
        postings: Postings,
        doc_len: Vec<u64>,
        forward: Vec<u32>,
        forward_off: Vec<usize>,
    ) -> Result<Self, Error> {
        let n = doc_len.len();
        if n == 0 {
            return Err(Error::ZeroDivision);
        }
        let num_doc = doc_len
            .iter()
            .try_fold(0u64, |a, &l| a.checked_add(l))
            .ok_or(Error::TooLarge)?;
        let avgdl = num_doc as f64 / n as f64;
        let nterms = postings.terms();
        let mut idf = Vec::with_capacity(nterms);
        let mut average_idf = None;
        match params.variant {
            Variant::Okapi { epsilon } => {
                if nterms == 0 {
                    return Err(Error::ZeroDivision);
                }
                let mut idf_sum = 0.0f64;
                let mut negative = Vec::new();
                for t in 0..nterms {
                    let freq = postings.df(t) as u64;
                    let v = ((n as u64 - freq) as f64 + 0.5).ln() - (freq as f64 + 0.5).ln();
                    idf.push(v);
                    idf_sum += v;
                    if v < 0.0 {
                        negative.push(t);
                    }
                }
                let average = idf_sum / nterms as f64;
                let eps = epsilon * average;
                for t in negative {
                    idf[t] = eps;
                }
                average_idf = Some(average);
            }
            Variant::L { .. } => {
                for t in 0..nterms {
                    idf.push(((n + 1) as f64).ln() - (postings.df(t) as f64 + 0.5).ln());
                }
            }
            Variant::Plus { .. } => {
                for t in 0..nterms {
                    idf.push(((n + 1) as f64 / postings.df(t) as f64).ln());
                }
            }
        }
        let Params { k1, b, variant } = params;
        let norm: Vec<f64> = doc_len
            .iter()
            .map(|&dl| {
                let l = (1.0 - b) + (b * dl as f64) / avgdl;
                match variant {
                    Variant::L { .. } => l,
                    _ => k1 * l,
                }
            })
            .collect();
        let kp1 = k1 + 1.0;
        let norm_ok = norm.iter().all(|&x| x != 0.0 && !x.is_nan());
        let sparse = norm_ok
            && kp1.is_finite()
            && match variant {
                Variant::Okapi { .. } => true,
                Variant::L { delta } => {
                    let denominator = (k1 + 0.0) + delta;
                    delta.is_finite() && denominator != 0.0 && !denominator.is_nan()
                }
                Variant::Plus { .. } => false,
            };
        let moderate = |x: f64| x.is_finite() && x.abs() <= 1e100;
        let extra = match variant {
            Variant::Okapi { epsilon } => epsilon,
            Variant::L { delta } | Variant::Plus { delta } => delta,
        };
        let always_finite = moderate(k1)
            && moderate(b)
            && moderate(extra)
            && norm.iter().all(|&x| x > 0.0 && x.is_finite())
            && idf.iter().all(|x| x.is_finite())
            && match variant {
                Variant::L { delta } => k1 >= 0.0 && delta >= 0.0 && k1 + delta > 0.0,
                _ => true,
            };
        Ok(Index {
            always_finite,
            params,
            avgdl,
            vocab,
            terms,
            idf,
            average_idf,
            max_tf: postings.tfs.iter().copied().max().unwrap_or(0),
            postings,
            doc_len,
            forward,
            forward_off,
            norm,
            kp1,
            sparse,
        })
    }

    pub fn params(&self) -> Params {
        self.params
    }

    /// Number of documents.
    pub fn corpus_size(&self) -> usize {
        self.doc_len.len()
    }

    pub fn avgdl(&self) -> f64 {
        self.avgdl
    }

    /// The average idf before flooring, for BM25Okapi.
    pub fn average_idf(&self) -> Option<f64> {
        self.average_idf
    }

    /// True when no query can produce a NaN or infinite score.
    pub fn always_finite(&self) -> bool {
        self.always_finite
    }

    /// The largest frequency of a term in a document.
    pub fn max_tf(&self) -> u32 {
        self.max_tf
    }

    pub fn doc_len(&self) -> &[u64] {
        &self.doc_len
    }

    /// Terms with their idf, in the order rank_bm25 stores them (first occurrence in the corpus).
    pub fn idf(&self) -> impl Iterator<Item = (&str, f64)> {
        self.terms
            .iter()
            .map(|t| &**t)
            .zip(self.idf.iter().copied())
    }

    /// The idf of a term returned by [`Index::term_id`].
    pub fn term_idf(&self, term: u32) -> f64 {
        self.idf[term as usize]
    }

    pub fn term_id(&self, term: &str) -> Option<u32> {
        self.vocab.get(term).copied()
    }

    /// The terms of document `doc` with their frequencies, in order of first occurrence.
    pub fn document(&self, doc: usize) -> impl Iterator<Item = (&str, u32)> + '_ {
        self.forward[self.forward_off[doc]..self.forward_off[doc + 1]]
            .iter()
            .map(move |&t| (&*self.terms[t as usize], self.tf(t, doc as u32)))
    }

    /// Frequency of term `term` in document `doc`.
    pub fn tf(&self, term: u32, doc: u32) -> u32 {
        let (docs, tfs) = self.postings.get(term);
        match docs.binary_search(&doc) {
            Ok(i) => tfs[i],
            Err(_) => 0,
        }
    }

    #[inline]
    fn contribution(&self, idf: f64, f: f64, d: usize) -> f64 {
        let Params { k1, variant, .. } = self.params;
        match variant {
            Variant::Okapi { .. } => idf * ((f * self.kp1) / (f + self.norm[d])),
            Variant::L { delta } => {
                let ctd = f / self.norm[d];
                (((idf * f) * self.kp1) * (ctd + delta)) / ((k1 + ctd) + delta)
            }
            Variant::Plus { delta } => idf * (delta + ((f * self.kp1) / (self.norm[d] + f))),
        }
    }

    /// Adds the contributions of `term` (None for a term not in the corpus) to `score`.
    pub fn add_term(&self, term: Option<u32>, score: &mut [f64]) {
        let (idf, docs, tfs): (f64, &[u32], &[u32]) = match term {
            Some(t) => {
                let (docs, tfs) = self.postings.get(t);
                (self.idf[t as usize], docs, tfs)
            }
            None => (0.0, &[], &[]),
        };
        if self.sparse && idf.is_finite() {
            for (&d, &tf) in docs.iter().zip(tfs) {
                score[d as usize] += self.contribution(idf, tf as f64, d as usize);
            }
            return;
        }
        let mut next = docs.iter().zip(tfs).peekable();
        for (d, s) in score.iter_mut().enumerate() {
            let f = match next.peek() {
                Some(&(&pd, &tf)) if pd as usize == d => {
                    next.next();
                    tf as f64
                }
                _ => 0.0,
            };
            *s += self.contribution(idf, f, d);
        }
    }

    /// `get_scores` for query terms already looked up with [`Index::term_id`].
    pub fn scores_into(&self, query: &[Option<u32>], score: &mut [f64]) {
        score.fill(0.0);
        for &term in query {
            self.add_term(term, score);
        }
    }

    /// The score of every document, as `get_scores` in rank_bm25.
    pub fn get_scores<S: AsRef<str>>(&self, query: &[S]) -> Vec<f64> {
        let ids: Vec<Option<u32>> = query.iter().map(|q| self.term_id(q.as_ref())).collect();
        let mut score = vec![0.0; self.corpus_size()];
        self.scores_into(&ids, &mut score);
        score
    }

    /// Serialises the index.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer(Vec::new());
        w.0.extend_from_slice(b"BM25R1");
        let (kind, extra) = match self.params.variant {
            Variant::Okapi { epsilon } => (0u8, epsilon),
            Variant::L { delta } => (1, delta),
            Variant::Plus { delta } => (2, delta),
        };
        w.0.push(kind);
        w.f64(self.params.k1);
        w.f64(self.params.b);
        w.f64(extra);
        w.u64(self.doc_len.len() as u64);
        for &dl in &self.doc_len {
            w.u64(dl);
        }
        w.u64(self.terms.len() as u64);
        for (t, term) in self.terms.iter().enumerate() {
            w.u64(term.len() as u64);
            w.0.extend_from_slice(term.as_bytes());
            let (docs, tfs) = self.postings.get(t as u32);
            w.u64(docs.len() as u64);
            for (&d, &tf) in docs.iter().zip(tfs) {
                w.u32(d);
                w.u32(tf);
            }
        }
        w.u64(self.forward.len() as u64);
        for &t in &self.forward {
            w.u32(t);
        }
        for &o in &self.forward_off {
            w.u64(o as u64);
        }
        w.0
    }

    /// Restores an index saved with [`Index::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut r = Reader { buf: bytes, pos: 0 };
        if r.take(6)? != b"BM25R1" {
            return Err(Error::InvalidData);
        }
        let kind = r.take(1)?[0];
        let k1 = r.f64()?;
        let b = r.f64()?;
        let extra = r.f64()?;
        let variant = match kind {
            0 => Variant::Okapi { epsilon: extra },
            1 => Variant::L { delta: extra },
            2 => Variant::Plus { delta: extra },
            _ => return Err(Error::InvalidData),
        };
        let n = r.len()?;
        let mut doc_len = Vec::with_capacity(n);
        for _ in 0..n {
            doc_len.push(r.u64()?);
        }
        let nterms = r.len()?;
        let mut vocab = FxHashMap::default();
        let mut terms = Vec::with_capacity(nterms);
        let mut postings = Postings {
            off: vec![0],
            ..Default::default()
        };
        for t in 0..nterms {
            let len = r.len()?;
            let term: Box<str> = std::str::from_utf8(r.take(len)?)
                .map_err(|_| Error::InvalidData)?
                .into();
            if vocab.insert(term.clone(), t as u32).is_some() {
                return Err(Error::InvalidData);
            }
            terms.push(term);
            let plen = r.len()?;
            let mut last = None;
            for _ in 0..plen {
                let d = r.u32()?;
                if d as usize >= n || last.is_some_and(|l| d <= l) {
                    return Err(Error::InvalidData);
                }
                last = Some(d);
                postings.docs.push(d);
                postings.tfs.push(r.u32()?);
            }
            postings.off.push(postings.docs.len());
        }
        let nf = r.len()?;
        let mut forward = Vec::with_capacity(nf);
        for _ in 0..nf {
            let t = r.u32()?;
            if t as usize >= nterms {
                return Err(Error::InvalidData);
            }
            forward.push(t);
        }
        let mut forward_off = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            let o = r.u64()? as usize;
            if o > nf || forward_off.last().is_some_and(|&p| o < p) {
                return Err(Error::InvalidData);
            }
            forward_off.push(o);
        }
        if r.pos != bytes.len()
            || forward_off.first() != Some(&0)
            || forward_off.last() != Some(&forward.len())
            || postings.docs.len() != forward.len()
            || postings.tfs.contains(&0)
            || (0..nterms).any(|t| postings.df(t) == 0)
        {
            return Err(Error::InvalidData);
        }
        // Every document's terms must match the postings and add up to its length.
        let mut seen = vec![u32::MAX; nterms];
        for d in 0..n {
            let mut len = 0u64;
            for &t in &forward[forward_off[d]..forward_off[d + 1]] {
                if seen[t as usize] == d as u32 {
                    return Err(Error::InvalidData);
                }
                seen[t as usize] = d as u32;
                let (docs, tfs) = postings.get(t);
                let i = docs
                    .binary_search(&(d as u32))
                    .map_err(|_| Error::InvalidData)?;
                len += tfs[i] as u64;
            }
            if len != doc_len[d] {
                return Err(Error::InvalidData);
            }
        }
        Index::new(
            Params { k1, b, variant },
            vocab,
            terms,
            postings,
            doc_len,
            forward,
            forward_off,
        )
    }
}

struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or(Error::InvalidData)?;
        let s = self.buf.get(self.pos..end).ok_or(Error::InvalidData)?;
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, Error> {
        Ok(f64::from_bits(self.u64()?))
    }
    fn len(&mut self) -> Result<usize, Error> {
        let v = self.u64()? as usize;
        if v > self.buf.len() {
            return Err(Error::InvalidData);
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(params: Params) -> Index {
        let mut b = Builder::new(params);
        b.add_document(["hello", "there", "good", "man"]).unwrap();
        b.add_document(["it", "is", "quite", "windy", "in", "london"])
            .unwrap();
        b.add_document(["how", "is", "the", "weather", "today"])
            .unwrap();
        b.build().unwrap()
    }

    // Values from rank_bm25 0.2.2.
    #[test]
    fn matches_rank_bm25() {
        let q = ["windy", "london"];
        assert_eq!(
            index(Params::okapi()).get_scores(&q),
            [0.0, 0.9372947225064051, 0.0]
        );
        assert_eq!(
            index(Params::l()).get_scores(&q),
            [0.0, 2.340615262868892, 0.0]
        );
        assert_eq!(
            index(Params::plus()).get_scores(&q),
            [2.772588722239781, 5.316248100441416, 2.772588722239781]
        );
    }

    #[test]
    fn document_keeps_first_occurrence_order() {
        let mut b = Builder::new(Params::okapi());
        b.add_document(["b", "a", "b", "c"]).unwrap();
        let idx = b.build().unwrap();
        assert_eq!(
            idx.document(0).collect::<Vec<_>>(),
            [("b", 2), ("a", 1), ("c", 1)]
        );
    }

    #[test]
    fn empty_corpus_is_an_error() {
        assert_eq!(
            Builder::new(Params::okapi()).build().unwrap_err(),
            Error::ZeroDivision
        );
        let mut b = Builder::new(Params::okapi());
        b.add_document(Vec::<&str>::new()).unwrap();
        assert_eq!(b.build().unwrap_err(), Error::ZeroDivision);
        let mut b = Builder::new(Params::l());
        b.add_document(Vec::<&str>::new()).unwrap();
        assert!(b.build().is_ok());
    }

    #[test]
    fn abandoned_terms_leave_no_trace() {
        let mut b = Builder::new(Params::okapi());
        b.add_document(["a"]).unwrap();
        b.push_token("discarded").unwrap();
        b.abandon_document();
        let with_abandon = b.build().unwrap();
        let mut c = Builder::new(Params::okapi());
        c.add_document(["a"]).unwrap();
        let plain = c.build().unwrap();
        assert_eq!(with_abandon.idf().count(), 1);
        assert_eq!(with_abandon.get_scores(&["a"]), plain.get_scores(&["a"]));
    }

    #[test]
    fn inconsistent_bytes_are_rejected() {
        let idx = index(Params::l());
        let bytes = idx.to_bytes();
        let mut zeroed = bytes.clone();
        let n = zeroed.len();
        zeroed[n - 16..].fill(0);
        assert!(Index::from_bytes(&zeroed).is_err());
        for i in 0..bytes.len() {
            let mut b = bytes.clone();
            b[i] ^= 0xff;
            let _ = Index::from_bytes(&b);
        }
    }

    #[test]
    fn terms_without_postings_are_rejected() {
        let mut w = Writer(Vec::new());
        w.0.extend_from_slice(b"BM25R1");
        w.0.push(0);
        for v in [1.5, 0.75, 0.25] {
            w.f64(v);
        }
        w.u64(1);
        w.u64(1);
        w.u64(2);
        for (term, postings) in [("a", 1u64), ("ghost", 0)] {
            w.u64(term.len() as u64);
            w.0.extend_from_slice(term.as_bytes());
            w.u64(postings);
            for _ in 0..postings {
                w.u32(0);
                w.u32(1);
            }
        }
        w.u64(1);
        w.u32(0);
        w.u64(0);
        w.u64(1);
        assert_eq!(Index::from_bytes(&w.0).unwrap_err(), Error::InvalidData);
    }

    #[test]
    fn index_is_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<Index>();
    }

    #[test]
    fn serialisation_round_trips() {
        for params in [Params::okapi(), Params::l(), Params::plus()] {
            let idx = index(params);
            let back = Index::from_bytes(&idx.to_bytes()).unwrap();
            let q = ["is", "windy", "nothing"];
            assert_eq!(back.get_scores(&q), idx.get_scores(&q));
        }
        assert!(Index::from_bytes(b"BM25R1").is_err());
    }
}
