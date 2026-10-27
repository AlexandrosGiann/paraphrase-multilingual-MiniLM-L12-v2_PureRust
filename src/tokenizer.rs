//! Tokenizer of paraphrase-multilingual-MiniLM-L12-v2 (XLM-R SentencePiece Unigram), giving the
//! same ids as Transformers.js 3.8.1:
//!   - added tokens (`<s> <pad> </s> <unk> <mask>`) split out first, longest match; `<mask>`
//!     strips the whitespace before it (lstrip)
//!   - the Precompiled normalizer as Transformers.js implements it: control characters
//!     removed, some spaces mapped to ' ', then NFKC (Unicode 17)
//!   - WhitespaceSplit + Metaspace ('\u{2581}' prepended to every word)
//!   - Unigram Viterbi with the same tie-breaking, unknown pieces fused
//!   - `<s> ... </s>`, truncation to 512 tokens
//!
//! Text is handled as code points where a lone surrogate is an element of its own, like a
//! JavaScript string, so `encode_utf16` reproduces Transformers.js even for ill-formed UTF-16.

use crate::json::Value;
use crate::nfkc::nfkc;
use std::collections::HashMap;

pub struct AddedToken {
    pub id: u32,
    pub content: Vec<u32>,
    pub lstrip: bool,
    pub rstrip: bool,
}

pub struct Tokenizer {
    ids: HashMap<String, u32>,
    scores: Vec<f64>,
    unk_id: u32,
    unk_score: f64,
    max_len: usize,
    added: Vec<AddedToken>,
    replacement: u32,
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub max_length: usize,
}

/// UTF-16 code units -> code points (a lone surrogate stays one element, like Array.from).
pub fn code_points(s: &[u16]) -> Vec<u32> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let c = s[i] as u32;
        i += 1;
        if (0xD800..=0xDBFF).contains(&c) && i < s.len() {
            let d = s[i] as u32;
            if (0xDC00..=0xDFFF).contains(&d) {
                i += 1;
                out.push((c - 0xD800) * 0x400 + (d - 0xDC00) + 0x10000);
                continue;
            }
        }
        out.push(c);
    }
    out
}

// Characters of JavaScript's \s (and String.prototype.trim) in current engines.
fn is_js_space(c: u32) -> bool {
    c == 0x20 || (0x09..=0x0D).contains(&c) || c == 0xA0 || c == 0x1680 || (0x2000..=0x200A).contains(&c)
        || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000 || c == 0xFEFF
}

// [\u0001-\u0008\u000B\u000E-\u001F\u007F\u008F\u009F]: removed
fn is_removed_control(c: u32) -> bool {
    (0x01..=0x08).contains(&c) || c == 0x0B || (0x0E..=0x1F).contains(&c) || c == 0x7F || c == 0x8F || c == 0x9F
}

// [\u0009\u000A\u000C\u000D   -‏    ▁　﻿�]: -> ' '
fn is_mapped_space(c: u32) -> bool {
    c == 0x09 || c == 0x0A || c == 0x0C || c == 0x0D || c == 0xA0 || c == 0x1680 || (0x2000..=0x200F).contains(&c)
        || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x2581 || c == 0x3000 || c == 0xFEFF || c == 0xFFFD
}

/// Code points -> UTF-16 code units (lone surrogates stay single units).
pub fn to_utf16(cps: &[u32]) -> Vec<u16> {
    let mut out = Vec::with_capacity(cps.len());
    for &c in cps {
        if c >= 0x10000 {
            let c = c - 0x10000;
            out.push(0xD800 + (c >> 10) as u16);
            out.push(0xDC00 + (c & 0x3FF) as u16);
        } else {
            out.push(c as u16);
        }
    }
    out
}

/// Transformers.js Precompiled normalizer (nmt_nfkc as it implements it). The replacements
/// work on UTF-16 code units, like the JavaScript regexes: removing a control character
/// between a lone high and a lone low surrogate joins them into one code point.
pub fn normalize(text: &[u32]) -> Vec<u32> {
    let units: Vec<u16> = to_utf16(text)
        .into_iter()
        .filter(|&c| !is_removed_control(c as u32))
        .map(|c| if is_mapped_space(c as u32) { 0x20 } else { c })
        .collect();
    let t = code_points(&units);
    let out = if t.contains(&0xFF5E) {
        let mut out = Vec::with_capacity(t.len());
        for (i, part) in t.split(|&c| c == 0xFF5E).enumerate() {
            if i > 0 {
                out.push(0xFF5E);
            }
            out.extend(nfkc(part));
        }
        out
    } else {
        nfkc(&t)
    };
    // back through a string, as in JavaScript (adjacent lone surrogates pair up)
    code_points(&to_utf16(&out))
}

/// The piece as a Rust string, or None if it holds a lone surrogate (no such piece exists).
fn piece_string(cps: &[u32]) -> Option<String> {
    let mut s = String::with_capacity(cps.len() * 2);
    for &c in cps {
        s.push(char::from_u32(c)?);
    }
    Some(s)
}

impl Tokenizer {
    /// vocab: one piece per line (vocab.txt), scores: little-endian f64 (scores.bin),
    /// cfg: the "tokenizer" object of manifest.json.
    pub fn new(vocab: &str, scores_bytes: &[u8], cfg: &Value) -> Tokenizer {
        let mut ids = HashMap::new();
        let mut max_len = 0;
        for (i, p) in vocab.split('\n').enumerate() {
            max_len = max_len.max(p.chars().count());
            ids.insert(p.to_string(), i as u32);
        }
        let mut scores: Vec<f64> = scores_bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(scores.len(), ids.len(), "vocab.txt and scores.bin do not match");
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let unk_id = cfg.get("unkId").usize() as u32;
        let unk_score = min - 10.0;
        scores[unk_id as usize] = unk_score;
        let added = cfg
            .get("added")
            .arr()
            .iter()
            .map(|a| AddedToken {
                id: a.get("id").usize() as u32,
                content: code_points(a.get("content").utf16()),
                lstrip: a.get("lstrip").bool(),
                rstrip: a.get("rstrip").bool(),
            })
            .collect();
        Tokenizer {
            ids,
            scores,
            unk_id,
            unk_score,
            max_len,
            added,
            replacement: code_points(cfg.get("replacement").utf16())[0],
            bos: cfg.get("bos").usize() as u32,
            eos: cfg.get("eos").usize() as u32,
            pad: cfg.get("pad").usize() as u32,
            max_length: cfg.get("maxLength").usize(),
        }
    }

    fn id_of(&self, piece: &[u32]) -> Option<u32> {
        piece_string(piece).and_then(|s| self.ids.get(&s).cloned())
    }

    fn added_token(&self, s: &[u32]) -> Option<&AddedToken> {
        self.added.iter().find(|a| a.content == s)
    }

    // DictionarySplitter: split around the added tokens (longest match)
    fn split_added(&self, text: &[u32]) -> Vec<Vec<u32>> {
        let (mut out, n, mut start, mut i) = (Vec::new(), text.len(), 0, 0);
        while i < n {
            let mut m: Option<&Vec<u32>> = None;
            for a in &self.added {
                if text[i..].starts_with(&a.content) && m.map_or(true, |x| a.content.len() > x.len()) {
                    m = Some(&a.content);
                }
            }
            match m {
                Some(t) => {
                    if i > start {
                        out.push(text[start..i].to_vec());
                    }
                    out.push(t.clone());
                    i += t.len();
                    start = i;
                }
                None => i += 1,
            }
        }
        if start < n {
            out.push(text[start..].to_vec());
        }
        out
    }

    // Unigram Viterbi over one pre-token; returns the pieces.
    fn viterbi(&self, chars: &[u32]) -> Vec<Vec<u32>> {
        let len = chars.len();
        // nodes in insertion order; node 0 = BOS (ends at 0), node 1 = EOS (begins at len)
        let mut pos = vec![0usize, len];
        let mut nlen = vec![0usize, 0];
        let mut score = vec![0f64, 0.0];
        let mut prev: Vec<isize> = vec![-1, -1];
        let mut bt = vec![0f64, 0.0];
        let mut begin: Vec<Vec<usize>> = vec![Vec::new(); len + 1];
        let mut end: Vec<Vec<usize>> = vec![Vec::new(); len + 1];
        end[0].push(0);
        begin[len].push(1);
        let mut piece = String::new();
        for p in 0..len {
            let mut single = false;
            piece.clear();
            let mut valid = true;
            let mut k = 1;
            while k <= self.max_len && p + k <= len {
                match char::from_u32(chars[p + k - 1]) {
                    Some(c) => piece.push(c),
                    None => valid = false,
                }
                if valid {
                    if let Some(&id) = self.ids.get(&piece) {
                        let node = pos.len();
                        pos.push(p);
                        nlen.push(k);
                        score.push(self.scores[id as usize]);
                        prev.push(-1);
                        bt.push(0.0);
                        begin[p].push(node);
                        end[p + k].push(node);
                        if k == 1 {
                            single = true;
                        }
                    }
                }
                k += 1;
            }
            if !single {
                let node = pos.len();
                pos.push(p);
                nlen.push(1);
                score.push(self.unk_score);
                prev.push(-1);
                bt.push(0.0);
                begin[p].push(node);
                end[p + 1].push(node);
            }
        }
        for p in 0..=len {
            if begin[p].is_empty() {
                return Vec::new();
            }
            for &r in &begin[p] {
                let mut best: isize = -1;
                let mut best_score = 0f64;
                for &l in &end[p] {
                    let s = bt[l] + score[r];
                    if best < 0 || s > best_score {
                        best = l as isize;
                        best_score = s;
                    }
                }
                if best < 0 {
                    return Vec::new();
                }
                prev[r] = best;
                bt[r] = best_score;
            }
        }
        let mut out = Vec::new();
        let mut node = prev[1];
        if node < 0 {
            return out;
        }
        while prev[node as usize] >= 0 {
            let n = node as usize;
            out.push(chars[pos[n]..pos[n] + nlen[n]].to_vec());
            node = prev[n];
        }
        out.reverse();
        out
    }

    fn fuse_unk(&self, arr: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        let unk = self.unk_id;
        let id_or = |t: &[u32]| self.id_of(t).unwrap_or(unk);
        let mut fused: Vec<Vec<u32>> = Vec::new();
        let mut i = 0;
        while i < arr.len() {
            fused.push(arr[i].clone());
            if id_or(&arr[i]) != unk {
                i += 1;
                continue;
            }
            i += 1;
            while i < arr.len() && id_or(&arr[i]) == unk {
                let last = fused.len() - 1;
                if self.id_of(&fused[last]) != Some(unk) {
                    let piece = arr[i].clone();
                    fused[last].extend(piece);
                }
                i += 1;
            }
        }
        fused
    }

    /// Token pieces of one text (no `<s>` `</s>`), as code points.
    pub fn tokenize(&self, text: &[u32]) -> Vec<Vec<u32>> {
        let mut sections = self.split_added(text);
        for i in 0..sections.len() {
            let (lstrip, rstrip) = match self.added_token(&sections[i]) {
                Some(a) => (a.lstrip, a.rstrip),
                None => continue,
            };
            if lstrip && i > 0 {
                let s = &mut sections[i - 1];
                while s.last().map_or(false, |&c| is_js_space(c)) {
                    s.pop();
                }
            }
            if rstrip && i + 1 < sections.len() {
                let s = &mut sections[i + 1];
                let b = s.iter().take_while(|&&c| is_js_space(c)).count();
                s.drain(..b);
            }
        }
        let mut tokens = Vec::new();
        for x in sections {
            if x.is_empty() {
                continue;
            }
            if self.added_token(&x).is_some() {
                tokens.push(x);
                continue;
            }
            let x = normalize(&x);
            if x.is_empty() {
                continue;
            }
            let mut pieces = Vec::new();
            for word in x.split(|&c| is_js_space(c)).filter(|w| !w.is_empty()) {
                let mut w = word.to_vec();
                if w[0] != self.replacement {
                    w.insert(0, self.replacement);
                }
                pieces.extend(self.viterbi(&w));
            }
            tokens.extend(self.fuse_unk(pieces));
        }
        tokens
    }

    /// Ids with `<s> ... </s>` of a UTF-16 text, truncated to max_length unless `no_truncate`
    /// (Transformers.js cuts the end, `</s>` included).
    pub fn encode_utf16(&self, text: &[u16], no_truncate: bool) -> Vec<u32> {
        let tokens = self.tokenize(&code_points(text));
        let mut ids = Vec::with_capacity(tokens.len() + 2);
        ids.push(self.bos);
        for t in &tokens {
            ids.push(self.id_of(t).unwrap_or(self.unk_id));
        }
        ids.push(self.eos);
        if !no_truncate && ids.len() > self.max_length {
            ids.truncate(self.max_length);
        }
        ids
    }

    /// Ids with `<s> ... </s>`, truncated to 512 tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let u: Vec<u16> = text.encode_utf16().collect();
        self.encode_utf16(&u, false)
    }
}
