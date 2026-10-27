//! Comparison with the Transformers.js / onnxruntime-web (WASM) reference (testdata/), bit for
//! bit. Used by `minilm verify` and by the tests.

use crate::json::{self, base64_decode, Value};
use crate::model::{mean_pool, l2_normalize, Pooling};
use crate::nfkc::nfkc;
use crate::MiniLM;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn read_json(path: &Path) -> Result<Value, String> {
    json::parse(&fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?)
}

fn f32s(b64: &Value) -> Vec<f32> {
    base64_decode(&b64.string()).chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// Number of values whose bit patterns differ (usize::MAX if the lengths differ).
pub fn bit_diff(a: &[f32], b: &[f32]) -> usize {
    if a.len() != b.len() {
        return usize::MAX;
    }
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// FNV-1a over the float32 bit patterns (the hash in reference.json).
pub fn fnv(data: &[f32]) -> String {
    let mut h: u32 = 0x811c9dc5;
    for v in data {
        for byte in v.to_bits().to_le_bytes() {
            h ^= byte as u32;
            h = h.wrapping_mul(0x01000193);
        }
    }
    format!("{:08x}", h)
}

fn ids_of(v: &Value) -> Vec<u32> {
    v.arr().iter().map(|x| x.usize() as u32).collect()
}

/// Embeddings of reference.json: ids, last hidden state, mean pooling and normalized
/// embedding of every text, and the padded batches. Returns (identical, total).
pub fn reference(m: &MiniLM, path: &Path, log: bool) -> Result<(usize, usize), String> {
    let r = read_json(path)?;
    let h = m.model.hidden;
    let (mut ok, mut total, mut tokens, mut ms) = (0, 0, 0, 0.0);
    for c in r.get("single").arr() {
        total += 1;
        let text = c.get("text").utf16();
        let ids = m.tokenizer.encode_utf16(text, false);
        let t0 = Instant::now();
        let e = m.model.embed(std::slice::from_ref(&ids), m.tokenizer.pad, Pooling::Mean, true)?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        ms += dt;
        tokens += ids.len();
        let mask = vec![vec![1u8; ids.len()]];
        let mean = mean_pool(&e.hidden, &mask, 1, ids.len(), h);
        let mut ref_ids = ids_of(c.get("ids"));
        ref_ids.truncate(512);
        let ids_ok = ref_ids == ids;
        let dh = bit_diff(&e.hidden, &f32s(c.get("hidden")));
        let dm = bit_diff(&mean, &f32s(c.get("mean")));
        let dp = bit_diff(&e.data, &f32s(c.get("pooled")));
        let dp2 = bit_diff(&l2_normalize(&mean, h), &f32s(c.get("pooled")));
        let good = ids_ok && dh == 0 && dm == 0 && dp == 0 && dp2 == 0;
        if good {
            ok += 1;
        }
        if log {
            let short: String = String::from_utf16_lossy(text).chars().take(40).collect();
            println!(
                "{} {} tokens, {:.0} ms{}: {}",
                if good { "SAME" } else { "DIFF" },
                ids.len(),
                dt,
                if good { String::new() } else { format!(" (ids {}, hidden {}, mean {}, pooled {}/{})", ids_ok, dh, dm, dp, dp2) },
                json::quote_utf16(&short.encode_utf16().collect::<Vec<u16>>())
            );
        }
    }
    for c in r.get("batches").arr() {
        total += 1;
        let lists: Vec<Vec<u32>> = c.get("texts").arr().iter().map(|t| m.tokenizer.encode_utf16(t.utf16(), false)).collect();
        let e = m.model.embed(&lists, m.tokenizer.pad, Pooling::Mean, true)?;
        let t_n = lists.iter().map(|l| l.len()).max().unwrap();
        let padded: Vec<Vec<u32>> = lists.iter().map(|l| (0..t_n).map(|j| if j < l.len() { l[j] } else { m.tokenizer.pad }).collect()).collect();
        let ref_ids: Vec<Vec<u32>> = c.get("ids").arr().iter().map(ids_of).collect();
        let ids_ok = ref_ids == padded;
        let hash_ok = fnv(&e.hidden) == c.get("hiddenHash").string();
        let dp = bit_diff(&e.data, &f32s(c.get("pooled")));
        let good = ids_ok && hash_ok && dp == 0;
        if good {
            ok += 1;
        }
        if log {
            println!(
                "{} batch {} x {}{}",
                if good { "SAME" } else { "DIFF" },
                lists.len(),
                t_n,
                if good { String::new() } else { format!(" (ids {}, hidden {}, pooled {})", ids_ok, hash_ok, dp) }
            );
        }
    }
    if log {
        println!("{}/{} identical, {} thread(s) (single texts: {} tokens in {:.0} ms)", ok, total, m.model.threads, tokens, ms);
    }
    Ok((ok, total))
}

/// Tokenizer ids versus Transformers.js (tokenizer-cases.json). Returns (identical, total).
pub fn tokenizer(m: &MiniLM, path: &Path, log: bool) -> Result<(usize, usize), String> {
    let r = read_json(path)?;
    let (mut ok, mut total) = (0, 0);
    for c in r.arr() {
        total += 1;
        let ids = m.tokenizer.encode_utf16(c.get("t").utf16(), true);
        if ids == ids_of(c.get("ids")) {
            ok += 1;
        } else if log && total - ok <= 10 {
            println!("MISMATCH {} ref {:?} ours {:?}", json::quote_utf16(c.get("t").utf16()), ids_of(c.get("ids")), ids);
        }
    }
    if log {
        println!("tokenizer: {}/{} identical", ok, total);
    }
    Ok((ok, total))
}

/// NFKC versus String.prototype.normalize('NFKC') of Node (nfkc-cases.json): every code point
/// (those not listed must stay unchanged) and the listed sequences. Returns (identical, total).
pub fn nfkc_cases(path: &Path, log: bool) -> Result<(usize, usize), String> {
    let r = read_json(path)?;
    let mut changed = std::collections::HashMap::new();
    for c in r.get("changed").arr() {
        let a = c.arr();
        changed.insert(a[0].usize() as u32, ids_of(&a[1]));
    }
    let (mut ok, mut total) = (0, 0);
    let mut check = |input: &[u32], want: &[u32]| {
        total += 1;
        let got = nfkc(input);
        if got == want {
            ok += 1;
        } else if log && total - ok <= 10 {
            println!("NFKC MISMATCH {:x?} want {:x?} got {:x?}", input, want, got);
        }
    };
    for cp in 0..=0x10FFFFu32 {
        match changed.get(&cp) {
            Some(want) => check(&[cp], want),
            None => check(&[cp], &[cp]),
        }
    }
    for s in r.get("seqs").arr() {
        let a = s.arr();
        check(&ids_of(&a[0]), &ids_of(&a[1]));
    }
    if log {
        println!("NFKC: {}/{} identical (Unicode {})", ok, total, r.get("unicode").string());
    }
    Ok((ok, total))
}
