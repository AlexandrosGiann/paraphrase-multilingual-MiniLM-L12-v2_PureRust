//! Bit-for-bit comparison with Transformers.js 3.8.1 + onnxruntime-web (WASM, q8) and with
//! Node's String.prototype.normalize (testdata/, made by tools/ref).
//!     cargo test --release

use minilm::{verify, MiniLM};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load(threads: usize) -> MiniLM {
    MiniLM::load(&root().join("model"), threads).expect("load model")
}

fn testdata(name: &str) -> PathBuf {
    root().join("testdata").join(name)
}

#[test]
fn nfkc_matches_node() {
    let (ok, total) = verify::nfkc_cases(&testdata("nfkc-cases.json"), true).unwrap();
    assert_eq!(ok, total);
}

#[test]
fn tokenizer_matches_transformers_js() {
    let m = load(1);
    let (ok, total) = verify::tokenizer(&m, &testdata("tokenizer-cases.json"), true).unwrap();
    assert_eq!(ok, total);
}

#[test]
fn embeddings_match_wasm_1_thread() {
    let (ok, total) = verify::reference(&load(1), &testdata("reference.json"), true).unwrap();
    assert_eq!(ok, total);
}

#[test]
fn embeddings_match_wasm_4_threads() {
    let (ok, total) = verify::reference(&load(4), &testdata("reference.json"), true).unwrap();
    assert_eq!(ok, total);
}

#[test]
fn simple_api() {
    let m = load(0);
    let e = m.embed(&["Η γάτα κάθεται στο χαλί.", "The cat sits on the mat.", "Η οικονομία αναπτύχθηκε φέτος."], false).unwrap();
    assert_eq!(e.len(), 3);
    assert_eq!(e[0].len(), 384);
    let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    assert!((dot(&e[0], &e[0]) - 1.0).abs() < 1e-5);
    assert!(dot(&e[0], &e[1]) > dot(&e[0], &e[2]));
    let _ = Path::new("");
}
