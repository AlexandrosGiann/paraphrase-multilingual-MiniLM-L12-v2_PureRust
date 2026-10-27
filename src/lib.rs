//! paraphrase-multilingual-MiniLM-L12-v2 (q8) sentence embeddings in pure Rust (standard
//! library only), bit-for-bit identical to Transformers.js 3.8.1 + onnxruntime-web (WASM,
//! dtype q8).
//!
//! ```no_run
//! let m = minilm::MiniLM::load(std::path::Path::new("model"), 0).unwrap();
//! let e = m.embed(&["Η γάτα κάθεται στο χαλί.", "The cat sits on the mat."], false).unwrap();
//! let sim: f32 = e[0].iter().zip(&e[1]).map(|(a, b)| a * b).sum();
//! println!("{}", sim);
//! ```

pub mod json;
pub mod kernels;
pub mod model;
pub mod nfkc;
mod nfkc_data;
pub mod tokenizer;
pub mod verify;

pub use model::{Embeddings, Model, Pooling};
pub use tokenizer::Tokenizer;

use std::fs;
use std::path::Path;

/// Tokenizer + model, loaded from a model directory (manifest.json, vocab.txt, scores.bin,
/// model-NN.bin).
pub struct MiniLM {
    pub tokenizer: Tokenizer,
    pub model: Model,
}

impl MiniLM {
    /// `threads` = 0: one per CPU core.
    pub fn load(dir: &Path, threads: usize) -> Result<MiniLM, String> {
        let manifest = json::parse(&fs::read_to_string(dir.join("manifest.json")).map_err(|e| format!("manifest.json: {}", e))?)?;
        let vocab = fs::read_to_string(dir.join("vocab.txt")).map_err(|e| format!("vocab.txt: {}", e))?;
        let scores = fs::read(dir.join("scores.bin")).map_err(|e| format!("scores.bin: {}", e))?;
        let tokenizer = Tokenizer::new(&vocab, &scores, manifest.get("tokenizer"));
        let model = Model::from_manifest(dir, &manifest, threads)?;
        Ok(MiniLM { tokenizer, model })
    }

    /// Sentence embeddings (mean pooling + L2 normalization), one 384-value vector per text.
    /// `batch`: all texts as one padded batch (Transformers.js `extractor([a, b, ...])`);
    /// otherwise one text at a time (`extractor(a)`, `extractor(b)`, ...).
    pub fn embed(&self, texts: &[&str], batch: bool) -> Result<Vec<Vec<f32>>, String> {
        let e = self.model.embed_texts(&self.tokenizer, texts, batch, Pooling::Mean, true)?;
        Ok(e.data.chunks_exact(self.model.hidden).map(|c| c.to_vec()).collect())
    }
}
