//! paraphrase-multilingual-MiniLM-L12-v2 (q8): the forward pass.
//!
//! Runs the same computation as Xenova/paraphrase-multilingual-MiniLM-L12-v2
//! onnx/model_quantized.onnx under onnxruntime-web (WASM), which is what Transformers.js uses
//! with { device: 'wasm', dtype: 'q8' }, after ONNX Runtime's graph optimizations:
//!   - embeddings: Gather + DequantizeLinear (uint8), (word + token type) + position, LayerNorm
//!   - q/k/v: DynamicQuantizeLinear + MatMulIntegerToFloat (int8 weights, per-column scale and
//!     zero point), + bias
//!   - FusedMatMul(Q, K^T, alpha = 1/sqrt(32)), attention mask, Softmax, probs x V, with the
//!     MLAS SGEMM accumulation order (one product and one sum per step, K never split)
//!   - attention output / intermediate / output: DynamicQuantizeMatMul (+ bias)
//!   - BiasGelu with the MLAS erf polynomial
//! The last hidden state is bit-for-bit identical. Mean pooling and L2 normalization follow
//! Transformers.js (feature-extraction pipeline, pooling 'mean', normalize true).
//!
//! Work is split over threads by output columns and attention heads; every value is computed
//! whole by one thread, so the results do not depend on the number of threads.

use crate::json::{self, Value};
use crate::kernels::{gelu, layer_norm_row, quantize_dynamic, softmax_row, Linear, FLT_LOWEST};
use crate::tokenizer::Tokenizer;
use std::fs;
use std::path::Path;

pub struct Layer {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub inter: Linear,
    pub out: Linear,
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
}

pub struct Model {
    pub hidden: usize,
    pub inter: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub max_pos: usize,
    eps: f32,
    alpha: f32,
    word: Vec<u8>,
    word_table: [f32; 256],
    pos: Vec<u8>,
    pos_table: [f32; 256],
    typ: Vec<u8>,
    type_table: [f32; 256],
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    pub layers: Vec<Layer>,
    pub threads: usize,
}

/// Output of `Model::embed`: pooled [batch x hidden] or the last hidden state
/// [batch x tokens x hidden].
pub struct Embeddings {
    pub data: Vec<f32>,
    pub dims: Vec<usize>,
    /// last hidden state [batch x tokens x hidden]
    pub hidden: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Pooling {
    None,
    Mean,
}

fn dequant_table(scale: f64, zp: f64) -> [f32; 256] {
    let s = scale as f32;
    let mut t = [0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = (i as i32 - zp as i32) as f32 * s;
    }
    t
}

/// Split [0, n) into `parts` ranges with boundaries on multiples of `align`.
fn split_range(n: usize, parts: usize, align: usize) -> Vec<(usize, usize)> {
    let units = n / align;
    let mut out = Vec::new();
    let mut start = 0;
    for i in 0..parts {
        let end = (units * (i + 1) / parts) * align;
        out.push((start, end));
        start = end;
    }
    out
}

/// Run `f(range, out_slice)` over the column ranges of an output [rows x n] on several
/// threads and assemble the result.
fn par_columns<F>(threads: usize, rows: usize, n: usize, align: usize, f: F) -> Vec<f32>
where
    F: Fn(usize, usize, &mut [f32]) + Sync,
{
    let ranges = split_range(n, threads.max(1).min(n / align).max(1), align);
    let mut parts: Vec<Vec<f32>> = ranges.iter().map(|r| vec![0f32; rows * (r.1 - r.0)]).collect();
    if parts.len() == 1 {
        f(0, n, &mut parts[0]);
    } else {
        std::thread::scope(|s| {
            for (r, part) in ranges.iter().zip(parts.iter_mut()) {
                let f = &f;
                s.spawn(move || f(r.0, r.1, part));
            }
        });
    }
    let mut out = vec![0f32; rows * n];
    for (r, part) in ranges.iter().zip(parts.iter()) {
        let w = r.1 - r.0;
        for row in 0..rows {
            out[row * n + r.0..row * n + r.1].copy_from_slice(&part[row * w..(row + 1) * w]);
        }
    }
    out
}

impl Model {
    /// Load from a directory with manifest.json and the model-NN.bin files.
    pub fn load(dir: &Path, threads: usize) -> Result<Model, String> {
        let manifest = json::parse(&fs::read_to_string(dir.join("manifest.json")).map_err(|e| e.to_string())?)?;
        Self::from_manifest(dir, &manifest, threads)
    }

    pub fn from_manifest(dir: &Path, m: &Value, threads: usize) -> Result<Model, String> {
        // the word embedding files are read straight into one table; the others are kept
        // until the tensors are copied out of them
        let e = m.get("embed");
        let word_files: Vec<usize> = e.get("word").get("files").arr().iter().map(|f| f.usize()).collect();
        let word_bytes: usize = word_files.iter().map(|&i| m.get("files").arr()[i].get("bytes").usize()).sum();
        let mut word: Vec<u8> = Vec::with_capacity(word_bytes);
        let mut files: Vec<Vec<u8>> = Vec::new();
        for (i, f) in m.get("files").arr().iter().enumerate() {
            let name = f.get("name").string();
            let b = fs::read(dir.join(&name)).map_err(|e| format!("{}: {}", name, e))?;
            if b.len() != f.get("bytes").usize() {
                return Err(format!("{}: expected {} bytes, got {}", name, f.get("bytes").usize(), b.len()));
            }
            if word_files.contains(&i) {
                word.extend_from_slice(&b);
                drop(b);
                files.push(Vec::new());
            } else {
                files.push(b);
            }
        }
        let bytes = |t: &Value, size: usize| -> &[u8] {
            let off = t.get("offset").usize();
            &files[t.get("file").usize()][off..off + t.get("length").usize() * size]
        };
        let f32s = |t: &Value| -> Vec<f32> { bytes(t, 4).chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect() };
        let i8s = |t: &Value| -> Vec<i16> { bytes(t, 1).iter().map(|&b| b as i8 as i16).collect() };
        let cfg = m.get("config");
        let lin = |t: &Value| -> Linear {
            Linear {
                k: t.get("k").usize(),
                n: t.get("n").usize(),
                w: i8s(t.get("w")),
                scale: f32s(t.get("scale")),
                zp: i8s(t.get("zp")).into_iter().map(|z| z as i32).collect(),
                bias: f32s(t.get("bias")),
            }
        };
        let layers = m
            .get("layers")
            .arr()
            .iter()
            .map(|l| Layer {
                q: lin(l.get("q")),
                k: lin(l.get("k")),
                v: lin(l.get("v")),
                o: lin(l.get("o")),
                inter: lin(l.get("inter")),
                out: lin(l.get("out")),
                ln1_w: f32s(l.get("ln1W")),
                ln1_b: f32s(l.get("ln1B")),
                ln2_w: f32s(l.get("ln2W")),
                ln2_b: f32s(l.get("ln2B")),
            })
            .collect();
        let threads = if threads == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) } else { threads };
        Ok(Model {
            hidden: cfg.get("hidden").usize(),
            inter: cfg.get("inter").usize(),
            heads: cfg.get("heads").usize(),
            head_dim: cfg.get("headDim").usize(),
            max_pos: cfg.get("maxPos").usize(),
            eps: cfg.get("eps").num() as f32,
            alpha: cfg.get("alpha").num() as f32,
            word,
            word_table: dequant_table(e.get("word").get("scale").num(), e.get("word").get("zp").num()),
            pos: bytes(e.get("position").get("data"), 1).to_vec(),
            pos_table: dequant_table(e.get("position").get("scale").num(), e.get("position").get("zp").num()),
            typ: bytes(e.get("tokenType").get("data"), 1).to_vec(),
            type_table: dequant_table(e.get("tokenType").get("scale").num(), e.get("tokenType").get("zp").num()),
            ln_w: f32s(e.get("lnW")),
            ln_b: f32s(e.get("lnB")),
            layers,
            threads: threads.max(1),
        })
    }

    /// The last hidden state [B x T x hidden] of B sequences of T tokens (padded) and their
    /// attention masks (1 = token, 0 = padding).
    pub fn forward(&self, ids: &[Vec<u32>], mask: &[Vec<u8>]) -> Result<Vec<f32>, String> {
        let (b_n, t_n, h_n, i_n) = (ids.len(), ids[0].len(), self.hidden, self.inter);
        let rows = b_n * t_n;
        if t_n > self.max_pos {
            return Err(format!("Sequence too long ({} > {} tokens)", t_n, self.max_pos));
        }
        let mut x = vec![0f32; rows * h_n];
        // embeddings: (word + token type 0) + position, then LayerNorm
        for b in 0..b_n {
            for t in 0..t_n {
                let id = ids[b][t] as usize;
                if (id + 1) * h_n > self.word.len() {
                    return Err(format!("Token id out of range: {}", id));
                }
                let w = &self.word[id * h_n..(id + 1) * h_n];
                let p = &self.pos[t * h_n..(t + 1) * h_n];
                let o = &mut x[(b * t_n + t) * h_n..(b * t_n + t + 1) * h_n];
                for i in 0..h_n {
                    o[i] = (self.word_table[w[i] as usize] + self.type_table[self.typ[i] as usize]) + self.pos_table[p[i] as usize];
                }
            }
        }
        let mut tmp = vec![0f32; h_n];
        for r in 0..rows {
            let row = &mut x[r * h_n..(r + 1) * h_n];
            layer_norm_row(row, &self.ln_w, &self.ln_b, &mut tmp, self.eps);
            row.copy_from_slice(&tmp);
        }
        // attention mask term: (1 - mask) * float lowest
        let mask_add: Vec<f32> = mask.iter().flat_map(|m| m.iter().map(|&v| (1.0f32 - v as f32) * FLT_LOWEST)).collect();
        let mut qa = vec![0i16; rows * i_n];
        let mut row_sums = vec![0i64; rows];

        for layer in &self.layers {
            let sc = quantize_dynamic(&x, h_n, &mut qa, &mut row_sums);
            let ctx = self.attention(layer, &qa, &row_sums, rows, sc, b_n, t_n, &mask_add);
            let sc = quantize_dynamic(&ctx, h_n, &mut qa, &mut row_sums);
            let o = self.linear(&layer.o, &qa, &row_sums, rows, sc, false);
            self.residual_norm(&mut x, o, &layer.ln1_w, &layer.ln1_b);
            let sc = quantize_dynamic(&x, h_n, &mut qa, &mut row_sums);
            let inter = self.linear(&layer.inter, &qa, &row_sums, rows, sc, true);
            let sc = quantize_dynamic(&inter, i_n, &mut qa, &mut row_sums);
            let out = self.linear(&layer.out, &qa, &row_sums, rows, sc, false);
            self.residual_norm(&mut x, out, &layer.ln2_w, &layer.ln2_b);
        }
        Ok(x)
    }

    // x = LayerNorm(add + x)
    fn residual_norm(&self, x: &mut [f32], mut add: Vec<f32>, g: &[f32], b: &[f32]) {
        let h = self.hidden;
        for (a, &v) in add.iter_mut().zip(x.iter()) {
            *a += v;
        }
        for (src, dst) in add.chunks_exact(h).zip(x.chunks_exact_mut(h)) {
            layer_norm_row(src, g, b, dst, self.eps);
        }
    }

    // A quantized linear layer over all columns (+ BiasGelu for the intermediate layer:
    // the bias is added the same way, float32 input + bias, before the activation).
    fn linear(&self, lin: &Linear, qa: &[i16], row_sums: &[i64], rows: usize, sc: f32, act: bool) -> Vec<f32> {
        par_columns(self.threads, rows, lin.n, 4, |n0, n1, out| {
            lin.columns(qa, row_sums, rows, sc, n0, n1, out);
            if act {
                gelu(out);
            }
        })
    }

    // q/k/v projections and attention, split by heads.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, l: &Layer, qa: &[i16], row_sums: &[i64], rows: usize, sc: f32, b_n: usize, t_n: usize, mask_add: &[f32]) -> Vec<f32> {
        let hd = self.head_dim;
        let alpha = self.alpha;
        par_columns(self.threads, rows, self.hidden, hd, |c0, c1, ctx| {
            let w = c1 - c0;
            let mut q = vec![0f32; rows * w];
            let mut k = vec![0f32; rows * w];
            let mut v = vec![0f32; rows * w];
            l.q.columns(qa, row_sums, rows, sc, c0, c1, &mut q);
            l.k.columns(qa, row_sums, rows, sc, c0, c1, &mut k);
            l.v.columns(qa, row_sums, rows, sc, c0, c1, &mut v);
            let mut scores = vec![0f32; t_n];
            let mut vt = vec![0f32; t_n * hd];
            for b in 0..b_n {
                let base = b * t_n;
                for h in 0..w / hd {
                    // V of this head, transposed to [d][j]
                    for j in 0..t_n {
                        for d in 0..hd {
                            vt[d * t_n + j] = v[(base + j) * w + h * hd + d];
                        }
                    }
                    for i in 0..t_n {
                        let qo = (base + i) * w + h * hd;
                        let qrow = &q[qo..qo + hd];
                        for j in 0..t_n {
                            let ko = (base + j) * w + h * hd;
                            let krow = &k[ko..ko + hd];
                            let mut acc = 0f32;
                            for d in 0..hd {
                                acc += qrow[d] * krow[d];
                            }
                            scores[j] = acc * alpha + mask_add[base + j];
                        }
                        softmax_row(&mut scores);
                        for d in 0..hd {
                            let vrow = &vt[d * t_n..(d + 1) * t_n];
                            let mut acc = 0f32;
                            for j in 0..t_n {
                                acc += scores[j] * vrow[j];
                            }
                            ctx[qo + d] = acc;
                        }
                    }
                }
            }
        })
    }

    /// Embeddings of token id lists (with `<s> ... </s>`), run as one padded batch like the
    /// Transformers.js pipeline does with an array of texts.
    pub fn embed(&self, id_lists: &[Vec<u32>], pad: u32, pooling: Pooling, normalize: bool) -> Result<Embeddings, String> {
        let b_n = id_lists.len();
        let t_n = id_lists.iter().map(|l| l.len()).max().unwrap_or(0);
        let h = self.hidden;
        let ids: Vec<Vec<u32>> = id_lists.iter().map(|l| (0..t_n).map(|j| if j < l.len() { l[j] } else { pad }).collect()).collect();
        let mask: Vec<Vec<u8>> = id_lists.iter().map(|l| (0..t_n).map(|j| (j < l.len()) as u8).collect()).collect();
        let hidden = self.forward(&ids, &mask)?;
        if pooling == Pooling::None {
            if normalize {
                return Err("normalize needs mean pooling".into());
            }
            return Ok(Embeddings { data: hidden.clone(), dims: vec![b_n, t_n, h], hidden });
        }
        let mut pooled = mean_pool(&hidden, &mask, b_n, t_n, h);
        if normalize {
            pooled = l2_normalize(&pooled, h);
        }
        Ok(Embeddings { data: pooled, dims: vec![b_n, h], hidden })
    }

    /// Tokenize and embed texts: one padded batch if `batch`, otherwise one text at a time
    /// (like extractor([a, b]) versus extractor(a), extractor(b) in Transformers.js).
    pub fn embed_texts(&self, tok: &Tokenizer, texts: &[&str], batch: bool, pooling: Pooling, normalize: bool) -> Result<Embeddings, String> {
        let lists: Vec<Vec<u32>> = texts.iter().map(|t| tok.encode(t)).collect();
        if batch || lists.len() == 1 {
            return self.embed(&lists, tok.pad, pooling, normalize);
        }
        let mut data = Vec::new();
        let mut hidden = Vec::new();
        for l in &lists {
            let e = self.embed(std::slice::from_ref(l), tok.pad, pooling, normalize)?;
            data.extend(e.data);
            hidden.extend(e.hidden);
        }
        let dims = if pooling == Pooling::Mean { vec![lists.len(), self.hidden] } else { vec![] };
        Ok(Embeddings { data, dims, hidden })
    }
}

/// Transformers.js mean_pooling: sum in double over the masked tokens, divided by their count,
/// stored as float32.
pub fn mean_pool(hidden: &[f32], mask: &[Vec<u8>], b_n: usize, t_n: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; b_n * h];
    for b in 0..b_n {
        for k in 0..h {
            let mut sum = 0f64;
            let mut count = 0f64;
            for j in 0..t_n {
                let attn = mask[b][j] as f64;
                count += attn;
                sum += hidden[(b * t_n + j) * h + k] as f64 * attn;
            }
            out[b * h + k] = (sum / count) as f32;
        }
    }
    out
}

/// Transformers.js Tensor.normalize(2, -1): float32 running sum of squares (each square exact
/// in double), ** 0.5, division in double stored as float32.
pub fn l2_normalize(x: &[f32], h: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (row, o) in x.chunks_exact(h).zip(out.chunks_exact_mut(h)) {
        let mut s = 0f32;
        for &v in row {
            s = (s as f64 + v as f64 * v as f64) as f32;
        }
        let n = (s as f64).sqrt() as f32 as f64;
        for (d, &v) in o.iter_mut().zip(row) {
            *d = (v as f64 / n) as f32;
        }
    }
    out
}
