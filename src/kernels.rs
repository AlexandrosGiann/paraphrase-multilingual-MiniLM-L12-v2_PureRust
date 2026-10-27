//! The arithmetic of the ONNX Runtime CPU kernels (MLAS, WebAssembly SIMD build) that run
//! paraphrase-multilingual-MiniLM-L12-v2 q8, in the same order and with the same float32
//! rounding. Rust f32 operations round every result to float32 and are never fused into FMA,
//! which is what the WASM build does (WebAssembly SIMD has no FMA).

pub const FLT_LOWEST: f32 = f32::MIN;

/// DynamicQuantizeLinear (uint8) over a whole tensor, as ORT computes it (also inside
/// DynamicQuantizeMatMul). Writes q - zero_point into `qa` and the per-row sums of those
/// values into `row_sums`; returns the scale.
pub fn quantize_dynamic(x: &[f32], row_len: usize, qa: &mut [i16], row_sums: &mut [i64]) -> f32 {
    let mut min = f32::MAX;
    let mut max = FLT_LOWEST;
    for &v in x {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max < 0.0 {
        max = 0.0;
    }
    let scale = if max == min { 1.0f32 } else { (max - min) / 255.0f32 };
    let izp = (0.0f32 - min / scale).clamp(0.0, 255.0);
    let zp = izp.round_ties_even() as i32;
    for (r, row) in x.chunks_exact(row_len).enumerate() {
        let mut s: i64 = 0;
        let out = &mut qa[r * row_len..(r + 1) * row_len];
        for (o, &v) in out.iter_mut().zip(row) {
            let q = ((v / scale).round_ties_even() as i32 + zp).clamp(0, 255) - zp;
            *o = q as i16;
            s += q as i64;
        }
        row_sums[r] = s;
    }
    scale
}

/// A quantized linear layer: int8 weights [n][k] with a scale and zero point per output column.
pub struct Linear {
    pub k: usize,
    pub n: usize,
    pub w: Vec<i16>,
    pub scale: Vec<f32>,
    pub zp: Vec<i32>,
    pub bias: Vec<f32>,
}

impl Linear {
    /// Output columns [n0, n1) for `rows` rows: out[r][n - n0] = bias[n] + float(sum_k qa[r][k] *
    /// (w[n][k] - zw[n])) * float(act_scale * w_scale[n]), as MatMulIntegerToFloat /
    /// DynamicQuantizeMatMul (MLAS scale processor) followed by Add(bias). The integer sums
    /// are exact, so their order does not matter.
    pub fn columns(&self, qa: &[i16], row_sums: &[i64], rows: usize, act_scale: f32, n0: usize, n1: usize, out: &mut [f32]) {
        let k = self.k;
        let width = n1 - n0;
        for n in n0..n1 {
            let mult = act_scale * self.scale[n];
            let w = &self.w[n * k..(n + 1) * k];
            let zw = self.zp[n] as i64;
            let bias = self.bias[n];
            for r in 0..rows {
                let a = &qa[r * k..(r + 1) * k];
                let mut s: i32 = 0;
                for (x, y) in a.iter().zip(w) {
                    s += *x as i32 * *y as i32;
                }
                let exact = s as i64 - zw * row_sums[r];
                out[r * width + (n - n0)] = bias + (exact as f32) * mult;
            }
        }
    }
}

/// LayerNormalization of one row (ORT layer_norm_impl.cc, float).
pub fn layer_norm_row(x: &[f32], g: &[f32], b: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len() as f32;
    let mut mean = 0f32;
    let mut ms = 0f32;
    for &v in x {
        mean += v;
        ms += v * v;
    }
    mean /= n;
    let sd = (ms / n - mean * mean + eps).sqrt();
    for i in 0..x.len() {
        out[i] = (x[i] - mean) / sd * g[i] + b[i];
    }
}

// MLAS exp constants (compute.cpp, MlasExpConstants).
const EXP_LOWER_SUMEXP: f32 = -88.3762626647949;
const EXP_ROUNDING_BIAS: f32 = 12582912.0;
const EXP_LOG2_RECIP: f32 = 1.44269504088896341;
const EXP_LOG2_HIGH: f32 = -6.93145752e-1;
const EXP_LOG2_LOW: f32 = -1.42860677e-6;
const EXP_P0: f32 = 0x1694000 as f32 / 0x1000000 as f32 / 1024.0; // 0x1.694000p-10
const EXP_P1: f32 = 0x1125edc as f32 / 0x1000000 as f32 / 128.0; // 0x1.125edcp-7
const EXP_P2: f32 = 0x1555b5a as f32 / 0x1000000 as f32 / 32.0; // 0x1.555b5ap-5
const EXP_P3: f32 = 0x1555450 as f32 / 0x1000000 as f32 / 8.0; // 0x1.555450p-3
const EXP_P4: f32 = 0x1fffff6 as f32 / 0x1000000 as f32 / 4.0; // 0x1.fffff6p-2
const EXP_P56: f32 = 1.0; // 0x1.000000p+0
const EXP_MAX_EXPONENT: i32 = 0x3F800000;

/// MlasComputeSumExpVector for one lane: exp(x - max) as MLAS computes it.
#[inline(always)]
fn sum_exp1(x: f32, neg_max: f32) -> f32 {
    let mut v = x + neg_max;
    if !(v >= EXP_LOWER_SUMEXP) {
        v = EXP_LOWER_SUMEXP;
    }
    let biased = v * EXP_LOG2_RECIP + EXP_ROUNDING_BIAS;
    let m = biased - EXP_ROUNDING_BIAS;
    v = m * EXP_LOG2_HIGH + v;
    v = m * EXP_LOG2_LOW + v;
    let scale = f32::from_bits(((biased.to_bits() as i32).wrapping_shl(23)).wrapping_add(EXP_MAX_EXPONENT) as u32);
    let mut p = EXP_P0;
    p = p * v + EXP_P1;
    p = p * v + EXP_P2;
    p = p * v + EXP_P3;
    p = p * v + EXP_P4;
    p = p * v + EXP_P56;
    p = p * v + EXP_P56;
    p * scale
}

/// MlasComputeSoftmax for one row, in place (float, not log, not smooth): 4-lane sums, two
/// vectors per step, then a scalar tail.
pub fn softmax_row(x: &mut [f32]) {
    let d = x.len();
    let mut max = FLT_LOWEST;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let neg_max = -max;
    let mut acc = 0f32;
    let mut n = d;
    let mut p = 0;
    if n >= 4 {
        let mut a = [0f32; 4];
        while n >= 8 {
            let mut e = [0f32; 8];
            for j in 0..8 {
                e[j] = sum_exp1(x[p + j], neg_max);
            }
            for j in 0..4 {
                a[j] = a[j] + e[j] + e[j + 4];
            }
            x[p..p + 8].copy_from_slice(&e);
            p += 8;
            n -= 8;
        }
        while n >= 4 {
            for j in 0..4 {
                let e = sum_exp1(x[p + j], neg_max);
                a[j] += e;
                x[p + j] = e;
            }
            p += 4;
            n -= 4;
        }
        acc = (a[0] + a[2]) + (a[1] + a[3]);
    }
    while n > 0 {
        let t = sum_exp1(x[p], neg_max);
        acc += t;
        x[p] = t;
        p += 1;
        n -= 1;
    }
    let scale = 1.0f32 / acc;
    for v in x.iter_mut() {
        *v = scale * *v;
    }
}

// MLAS erf constants (erf.cpp, MlasErfConstants).
const ERF_UPPER: f32 = 3.925;
const ERF_SPLIT: f32 = 0.921875;
const ERF_S: [f32; 6] = [-5.99104969e-4, 4.99339588e-3, -2.67667342e-2, 1.12818025e-1, -3.76124859e-1, 1.28379151e-1];
const ERF_B: [f32; 7] = [1.72948930e-5, -3.83208680e-4, 3.88393435e-3, -2.42545605e-2, 1.06777847e-1, 6.34846687e-1, 1.28717512e-1];
const ERF_EXP_LOWER: f32 = -88.3762626647949;
const ERF_LOG2R: f32 = 1.44269504088896341;
const ERF_LOG2_HI: f32 = -6.93145752e-1;
const ERF_LOG2_LO: f32 = -1.42860677e-6;
const ERF_P: [f32; 5] = [1.38319808e-3, 8.37550033e-3, 4.16689515e-2, 1.66664466e-1, 4.99999851e-1];
const ERF_C: f32 = 12582912.0;

/// MlasErfKernel, vector path (the GELU rows have 1536 values, a multiple of 4).
pub fn erf(x: f32) -> f32 {
    let neg = x.is_sign_negative();
    let mut a = x.abs();
    if !(a <= ERF_UPPER) {
        a = ERF_UPPER;
    }
    let r = if a > ERF_SPLIT {
        let mut rb = ERF_B[0];
        for &c in &ERF_B[1..] {
            rb = rb * a + c;
        }
        rb = rb * a + a;
        rb = -rb;
        if rb < ERF_EXP_LOWER {
            rb = ERF_EXP_LOWER;
        }
        let mut m = ERF_LOG2R * rb + ERF_C;
        m -= ERF_C;
        let mut fx = m * ERF_LOG2_HI + rb;
        fx = m * ERF_LOG2_LO + fx;
        let mut y = ERF_P[0];
        for &c in &ERF_P[1..] {
            y = y * fx + c;
        }
        y = y * fx + 1.0;
        y = y * fx + 1.0;
        y *= f32::from_bits((((m as i32) + 127) << 23) as u32);
        1.0 - y
    } else {
        let sq = a * a;
        let mut r = ERF_S[0];
        for &c in &ERF_S[1..] {
            r = r * sq + c;
        }
        r * a + a
    };
    if neg {
        -r
    } else {
        r
    }
}

const SQRT1_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// BiasGelu (erf variant) on values that already hold input + bias, in place:
/// 0.5 * x * (erf(x * sqrt(1/2)) + 1), as (x * 0.5) * (erf(..) + 1).
pub fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let t = *v;
        *v = (t * 0.5) * (erf(t * SQRT1_2) + 1.0);
    }
}
