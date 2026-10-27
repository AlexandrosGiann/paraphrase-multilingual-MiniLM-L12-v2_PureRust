//! Unicode NFKC normalization (Unicode 17.0, the version behind the reference's
//! `String.prototype.normalize`). Works on code points, where a lone surrogate (0xD800..0xDFFF)
//! is an element of its own, like a JavaScript string iterated with `Array.from`; such
//! elements pass through unchanged.

use crate::nfkc_data::{CCC, COMPOSE, DECOMP, DECOMP_CPS};

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

fn decomposition(c: u32) -> Option<&'static [u32]> {
    DECOMP
        .binary_search_by_key(&c, |e| e.0)
        .ok()
        .map(|i| &DECOMP_CPS[DECOMP[i].1 as usize..(DECOMP[i].1 + DECOMP[i].2) as usize])
}

/// Canonical combining class.
pub fn ccc(c: u32) -> u8 {
    match CCC.binary_search_by(|r| {
        if r.1 < c {
            std::cmp::Ordering::Less
        } else if r.0 > c {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => CCC[i].2,
        Err(_) => 0,
    }
}

fn compose_pair(a: u32, b: u32) -> Option<u32> {
    COMPOSE.binary_search_by_key(&(a, b), |e| (e.0, e.1)).ok().map(|i| COMPOSE[i].2)
}

/// NFKC of a code point sequence.
pub fn nfkc(cps: &[u32]) -> Vec<u32> {
    if cps.iter().all(|&c| c < 0x80) {
        return cps.to_vec();
    }
    // full compatibility decomposition
    let mut d: Vec<u32> = Vec::with_capacity(cps.len() + 8);
    for &c in cps {
        let s = c.wrapping_sub(S_BASE);
        if c >= S_BASE && s < S_COUNT {
            d.push(L_BASE + s / N_COUNT);
            d.push(V_BASE + (s % N_COUNT) / T_COUNT);
            let t = s % T_COUNT;
            if t != 0 {
                d.push(T_BASE + t);
            }
            continue;
        }
        match decomposition(c) {
            Some(m) => d.extend_from_slice(m),
            None => d.push(c),
        }
    }
    // canonical ordering: stable sort of each run of non-starters by combining class
    let mut cls: Vec<u8> = d.iter().map(|&c| ccc(c)).collect();
    for i in 1..d.len() {
        if cls[i] == 0 {
            continue;
        }
        let mut k = i;
        while k > 0 && cls[k - 1] > cls[k] {
            d.swap(k, k - 1);
            cls.swap(k, k - 1);
            k -= 1;
        }
    }
    // canonical composition
    let mut out: Vec<u32> = Vec::with_capacity(d.len());
    let mut starter: Option<usize> = None;
    let mut last_cls: i32 = -1;
    for i in 0..d.len() {
        let (c, cc) = (d[i], cls[i] as i32);
        if let Some(st) = starter {
            if last_cls == 0 || last_cls < cc {
                let a = out[st];
                let l_index = a.wrapping_sub(L_BASE);
                let v_index = c.wrapping_sub(V_BASE);
                let comp = if last_cls == 0 && a >= L_BASE && l_index < L_COUNT && c >= V_BASE && v_index < V_COUNT {
                    Some(S_BASE + (l_index * V_COUNT + v_index) * T_COUNT)
                } else if last_cls == 0 && a >= S_BASE && a - S_BASE < S_COUNT && (a - S_BASE) % T_COUNT == 0 && c > T_BASE && c - T_BASE < T_COUNT {
                    Some(a + (c - T_BASE))
                } else {
                    compose_pair(a, c)
                };
                if let Some(x) = comp {
                    out[st] = x;
                    continue;
                }
            }
        }
        if cc == 0 {
            starter = Some(out.len());
        }
        out.push(c);
        last_cls = cc;
    }
    out
}
