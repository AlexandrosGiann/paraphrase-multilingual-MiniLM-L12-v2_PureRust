//! Command line interface.
//!
//!   minilm embed [options] TEXT...        embeddings as JSON
//!   minilm similarity [options] TEXT...   cosine similarity matrix
//!   minilm search [options] QUERY TEXT... texts ranked by similarity to QUERY
//!   minilm verify [options]               bit-for-bit comparison with the WASM reference
//!   minilm bench [options] [TEXT]         speed
//!
//! Options: --model DIR (default: ./model, then the model/ of this crate), --threads N
//! (0 = one per core, default), --batch (all texts as one padded batch), --pooling mean|none,
//! --no-normalize, --testdata DIR.

use minilm::json::quote_utf16;
use minilm::{verify, MiniLM, Pooling};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

struct Opts {
    model: Option<PathBuf>,
    testdata: Option<PathBuf>,
    threads: usize,
    batch: bool,
    pooling: Pooling,
    normalize: bool,
    args: Vec<String>,
}

fn usage() -> ! {
    eprintln!(
        "usage: minilm <embed|similarity|search|verify|bench> [--model DIR] [--threads N] [--batch]\n\
         \x20             [--pooling mean|none] [--no-normalize] [--testdata DIR] [TEXT...]"
    );
    exit(2);
}

fn parse(args: Vec<String>) -> Opts {
    let mut o = Opts { model: None, testdata: None, threads: 0, batch: false, pooling: Pooling::Mean, normalize: true, args: Vec::new() };
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => o.model = Some(PathBuf::from(it.next().unwrap_or_else(|| usage()))),
            "--testdata" => o.testdata = Some(PathBuf::from(it.next().unwrap_or_else(|| usage()))),
            "--threads" => o.threads = it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--batch" => o.batch = true,
            "--no-normalize" => o.normalize = false,
            "--pooling" => {
                o.pooling = match it.next().as_deref() {
                    Some("mean") => Pooling::Mean,
                    Some("none") => Pooling::None,
                    _ => usage(),
                }
            }
            "--" => o.args.extend(it.by_ref()),
            _ => o.args.push(a),
        }
    }
    if o.pooling == Pooling::None {
        o.normalize = false;
    }
    o
}

// a directory: the given one, ./name if it exists, or the one in this crate (cargo run)
fn find_dir(given: &Option<PathBuf>, name: &str) -> PathBuf {
    if let Some(p) = given {
        return p.clone();
    }
    let here = PathBuf::from(name);
    if here.is_dir() {
        return here;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join(name)
}

fn load(o: &Opts) -> MiniLM {
    let dir = find_dir(&o.model, "model");
    MiniLM::load(&dir, o.threads).unwrap_or_else(|e| {
        eprintln!("cannot load the model from {}: {}", dir.display(), e);
        exit(1);
    })
}

fn texts(o: &Opts) -> Vec<&str> {
    if o.args.is_empty() {
        usage();
    }
    o.args.iter().map(|s| s.as_str()).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn fail(e: String) -> ! {
    eprintln!("{}", e);
    exit(1);
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let cmd = args.remove(0);
    let o = parse(args);
    match cmd.as_str() {
        "embed" => {
            let m = load(&o);
            let t = texts(&o);
            let ids: Vec<Vec<u32>> = t.iter().map(|s| m.tokenizer.encode(s)).collect();
            let e = m.model.embed_texts(&m.tokenizer, &t, o.batch, o.pooling, o.normalize).unwrap_or_else(|e| fail(e));
            // {:?} prints the shortest decimal that reads back as the same float32
            let data: Vec<String> = e.data.iter().map(|v| format!("{:?}", v)).collect();
            let ids_s: Vec<String> = ids.iter().map(|l| format!("{:?}", l)).collect();
            let texts_s: Vec<String> = t.iter().map(|s| quote_utf16(&s.encode_utf16().collect::<Vec<u16>>())).collect();
            println!("{{\"texts\":[{}],\"ids\":[{}],\"dims\":{:?},\"data\":[{}]}}", texts_s.join(","), ids_s.join(","), e.dims, data.join(","));
        }
        "similarity" => {
            let m = load(&o);
            let t = texts(&o);
            let v = m.embed(&t, o.batch).unwrap_or_else(|e| fail(e));
            for (i, a) in v.iter().enumerate() {
                let row: Vec<String> = v.iter().map(|b| format!("{:6.3}", dot(a, b))).collect();
                println!("{}  {}. {}", row.join(" "), i + 1, t[i]);
            }
        }
        "search" => {
            let m = load(&o);
            let t = texts(&o);
            if t.len() < 2 {
                usage();
            }
            let q = m.embed(&t[..1], false).unwrap_or_else(|e| fail(e));
            let v = m.embed(&t[1..], o.batch).unwrap_or_else(|e| fail(e));
            let mut ranked: Vec<(f32, &str)> = v.iter().zip(&t[1..]).map(|(e, s)| (dot(&q[0], e), *s)).collect();
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (s, text) in ranked {
                println!("{:6.3}  {}", s, text);
            }
        }
        "verify" => {
            let m = load(&o);
            let dir = find_dir(&o.testdata, "testdata");
            println!("threads: {}", m.model.threads);
            let results = [
                verify::reference(&m, &dir.join("reference.json"), true),
                verify::tokenizer(&m, &dir.join("tokenizer-cases.json"), true),
                verify::nfkc_cases(&dir.join("nfkc-cases.json"), true),
            ];
            let mut all = true;
            for r in results {
                match r {
                    Ok((ok, total)) => all &= ok == total,
                    Err(e) => {
                        eprintln!("{}", e);
                        all = false;
                    }
                }
            }
            println!("{}", if all { "ALL IDENTICAL" } else { "DIFFERENCES FOUND" });
            exit(if all { 0 } else { 1 });
        }
        "bench" => {
            let m = load(&o);
            let text = o.args.first().map(|s| s.as_str()).unwrap_or("Ο Αλέξανδρος έγραψε έναν κώδικα σε καθαρή JavaScript για ένα παλιό κινητό.");
            let ids = m.tokenizer.encode(text);
            let mut best = f64::INFINITY;
            for _ in 0..10 {
                let t0 = Instant::now();
                m.model.embed(std::slice::from_ref(&ids), m.tokenizer.pad, Pooling::Mean, true).unwrap_or_else(|e| fail(e));
                best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
            }
            println!("{} thread(s), {} tokens: best {:.1} ms ({:.2} ms/token)", m.model.threads, ids.len(), best, best / ids.len() as f64);
        }
        _ => usage(),
    }
}
