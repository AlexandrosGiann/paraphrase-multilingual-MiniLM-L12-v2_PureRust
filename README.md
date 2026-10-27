# paraphrase-multilingual-MiniLM-L12-v2 σε καθαρή Rust

Embeddings προτάσεων σε 50+ γλώσσες με το μοντέλο
[sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
σε Rust χωρίς καμία εξάρτηση: μόνο η standard library. Δεν χρειάζεται ONNX Runtime, WebAssembly, BLAS ή
βιβλιοθήκες C. Τρέχει το ίδιο αρχείο που χρησιμοποιεί το Transformers.js
(`Xenova/paraphrase-multilingual-MiniLM-L12-v2`, `onnx/model_quantized.onnx`, q8).

Δίνει **τα ίδια αποτελέσματα, bit προς bit,** με την έκδοση WebAssembly (Transformers.js 3.8.1 + ONNX Runtime Web,
`device: 'wasm'`, `dtype: 'q8'`): ίδια token ids, ίδιο last hidden state, ίδιο τελικό embedding (mean pooling +
κανονικοποίηση). Είναι η αδελφή έκδοση του
[paraphrase-multilingual-MiniLM-L12-v2_PureJS](https://github.com/AlexandrosGiann/paraphrase-multilingual-MiniLM-L12-v2_PureJS)
και κάνει ακριβώς τους ίδιους υπολογισμούς.

## Εκτέλεση

Χρειάζεται μόνο Rust (stable, 1.77 ή νεότερη· δοκιμάστηκε με 1.99):

```bash
cargo build --release
```

```bash
./target/release/minilm similarity "Η γάτα κάθεται στο χαλί." "The cat sits on the mat." "Το σκυλί παίζει στον κήπο."
```

```
 1.000  0.736  0.096  1. Η γάτα κάθεται στο χαλί.
 0.736  1.000  0.065  2. The cat sits on the mat.
 0.096  0.065  1.000  3. Το σκυλί παίζει στον κήπο.
```

Εντολές:

| Εντολή | Τι κάνει |
|---|---|
| `minilm embed ΚΕΙΜΕΝΟ...` | embeddings σε JSON (`texts`, `ids`, `dims`, `data`) |
| `minilm similarity ΚΕΙΜΕΝΟ...` | πίνακας συνημιτόνου ομοιότητας |
| `minilm search ΕΡΩΤΗΣΗ ΚΕΙΜΕΝΟ...` | κατάταξη των κειμένων ως προς την ερώτηση |
| `minilm verify` | σύγκριση bit προς bit με την έκδοση WASM (`testdata/`) |
| `minilm bench [ΚΕΙΜΕΝΟ]` | ταχύτητα |

Επιλογές:

- `--threads N`: νήματα (`0` = ένα ανά πυρήνα, η προεπιλογή). Το αποτέλεσμα είναι ίδιο με όποιον αριθμό νημάτων,
  γιατί κάθε τιμή εξόδου την υπολογίζει ολόκληρη ένα νήμα.
- `--batch`: όλα τα κείμενα ως ένα batch με padding, σαν το `extractor([a, b, ...])` του Transformers.js. Χωρίς
  αυτό, κάθε κείμενο υπολογίζεται μόνο του, σαν το `extractor(a)`. Τα δύο διαφέρουν ελάχιστα μεταξύ τους (στο batch η
  κβάντιση των ενεργοποιήσεων γίνεται σε όλο τον τανυστή). Και τα δύο ταυτίζονται bit προς bit με το αντίστοιχο
  κάλεσμα του Transformers.js.
- `--pooling none`: το last hidden state (ένα διάνυσμα ανά token), `--no-normalize`: mean pooling χωρίς
  κανονικοποίηση.
- `--model DIR`: ο φάκελος του μοντέλου (προεπιλογή: `./model`).

Ως βιβλιοθήκη:

```rust
let m = minilm::MiniLM::load(std::path::Path::new("model"), 0)?;
let e = m.embed(&["Η γάτα κάθεται στο χαλί.", "The cat sits on the mat."], false)?;
let similarity: f32 = e[0].iter().zip(&e[1]).map(|(a, b)| a * b).sum();
```

## Επαλήθευση

```bash
cargo test --release
```

| Έλεγχος | Αποτέλεσμα |
|---|---|
| 40 κείμενα (12 γλώσσες, κενά, emoji, ειδικά tokens, `<mask>`, NFKC, μοναχικά surrogates, άγνωστοι χαρακτήρες, 122 και 1618→512 tokens) + 3 batches με padding, έναντι Transformers.js 3.8.1 + onnxruntime-web (`testdata/reference.json`): ids, last hidden state, mean pooling, κανονικοποιημένο embedding | **43/43 bit-for-bit ίδια** με 1, 4 και 8 νήματα (`reports/verify-*.log`) |
| Tokenizer έναντι Transformers.js: τα κείμενα των ελέγχων + 20.000 τυχαία, μαζί με ειδικά tokens μέσα στο κείμενο και μισά ζεύγη surrogates (`testdata/tokenizer-cases.json`) | **20.092/20.092 ίδια** |
| NFKC έναντι `String.prototype.normalize('NFKC')` (Node 24, Unicode 17): κάθε code point από το 0 ως το 10FFFF και 60.000 τυχαίες ακολουθίες με combining χαρακτήρες και Hangul (`testdata/nfkc-cases.json`) | **1.174.112/1.174.112 ίδια** |

Οι πράξεις float32 της Rust στρογγυλεύουν κάθε αποτέλεσμα σε float32 και δεν συγχωνεύονται ποτέ σε FMA. Αυτό
κάνει και το ONNX Runtime σε WebAssembly. Γι' αυτό τα αποτελέσματα δεν εξαρτώνται από τον επεξεργαστή.

## Τι ακριβώς αναπαράγεται

Αναφορά: `pipeline('feature-extraction', 'Xenova/paraphrase-multilingual-MiniLM-L12-v2', { device: 'wasm', dtype: 'q8' })`
(το q8 είναι η προεπιλογή του Transformers.js για WASM), και κάλεσμα με `{ pooling: 'mean', normalize: true }`.

- **Tokenizer** (`src/tokenizer.rs`, XLM-R SentencePiece Unigram, όπως το υλοποιεί το Transformers.js):
  - ειδικά tokens πρώτα, με το `lstrip` του `<mask>`
  - ο normalizer `Precompiled`: αφαίρεση χαρακτήρων ελέγχου και κάποια κενά σε `' '` πάνω σε μονάδες UTF-16,
    όπως τα regex της JavaScript, και μετά NFKC
  - `WhitespaceSplit` + `Metaspace`, Viterbi με την ίδια επιλογή σε ισοπαλίες, συγχώνευση άγνωστων κομματιών,
    `<s> … </s>`, κόψιμο στα 512 tokens
  - το κείμενο κρατιέται ως code points όπου ένα μοναχικό surrogate είναι στοιχείο από μόνο του, όπως ένα string της
    JavaScript. Το `encode_utf16` δίνει τα ίδια ids ακόμα και για κακοσχηματισμένο UTF-16.
- **NFKC** (`src/nfkc.rs`, δεδομένα στο `src/nfkc_data.rs` από το UCD 17 με `cargo run --example gen_nfkc`).
- **Μοντέλο** (`src/model.rs`, `src/kernels.rs`), όπως τον γράφο ONNX μετά τις βελτιστοποιήσεις του ONNX Runtime:
  - embeddings: `Gather` + `DequantizeLinear` (uint8), (λέξη + token type) + θέση, `LayerNormalization`
  - q/k/v: `DynamicQuantizeLinear` + `MatMulIntegerToFloat` (βάρη int8 με scale και zero point ανά στήλη): ακριβή
    ακέραια αθροίσματα, μετά `float(άθροισμα) * float(scale_A * scale_B)`, μετά + bias
  - `FusedMatMul(Q, Kᵀ, alpha = 1/√32)`, μάσκα, `Softmax` του MLAS με το πολυώνυμο exp του, `probs × V`, με τη σειρά
    άθροισης του SGEMM του MLAS
  - attention output / intermediate / output: `DynamicQuantizeMatMul`
  - `BiasGelu` με το πολυώνυμο erf του MLAS
- **Pooling**: το `mean_pooling` του Transformers.js (άθροισμα σε double, αποθήκευση σε float32) και το
  `normalize(2, -1)` (άθροισμα τετραγώνων σε float32, `** 0.5`, διαίρεση).

## Αρχεία

```
Cargo.toml               χωρίς εξαρτήσεις
src/
  main.rs                το πρόγραμμα minilm (embed, similarity, search, verify, bench)
  lib.rs                 MiniLM::load / embed
  model.rs               forward pass, νήματα, pooling, φόρτωση βαρών
  kernels.rs             οι πυρήνες του MLAS: κβάντιση, γραμμικά q8, LayerNorm, softmax/exp, erf/GELU
  tokenizer.rs           tokenizer (Unigram)
  nfkc.rs, nfkc_data.rs  NFKC (Unicode 17)
  json.rs                μικρός αναγνώστης JSON (strings σε UTF-16, όπως στη JavaScript)
  verify.rs              σύγκριση με το reference
tests/parity.rs          cargo test --release
examples/gen_nfkc.rs     UCD 17 (tools/ucd) -> src/nfkc_data.rs
model/                   manifest.json, vocab.txt, scores.bin, 9 αρχεία .bin (118 MB, τα ίδια bytes με το q8 ONNX)
testdata/                reference.json, tokenizer-cases.json, nfkc-cases.json, cases.json
tools/
  convert_onnx.py        ONNX -> model/ (τα βάρη αντιγράφονται bit-for-bit)
  ref/                   Transformers.js 3.8.1 + onnxruntime-web για τα reference (ref_embed.mjs,
                         gen_testdata.mjs, save_opt.mjs: ο γράφος μετά τις βελτιστοποιήσεις του ORT)
  ortsrc/                τα κομμάτια του κώδικα ONNX Runtime (commit 89f8206ba4) που αναπαράγονται
  ucd/                   Unicode Character Database 17.0
reports/                 logs του minilm verify
```

## Ξανά από την αρχή

```bash
mkdir -p hf/onnx && for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json onnx/model_quantized.onnx; do curl -L -o hf/$f https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/$f; done
```

```bash
python tools/convert_onnx.py hf model
cargo run --release --example gen_nfkc
cd tools/ref && npm install && node ref_embed.mjs ../../testdata/cases.json ../../testdata/reference.json && node gen_testdata.mjs && cd ../..
cargo test --release
```

## Ταχύτητα και μνήμη

Σε λάπτοπ με 8 νήματα (release build):

| Κείμενο | 1 νήμα | 4 νήματα | 8 νήματα |
|---|---|---|---|
| 22 tokens | 80 ms | 50 ms | 45 ms |
| 122 tokens | 0,52 s | | |
| 512 tokens | 3,6 s | | |

Το attention μεγαλώνει με το τετράγωνο του μήκους. Η μνήμη φτάνει τα 187 MB: ο πίνακας λέξεων (96 MB, 250.037 ×
384, uint8), τα γραμμικά επίπεδα (22 MB int8, κρατιούνται ως i16 για ταχύτερους ακέραιους πολλαπλασιασμούς) και
ο tokenizer.
