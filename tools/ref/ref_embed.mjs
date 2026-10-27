// Reference: Transformers.js 3.8.1 feature-extraction pipeline on onnxruntime-web (WASM,
// 1 thread), dtype q8 (onnx/model_quantized.onnx), the default for device 'wasm'.
// node ref_embed.mjs ../../testdata/cases.json ../../testdata/reference.json
import * as ort from 'onnxruntime-web';
import fs from 'fs';
import path from 'path';
import { pathToFileURL } from 'url';
ort.env.wasm.wasmPaths = pathToFileURL(path.resolve('node_modules/onnxruntime-web/dist')).href + '/';
ort.env.wasm.numThreads = 1;
globalThis[Symbol.for('onnxruntime')] = ort;
const { pipeline, env } = await import('@huggingface/transformers');
env.allowRemoteModels = false;
env.localModelPath = path.resolve('../..') + '/';
env.backends.onnx.wasm.numThreads = 1;
const [casesFile, outFile] = process.argv.slice(2);
const cases = JSON.parse(fs.readFileSync(casesFile, 'utf8'));
const extractor = await pipeline('feature-extraction', 'hf', { dtype: 'q8', device: 'auto' });
const b64 = t => Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength).toString('base64');
// FNV-1a over the float32 bit patterns (the same hash is computed in the browser)
function fnv(data) {
  const u = new Uint32Array(data.buffer, data.byteOffset, data.length);
  let h = 0x811c9dc5;
  for (let i = 0; i < u.length; i++) {
    for (let s = 0; s < 32; s += 8) { h ^= (u[i] >>> s) & 255; h = Math.imul(h, 0x01000193) >>> 0; }
  }
  return h.toString(16).padStart(8, '0');
}
const single = [];
for (const text of cases.single) {
  const enc = extractor.tokenizer(text);
  const ids = Array.from(enc.input_ids.data, Number);
  const t0 = Date.now();
  const hidden = await extractor(text, { pooling: 'none' });
  const pooled = await extractor(text, { pooling: 'mean', normalize: true });
  const mean = await extractor(text, { pooling: 'mean', normalize: false });
  single.push({ text, ids, hiddenDims: hidden.dims, hiddenHash: fnv(hidden.data), hidden: b64(hidden), mean: b64(mean), pooled: b64(pooled) });
  console.log(`${ids.length} tokens, ${Date.now() - t0} ms, hash ${fnv(hidden.data)}: ${JSON.stringify(text.slice(0, 50))}`);
}
const batches = [];
for (const texts of cases.batches) {
  const enc = extractor.tokenizer(texts, { padding: true, truncation: true });
  const hidden = await extractor(texts, { pooling: 'none' });
  const pooled = await extractor(texts, { pooling: 'mean', normalize: true });
  batches.push({ texts, ids: enc.input_ids.tolist().map(r => r.map(Number)), hiddenDims: hidden.dims, hiddenHash: fnv(hidden.data), pooled: b64(pooled) });
  console.log(`batch ${texts.length} x ${hidden.dims[1]}, hash ${fnv(hidden.data)}`);
}
fs.mkdirSync(path.dirname(outFile), { recursive: true });
fs.writeFileSync(outFile, JSON.stringify({ engine: 'transformers.js 3.8.1 + onnxruntime-web 1.22.0-dev.20250409-89f8206ba4 (wasm, 1 thread), dtype q8', single, batches }));
