// Save the graph after ORT's graph optimizations (onnxruntime-node, level 'all').
import * as ort from 'onnxruntime-node';
const [inp, out] = process.argv.slice(2);
await ort.InferenceSession.create(inp, { graphOptimizationLevel: 'all', optimizedModelFilePath: out });
console.log('saved', out);
