"""
Convert Xenova/paraphrase-multilingual-MiniLM-L12-v2 onnx/model_quantized.onnx (q8) and
tokenizer.json into the files the pure-JS engine loads (web/model/).

    python tools/convert_onnx.py hf web/model

The quantized weights are copied bit for bit (int8 / uint8 values, float32 scales and
biases). Linear weights are stored transposed ([out][in]) so the JS matmul reads them
row by row. Nothing is re-quantized or rounded.
"""
import json
import os
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

MAX_FILE = 16 * 1024 * 1024
EMBED_ROWS_PER_FILE = 40000


def main(hf_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(os.path.join(hf_dir, 'onnx', 'model_quantized.onnx'))
    g = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    consts = {}
    for n in g.node:
        if n.op_type == 'Constant':
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    cfg = json.load(open(os.path.join(hf_dir, 'config.json'), encoding='utf8'))
    H, I, L = cfg['hidden_size'], cfg['intermediate_size'], cfg['num_hidden_layers']
    heads = cfg['num_attention_heads']

    # MatMulInteger nodes by name -> (weight, scale, zero point). The weight scale is the
    # input of the Mul that follows the Cast of the MatMulInteger output.
    by_out = {}
    for n in g.node:
        for o in n.input:
            by_out.setdefault(o, []).append(n)
    linear = {}
    for n in g.node:
        if n.op_type != 'MatMulInteger':
            continue
        wq, wzp = n.input[1], n.input[3]
        assert init[wq].dtype == np.int8 and init[wzp].dtype == np.int8, n.name
        scale = wq[:-len('_quantized')] + '_scale'
        prefix = n.name[:-len('/MatMul_quant')]
        linear[prefix] = (init[wq], init[scale], init[wzp])

    files = []          # [bytearray]
    manifest = {'config': {
        'hidden': H, 'inter': I, 'layers': L, 'heads': heads, 'headDim': H // heads,
        'vocab': cfg['vocab_size'], 'maxPos': cfg['max_position_embeddings'],
        'eps': float(np.float32(cfg['layer_norm_eps'])),
    }}

    def new_file():
        files.append(bytearray())

    def put(arr):
        """Append a tensor (aligned to 4 bytes) and return its location."""
        b = np.ascontiguousarray(arr).tobytes()
        if not files or len(files[-1]) + len(b) + 4 > MAX_FILE:
            new_file()
        f = files[-1]
        while len(f) % 4:
            f.append(0)
        off = len(f)
        f.extend(b)
        return {'file': len(files) - 1, 'offset': off, 'length': int(arr.size)}

    # word embeddings: uint8 rows, own files
    we = init['embeddings.word_embeddings.weight_quantized']
    assert we.dtype == np.uint8 and we.shape == (cfg['vocab_size'], H)
    emb_files = []
    for r0 in range(0, we.shape[0], EMBED_ROWS_PER_FILE):
        new_file()
        files[-1].extend(we[r0:r0 + EMBED_ROWS_PER_FILE].tobytes())
        emb_files.append(len(files) - 1)
    new_file()

    def scalar(name):
        a = init[name]
        assert a.size == 1
        return a.reshape(()).item()

    def emb(name):
        return {
            'data': put(init['embeddings.%s.weight_quantized' % name]),
            'scale': float(np.float32(scalar('embeddings.%s.weight_scale' % name))),
            'zp': int(scalar('embeddings.%s.weight_zero_point' % name)),
        }

    word = {'files': emb_files, 'rowsPerFile': EMBED_ROWS_PER_FILE,
            'scale': float(np.float32(scalar('embeddings.word_embeddings.weight_scale'))),
            'zp': int(scalar('embeddings.word_embeddings.weight_zero_point'))}
    manifest['embed'] = {
        'word': word, 'position': emb('position_embeddings'), 'tokenType': emb('token_type_embeddings'),
        'lnW': put(init['embeddings.LayerNorm.weight']), 'lnB': put(init['embeddings.LayerNorm.bias']),
    }

    def lin(prefix, bias):
        w, s, z = linear[prefix]
        k, n = w.shape
        assert s.shape == (n,) and z.shape == (n,)
        return {'k': int(k), 'n': int(n), 'w': put(np.ascontiguousarray(w.T)),
                'scale': put(s.astype(np.float32)), 'zp': put(z), 'bias': put(init[bias])}

    # attention scale: ORT fuses MatMul + Div(sqrt(head_dim)) into FusedMatMul(alpha = 1 / d)
    div = consts['/encoder/layer.0/attention/self/Constant_12_output_0']
    alpha = np.float32(1.0) / np.float32(div)
    manifest['config']['alpha'] = float(alpha)
    manifest['layers'] = []
    for l in range(L):
        p = '/encoder/layer.%d' % l
        e = 'encoder.layer.%d.' % l
        assert np.float32(consts[p + '/attention/self/Constant_12_output_0']) == np.float32(div)
        manifest['layers'].append({
            'q': lin(p + '/attention/self/query', e + 'attention.self.query.bias'),
            'k': lin(p + '/attention/self/key', e + 'attention.self.key.bias'),
            'v': lin(p + '/attention/self/value', e + 'attention.self.value.bias'),
            'o': lin(p + '/attention/output/dense', e + 'attention.output.dense.bias'),
            'ln1W': put(init[e + 'attention.output.LayerNorm.weight']),
            'ln1B': put(init[e + 'attention.output.LayerNorm.bias']),
            'inter': lin(p + '/intermediate/dense', e + 'intermediate.dense.bias'),
            'out': lin(p + '/output/dense', e + 'output.dense.bias'),
            'ln2W': put(init[e + 'output.LayerNorm.weight']),
            'ln2B': put(init[e + 'output.LayerNorm.bias']),
        })

    names = []
    for i, f in enumerate(files):
        name = 'model-%02d.bin' % i
        with open(os.path.join(out_dir, name), 'wb') as fh:
            fh.write(f)
        names.append({'name': name, 'bytes': len(f)})
    manifest['files'] = names

    # tokenizer: Unigram vocabulary (one piece per line) and scores (float64, little endian)
    tok = json.load(open(os.path.join(hf_dir, 'tokenizer.json'), encoding='utf8'))
    model_t = tok['model']
    assert model_t['type'] == 'Unigram'
    vocab = model_t['vocab']
    for piece, _ in vocab:
        assert '\n' not in piece and '\r' not in piece and piece
    with open(os.path.join(out_dir, 'vocab.txt'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(p for p, _ in vocab))
    with open(os.path.join(out_dir, 'scores.bin'), 'wb') as fh:
        fh.write(struct.pack('<%dd' % len(vocab), *[s for _, s in vocab]))
    added = [{'id': a['id'], 'content': a['content'], 'lstrip': a['lstrip'], 'rstrip': a['rstrip']}
             for a in tok['added_tokens']]
    tcfg = json.load(open(os.path.join(hf_dir, 'tokenizer_config.json'), encoding='utf8'))
    manifest['tokenizer'] = {
        'vocabSize': len(vocab), 'unkId': model_t['unk_id'], 'added': added,
        'bos': 0, 'eos': 2, 'pad': tok['padding']['pad_id'] if tok.get('padding') else 1,
        'maxLength': tcfg['model_max_length'],
        'replacement': tok['pre_tokenizer']['pretokenizers'][1]['replacement'],
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=1)
    total = sum(n['bytes'] for n in names)
    print('wrote %d weight files (%.1f MB), vocab %d, alpha %r' % (len(names), total / 1e6, len(vocab), float(alpha)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
