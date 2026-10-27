// Test data for the Rust tests, produced by the reference implementations:
//   testdata/tokenizer-cases.json  texts and their ids from Transformers.js 3.8.1 (no truncation)
//   testdata/nfkc-cases.json       String.prototype.normalize('NFKC') of Node (Unicode 17):
//                                  every code point that changes, plus random sequences
// node gen_testdata.mjs [randomTexts] [randomSequences]
import { AutoTokenizer, env } from '@huggingface/transformers';
import fs from 'fs';
import path from 'path';
env.allowRemoteModels = false;
env.localModelPath = path.resolve('../..') + '/';
const tok = await AutoTokenizer.from_pretrained('hf');
const cases = JSON.parse(fs.readFileSync('../../testdata/cases.json', 'utf8'));
const nTexts = +(process.argv[2] || 20000), nSeqs = +(process.argv[3] || 60000);

// deterministic random numbers (mulberry32), so the files are reproducible
let seed = 20261027;
function rand() { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }
function rnd(n) { return Math.floor(rand() * n); }

const fixed = cases.single.concat([].concat(...cases.batches), [
  ' ', '▁', '▁▁x', 'x▁y', '　x', ' x ', '\u0085NEL', '᠎MVS', '\u0000nul', 'a\u000cb', '�repl',
  '<mask', 'mask>', '<<mask>>', ' <mask> <mask>  ', '<s><s></s>', '<unk>x', 'x<pad>y', '<s>　<mask>', '\t<mask>',
  '～', '～～', 'a～Ａ～b', '～　～', 'constructor', '__proto__', 'hasOwnProperty toString',
  '\ud83d', '\udc00\ud83d', 'x𐀀y', '각', 'Å', 'Ǆǅǆ', 'ﾊﾟﾋﾟ',
  '①②③', '㍿', 'ﷺ', 'ꙮꙮꙮ', '𓀀𓀁', 'ꙮ a ꙮ', '͸͹', '',
]);
const pools = [
  'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', '0123456789', ' ', '  \t\n\r 　 ▁​﻿', "'.,;:!?-_()[]{}<>|/@#$%^&*+=~`\"",
  'αβγδεζηθικλμνξοπρστυφχψωΆΈΉΊΌΎΏάέήίόύώΑΒΓ', 'こんにちは世界日本語漢字ｶﾀｶﾅ', '한국어조선말', 'ภาษาไทย', 'हिन्दी', 'عربي', 'русский',
  '😀🌅👍🏽🇬🇷', '١٢٣٤²³¼½Ⅻ１２ﬁ①㎞™', 'éèêëàâäôöûüçñ', '́̈‍̣', '\u0001\u0002\u007f\u008f\u009f\u0085',
  'ꙮ𓀀𐎀ᚠ', '～~', '𐏿',
];
const specials = ['<s>', '</s>', '<pad>', '<unk>', '<mask>'];
function randomString() {
  let s = '', len = 1 + rnd(40);
  while (s.length < len) {
    if (rnd(25) === 0) { s += specials[rnd(specials.length)]; continue; }
    const p = pools[rnd(pools.length)];
    // code units, not code points, so lone surrogates appear too
    const units = rnd(4) === 0 ? p.split('') : Array.from(p);
    s += units[rnd(units.length)];
  }
  return s;
}
const texts = fixed.concat(Array.from({ length: nTexts }, randomString));
const tokCases = texts.map(t => ({ t, ids: Array.from(tok.encode(t), Number) }));
fs.writeFileSync('../../testdata/tokenizer-cases.json', JSON.stringify(tokCases));

const changed = [];
for (let cp = 0; cp <= 0x10FFFF; cp++) {
  const s = cp >= 0xD800 && cp <= 0xDFFF ? String.fromCharCode(cp) : String.fromCodePoint(cp);
  const n = s.normalize('NFKC');
  if (n !== s) changed.push([cp, Array.from(n, c => c.codePointAt(0))]);
}
const cpPool = [];
for (let cp = 0x300; cp < 0x370; cp++) cpPool.push(cp);
[0x41, 0x61, 0x45, 0x65, 0x4F, 0x6F, 0x55, 0x3B1, 0x3B5, 0x3B7, 0x3C9, 0x391, 0x1100, 0x1161, 0x11A8, 0x11C2, 0xAC00, 0xAC01, 0xD7A3,
 0x3099, 0x309A, 0x304B, 0x30AB, 0xFF76, 0xFF9E, 0x0915, 0x093C, 0x094D, 0x0F71, 0x0F72, 0x0F80, 0x05B0, 0x05BC, 0x1E0A, 0x0323,
 0x212B, 0x2126, 0xFB01, 0x2460, 0x1D400, 0x1F600, 0x0344, 0x0F73, 0x0F75, 0x1D15E, 0x1D165, 0x1D16E, 0x0655, 0x0627, 0x0653, 0x0654,
 0x00A8, 0x02DC, 0x0F77, 0x0F79, 0xFDFA, 0x3300, 0x1F201, 0x16D4A, 0x11131, 0x1E944, 0x0B47, 0x0B3E, 0x0BC6, 0x0BBE, 0x0CC6, 0x0CD5,
 0xD800, 0xDC00].forEach(c => cpPool.push(c));
const seqs = [];
for (let i = 0; i < nSeqs; i++) {
  const len = 1 + rnd(8), cps = [];
  for (let k = 0; k < len; k++) cps.push(cpPool[rnd(cpPool.length)]);
  const s = cps.map(c => c >= 0xD800 && c <= 0xDFFF ? String.fromCharCode(c) : String.fromCodePoint(c)).join('');
  // a high and a low surrogate drawn next to each other form a pair; keep the code point view of the string
  seqs.push([Array.from(s, c => c.codePointAt(0)), Array.from(s.normalize('NFKC'), c => c.codePointAt(0))]);
}
fs.writeFileSync('../../testdata/nfkc-cases.json', JSON.stringify({ unicode: process.versions.unicode, changed, seqs }));
console.log(`tokenizer cases ${tokCases.length}, nfkc: ${changed.length} changed code points, ${seqs.length} sequences`);
