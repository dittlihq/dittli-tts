/**
 * English grapheme-to-phoneme.
 * Mirrors dittli_tts/text/english.py: CMU dictionary lookup with a neural G2P
 * fallback (encoder.onnx + decoder_step.onnx, run via @dittli/tts-core's
 * createOnnxG2p) for OOV words.
 *
 * Assets (CMU dict ~5 MB, G2P graphs ~1.7 MB) are fetched lazily on first use
 * via `graphemeToPhonemeEN.prepare(...)`. Core calls `prepare()` from inside
 * `Engine.load()` with the consumer's `assetBase` and injected ORT primitives —
 * never at module load. The OOV path is async (ORT inference returns a Promise).
 */

import { createOnnxG2p } from "@dittli/tts-core/internal";

let _cmu = null;
let _cmuPromise = null;

// OOV fallback: encoder.onnx + decoder_step.onnx run on the shared ORT runtime
// (replaces the old hand-rolled JS GRU in g2p_predict.js). Set by prepare().
let _onnxPredict = null;
let _onnxPromise = null;

async function _fetchBytes(url, signal) {
  const res = await fetch(url, signal ? { signal } : undefined);
  if (!res.ok) throw new Error(`Failed to fetch ${url} (${res.status})`);
  return new Uint8Array(await res.arrayBuffer());
}

async function _loadOnnxG2p({ assetBase, signal, onProgress, ort, executionProviders }) {
  if (!ort) {
    throw new Error("graphemeToPhonemeEN.prepare requires injected { ort } (call via Engine.load)");
  }
  const [encoderBytes, decoderBytes, vocab] = await Promise.all([
    _fetchBytes(`${assetBase}en/g2p_encoder.onnx`, signal),
    _fetchBytes(`${assetBase}en/g2p_decoder_step.onnx`, signal),
    fetch(`${assetBase}en/g2p_vocab.json`, signal ? { signal } : undefined).then((r) => r.json()),
  ]);
  _onnxPredict = await createOnnxG2p({
    ort,
    encoderBytes,
    decoderBytes,
    vocab,
    executionProviders,
  });
  if (onProgress) {
    onProgress({ asset: "g2p_onnx", language: "en", loaded: 1, total: 1 });
  }
}

async function _loadCMU({ assetBase, signal, onProgress }) {
  const url = `${assetBase}en/cmudict.json`;
  const res = await fetch(url, signal ? { signal } : undefined);
  if (!res.ok) {
    throw new Error(`Failed to fetch cmudict.json (${res.status}) from ${url}`);
  }
  const data = await res.json();
  if (onProgress) {
    onProgress({ asset: "cmudict", language: "en", loaded: 1, total: 1 });
  }
  return data;
}

function _getCMU() {
  return _cmu || {};
}

function _parsePhone(phn) {
  const m = phn.match(/(\d)$/);
  if (m) return [phn.slice(0, -1).toLowerCase(), parseInt(m[1], 10) + 1];
  return [phn.toLowerCase(), 0];
}

function _parseSyllables(syllables) {
  const phones = [];
  const tones = [];
  for (const syl of syllables) {
    for (const phn of syl) {
      const [ph, tone] = _parsePhone(phn);
      phones.push(ph);
      tones.push(tone);
    }
  }
  return [phones, tones];
}

function _mapPhoneme(ph, symbolSet) {
  const rep = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "\xB7": ",",
    "、": ",",
    "...": "…",
    v: "V",
  };
  if (rep[ph] !== undefined) ph = rep[ph];
  if (symbolSet && !symbolSet.has(ph)) return "UNK";
  return ph;
}

export async function graphemeToPhonemeEN(text, opts = {}) {
  const { symbolSet = null, padStartEnd = true } = opts;
  text = text.toLowerCase().trim();
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  const allPhones = [];
  const allTones = [];
  const word2ph = [];

  for (const word of words) {
    const lead = (word.match(/^[^a-z0-9]*/) || [""])[0];
    const trail = (word.match(/[^a-z0-9']*$/) || [""])[0];
    const core = word.slice(lead.length, word.length - trail.length);

    for (const ch of lead) {
      allPhones.push(_mapPhoneme(ch, symbolSet));
      allTones.push(0);
      word2ph.push(1);
    }

    if (core.length > 0) {
      let resolved = false;

      if (core.includes("'")) {
        const parts = core.split("'");
        const partPhones = [];
        const partTones = [];
        let allFound = true;
        for (let pi = 0; pi < parts.length; pi++) {
          const part = parts[pi];
          if (pi > 0) {
            partPhones.push("'");
            partTones.push(0);
          }
          if (part.length === 0) continue;
          const upper = part.toUpperCase();
          if (_getCMU()[upper]) {
            const [ph, tn] = _parseSyllables([_getCMU()[upper]]);
            partPhones.push(...ph);
            partTones.push(...tn);
          } else {
            const preds = _onnxPredict ? await _onnxPredict(part) : null;
            if (preds && preds.length > 0) {
              for (const phn of preds) {
                const [ph2, tn2] = _parsePhone(phn);
                partPhones.push(ph2);
                partTones.push(tn2);
              }
            } else {
              allFound = false;
              break;
            }
          }
        }
        if (allFound && partPhones.length > 0) {
          for (const p of partPhones) allPhones.push(_mapPhoneme(p, symbolSet));
          allTones.push(...partTones);
          word2ph.push(partPhones.length);
          resolved = true;
        }
      }

      if (!resolved) {
        const upper = core.toUpperCase();
        if (_getCMU()[upper]) {
          const [phones, tones] = _parseSyllables([_getCMU()[upper]]);
          for (const p of phones) allPhones.push(_mapPhoneme(p, symbolSet));
          allTones.push(...tones);
          word2ph.push(phones.length);
          resolved = true;
        }
      }

      if (!resolved) {
        const preds = _onnxPredict ? await _onnxPredict(core) : null;
        if (preds && preds.length > 0) {
          const [ph, tn] = _parseSyllables([preds]);
          for (const p of ph) allPhones.push(_mapPhoneme(p, symbolSet));
          allTones.push(...tn);
          word2ph.push(ph.length);
        } else {
          for (const ch of core) {
            if (ch === "'") continue;
            allPhones.push(_mapPhoneme(ch.toLowerCase(), symbolSet));
            allTones.push(0);
          }
          word2ph.push(core.replace(/'/g, "").length);
        }
      }
    }

    for (const ch of trail) {
      allPhones.push(_mapPhoneme(ch, symbolSet));
      allTones.push(0);
      word2ph.push(1);
    }
  }

  if (padStartEnd) {
    allPhones.unshift("_");
    allPhones.push("_");
    allTones.unshift(0);
    allTones.push(0);
    word2ph.unshift(1);
    word2ph.push(1);
  }
  return { phones: allPhones, tones: allTones, word2ph };
}

graphemeToPhonemeEN.prepare = async function prepare(opts) {
  const { assetBase, signal, onProgress, ort, executionProviders } = opts || {};
  if (!assetBase) {
    throw new Error("graphemeToPhonemeEN.prepare requires { assetBase }");
  }
  await Promise.all([
    _cmu
      ? Promise.resolve()
      : (() => {
          if (!_cmuPromise) _cmuPromise = _loadCMU({ assetBase, signal, onProgress });
          return _cmuPromise.then((d) => {
            _cmu = d;
          });
        })(),
    _onnxPredict
      ? Promise.resolve()
      : (() => {
          if (!_onnxPromise) {
            _onnxPromise = _loadOnnxG2p({ assetBase, signal, onProgress, ort, executionProviders });
          }
          return _onnxPromise;
        })(),
  ]);
};
