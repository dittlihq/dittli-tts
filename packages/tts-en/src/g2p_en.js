/**
 * English grapheme-to-phoneme.
 * Mirrors dittli_tts/text/english.py: CMU dictionary lookup with a neural G2P
 * fallback (g2p_predict.js) for OOV words.
 *
 * Assets (CMU dict ~5 MB, neural G2P weights ~4 MB) are fetched lazily on
 * first use via `graphemeToPhonemeEN.prepare({ assetBase, signal, onProgress })`.
 * Core calls `prepare()` from inside `Engine.load()` with the consumer's
 * `assetBase` — never at module load.
 */

import { predict as _g2pPredict, prepare as _prepareG2pPredict } from "./g2p_predict.js";

let _cmu = null;
let _cmuPromise = null;

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

export function graphemeToPhonemeEN(text, opts = {}) {
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
            const preds = _g2pPredict(part);
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
        const preds = _g2pPredict(core);
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
  const { assetBase, signal, onProgress } = opts || {};
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
    _prepareG2pPredict({ assetBase, signal, onProgress }),
  ]);
};
