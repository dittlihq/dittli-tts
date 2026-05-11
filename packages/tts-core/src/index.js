/**
 * DittliTTS — text-to-speech for the browser via ONNX Runtime Web.
 *
 * This package is the inference engine only. Import a language pack so that
 * its G2P implementation, default model URL, and default metadata URL are
 * registered:
 *
 *   import { DittliTTS } from "@dittli/tts-en";
 *   const tts = new DittliTTS({ language: "en" });
 *   const wavBytes = await tts.speak("hello");
 *
 * The returned bytes are a Uint8Array of a complete WAV file. Wrap in a Blob
 * to play it back:
 *
 *   const url = URL.createObjectURL(new Blob([wavBytes], { type: "audio/wav" }));
 *   new Audio(url).play();
 */

import * as ort from "onnxruntime-web";

const G2P_BY_LANG = {};
const DEFAULT_METADATA_BY_LANG = {};
const DEFAULT_MODEL_BY_LANG = {};

function _toUrl(value) {
  if (value == null) return null;
  if (value instanceof URL) return value;
  return new URL(value, typeof window !== "undefined" ? window.location.href : "file:///");
}

async function _fetchArrayBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch failed (${res.status}) for ${url}`);
  return await res.arrayBuffer();
}

async function _fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch failed (${res.status}) for ${url}`);
  return await res.json();
}

function _validateMetadata(meta, source) {
  for (const k of ["language", "language_id", "tone_offset", "sample_rate", "symbols"]) {
    if (meta[k] === undefined) {
      throw new Error(`metadata sidecar missing field "${k}": ${source}`);
    }
  }
  return meta;
}

function _phonemesToIds(phones, tones, meta) {
  const symbolSet = meta._symbolMap;
  const unkId = symbolSet.UNK;
  const phoneIds = phones.map((p) => (symbolSet[p] !== undefined ? symbolSet[p] : unkId));
  const toneIds = tones.map((t) => t + meta.tone_offset);
  const langIds = new Array(phoneIds.length).fill(meta.language_id);
  return [phoneIds, toneIds, langIds];
}

function _insertBlanks(arr) {
  const n = arr.length;
  const out = new Array(n * 2 + 1).fill(0);
  for (let i = 0; i < n; i++) out[1 + i * 2] = arr[i];
  return out;
}

/**
 * Build a WAV file (Float32 → 16-bit PCM) and return its bytes.
 * Mono only — the engine produces a single channel.
 */
function _floatToWav(samples, sampleRate) {
  const numSamples = samples.length;
  const headerSize = 44;
  const dataSize = numSamples * 2;
  const buf = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buf);

  const writeAscii = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // channels
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (channels * sampleRate * bytesPerSample)
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeAscii(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = headerSize;
  for (let i = 0; i < numSamples; i++) {
    let s = samples[i];
    if (s > 1) s = 1;
    else if (s < -1) s = -1;
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Uint8Array(buf);
}

export class DittliTTS {
  constructor(opts = {}) {
    this.modelUrl = opts.modelUrl != null ? _toUrl(opts.modelUrl) : null;
    this.metadataUrl = opts.metadataUrl != null ? _toUrl(opts.metadataUrl) : null;
    this.language = opts.language || null;
    this.executionProviders = opts.executionProviders || ["wasm"];
    this.session = null;
    this.metadata = null;
    this._symbolSet = null;
    this._initialized = false;
    this._initPromise = null;
  }

  static registerLanguage(lang, g2pFn) {
    G2P_BY_LANG[lang] = g2pFn;
  }

  static registerDefaultMetadata(lang, metadataUrl) {
    DEFAULT_METADATA_BY_LANG[lang] = _toUrl(metadataUrl);
  }

  static registerDefaultModel(lang, modelUrl) {
    DEFAULT_MODEL_BY_LANG[lang] = _toUrl(modelUrl);
  }

  async init() {
    if (this._initialized) return;
    if (this._initPromise) return this._initPromise;
    this._initPromise = this._init();
    try {
      await this._initPromise;
    } finally {
      this._initPromise = null;
    }
  }

  async _init() {
    let modelUrl = this.modelUrl;
    if (!modelUrl) {
      if (this.language && DEFAULT_MODEL_BY_LANG[this.language]) {
        modelUrl = DEFAULT_MODEL_BY_LANG[this.language];
      } else {
        const registered = Object.keys(DEFAULT_MODEL_BY_LANG);
        if (registered.length === 1) {
          modelUrl = DEFAULT_MODEL_BY_LANG[registered[0]];
        }
      }
    }
    if (!modelUrl) {
      throw new Error(
        "No modelUrl provided and no language pack supplied a default. " +
          "Import a language pack (e.g. @dittli/tts-en) or pass { modelUrl }." +
          (Object.keys(DEFAULT_MODEL_BY_LANG).length > 1
            ? ` Loaded packs: ${Object.keys(DEFAULT_MODEL_BY_LANG).join(", ")} — pass { language: 'en' } to disambiguate.`
            : ""),
      );
    }

    let metadataUrl = this.metadataUrl;
    if (!metadataUrl) {
      if (this.language && DEFAULT_METADATA_BY_LANG[this.language]) {
        metadataUrl = DEFAULT_METADATA_BY_LANG[this.language];
      } else {
        const registered = Object.keys(DEFAULT_METADATA_BY_LANG);
        if (registered.length === 1) {
          metadataUrl = DEFAULT_METADATA_BY_LANG[registered[0]];
        }
      }
    }
    if (!metadataUrl) {
      throw new Error(
        "No metadataUrl provided and no language pack supplied a default. " +
          "Import a language pack or pass { metadataUrl }.",
      );
    }

    const meta = _validateMetadata(await _fetchJson(metadataUrl), String(metadataUrl));
    const symMap = {};
    for (let i = 0; i < meta.symbols.length; i++) symMap[meta.symbols[i]] = i;
    meta._symbolMap = symMap;
    this._symbolSet = new Set(meta.symbols);
    this.metadata = meta;

    const g2p = G2P_BY_LANG[meta.language];
    if (!g2p) {
      throw new Error(
        `No G2P registered for language "${meta.language}". ` +
          `Import the matching language pack: @dittli/tts-${meta.language}`,
      );
    }

    const modelBytes = new Uint8Array(await _fetchArrayBuffer(modelUrl));
    this.session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: this.executionProviders,
    });

    if (typeof g2p.prepare === "function") {
      await g2p.prepare();
    }

    this._initialized = true;
  }

  textToPhonemeIds(text) {
    const g2p = G2P_BY_LANG[this.metadata.language];
    const { phones, tones } = g2p(text, { symbolSet: this._symbolSet });
    const [phoneIds, toneIds, langIds] = _phonemesToIds(phones, tones, this.metadata);
    return {
      phoneIds: _insertBlanks(phoneIds),
      toneIds: _insertBlanks(toneIds),
      langIds: _insertBlanks(langIds),
    };
  }

  async speak(text, options = {}) {
    await this.init();

    const speakerName = options.speaker || null;
    const speed = options.speed || 1.0;

    const meta = this.metadata;
    const spk2id = meta.spk2id || {};
    let sidValue = 0;
    if (speakerName && spk2id[speakerName] !== undefined) {
      sidValue = spk2id[speakerName];
    }

    const { phoneIds, toneIds, langIds } = this.textToPhonemeIds(text);
    const seqLen = phoneIds.length;

    const feeds = {
      x: new ort.Tensor("int64", BigInt64Array.from(phoneIds.map((v) => BigInt(v))), [1, seqLen]),
      x_lengths: new ort.Tensor("int64", [BigInt(seqLen)], [1]),
      sid: new ort.Tensor("int64", [BigInt(sidValue)], [1]),
      tone: new ort.Tensor("int64", BigInt64Array.from(toneIds.map((v) => BigInt(v))), [1, seqLen]),
      language: new ort.Tensor("int64", BigInt64Array.from(langIds.map((v) => BigInt(v))), [
        1,
        seqLen,
      ]),
      bert: new ort.Tensor("float32", new Float32Array(seqLen * 1024), [1, 1024, seqLen]),
      ja_bert: new ort.Tensor("float32", new Float32Array(seqLen * 768), [1, 768, seqLen]),
      noise_scale: new ort.Tensor("float32", [0.667], [1]),
      noise_scale_w: new ort.Tensor("float32", [0.8], [1]),
      length_scale: new ort.Tensor("float32", [1.0 / speed], [1]),
    };

    const results = await this.session.run(feeds);
    const audio = results.audio.data;
    return _floatToWav(audio, meta.sample_rate);
  }

  async dispose() {
    if (this.session) {
      await this.session.release();
      this.session = null;
      this._initialized = false;
    }
  }
}

export default DittliTTS;
