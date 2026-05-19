/**
 * Per-language `Engine` — owns one ONNX session + its metadata, the
 * resolved language pack, and the inference call.
 *
 * `DittliTTS` keeps a `Map<normalizedLang, Engine>` and dispatches
 * `synthesize()` calls to the engine for the requested language.
 */

import { _resolveAsset } from "./internal.js";
import { createSession, releaseSession, runSession, tensor } from "./ort.js";

function _validateMetadata(meta, source) {
  for (const k of ["language", "language_id", "tone_offset", "sample_rate", "symbols"]) {
    if (meta[k] === undefined) {
      throw new Error(`metadata sidecar missing field "${k}": ${source}`);
    }
  }
  return meta;
}

async function _fetchJson(url, signal) {
  const res = await fetch(url, signal ? { signal } : undefined);
  if (!res.ok) throw new Error(`Fetch failed (${res.status}) for ${url}`);
  return await res.json();
}

async function _fetchArrayBuffer(url, signal) {
  const res = await fetch(url, signal ? { signal } : undefined);
  if (!res.ok) throw new Error(`Fetch failed (${res.status}) for ${url}`);
  return await res.arrayBuffer();
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

export class Engine {
  constructor({ pack, assetBase, executionProviders }) {
    this.pack = pack;
    this.assetBase = assetBase;
    this.executionProviders = executionProviders;
    this.session = null;
    this.metadata = null;
    this._symbolSet = null;
  }

  async load({ signal, onProgress } = {}) {
    const metadataUrl = _resolveAsset(this.assetBase, this.pack.assets.metadata);
    const modelUrl = _resolveAsset(this.assetBase, this.pack.assets.model);

    const meta = _validateMetadata(await _fetchJson(metadataUrl, signal), metadataUrl);
    const symMap = {};
    for (let i = 0; i < meta.symbols.length; i++) symMap[meta.symbols[i]] = i;
    meta._symbolMap = symMap;
    this._symbolSet = new Set(meta.symbols);
    this.metadata = meta;
    if (onProgress) {
      onProgress({ asset: "metadata", language: this.pack.language, loaded: 1, total: 1 });
    }

    const modelBytes = new Uint8Array(await _fetchArrayBuffer(modelUrl, signal));
    if (onProgress) {
      onProgress({
        asset: "model",
        language: this.pack.language,
        loaded: modelBytes.byteLength,
        total: modelBytes.byteLength,
      });
    }

    this.session = await createSession(modelBytes, {
      executionProviders: this.executionProviders,
    });

    if (typeof this.pack.g2p.prepare === "function") {
      await this.pack.g2p.prepare({ assetBase: this.assetBase, signal, onProgress });
    }
  }

  textToPhonemeIds(text) {
    const { phones, tones } = this.pack.g2p(text, { symbolSet: this._symbolSet });
    const [phoneIds, toneIds, langIds] = _phonemesToIds(phones, tones, this.metadata);
    return {
      phoneIds: _insertBlanks(phoneIds),
      toneIds: _insertBlanks(toneIds),
      langIds: _insertBlanks(langIds),
    };
  }

  async synthesize(text, { speaker, speed = 1.0, signal } = {}) {
    if (!this.session) {
      throw new Error("Engine.synthesize called before load()");
    }
    const meta = this.metadata;
    const spk2id = meta.spk2id || {};
    let sidValue = 0;
    if (speaker && spk2id[speaker] !== undefined) {
      sidValue = spk2id[speaker];
    }

    const { phoneIds, toneIds, langIds } = this.textToPhonemeIds(text);
    const seqLen = phoneIds.length;

    const feeds = {
      x: tensor("int64", BigInt64Array.from(phoneIds.map((v) => BigInt(v))), [1, seqLen]),
      x_lengths: tensor("int64", [BigInt(seqLen)], [1]),
      sid: tensor("int64", [BigInt(sidValue)], [1]),
      tone: tensor("int64", BigInt64Array.from(toneIds.map((v) => BigInt(v))), [1, seqLen]),
      language: tensor("int64", BigInt64Array.from(langIds.map((v) => BigInt(v))), [1, seqLen]),
      bert: tensor("float32", new Float32Array(seqLen * 1024), [1, 1024, seqLen]),
      ja_bert: tensor("float32", new Float32Array(seqLen * 768), [1, 768, seqLen]),
      noise_scale: tensor("float32", [0.667], [1]),
      noise_scale_w: tensor("float32", [0.8], [1]),
      length_scale: tensor("float32", [1.0 / speed], [1]),
    };

    const results = await runSession(this.session, feeds, signal);
    return {
      samples: results.audio.data,
      sampleRate: meta.sample_rate,
    };
  }

  async dispose() {
    await releaseSession(this.session);
    this.session = null;
  }
}
