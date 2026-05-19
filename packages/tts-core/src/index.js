/**
 * DittliTTS — text-to-speech for the browser via ONNX Runtime Web.
 *
 *   import { DittliTTS } from "@dittli/tts-core";
 *   import "@dittli/tts-de";  // side-effect registers the German pack
 *
 *   const tts = new DittliTTS({ language: "de", assetBase: "/tts/" });
 *   await tts.play("Hallo Welt");
 *
 * The consumer copies `node_modules/@dittli/tts-{core,en,de}/assets`
 * (and `tts-core/ort-wasm/`) into one tree at `assetBase`. See README.
 */

import { createAudioContext, playSamples } from "./audio.js";
import { Engine } from "./engine.js";
import {
  _defaultOrtAssetBase,
  _getLanguagePack,
  _normalizeLanguage,
  registerLanguagePack,
} from "./internal.js";
import { configureOrt } from "./ort.js";

export { AudioContextLockedError, floatToWav } from "./audio.js";

export class DittliTTS {
  constructor(opts = {}) {
    if (!opts.language) {
      throw new Error("DittliTTS: { language } is required");
    }
    if (!opts.assetBase) {
      throw new Error("DittliTTS: { assetBase } is required");
    }

    this._primaryLanguage = _normalizeLanguage(opts.language);
    this._assetBase = opts.assetBase;
    this._ortAssetBase = opts.ortAssetBase || _defaultOrtAssetBase(opts.assetBase);
    this._executionProviders = opts.executionProviders || ["wasm"];
    this._verbose = opts.verbose === true;
    this._onProgress = opts.onProgress || null;
    this._skipWarmup = opts.skipWarmup === true;
    this._extraPacks = opts.packs || [];

    this._engines = new Map();
    this._inflight = new Map();
    this._audioContext = null;
    this._currentPlayController = null;
    this._initPromise = null;
    this._disposed = false;

    for (const pack of this._extraPacks) {
      registerLanguagePack(pack);
    }

    configureOrt({ wasmPaths: this._ortAssetBase, verbose: this._verbose });
  }

  /**
   * Idempotent. Loads the primary language and runs a kernel warmup.
   *
   * A successful init is sticky — subsequent calls return the cached
   * promise. A failed init (e.g., transient network error during asset
   * fetch) resets internal state so the next `init()` call retries.
   */
  async init() {
    if (this._initPromise) return this._initPromise;
    const promise = this._init();
    this._initPromise = promise;
    try {
      await promise;
    } catch (err) {
      if (this._initPromise === promise) this._initPromise = null;
      throw err;
    }
    return promise;
  }

  async _init() {
    await this.loadLanguage(this._primaryLanguage);
    if (!this._skipWarmup) {
      try {
        await this.synthesize("a", { language: this._primaryLanguage });
      } catch (e) {
        // Warmup failures are non-fatal — the real synthesize() call
        // will surface the same error with the actual input.
        if (this._verbose) console.warn("DittliTTS warmup failed:", e);
      }
    }
  }

  /** Load a second (or third) language onto this instance. Idempotent. */
  async loadLanguage(language, { signal } = {}) {
    if (this._disposed) throw new Error("DittliTTS: instance has been disposed");
    const key = _normalizeLanguage(language);
    if (this._engines.has(key)) return;
    if (this._inflight.has(key)) {
      await this._inflight.get(key);
      return;
    }
    const promise = this._loadLanguage(key, signal);
    this._inflight.set(key, promise);
    try {
      const engine = await promise;
      this._engines.set(key, engine);
    } finally {
      this._inflight.delete(key);
    }
  }

  async _loadLanguage(key, signal) {
    const pack = _getLanguagePack(key);
    if (!pack) {
      throw new Error(
        `No language pack registered for "${key}". ` +
          `Import the matching package: import "@dittli/tts-${key}"`,
      );
    }
    const engine = new Engine({
      pack,
      assetBase: this._assetBase,
      executionProviders: this._executionProviders,
    });
    await engine.load({ signal, onProgress: this._onProgress });
    return engine;
  }

  /** Pure inference. Returns decoded float samples + sample rate. */
  async synthesize(text, opts = {}) {
    if (this._disposed) throw new Error("DittliTTS: instance has been disposed");
    const lang = _normalizeLanguage(opts.language || this._primaryLanguage);
    let engine = this._engines.get(lang);
    if (!engine) {
      await this.loadLanguage(lang, { signal: opts.signal });
      engine = this._engines.get(lang);
    }
    return engine.synthesize(text, {
      speaker: opts.speaker,
      speed: opts.speed,
      signal: opts.signal,
    });
  }

  /** Synthesize and play through an instance-owned AudioContext. */
  async play(text, opts = {}) {
    if (this._disposed) throw new Error("DittliTTS: instance has been disposed");

    this._currentPlayController?.abort();
    const internal = new AbortController();
    this._currentPlayController = internal;

    const userSignal = opts.signal;
    const onUserAbort = () => internal.abort(userSignal?.reason);
    if (userSignal) {
      if (userSignal.aborted) internal.abort(userSignal.reason);
      else userSignal.addEventListener("abort", onUserAbort, { once: true });
    }

    try {
      const { samples, sampleRate } = await this.synthesize(text, {
        ...opts,
        signal: internal.signal,
      });
      if (!this._audioContext) {
        this._audioContext = createAudioContext();
      }
      await playSamples({
        samples,
        sampleRate,
        audioContext: this._audioContext,
        signal: internal.signal,
      });
    } finally {
      if (userSignal) userSignal.removeEventListener("abort", onUserAbort);
      if (this._currentPlayController === internal) {
        this._currentPlayController = null;
      }
    }
  }

  /** Stop any current playback without disposing the engine. */
  stop() {
    this._currentPlayController?.abort();
    this._currentPlayController = null;
  }

  /** Release all ORT sessions and close the AudioContext. */
  async dispose() {
    this._disposed = true;
    this.stop();
    for (const engine of this._engines.values()) {
      await engine.dispose();
    }
    this._engines.clear();
    if (this._audioContext) {
      try {
        await this._audioContext.close();
      } catch {
        // already closed
      }
      this._audioContext = null;
    }
  }

  /**
   * Construct + init in one call, scheduled via requestIdleCallback
   * (falls back to setTimeout(0) on Safari). Returns the live instance.
   */
  static preloadWhenIdle(opts) {
    return new Promise((resolve, reject) => {
      const schedule =
        typeof globalThis.requestIdleCallback === "function"
          ? globalThis.requestIdleCallback
          : (cb) => setTimeout(cb, 0);
      schedule(async () => {
        try {
          const tts = new DittliTTS(opts);
          await tts.init();
          resolve(tts);
        } catch (e) {
          reject(e);
        }
      });
    });
  }
}

export default DittliTTS;
