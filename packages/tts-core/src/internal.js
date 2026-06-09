/**
 * Internal registry + URL helpers. Public-ish surface for power users
 * via `@dittli/tts-core/internal`, but not part of the documented API.
 */

export { createOnnxG2p } from "./g2p_onnx.js";

export const _LANG_REGISTRY = new Map();

/**
 * Register a language pack so `new DittliTTS({ language: "<lang>" })`
 * can find its G2P + asset paths. Called as a side-effect by
 * `import "@dittli/tts-<lang>"`.
 */
export function registerLanguagePack(pack) {
  if (!pack || typeof pack !== "object") {
    throw new Error("registerLanguagePack: pack must be an object");
  }
  if (!pack.language || typeof pack.language !== "string") {
    throw new Error("registerLanguagePack: pack.language is required");
  }
  if (typeof pack.g2p !== "function") {
    throw new Error("registerLanguagePack: pack.g2p must be a function");
  }
  if (!pack.assets || typeof pack.assets !== "object") {
    throw new Error("registerLanguagePack: pack.assets is required");
  }
  const key = _normalizeLanguage(pack.language);
  _LANG_REGISTRY.set(key, pack);
}

export function _getLanguagePack(lang) {
  return _LANG_REGISTRY.get(_normalizeLanguage(lang));
}

/**
 * Normalize a BCP-47 or 2-letter tag to a 2-letter key.
 *   "de", "de-DE", "de-AT", "de-CH"  → "de"
 *   "EN", "en-US", "en_GB"            → "en"
 *
 * If we ever ship region-specific models the registry key becomes
 * the full tag; for now the leading subtag is the registry key.
 */
export function _normalizeLanguage(tag) {
  if (typeof tag !== "string" || tag.length === 0) {
    throw new Error("_normalizeLanguage: tag must be a non-empty string");
  }
  return tag.toLowerCase().split(/[-_]/)[0];
}

/**
 * Resolve a relative asset path against an `assetBase`. The base is
 * normalized to end with `/`.
 */
export function _resolveAsset(base, rel) {
  if (typeof base !== "string" || base.length === 0) {
    throw new Error("_resolveAsset: base must be a non-empty string");
  }
  const normalized = base.endsWith("/") ? base : `${base}/`;
  const relTrimmed = rel.startsWith("/") ? rel.slice(1) : rel;
  return `${normalized}${relTrimmed}`;
}

/**
 * Default ORT asset base derived from `assetBase`: `${assetBase}ort/`.
 */
export function _defaultOrtAssetBase(assetBase) {
  return _resolveAsset(assetBase, "ort/");
}

/**
 * Convert an `AbortSignal`'s state into an Error suitable for rejecting a
 * Promise. Mirrors what the DOM does for AbortSignal-aware APIs.
 */
export function _abortError(signal) {
  if (signal?.reason instanceof Error) return signal.reason;
  const err = new Error(signal?.reason || "Aborted");
  err.name = "AbortError";
  return err;
}
