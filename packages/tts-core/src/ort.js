/**
 * The only file in dittli-tts that touches `onnxruntime-web` directly.
 *
 * Everything else in core goes through this module so that ORT setup
 * (WASM paths, warning filter, session lifecycle, AbortSignal plumbing)
 * lives in exactly one place.
 */

// Import the WASM-only entry point, NOT the default "onnxruntime-web".
// The default entry's loader hard-references `ort-wasm-simd-threaded.jsep.wasm`
// (~25 MB, the WebGPU/JSEP build) and fetches it at runtime even though we only
// use the CPU `wasm` execution provider. The `/wasm` subpath references the
// non-jsep `ort-wasm-simd-threaded.wasm` (~12 MB) — which is the only binary
// `scripts/copy-ort-wasm.js` ships — and pulls a smaller JS bundle too.
import * as ort from "onnxruntime-web/wasm";

import { _abortError } from "./internal.js";

let _ortConfigured = false;
let _filterInstalled = false;

/**
 * Install a `console.warn` filter that drops ORT's `[W:onnxruntime:...]`
 * messages. Called once per page when `verbose: false`.
 *
 * Yes, monkey-patching `console.warn` from a library is normally a sin.
 * The README documents this; consumers who want the warnings pass
 * `{ verbose: true }`.
 */
function _installWarnFilter() {
  if (_filterInstalled) return;
  _filterInstalled = true;
  const original = console.warn.bind(console);
  console.warn = (...args) => {
    if (args.length > 0 && typeof args[0] === "string" && args[0].startsWith("[W:onnxruntime")) {
      return;
    }
    original(...args);
  };
}

/**
 * Configure ORT once per page. Subsequent calls with the same `wasmPaths`
 * are no-ops; a different `wasmPaths` will overwrite.
 *
 * Note: when `verbose: false`, the `console.warn` filter is installed
 * exactly once per page and **cannot be reverted**. A later call with
 * `verbose: true` will not restore the original `console.warn`; reload
 * the page if you need ORT warnings back during a debugging session.
 */
export function configureOrt({ wasmPaths, verbose = false } = {}) {
  if (!verbose) _installWarnFilter();
  if (wasmPaths) {
    ort.env.wasm.wasmPaths = wasmPaths;
  }
  _ortConfigured = true;
}

export function isOrtConfigured() {
  return _ortConfigured;
}

export async function createSession(modelBytes, opts = {}) {
  return await ort.InferenceSession.create(modelBytes, {
    executionProviders: opts.executionProviders || ["wasm"],
  });
}

/**
 * Run a session with optional AbortSignal support.
 *
 * ORT doesn't accept an AbortSignal on `session.run`. We `Promise.race`
 * the run against an abort-rejection: the JS promise rejects promptly,
 * but the WASM-side compute keeps running to completion in the
 * background. Aborting saves caller wall-time, not CPU.
 */
export async function runSession(session, feeds, signal) {
  if (!signal) return await session.run(feeds);
  if (signal.aborted) throw _abortError(signal);

  let onAbort;
  const abortPromise = new Promise((_, reject) => {
    onAbort = () => reject(_abortError(signal));
    signal.addEventListener("abort", onAbort, { once: true });
  });

  try {
    return await Promise.race([session.run(feeds), abortPromise]);
  } finally {
    if (onAbort) signal.removeEventListener("abort", onAbort);
  }
}

export function tensor(type, data, shape) {
  return new ort.Tensor(type, data, shape);
}

export async function releaseSession(session) {
  if (session) await session.release();
}
