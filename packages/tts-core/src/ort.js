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
let _verbose = false;

/**
 * Configure ORT once per page. Subsequent calls with the same `wasmPaths`
 * are no-ops; a different `wasmPaths` will overwrite.
 *
 * ORT emits `[W:onnxruntime:...]` graph-optimisation chatter (e.g. "Could
 * not find a CPU kernel and hence can't constant fold Exp node ...") through
 * the wasm logger, which emscripten binds to `console.error` — so it can't be
 * filtered from the JS side without also swallowing genuine errors. Instead we
 * raise ORT's own log level: at `"error"` the WARNING-level lines are dropped
 * at the source, while real errors still surface. `createSession` mirrors this
 * with a matching per-session `logSeverityLevel`.
 */
export function configureOrt({ wasmPaths, verbose = false } = {}) {
  _verbose = verbose;
  ort.env.logLevel = verbose ? "warning" : "error";
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
    // 3 = ERROR. Matches env.logLevel so the graph-optimisation warnings the
    // optimiser logs during session creation stay suppressed unless verbose.
    logSeverityLevel: _verbose ? 0 : 3,
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
