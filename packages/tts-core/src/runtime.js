/**
 * The only file in dittli-tts that touches the rten-backed Rust WASM runtime.
 *
 * Inference runs in a dedicated Web Worker (runtime-worker.js) so the main
 * thread is never blocked during synthesis. The session API is unchanged from
 * the caller's perspective — createSession / runSession / releaseSession are
 * all async Promises.
 *
 * The wasm-bindgen JS glue (`dittli_runtime.js`) and `.wasm` binary live in
 * `runtime-wasm/` (populated by `scripts/copy-runtime-wasm.js` before publish
 * or dev). Only the `.wasm` binary is fetched at runtime from the consumer's
 * `wasmBase` URL.
 */

import { _abortError } from "./internal.js";

let _runtimeBase = null;
let _worker = null;
let _nextMsgId = 0;
let _nextSid = 0;
const _pending = new Map();

export function configureRuntime({ wasmBase } = {}) {
  if (wasmBase) {
    _runtimeBase = wasmBase.endsWith("/") ? wasmBase : `${wasmBase}/`;
  }
}

export function isRuntimeConfigured() {
  return _runtimeBase !== null;
}

function _getWorker() {
  if (!_worker) {
    _worker = new Worker(new URL("./runtime-worker.js", import.meta.url), { type: "module" });
    _worker.onmessage = ({ data }) => {
      const settle = _pending.get(data.id);
      if (settle) {
        _pending.delete(data.id);
        if (data.type === "err") settle.reject(new Error(data.message));
        else settle.resolve(data);
      }
    };
    _worker.onerror = (e) => {
      for (const { reject } of _pending.values()) {
        reject(new Error(e.message ?? "Worker error"));
      }
      _pending.clear();
    };
  }
  return _worker;
}

function _post(msg, transfer = []) {
  const id = _nextMsgId++;
  return new Promise((resolve, reject) => {
    _pending.set(id, { resolve, reject });
    _getWorker().postMessage({ ...msg, id }, transfer);
  });
}

export async function createSession(modelBytes, _opts = {}) {
  if (!_runtimeBase) {
    throw new Error("DittliTTS: configureRuntime({ wasmBase }) must be called before use");
  }
  const sid = _nextSid++;
  const transfer = modelBytes.buffer instanceof ArrayBuffer ? [modelBytes.buffer] : [];
  await _post({ type: "createSession", sid, wasmBase: _runtimeBase, bytes: modelBytes }, transfer);
  return { _sid: sid };
}

export async function runSession(session, feeds, signal) {
  if (signal?.aborted) throw _abortError(signal);

  // Transfer TypedArray buffers zero-copy; plain Arrays are structured-cloned.
  const transfer = [];
  for (const v of Object.values(feeds)) {
    if (ArrayBuffer.isView(v.data) && v.data.buffer instanceof ArrayBuffer) {
      transfer.push(v.data.buffer);
    }
  }

  const runPromise = _post({ type: "run", sid: session._sid, feeds }, transfer).then(
    (r) => ({ audio: { data: r.out } }),
  );

  if (!signal) return runPromise;

  let onAbort;
  const abortPromise = new Promise((_, reject) => {
    onAbort = () => reject(_abortError(signal));
    signal.addEventListener("abort", onAbort, { once: true });
  });
  try {
    return await Promise.race([runPromise, abortPromise]);
  } finally {
    if (onAbort) signal.removeEventListener("abort", onAbort);
  }
}

export function tensor(type, data, shape) {
  return { type, data, shape };
}

export async function releaseSession(session) {
  if (session?._sid == null) return;
  await _post({ type: "release", sid: session._sid });
}
