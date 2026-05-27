/**
 * Web Worker — runs DittliSession inference off the main thread so the
 * browser UI stays responsive during synchronous WASM execution.
 *
 * Protocol (all messages carry a numeric `id` for request/response correlation):
 *   → { id, type: "createSession", sid, wasmBase, bytes: Uint8Array }
 *   ← { id, type: "ok" }
 *
 *   → { id, type: "run", sid, feeds }   (TypedArray buffers may be transferred)
 *   ← { id, type: "ok", out: Float32Array }  (out buffer transferred back)
 *
 *   → { id, type: "release", sid }
 *   ← { id, type: "ok" }
 *
 * On any error: ← { id, type: "err", message: string }
 */

let _DittliSession = null;
let _initPromise = null;
const _sessions = new Map();

function ensureInit(wasmBase) {
  if (!_initPromise) {
    _initPromise = (async () => {
      // @vite-ignore split prevents Vite from resolving the specifier at build time.
      const mod = await import(/* @vite-ignore */ "../runtime-wasm/" + "dittli_runtime.js");
      await mod.default({ module_or_path: `${wasmBase}dittli_runtime_bg.wasm` });
      _DittliSession = mod.DittliSession;
    })();
  }
  return _initPromise;
}

self.onmessage = async ({ data: msg }) => {
  try {
    switch (msg.type) {
      case "createSession": {
        await ensureInit(msg.wasmBase);
        _sessions.set(msg.sid, new _DittliSession(msg.bytes));
        self.postMessage({ id: msg.id, type: "ok" });
        break;
      }
      case "run": {
        const session = _sessions.get(msg.sid);
        const out = session.run(msg.feeds);
        self.postMessage({ id: msg.id, type: "ok", out }, [out.buffer]);
        break;
      }
      case "release": {
        _sessions.get(msg.sid)?.release();
        _sessions.delete(msg.sid);
        self.postMessage({ id: msg.id, type: "ok" });
        break;
      }
      default:
        self.postMessage({ id: msg.id, type: "err", message: `unknown message type: ${msg.type}` });
    }
  } catch (err) {
    self.postMessage({ id: msg.id, type: "err", message: err?.message ?? String(err) });
  }
};
