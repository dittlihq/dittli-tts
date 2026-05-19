/**
 * Audio helpers: float samples → WAV bytes, and direct playback via
 * Web Audio API.
 *
 * The `DittliTTS` instance owns the `AudioContext` — `playSamples`
 * takes one as a parameter rather than constructing its own, so that
 * test environments and multi-instance apps don't get a stray context
 * per `play()` call.
 */

const HEADER_SIZE = 44;

/**
 * Build a WAV file (Float32 → 16-bit PCM) and return its bytes.
 * Mono only — the engine produces a single channel.
 */
export function floatToWav(samples, sampleRate) {
  const numSamples = samples.length;
  const dataSize = numSamples * 2;
  const buf = new ArrayBuffer(HEADER_SIZE + dataSize);
  const view = new DataView(buf);

  const writeAscii = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = HEADER_SIZE;
  for (let i = 0; i < numSamples; i++) {
    let s = samples[i];
    if (s > 1) s = 1;
    else if (s < -1) s = -1;
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Uint8Array(buf);
}

export class AudioContextLockedError extends Error {
  constructor(cause) {
    super(
      "AudioContext could not be created — likely blocked by the browser's " +
        "autoplay policy. Call play() from within a user-gesture handler " +
        "(click, tap, keydown).",
    );
    this.name = "AudioContextLockedError";
    if (cause) this.cause = cause;
  }
}

/**
 * Construct an `AudioContext`. Throws `AudioContextLockedError` if the
 * browser refuses (autoplay policy outside a user gesture).
 */
export function createAudioContext() {
  const Ctor = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Ctor) {
    throw new Error("Web Audio API not available in this environment. play() is browser-only.");
  }
  try {
    return new Ctor();
  } catch (e) {
    throw new AudioContextLockedError(e);
  }
}

/**
 * Play `samples` through `audioContext`. Resolves when playback ends
 * or `signal` aborts. The returned cleanup is internal: this owns the
 * `AudioBufferSourceNode` for the duration of the call.
 */
export function playSamples({ samples, sampleRate, audioContext, signal }) {
  if (!audioContext) {
    throw new Error("playSamples requires an audioContext");
  }
  if (signal?.aborted) {
    return Promise.reject(_abortError(signal));
  }

  return new Promise((resolve, reject) => {
    const buffer = audioContext.createBuffer(1, samples.length, sampleRate);
    buffer.getChannelData(0).set(samples);

    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);

    let settled = false;
    const settle = (fn, value) => {
      if (settled) return;
      settled = true;
      if (signal && onAbort) signal.removeEventListener("abort", onAbort);
      try {
        source.disconnect();
      } catch {
        // already disconnected
      }
      fn(value);
    };

    source.onended = () => settle(resolve);

    const onAbort = () => {
      try {
        source.stop();
      } catch {
        // already stopped
      }
      settle(reject, _abortError(signal));
    };
    if (signal) signal.addEventListener("abort", onAbort, { once: true });

    if (audioContext.state === "suspended") {
      audioContext.resume().catch(() => {});
    }
    source.start();
  });
}

function _abortError(signal) {
  if (signal?.reason instanceof Error) return signal.reason;
  const err = new Error(signal?.reason || "Aborted");
  err.name = "AbortError";
  return err;
}
