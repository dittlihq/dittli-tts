import { AudioContextLockedError, floatToWav } from "@dittli/tts-core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createAudioContext, playSamples } from "../../packages/tts-core/src/audio.js";

describe("floatToWav", () => {
  it("emits a valid RIFF/WAVE header", () => {
    const samples = new Float32Array([0, 0.5, -0.5, 1]);
    const wav = floatToWav(samples, 16000);
    const view = new DataView(wav.buffer);

    // "RIFF" + "WAVE"
    expect(String.fromCharCode(...wav.slice(0, 4))).toBe("RIFF");
    expect(String.fromCharCode(...wav.slice(8, 12))).toBe("WAVE");
    expect(String.fromCharCode(...wav.slice(12, 16))).toBe("fmt ");
    expect(String.fromCharCode(...wav.slice(36, 40))).toBe("data");

    // PCM, mono, 16-bit
    expect(view.getUint16(20, true)).toBe(1);
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint16(34, true)).toBe(16);

    expect(view.getUint32(24, true)).toBe(16000);
    expect(view.getUint32(28, true)).toBe(16000 * 2);

    // Total size 44 header + 4 samples × 2 bytes
    expect(wav.byteLength).toBe(44 + 8);
    expect(view.getUint32(40, true)).toBe(8);
  });

  it("clips samples outside [-1, 1]", () => {
    const samples = new Float32Array([2, -2, 1, -1]);
    const wav = floatToWav(samples, 8000);
    const view = new DataView(wav.buffer);
    // 2 → +1 → 0x7fff
    expect(view.getInt16(44, true)).toBe(0x7fff);
    // -2 → -1 → -0x8000
    expect(view.getInt16(46, true)).toBe(-0x8000);
    expect(view.getInt16(48, true)).toBe(0x7fff);
    expect(view.getInt16(50, true)).toBe(-0x8000);
  });
});

describe("createAudioContext", () => {
  let savedCtor;
  beforeEach(() => {
    savedCtor = globalThis.AudioContext;
  });
  afterEach(() => {
    globalThis.AudioContext = savedCtor;
    globalThis.webkitAudioContext = undefined;
  });

  it("returns an AudioContext instance", () => {
    const ctx = createAudioContext();
    expect(ctx).toBeInstanceOf(savedCtor);
  });

  it("falls back to webkitAudioContext", () => {
    globalThis.AudioContext = undefined;
    globalThis.webkitAudioContext = savedCtor;
    const ctx = createAudioContext();
    expect(ctx).toBeInstanceOf(savedCtor);
  });

  it("throws when no AudioContext is available", () => {
    globalThis.AudioContext = undefined;
    globalThis.webkitAudioContext = undefined;
    expect(() => createAudioContext()).toThrow(/Web Audio API not available/);
  });

  it("wraps construction failures in AudioContextLockedError", () => {
    class BrokenCtx {
      constructor() {
        throw new Error("autoplay blocked");
      }
    }
    globalThis.AudioContext = BrokenCtx;
    expect(() => createAudioContext()).toThrow(AudioContextLockedError);
  });
});

describe("playSamples", () => {
  it("plays through to onended", async () => {
    const ctx = new globalThis.AudioContext();
    const samples = new Float32Array(16);
    await playSamples({ samples, sampleRate: 16000, audioContext: ctx });
    // No assertion needed; resolved means onended fired.
  });

  it("rejects synchronously when signal is already aborted", async () => {
    const ctx = new globalThis.AudioContext();
    const controller = new AbortController();
    controller.abort();
    await expect(
      playSamples({
        samples: new Float32Array(8),
        sampleRate: 8000,
        audioContext: ctx,
        signal: controller.signal,
      }),
    ).rejects.toMatchObject({ name: "AbortError" });
  });

  it("resumes a suspended AudioContext before starting", async () => {
    const ctx = new globalThis.AudioContext();
    ctx.state = "suspended";
    const samples = new Float32Array(8);
    await playSamples({ samples, sampleRate: 8000, audioContext: ctx });
    expect(ctx._resumeCount).toBe(1);
    expect(ctx.state).toBe("running");
  });

  it("throws when audioContext is missing", async () => {
    await expect(playSamples({ samples: new Float32Array(1), sampleRate: 8000 })).rejects.toThrow(
      /audioContext/,
    );
  });

  it("aborts mid-playback when the signal fires", async () => {
    // Use a custom AudioContext whose source doesn't auto-end so we can
    // observe abort behaviour.
    class StallingSource {
      constructor() {
        this._started = false;
        this._stopped = false;
        this.onended = null;
      }
      connect() {}
      disconnect() {}
      start() {
        this._started = true;
      }
      stop() {
        this._stopped = true;
      }
    }
    class StallingCtx {
      constructor() {
        this.state = "running";
        this.destination = {};
      }
      createBuffer(_c, length, sampleRate) {
        return {
          numberOfChannels: 1,
          length,
          sampleRate,
          getChannelData: () => new Float32Array(length),
        };
      }
      createBufferSource() {
        return new StallingSource();
      }
      async resume() {
        this.state = "running";
      }
    }
    const ctx = new StallingCtx();
    const controller = new AbortController();
    const promise = playSamples({
      samples: new Float32Array(8),
      sampleRate: 8000,
      audioContext: ctx,
      signal: controller.signal,
    });
    queueMicrotask(() => controller.abort());
    await expect(promise).rejects.toMatchObject({ name: "AbortError" });
  });
});
