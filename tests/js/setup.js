/**
 * Vitest setup: install lightweight Web Audio stubs and a fetch
 * placeholder. Individual tests override `globalThis.fetch` per case
 * via `vi.fn` so they can assert what was requested.
 *
 * happy-dom doesn't ship Web Audio, so we install just enough
 * surface area for `audio.js` and `DittliTTS.play()` to exercise.
 */

import { vi } from "vitest";

class FakeAudioBuffer {
  constructor(channels, length, sampleRate) {
    this.numberOfChannels = channels;
    this.length = length;
    this.sampleRate = sampleRate;
    this._data = new Float32Array(length);
  }
  getChannelData() {
    return this._data;
  }
}

class FakeAudioBufferSourceNode {
  constructor() {
    this.buffer = null;
    this.onended = null;
    this._started = false;
    this._stopped = false;
    this._connected = false;
  }
  connect() {
    this._connected = true;
  }
  disconnect() {
    this._connected = false;
  }
  start() {
    if (this._started) throw new Error("AudioBufferSourceNode already started");
    this._started = true;
    // Schedule onended on a microtask so tests can attach abort handlers etc.
    queueMicrotask(() => {
      if (this._stopped) return;
      if (this.onended) this.onended();
    });
  }
  stop() {
    if (!this._started) throw new InvalidStateError();
    this._stopped = true;
  }
}

class InvalidStateError extends Error {
  constructor() {
    super("source has not been started");
    this.name = "InvalidStateError";
  }
}

class FakeAudioContext {
  constructor() {
    this.state = "running";
    this.destination = {};
    this._closed = false;
    this._resumeCount = 0;
  }
  createBuffer(channels, length, sampleRate) {
    return new FakeAudioBuffer(channels, length, sampleRate);
  }
  createBufferSource() {
    return new FakeAudioBufferSourceNode();
  }
  async resume() {
    this._resumeCount++;
    this.state = "running";
  }
  async close() {
    this._closed = true;
    this.state = "closed";
  }
}

globalThis.AudioContext = FakeAudioContext;
globalThis.AudioBuffer = FakeAudioBuffer;
globalThis.AudioBufferSourceNode = FakeAudioBufferSourceNode;
globalThis.__FakeAudio = { FakeAudioContext, FakeAudioBufferSourceNode };

// Default fetch is a clear failure — every test that hits the network
// must install its own mock.
globalThis.fetch = vi.fn(() => {
  throw new Error("fetch not mocked for this test");
});
