/**
 * Engine tests with mocked runtime.js. Verifies the asset-fetch
 * flow, metadata validation, AbortSignal propagation, and the
 * Promise.race abort behaviour around runSession.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../packages/tts-core/src/runtime.js", () => {
  const calls = { create: [], run: [], release: [] };
  let pending = null;

  return {
    configureRuntime: vi.fn(),
    isRuntimeConfigured: vi.fn(() => true),
    createSession: vi.fn(async (bytes) => {
      calls.create.push(bytes);
      return { _fake: true };
    }),
    runSession: vi.fn(async (_session, feeds, signal) => {
      calls.run.push(feeds);
      if (signal?.aborted) {
        const err = new Error("Aborted");
        err.name = "AbortError";
        throw err;
      }
      if (pending) {
        // Race the stall against the AbortSignal so abort tests work.
        await new Promise((resolve, reject) => {
          if (signal) {
            signal.addEventListener(
              "abort",
              () => {
                const err = new Error("Aborted");
                err.name = "AbortError";
                reject(err);
              },
              { once: true },
            );
          }
          pending.then(resolve, reject);
        });
      }
      return { audio: { data: new Float32Array(16) } };
    }),
    tensor: vi.fn((type, data, shape) => ({ type, data, shape })),
    releaseSession: vi.fn(async () => {
      calls.release.push(true);
    }),
    __calls: calls,
    __stall(promise) {
      pending = promise;
    },
    __reset() {
      pending = null;
      calls.create.length = 0;
      calls.run.length = 0;
      calls.release.length = 0;
    },
  };
});

const TEXT_DECODER = new TextDecoder();

function makeMeta(overrides = {}) {
  return {
    language: "xx",
    language_id: 0,
    tone_offset: 0,
    sample_rate: 16000,
    symbols: ["_", "a", "b", "UNK"],
    spk2id: {},
    ...overrides,
  };
}

function installFetch(routes) {
  globalThis.fetch = vi.fn(async (url, opts) => {
    const key = String(url);
    if (!(key in routes)) {
      throw new Error(`unexpected fetch: ${key}`);
    }
    const entry = routes[key];
    if (opts?.signal?.aborted) {
      const err = new Error("aborted");
      err.name = "AbortError";
      throw err;
    }
    if (typeof entry === "function") return entry(opts);
    if (entry instanceof Error) throw entry;
    return entry;
  });
}

function jsonResponse(obj) {
  return {
    ok: true,
    status: 200,
    json: async () => obj,
    arrayBuffer: async () => new TextEncoder().encode(JSON.stringify(obj)).buffer,
  };
}

function bytesResponse(bytes) {
  return {
    ok: true,
    status: 200,
    json: async () => JSON.parse(TEXT_DECODER.decode(bytes)),
    arrayBuffer: async () =>
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  };
}

function fakePack(opts = {}) {
  return {
    language: opts.language || "xx",
    g2p: Object.assign(
      vi.fn((_text, _o) => ({ phones: ["a", "b"], tones: [0, 0] })),
      { prepare: opts.prepare },
    ),
    assets: {
      metadata: `${opts.language || "xx"}/metadata.json`,
      model: `${opts.language || "xx"}/model.onnx`,
    },
  };
}

beforeEach(async () => {
  const runtime = await import("../../packages/tts-core/src/runtime.js");
  runtime.__reset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Engine.load", () => {
  it("fetches metadata + model, then calls pack.g2p.prepare", async () => {
    const { Engine } = await import("../../packages/tts-core/src/engine.js");
    const prepare = vi.fn(async () => {});
    const pack = fakePack({ language: "xx", prepare });

    installFetch({
      "/base/xx/metadata.json": jsonResponse(makeMeta()),
      "/base/xx/model.onnx": bytesResponse(new Uint8Array([1, 2, 3, 4])),
    });

    const engine = new Engine({ pack, assetBase: "/base/", executionProviders: ["wasm"] });
    const onProgress = vi.fn();
    await engine.load({ onProgress });

    expect(globalThis.fetch).toHaveBeenCalledTimes(2);
    expect(prepare).toHaveBeenCalledWith(
      expect.objectContaining({ assetBase: "/base/", onProgress }),
    );
    expect(engine.metadata.language).toBe("xx");
    expect(engine.session).not.toBeNull();
    expect(onProgress).toHaveBeenCalledWith(
      expect.objectContaining({ asset: "metadata", language: "xx" }),
    );
    expect(onProgress).toHaveBeenCalledWith(
      expect.objectContaining({ asset: "model", language: "xx" }),
    );
  });

  it("throws when metadata is missing required fields", async () => {
    const { Engine } = await import("../../packages/tts-core/src/engine.js");
    installFetch({
      "/base/xx/metadata.json": jsonResponse({ language: "xx" }), // missing other fields
      "/base/xx/model.onnx": bytesResponse(new Uint8Array([1])),
    });
    const engine = new Engine({
      pack: fakePack(),
      assetBase: "/base/",
      executionProviders: ["wasm"],
    });
    await expect(engine.load()).rejects.toThrow(/metadata sidecar/);
  });

  it("propagates AbortSignal to fetch", async () => {
    const { Engine } = await import("../../packages/tts-core/src/engine.js");
    const seenSignals = [];
    globalThis.fetch = vi.fn(async (_url, opts) => {
      seenSignals.push(opts?.signal);
      return jsonResponse(makeMeta());
    });
    const engine = new Engine({
      pack: fakePack(),
      assetBase: "/base/",
      executionProviders: ["wasm"],
    });
    const controller = new AbortController();
    await engine.load({ signal: controller.signal }).catch(() => {});
    expect(seenSignals[0]).toBe(controller.signal);
  });

  it("surfaces a non-OK fetch as an error", async () => {
    const { Engine } = await import("../../packages/tts-core/src/engine.js");
    globalThis.fetch = vi.fn(async () => ({
      ok: false,
      status: 404,
      json: async () => ({}),
      arrayBuffer: async () => new ArrayBuffer(0),
    }));
    const engine = new Engine({
      pack: fakePack(),
      assetBase: "/base/",
      executionProviders: ["wasm"],
    });
    await expect(engine.load()).rejects.toThrow(/404/);
  });
});

describe("Engine.synthesize", () => {
  it("returns float samples + sample rate", async () => {
    const { Engine } = await import("../../packages/tts-core/src/engine.js");
    installFetch({
      "/base/xx/metadata.json": jsonResponse(makeMeta({ sample_rate: 22050 })),
      "/base/xx/model.onnx": bytesResponse(new Uint8Array([1])),
    });
    const engine = new Engine({
      pack: fakePack(),
      assetBase: "/base/",
      executionProviders: ["wasm"],
    });
    await engine.load();
    const out = await engine.synthesize("hello");
    expect(out.sampleRate).toBe(22050);
    expect(out.samples).toBeInstanceOf(Float32Array);
    expect(out.samples.length).toBe(16);
  });

  it("rejects synchronously on already-aborted signal", async () => {
    const { Engine } = await import("../../packages/tts-core/src/engine.js");
    installFetch({
      "/base/xx/metadata.json": jsonResponse(makeMeta()),
      "/base/xx/model.onnx": bytesResponse(new Uint8Array([1])),
    });
    const engine = new Engine({
      pack: fakePack(),
      assetBase: "/base/",
      executionProviders: ["wasm"],
    });
    await engine.load();
    const controller = new AbortController();
    controller.abort();
    await expect(engine.synthesize("a", { signal: controller.signal })).rejects.toMatchObject({
      name: "AbortError",
    });
  });

  it("aborts an in-flight runSession via Promise.race", async () => {
    const runtime = await import("../../packages/tts-core/src/runtime.js");
    const { Engine } = await import("../../packages/tts-core/src/engine.js");
    installFetch({
      "/base/xx/metadata.json": jsonResponse(makeMeta()),
      "/base/xx/model.onnx": bytesResponse(new Uint8Array([1])),
    });
    // Stall runSession forever so the abort wins the race.
    runtime.__stall(new Promise(() => {}));

    const engine = new Engine({
      pack: fakePack(),
      assetBase: "/base/",
      executionProviders: ["wasm"],
    });
    await engine.load();

    const controller = new AbortController();
    const promise = engine.synthesize("a", { signal: controller.signal });
    queueMicrotask(() => controller.abort("user cancelled"));
    await expect(promise).rejects.toMatchObject({ name: "AbortError" });
  });

  it("uses spk2id when speaker is provided", async () => {
    const runtime = await import("../../packages/tts-core/src/runtime.js");
    const { Engine } = await import("../../packages/tts-core/src/engine.js");
    installFetch({
      "/base/xx/metadata.json": jsonResponse(makeMeta({ spk2id: { ALICE: 3, BOB: 7 } })),
      "/base/xx/model.onnx": bytesResponse(new Uint8Array([1])),
    });
    const engine = new Engine({
      pack: fakePack(),
      assetBase: "/base/",
      executionProviders: ["wasm"],
    });
    await engine.load();
    await engine.synthesize("hello", { speaker: "BOB" });
    const lastFeeds = runtime.__calls.run.at(-1);
    expect(Number(lastFeeds.sid.data[0])).toBe(7);
  });
});

describe("Engine.dispose", () => {
  it("releases the underlying session", async () => {
    const runtime = await import("../../packages/tts-core/src/runtime.js");
    const { Engine } = await import("../../packages/tts-core/src/engine.js");
    installFetch({
      "/base/xx/metadata.json": jsonResponse(makeMeta()),
      "/base/xx/model.onnx": bytesResponse(new Uint8Array([1])),
    });
    const engine = new Engine({
      pack: fakePack(),
      assetBase: "/base/",
      executionProviders: ["wasm"],
    });
    await engine.load();
    await engine.dispose();
    expect(engine.session).toBeNull();
    expect(runtime.__calls.release.length).toBe(1);
  });
});
