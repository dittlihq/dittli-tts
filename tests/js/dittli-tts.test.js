/**
 * High-level DittliTTS tests with mocked ORT, mocked fetch, and our
 * stub Web Audio API from setup.js.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("onnxruntime-web/wasm", () => {
  class FakeTensor {
    constructor(type, data, shape) {
      this.type = type;
      this.data = data;
      this.dims = shape;
    }
  }
  const InferenceSession = {
    create: vi.fn(async () => ({
      run: vi.fn(async () => ({ audio: { data: new Float32Array(8) } })),
      release: vi.fn(async () => {}),
    })),
  };
  return {
    InferenceSession,
    Tensor: FakeTensor,
    env: { wasm: { wasmPaths: undefined } },
  };
});

function makeMeta(language = "xx") {
  return {
    language,
    language_id: 0,
    tone_offset: 0,
    sample_rate: 16000,
    symbols: ["_", "a", "b", "UNK"],
    spk2id: {},
  };
}

function installFetch(routes) {
  globalThis.fetch = vi.fn(async (url, opts) => {
    const key = String(url);
    if (!(key in routes)) throw new Error(`unexpected fetch: ${key}`);
    if (opts?.signal?.aborted) {
      const err = new Error("aborted");
      err.name = "AbortError";
      throw err;
    }
    const entry = routes[key];
    if (typeof entry === "function") return entry();
    return entry;
  });
}

function jsonResp(obj) {
  return {
    ok: true,
    status: 200,
    json: async () => obj,
    arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
  };
}

function modelResp() {
  return {
    ok: true,
    status: 200,
    json: async () => ({}),
    arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
  };
}

async function freshCore() {
  // Re-import on a fresh module graph so each test starts with an empty
  // language registry (registerLanguagePack mutates module state).
  vi.resetModules();
  const core = await import("../../packages/tts-core/src/index.js");
  const internal = await import("../../packages/tts-core/src/internal.js");
  internal._LANG_REGISTRY.clear();
  return { core, internal };
}

function makeRoutes(lang = "xx") {
  return {
    [`/tts/${lang}/metadata.json`]: jsonResp(makeMeta(lang)),
    [`/tts/${lang}/model.onnx`]: modelResp(),
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("DittliTTS constructor", () => {
  it("requires language", async () => {
    const { core } = await freshCore();
    expect(() => new core.DittliTTS({ assetBase: "/tts/" })).toThrow(/language/);
  });

  it("requires assetBase", async () => {
    const { core } = await freshCore();
    expect(() => new core.DittliTTS({ language: "en" })).toThrow(/assetBase/);
  });

  it("registers explicit packs from opts.packs", async () => {
    const { core, internal } = await freshCore();
    const pack = {
      language: "xx",
      g2p: () => ({ phones: [], tones: [] }),
      assets: { metadata: "xx/m", model: "xx/o" },
    };
    new core.DittliTTS({ language: "xx", assetBase: "/tts/", packs: [pack] });
    expect(internal._LANG_REGISTRY.has("xx")).toBe(true);
  });

  it("normalizes the language tag", async () => {
    const { core } = await freshCore();
    const tts = new core.DittliTTS({ language: "de-DE", assetBase: "/tts/" });
    expect(tts._primaryLanguage).toBe("de");
  });
});

describe("DittliTTS.init", () => {
  it("loads the primary language and runs a warmup synth", async () => {
    const { core, internal } = await freshCore();
    const pack = {
      language: "xx",
      g2p: Object.assign(
        vi.fn(() => ({ phones: ["a"], tones: [0] })),
        { prepare: vi.fn(async () => {}) },
      ),
      assets: { metadata: "xx/metadata.json", model: "xx/model.onnx" },
    };
    internal.registerLanguagePack(pack);
    installFetch(makeRoutes());

    const tts = new core.DittliTTS({ language: "xx", assetBase: "/tts/" });
    await tts.init();

    expect(pack.g2p.prepare).toHaveBeenCalledTimes(1);
    // Warmup ran => g2p invoked at least once for the "a" warmup call.
    expect(pack.g2p).toHaveBeenCalled();
  });

  it("skips warmup when skipWarmup is true", async () => {
    const { core, internal } = await freshCore();
    const g2p = vi.fn(() => ({ phones: ["a"], tones: [0] }));
    internal.registerLanguagePack({
      language: "xx",
      g2p,
      assets: { metadata: "xx/metadata.json", model: "xx/model.onnx" },
    });
    installFetch(makeRoutes());

    const tts = new core.DittliTTS({
      language: "xx",
      assetBase: "/tts/",
      skipWarmup: true,
    });
    await tts.init();
    expect(g2p).not.toHaveBeenCalled();
  });

  it("is idempotent on success", async () => {
    const { core, internal } = await freshCore();
    internal.registerLanguagePack({
      language: "xx",
      g2p: () => ({ phones: ["a"], tones: [0] }),
      assets: { metadata: "xx/metadata.json", model: "xx/model.onnx" },
    });
    installFetch(makeRoutes());
    const tts = new core.DittliTTS({ language: "xx", assetBase: "/tts/", skipWarmup: true });
    await tts.init();
    const fetchCount = globalThis.fetch.mock.calls.length;
    await tts.init();
    await tts.init();
    expect(globalThis.fetch.mock.calls.length).toBe(fetchCount);
  });

  it("allows retry after a failed init", async () => {
    const { core, internal } = await freshCore();
    internal.registerLanguagePack({
      language: "xx",
      g2p: () => ({ phones: ["a"], tones: [0] }),
      assets: { metadata: "xx/metadata.json", model: "xx/model.onnx" },
    });

    let failOnce = true;
    globalThis.fetch = vi.fn(async (url) => {
      if (failOnce) {
        failOnce = false;
        throw new Error("network down");
      }
      const routes = makeRoutes();
      return routes[String(url)] ?? Promise.reject(new Error("no route"));
    });

    const tts = new core.DittliTTS({ language: "xx", assetBase: "/tts/", skipWarmup: true });
    await expect(tts.init()).rejects.toThrow(/network down/);
    // Retry should now succeed.
    await tts.init();
  });
});

describe("DittliTTS.loadLanguage", () => {
  it("dedups concurrent calls for the same language", async () => {
    const { core, internal } = await freshCore();
    internal.registerLanguagePack({
      language: "xx",
      g2p: () => ({ phones: [], tones: [] }),
      assets: { metadata: "xx/metadata.json", model: "xx/model.onnx" },
    });
    installFetch(makeRoutes());
    const tts = new core.DittliTTS({ language: "xx", assetBase: "/tts/", skipWarmup: true });

    await Promise.all([tts.loadLanguage("xx"), tts.loadLanguage("xx"), tts.loadLanguage("xx")]);
    // Two fetches: metadata + model. Concurrent calls share the in-flight promise.
    expect(globalThis.fetch.mock.calls.length).toBe(2);
  });

  it("loads a second language onto the same instance", async () => {
    const { core, internal } = await freshCore();
    const xxPack = {
      language: "xx",
      g2p: () => ({ phones: [], tones: [] }),
      assets: { metadata: "xx/metadata.json", model: "xx/model.onnx" },
    };
    const yyPack = {
      language: "yy",
      g2p: () => ({ phones: [], tones: [] }),
      assets: { metadata: "yy/metadata.json", model: "yy/model.onnx" },
    };
    internal.registerLanguagePack(xxPack);
    internal.registerLanguagePack(yyPack);

    installFetch({ ...makeRoutes("xx"), ...makeRoutes("yy") });

    const tts = new core.DittliTTS({ language: "xx", assetBase: "/tts/", skipWarmup: true });
    await tts.init();
    await tts.loadLanguage("yy");

    expect(tts._engines.has("xx")).toBe(true);
    expect(tts._engines.has("yy")).toBe(true);
  });

  it("throws when no pack is registered for the language", async () => {
    const { core } = await freshCore();
    installFetch({});
    const tts = new core.DittliTTS({ language: "xx", assetBase: "/tts/", skipWarmup: true });
    await expect(tts.init()).rejects.toThrow(/No language pack registered/);
  });
});

describe("DittliTTS.synthesize + play", () => {
  async function makeReady(language = "xx") {
    const { core, internal } = await freshCore();
    internal.registerLanguagePack({
      language,
      g2p: () => ({ phones: ["a", "b"], tones: [0, 0] }),
      assets: { metadata: `${language}/metadata.json`, model: `${language}/model.onnx` },
    });
    installFetch(makeRoutes(language));
    const tts = new core.DittliTTS({
      language,
      assetBase: "/tts/",
      skipWarmup: true,
    });
    await tts.init();
    return { core, tts };
  }

  it("returns decoded float samples + sample rate", async () => {
    const { tts } = await makeReady();
    const out = await tts.synthesize("hello");
    expect(out.samples).toBeInstanceOf(Float32Array);
    expect(out.sampleRate).toBe(16000);
  });

  it("plays through to completion", async () => {
    const { tts } = await makeReady();
    await tts.play("hello");
    // resolved means onended fired through the AudioContext stub
  });

  it("aborts the previous play when called again", async () => {
    const { tts } = await makeReady();
    // First play: don't await, so the second call interrupts.
    const first = tts.play("one");
    const second = tts.play("two");
    await expect(first).rejects.toMatchObject({ name: "AbortError" });
    await second;
  });

  it("stop() aborts the current play", async () => {
    const { tts } = await makeReady();
    const promise = tts.play("hello");
    tts.stop();
    await expect(promise).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("DittliTTS.dispose", () => {
  it("releases sessions and prevents further use", async () => {
    const { core, internal } = await freshCore();
    internal.registerLanguagePack({
      language: "xx",
      g2p: () => ({ phones: [], tones: [] }),
      assets: { metadata: "xx/metadata.json", model: "xx/model.onnx" },
    });
    installFetch(makeRoutes());
    const tts = new core.DittliTTS({ language: "xx", assetBase: "/tts/", skipWarmup: true });
    await tts.init();
    await tts.dispose();
    expect(tts._disposed).toBe(true);
    expect(tts._engines.size).toBe(0);
    await expect(tts.synthesize("a")).rejects.toThrow(/disposed/);
    await expect(tts.loadLanguage("xx")).rejects.toThrow(/disposed/);
    await expect(tts.play("a")).rejects.toThrow(/disposed/);
  });
});

describe("DittliTTS.preloadWhenIdle", () => {
  it("returns an inited instance", async () => {
    const { core, internal } = await freshCore();
    internal.registerLanguagePack({
      language: "xx",
      g2p: () => ({ phones: [], tones: [] }),
      assets: { metadata: "xx/metadata.json", model: "xx/model.onnx" },
    });
    installFetch(makeRoutes());
    const tts = await core.DittliTTS.preloadWhenIdle({
      language: "xx",
      assetBase: "/tts/",
      skipWarmup: true,
    });
    expect(tts).toBeInstanceOf(core.DittliTTS);
    expect(tts._engines.has("xx")).toBe(true);
  });
});
