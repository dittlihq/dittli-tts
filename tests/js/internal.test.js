import {
  _abortError,
  _defaultOrtAssetBase,
  _getLanguagePack,
  _LANG_REGISTRY,
  _normalizeLanguage,
  _resolveAsset,
  registerLanguagePack,
} from "@dittli/tts-core/internal";
import { describe, expect, it } from "vitest";

function makePack(lang) {
  return {
    language: lang,
    g2p: (text) => ({ phones: [...text], tones: text.split("").map(() => 0) }),
    assets: { metadata: `${lang}/metadata.json`, model: `${lang}/model.onnx` },
  };
}

describe("_normalizeLanguage", () => {
  it("lowercases two-letter tags", () => {
    expect(_normalizeLanguage("DE")).toBe("de");
    expect(_normalizeLanguage("En")).toBe("en");
  });

  it("strips BCP-47 region subtags", () => {
    expect(_normalizeLanguage("de-DE")).toBe("de");
    expect(_normalizeLanguage("de-AT")).toBe("de");
    expect(_normalizeLanguage("en-US")).toBe("en");
  });

  it("accepts underscore separators", () => {
    expect(_normalizeLanguage("en_GB")).toBe("en");
  });

  it("strips script subtags too", () => {
    expect(_normalizeLanguage("zh-Hant")).toBe("zh");
  });

  it("throws on invalid input", () => {
    expect(() => _normalizeLanguage("")).toThrow();
    expect(() => _normalizeLanguage(null)).toThrow();
    expect(() => _normalizeLanguage(undefined)).toThrow();
  });
});

describe("_resolveAsset", () => {
  it("appends a trailing slash when missing", () => {
    expect(_resolveAsset("/tts", "de/model.onnx")).toBe("/tts/de/model.onnx");
  });

  it("preserves an existing trailing slash", () => {
    expect(_resolveAsset("/tts/", "de/model.onnx")).toBe("/tts/de/model.onnx");
  });

  it("strips a leading slash on the rel path", () => {
    expect(_resolveAsset("/tts/", "/de/model.onnx")).toBe("/tts/de/model.onnx");
  });

  it("works with absolute URLs as base", () => {
    expect(_resolveAsset("https://cdn.example.com", "en/model.onnx")).toBe(
      "https://cdn.example.com/en/model.onnx",
    );
  });

  it("throws on empty base", () => {
    expect(() => _resolveAsset("", "x")).toThrow();
  });
});

describe("_defaultOrtAssetBase", () => {
  it("appends ort/", () => {
    expect(_defaultOrtAssetBase("/tts/")).toBe("/tts/ort/");
    expect(_defaultOrtAssetBase("/tts")).toBe("/tts/ort/");
  });
});

describe("_abortError", () => {
  it("returns signal.reason if it's already an Error", () => {
    const err = new Error("custom reason");
    const signal = { reason: err };
    expect(_abortError(signal)).toBe(err);
  });

  it("wraps a string reason in an AbortError", () => {
    const result = _abortError({ reason: "user cancelled" });
    expect(result).toBeInstanceOf(Error);
    expect(result.name).toBe("AbortError");
    expect(result.message).toBe("user cancelled");
  });

  it("creates a generic AbortError when reason is missing", () => {
    const result = _abortError({});
    expect(result.name).toBe("AbortError");
    expect(result.message).toBe("Aborted");
  });

  it("handles a null signal", () => {
    const result = _abortError(null);
    expect(result.name).toBe("AbortError");
  });
});

describe("registerLanguagePack + _getLanguagePack", () => {
  it("registers under the normalized language key", () => {
    registerLanguagePack(makePack("xx"));
    expect(_LANG_REGISTRY.has("xx")).toBe(true);
    expect(_getLanguagePack("xx")).toMatchObject({ language: "xx" });
  });

  it("looks up by normalized form", () => {
    registerLanguagePack(makePack("yy"));
    expect(_getLanguagePack("YY-CA")).toMatchObject({ language: "yy" });
  });

  it("rejects packs without a language string", () => {
    expect(() => registerLanguagePack({})).toThrow(/language/);
    expect(() => registerLanguagePack({ language: 42 })).toThrow(/language/);
  });

  it("rejects packs without a g2p function", () => {
    expect(() => registerLanguagePack({ language: "zz", assets: {} })).toThrow(/g2p/);
  });

  it("rejects packs without assets", () => {
    expect(() =>
      registerLanguagePack({ language: "zz", g2p: () => ({ phones: [], tones: [] }) }),
    ).toThrow(/assets/);
  });

  it("rejects non-object packs", () => {
    expect(() => registerLanguagePack(null)).toThrow();
    expect(() => registerLanguagePack("string")).toThrow();
  });
});
