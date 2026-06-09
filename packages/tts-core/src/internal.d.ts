import type { LanguageCode, LanguagePack } from "./index";

export function createOnnxG2p(opts: {
  ort: {
    createSession: (bytes: Uint8Array, opts?: object) => Promise<unknown>;
    runSession: (
      session: unknown,
      feeds: object,
      signal?: AbortSignal,
    ) => Promise<Record<string, { data: ArrayLike<number> }>>;
    tensor: (type: string, data: unknown, shape: number[]) => unknown;
  };
  encoderBytes: Uint8Array;
  decoderBytes: Uint8Array;
  vocab: {
    graphemes: string[];
    phonemes: string[];
    start_id: number;
    eos_id: number;
    max_decode?: number;
  };
  executionProviders?: string[];
}): Promise<(word: string) => Promise<string[]>>;

export const _LANG_REGISTRY: Map<string, LanguagePack>;

export function registerLanguagePack(pack: LanguagePack): void;

export function _getLanguagePack(lang: LanguageCode): LanguagePack | undefined;

export function _normalizeLanguage(tag: string): string;

export function _resolveAsset(base: string, rel: string): string;

export function _defaultOrtAssetBase(assetBase: string): string;

export function _abortError(signal?: AbortSignal): Error;

export type { LanguageCode, LanguagePack };
