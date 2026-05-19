import type { LanguageCode, LanguagePack } from "./index";

export const _LANG_REGISTRY: Map<string, LanguagePack>;

export function registerLanguagePack(pack: LanguagePack): void;

export function _getLanguagePack(lang: LanguageCode): LanguagePack | undefined;

export function _normalizeLanguage(tag: string): string;

export function _resolveAsset(base: string, rel: string): string;

export function _defaultOrtAssetBase(assetBase: string): string;

export function _abortError(signal?: AbortSignal): Error;

export type { LanguageCode, LanguagePack };
