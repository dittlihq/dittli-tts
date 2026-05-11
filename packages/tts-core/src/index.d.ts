export interface ModelMetadata {
  language: string;
  language_id: number;
  tone_offset: number;
  sample_rate: number;
  symbols: string[];
  phoneme_set?: string;
  n_speakers?: number;
  spk2id?: Record<string, number>;
}

export interface G2PResult {
  phones: string[];
  tones: number[];
  word2ph?: number[];
}

export type G2PFunction = ((
  text: string,
  opts?: { symbolSet?: Set<string>; padStartEnd?: boolean },
) => G2PResult) & {
  /** Optional async loader for assets that should be fetched on first use. */
  prepare?: () => Promise<void>;
};

export type UrlLike = string | URL;

export interface DittliTTSOptions {
  /** URL (string or URL) to the .onnx model. Falls back to the language pack's default. */
  modelUrl?: UrlLike;
  /** URL (string or URL) to the metadata JSON. Falls back to the language pack's default. */
  metadataUrl?: UrlLike;
  /** Language hint ("en", "de", ...) — used to pick a default when multiple packs are loaded. */
  language?: string;
  /** Execution providers passed to onnxruntime-web. Defaults to ["wasm"]. */
  executionProviders?: string[];
}

export interface SpeakOptions {
  speaker?: string;
  speed?: number;
}

export class DittliTTS {
  constructor(options?: DittliTTSOptions);

  static registerLanguage(lang: string, g2pFn: G2PFunction): void;
  static registerDefaultMetadata(lang: string, metadataUrl: UrlLike): void;
  static registerDefaultModel(lang: string, modelUrl: UrlLike): void;

  metadata: ModelMetadata | null;

  init(): Promise<void>;

  textToPhonemeIds(text: string): {
    phoneIds: number[];
    toneIds: number[];
    langIds: number[];
  };

  /** Synthesizes `text` and returns the WAV file bytes. */
  speak(text: string, options?: SpeakOptions): Promise<Uint8Array>;

  dispose(): Promise<void>;
}

export default DittliTTS;
