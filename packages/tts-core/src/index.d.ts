export type LanguageCode = string;

export interface AssetProgressEvent {
  asset: "model" | "metadata" | "cmudict" | "g2p_model";
  language?: string;
  loaded: number;
  total: number;
}

export interface AssetLayout {
  /** Base URL prefix for per-language assets. Trailing "/" is normalized. */
  assetBase: string;
  /** Separate base for ORT WASMs. Defaults to `${assetBase}ort/`. */
  ortAssetBase?: string;
}

export interface DittliTTSOptions extends AssetLayout {
  /** Primary language. Accepts "de", "de-DE", "en-US", etc. */
  language: LanguageCode;
  /** Default false. When false, library installs a console.warn filter for [W:onnxruntime] noise. */
  verbose?: boolean;
  /** Defaults to ["wasm"]. */
  executionProviders?: string[];
  /** Per-asset download progress. */
  onProgress?: (e: AssetProgressEvent) => void;
  /** Default false. When true, init() skips the kernel-warmup inference. */
  skipWarmup?: boolean;
  /** Explicit language packs (alternative to side-effect imports). */
  packs?: LanguagePack[];
}

export interface SpeakOptions {
  /** Override the instance's primary language for this call. */
  language?: LanguageCode;
  /** Cancels in-flight asset fetches AND aborts the ORT session.run. */
  signal?: AbortSignal;
  speed?: number;
  speaker?: string;
}

export interface DecodedAudio {
  samples: Float32Array;
  sampleRate: number;
}

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

export interface G2PPrepareOptions {
  assetBase: string;
  signal?: AbortSignal;
  onProgress?: (e: AssetProgressEvent) => void;
}

export type G2PFunction = ((
  text: string,
  opts?: { symbolSet?: Set<string>; padStartEnd?: boolean },
) => G2PResult) & {
  /** Loader for assets that should be fetched on first use. */
  prepare?: (opts: G2PPrepareOptions) => Promise<void>;
};

export interface LanguagePack {
  language: LanguageCode;
  g2p: G2PFunction;
  /** Relative paths under `assetBase`. */
  assets: { metadata: string; model: string };
}

export class AudioContextLockedError extends Error {
  readonly name: "AudioContextLockedError";
}

export class DittliTTS {
  constructor(opts: DittliTTSOptions);

  init(): Promise<void>;
  loadLanguage(language: LanguageCode, opts?: { signal?: AbortSignal }): Promise<void>;
  synthesize(text: string, opts?: SpeakOptions): Promise<DecodedAudio>;
  play(text: string, opts?: SpeakOptions): Promise<void>;
  stop(): void;
  dispose(): Promise<void>;

  static preloadWhenIdle(opts: DittliTTSOptions): Promise<DittliTTS>;
}

/** Build a WAV file (Float32 → 16-bit PCM) and return its bytes. */
export function floatToWav(samples: Float32Array, sampleRate: number): Uint8Array;

export default DittliTTS;
