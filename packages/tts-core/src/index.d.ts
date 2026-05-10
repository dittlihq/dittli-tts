interface ModelMetadata {
  language: string;
  language_id: number;
  tone_offset: number;
  sample_rate: number;
  symbols: string[];
  phoneme_set?: string;
  n_speakers?: number;
  spk2id?: Record<string, number>;
}

declare class DittliTTS {
  constructor(options?: {
    modelPath?: string;
    metadataPath?: string;
    language?: string;
    device?: "cpu" | "gpu";
  });

  static registerLanguage(lang: string, g2pFn: (text: string, opts?: object) => { phones: string[]; tones: number[] }): void;
  static registerDefaultMetadata(lang: string, metadataPath: string): void;
  static registerDefaultModel(lang: string, modelPath: string): void;

  metadata: ModelMetadata | null;

  init(): Promise<void>;

  textToPhonemeIds(text: string): {
    phoneIds: number[];
    toneIds: number[];
    langIds: number[];
  };

  speak(
    text: string,
    options?: {
      output?: string;
      speaker?: string;
      speed?: number;
    },
  ): Promise<Buffer>;

  speak(text: string, outputPath?: string): Promise<Buffer>;

  dispose(): Promise<void>;
}

export = DittliTTS;
