declare class TinyTTS {
  constructor(options?: {
    modelPath?: string;
    device?: 'cpu' | 'gpu';
  });

  init(): Promise<void>;

  speak(text: string, options?: {
    output?: string;
    speaker?: string;
    speed?: number;
  }): Promise<Buffer>;

  speak(text: string, outputPath?: string): Promise<Buffer>;

  dispose(): Promise<void>;
}

export default TinyTTS;
