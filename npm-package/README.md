# DittliTTS

**Ultra-lightweight Text-to-Speech for Node.js — 1.6M params, 44.1kHz, ~53x real-time on CPU.**

[![npm](https://img.shields.io/npm/v/dittli-tts?color=green)](https://www.npmjs.com/package/dittli-tts)
[![PyPI](https://img.shields.io/pypi/v/dittli-tts?color=blue)](https://pypi.org/project/dittli-tts/)

Pure Node.js offline TTS inference via ONNX Runtime. **Zero Python dependency.** The ONNX model (~6 MB) is auto-downloaded from HuggingFace on first use.

## Installation

```bash
npm install dittli-tts
```

## Quick Start

```javascript
const DittliTTS = require('dittli-tts');

const tts = new DittliTTS();

// Synthesize and save to WAV
await tts.speak('Hello world!', { output: 'hello.wav' });

// With options
await tts.speak('This is a fast speech test.', {
  output: 'fast.wav',
  speaker: 'MALE',
  speed: 1.5
});

// Clean up
await tts.dispose();
```

## CLI

```bash
# Basic usage
npx dittli-tts "Hello world!" -o hello.wav

# With options
npx dittli-tts "The weather is nice today." -o output.wav -s MALE --speed 1.2
```

## Features

- **Offline inference** — no server, no API calls, no Python needed
- **ONNX Runtime** — fast CPU inference (~53x real-time)
- **Neural G2P** — ported g2p_en GRU model for accurate pronunciation of any English word
- **Full CMU dictionary** — 123,463 entries for precise phoneme lookup
- **100% G2P match** with Python (PyPI) version across 542 test sentences
- **Auto model download** — ONNX model fetched from HuggingFace on first run

## API

### `new DittliTTS(options?)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| modelPath | string | *(auto-download)* | Path to ONNX model file |

### `tts.speak(text, options?)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| output | string | `'output.wav'` | Output WAV file path |
| speaker | string | `'MALE'` | Speaker ID (`MALE` or `FEMALE`) |
| speed | number | `1.0` | Speech speed (0.3–3.0) |

Returns: `Promise<Buffer>` — WAV audio data (also saves to file if `output` is set)

### `tts.dispose()`

Release ONNX session resources.

## How It Works

1. **Text normalization** — expands numbers, abbreviations, time expressions
2. **G2P pipeline** — converts text to phonemes:
   - Apostrophe-aware word splitting (matches Python BERT tokenizer behavior)
   - CMU dictionary lookup (123K entries)
   - Neural G2P fallback (GRU encoder-decoder, identical to Python `g2p_en`)
3. **Phoneme → IDs** — maps phonemes + tones to model input tensors
4. **ONNX inference** — generates 44.1kHz audio waveform
5. **WAV output** — saves as standard WAV file

## Model Info

| Metric | Value |
|--------|-------|
| Parameters | 1.6M |
| Model size | ~3.4 MB (ONNX FP16) |
| Sample rate | 44.1 kHz |
| CPU speed | ~53x real-time |
| Language | English |

## Python Version

Also available on PyPI with the same G2P output:

```bash
pip install dittli-tts
```

```python
from dittli_tts import DittliTTS
tts = DittliTTS()
tts.speak("Hello world!", output_path="hello.wav")
```

## License

Apache License 2.0

## Links

- **GitHub**: [github.com/tronghieuit/dittli-tts](https://github.com/tronghieuit/dittli-tts)
- **PyPI**: [pypi.org/project/dittli-tts](https://pypi.org/project/dittli-tts/)
- **npm**: [npmjs.com/package/dittli-tts](https://www.npmjs.com/package/dittli-tts)
- **Demo**: [huggingface.co/spaces/backtracking/dittli-tts-demo](https://huggingface.co/spaces/backtracking/dittli-tts-demo)
