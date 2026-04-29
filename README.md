<p align="center">
  <img src="dittli_tts.jpg" alt="Dittli TTS" width="480"/>
</p>

<h1 align="center">Dittli TTS</h1>

<p align="center">
  Browser-first lightweight text-to-speech — forked from <a href="https://github.com/tronghieuit/tiny-tts">TinyTTS</a> by tronghieuit
</p>

---

~1.6M parameters · ~3.4 MB ONNX FP16 · 44.1 kHz · runs on CPU

Dittli TTS extends the original TinyTTS with German language support, a full adversarial training pipeline, and multi-language browser runtime.

## Languages

| Language | Voice | Status |
|---|---|---|
| English | Male (original) | Shipped |
| German | Thorsten Voice (CC0, 100k steps) | Shipped |

## Install

### Node.js (npm)

```bash
npm install dittli-tts
```

```javascript
const DittliTTS = require("dittli-tts");

// English
const en = new DittliTTS({ modelPath: "./dittli-en.onnx" });
await en.speak("Hello world!", "en.wav");

// German — same API, different model + sidecar
const de = new DittliTTS({ modelPath: "./dittli-de.onnx" });
await de.speak("Guten Morgen, wie geht es dir?", "de.wav");
```

CLI:

```bash
node bin/cli.js "Hello world" --model models/dittli-en.onnx -o out.wav
node bin/cli.js "Guten Morgen" --model models/dittli-de.onnx -o out.wav
```

### Python

```bash
pip install dittli-tts
```

```python
from dittli_tts import DittliTTS

tts = DittliTTS()
tts.speak("Hello, world!", output_path="hello.wav")
```

German inference (requires trained checkpoint):

```bash
python -m dittli_tts.infer --lang DE --checkpoint G_de.pth \
    --text "Guten Morgen, wie geht es dir?"
```

## Training

See [TRAINING_DE.md](TRAINING_DE.md) for the full cloud training guide (Modal, Kaggle, Vast.ai).

Quick start:

```bash
bash scripts/setup_de_data.sh                          # fetch Thorsten dataset
python -m dittli_tts.data.preprocess \
    --metadata data/thorsten/metadata.csv \
    --wavs-dir data/thorsten/wavs
python scripts/finetune_de.py \
    --metadata data/thorsten/metadata.csv \
    --wavs-dir data/thorsten/wavs \
    --english-ckpt checkpoints/G.pth \
    --ckpt-dir checkpoints_de/ --device cuda
```

## License

Dittli TTS additions © 2026 Dittli TTS contributors, [Apache 2.0](LICENSE).

Original TinyTTS © 2025 tronghieuit, Apache 2.0. See [NOTICE](NOTICE).
