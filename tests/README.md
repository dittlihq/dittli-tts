# Tests

`package.json` is the canonical entry point — every command below has a wrapping `npm run`:

| Command | What it runs | Speed |
|---|---|---|
| `npm test` | Default suite: unit + parity + integration (no `slow` marker) | seconds–minutes |
| `npm run test:unit` | `tests/unit/` — pure functions, no I/O | < 5 s |
| `npm run test:parity` | `tests/parity/` — Python G2P vs JS G2P (needs Node) | < 5 s |
| `npm run test:integration` | `tests/integration/` — uses `checkpoints/G.pth` (committed) | ~30 s on CPU |
| `npm run test:slow` | `tests/slow/` — needs the Thorsten dataset on disk | ~30 s after preprocess |
| `npm run test:all` | Everything, including `slow` | minutes |

Equivalent `pytest` invocations work too — the npm scripts just call them.

## Markers

- `slow` — requires external data (Thorsten ~3.7 GB). Excluded by default.
- `gpu` — requires CUDA.
- `node` — requires Node.js. Auto-skipped if `node` isn't on PATH.
- `onnx` — requires an exported ONNX model under `onnx/`.

Run a marked subset with `pytest -m <marker>`. Run *only* unmarked-or-`slow` with `pytest -m ''`.

## Layout

```
tests/
├── conftest.py            # shared fixtures (paths, model load, subprocess helper)
├── unit/                  # pure functions — G2P, normalize, symbols, audio
├── parity/                # Python ↔ JS G2P drift detector
├── integration/           # end-to-end inference (PyTorch / ONNX / Node CLI)
└── slow/                  # training-pipeline smoke (was scripts/smoke_de.py)
```

## Common skip conditions in CI

- NLTK tagger / cmudict not yet downloaded → English-using integration skipped
- `onnxruntime` not installed → ONNX integration skipped
- `npm install` not run → Node CLI integration skipped
- `models/dittli-en.onnx` missing → Node CLI integration skipped
- `data/thorsten/` missing → slow smoke skipped

All skips print the exact missing dep so reproducing locally is straightforward.
