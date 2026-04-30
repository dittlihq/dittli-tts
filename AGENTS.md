# AGENTS.md

Guidance for AI coding assistants (Claude Code, Cursor, Aider, Copilot,
Codex, etc.) working in this repo. Human contributors: see
[README.md](./README.md) and [docs/](./docs/).

## What this project is

DittliTTS is a polyglot library shipping the same VITS-derived ~1.6 M-param
TTS model as both:

- **Python package** (`src/dittli_tts/`) — full training + inference,
  published to PyPI as `dittli-tts`.
- **Node package** (`src/node/`) — pure ONNX-runtime inference, published
  to npm as `dittli-tts`. **The Node package is the canonical deployment
  target.**

Two languages: English (original) and German (Thorsten Voice fine-tune).

## Layout

```
.
├── pyproject.toml, uv.lock          # Python config (canonical)
├── package.json, package-lock.json  # Node config + canonical command runner
├── biome.json
│
├── src/
│   ├── dittli_tts/                  # Python package
│   │   ├── inference/               # engine.py, onnx.py, export.py
│   │   ├── training/                # trainer.py, losses.py, modal.py
│   │   ├── data/                    # dataset.py, preprocess.py
│   │   ├── models/, nn/, alignment/
│   │   └── text/                    # G2P (english.py, german.py) + symbols
│   └── node/                        # Node package source
│       ├── index.js, bin/cli.js
│       └── g2p_en.js, g2p_de.js, ...
│
├── tools/                           # Python dev tools (app.py, benchmarks)
├── scripts/                         # Data prep + codegen utilities
├── tests/                           # See tests/README.md
├── models/                          # ONNX sidecar metadata (shared)
├── checkpoints/                     # G.pth (committed) + symbol snapshot
└── docs/                            # Training guide + historical notes
```

## Build / dev tooling

- **Python:** `uv` only — never use `pip install -r requirements.txt`,
  there is no `requirements.txt`. Use `uv sync` to install,
  `uv run <cmd>` to invoke. The lockfile is `uv.lock`.
- **Node:** `npm` (or `pnpm`/`yarn` if user prefers). Lockfile is
  `package-lock.json`.
- **Linting:** `npm run lint` (combines ruff + biome), `npm run lint:fix`.
  Specific subsets: `lint:py`, `lint:js`. Always run after a non-trivial
  change.

## Tests

`package.json` is the canonical command runner. See
[tests/README.md](./tests/README.md) for the full matrix.

Most useful invocations:

```bash
npm test                  # default suite (unit + parity + integration)
npm run test:unit         # < 5 s, no I/O
npm run test:integration  # uses checkpoints/G.pth (committed)
npm run test:slow         # opt-in: needs Thorsten dataset
```

Markers: `slow` (external data), `gpu`, `node`, `onnx`. Tests skip with
clear messages when their dependency is missing — never invent fixtures
or commit large binaries to make a test pass.

## Authoritative invariants — DO NOT regress

1. **License attribution stays.** `LICENSE` keeps `Copyright 2025
   tronghieuit`; `NOTICE` and the README's "Original TinyTTS …" lines
   are required by the upstream Apache-2.0 fork. Renaming is fine
   anywhere else.
2. **Python ↔ JS G2P parity.** `src/dittli_tts/text/german.py` is the
   source of truth; `src/node/g2p_de.js` mirrors it. Drift is the most
   common silent training/inference bug in this repo, so
   `tests/parity/test_g2p_de_parity.py` runs on every `npm test`. If
   you change either side, run `python scripts/gen_de_rules.py` to
   regenerate `src/node/g2p_de_rules.json` and re-run the parity test.
3. **Symbol-table compatibility.** `src/dittli_tts/text/symbols.py`
   defines the language ordering and tone offsets the trained
   checkpoints expect. Adding entries is OK; reordering or deleting
   breaks every existing checkpoint. The English symbol snapshot at
   `checkpoints/symbols_v1_en.txt` is used by the embedding remapper —
   leave it alone.
4. **`G.pth` / `D.pth` naming is canonical.** It comes from VITS:
   `G_<step>.pth` is the generator, `D_<step>.pth` the discriminator.
   Don't rename. `G.pth` (no step) is the published English warm-start
   and lives in `checkpoints/`.
5. **No new HuggingFace dependency.** The repo deliberately does not
   use HF Hub or `huggingface_hub` at runtime. The npm package's HF
   model URL (`src/node/index.js:HF_URL`) is being migrated away from;
   don't add new HF call sites.
6. **Top-level `import dittli_tts` stays cheap.** `src/dittli_tts/__init__.py`
   lazy-loads torch / inference engine / english G2P inside method
   bodies. Don't move heavy imports back to module scope; that breaks
   offline tooling and unit-test collection.

## Branches

Long-lived development branches:

- `develop` — primary integration branch
- `cleanup` — current refactor base
- `claude/cleanup-rename-reorganize-*` — agent work branches

Default to working on `claude/<short-task-name>-<random-suffix>`. Only
push to the branch the user names; never to `main`/`master`/`develop`
without explicit instruction.

## House style

- **Tone in code & PRs:** terse, factual, no marketing. The README's
  "ultra-lightweight" / "blazingly fast" is enough for users; internal
  comments and commit messages are technical.
- **Comments:** only when the *why* isn't obvious from a well-named
  identifier. Don't restate what the code does.
- **No emojis** in code, comments, commit messages, or docs unless the
  user explicitly asks.
- **Commit messages:** subject summarises the *why*, body lists the
  concrete changes. Use a single-line subject under ~72 chars; group
  related work into one commit rather than mechanical per-file commits.
- **Backwards-compat shims:** don't add them unless asked. If a
  refactor breaks a caller, change the caller too in the same commit.

## Things to ask the user before doing

- **Renaming a published artifact** (PyPI `dittli-tts`, npm
  `dittli-tts`, HF spaces) — version bump implications.
- **Force-push, history rewrite, branch deletion.** Always confirm.
- **Adding a new dependency** unless it directly replaces a heavier one.
- **Creating a PR.** Don't open one without explicit instruction.
- **Touching `LICENSE` / `NOTICE`** — you almost certainly shouldn't.

## Things you can do without asking

- Edit / refactor / write tests inside `src/`, `tools/`, `scripts/`,
  `tests/`.
- Run `pytest`, `npm test`, `npm run lint`, `uv sync`.
- Update docs in `docs/` and READMEs to match code reality.
- Delete obviously stale temporary files (debug scripts, sample wavs,
  scratch notebooks) — but check `git log -- <path>` first.
