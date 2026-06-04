#!/usr/bin/env python
"""Scaffold a new language pack so contributors start from a working tree.

Creates `packages/tts-<code>/` (package.json, src/index.js, src/index.d.ts) and
an empty `assets/<code>/` directory, then prints the next-steps checklist from
docs/ADDING_A_LANGUAGE.md with the code filled in. It does NOT write the
language-specific G2P or model — those are Steps 2–4 of the guide.

    python scripts/new_language.py fr --name "French"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

INDEX_JS = """\
import {{ registerLanguagePack }} from "@dittli/tts-core/internal";
// TODO: implement the G2P (rule-based g2p_{code}.js, or wire createOnnxG2p for
// a neural pack — see packages/tts-en/src/g2p_en.js). See docs/ADDING_A_LANGUAGE.md.
import {{ graphemeToPhoneme{CODE} }} from "./g2p_{code}.js";

export const {code}Pack = {{
  language: "{code}",
  g2p: graphemeToPhoneme{CODE},
  assets: {{
    metadata: "{code}/metadata.json",
    model: "{code}/model.onnx",
  }},
}};

registerLanguagePack({code}Pack);

export {{ graphemeToPhoneme{CODE} }};
"""

INDEX_DTS = """\
import type {{ G2PFunction }} from "@dittli/tts-core";

export const graphemeToPhoneme{CODE}: G2PFunction;
"""


def _package_json(code: str, name: str) -> dict:
    return {
        "name": f"@dittli/tts-{code}",
        "version": "0.5.0",
        "description": f"DittliTTS {name} language pack — G2P + model metadata for @dittli/tts-core. Browser-targeted.",
        "type": "module",
        "main": "src/index.js",
        "module": "src/index.js",
        "types": "src/index.d.ts",
        "exports": {
            ".": {
                "types": "./src/index.d.ts",
                "import": "./src/index.js",
                "default": "./src/index.js",
            },
            "./assets/*": "./assets/*",
        },
        "files": ["src/**/*.js", "src/**/*.d.ts", "src/**/*.json", "assets/**"],
        "sideEffects": ["./src/index.js"],
        "scripts": {"prepublishOnly": f"node ../../scripts/check-publish-assets.js tts-{code}"},
        "keywords": ["tts", "text-to-speech", name.lower(), "g2p", "browser", "dittli"],
        "author": "Dittli TTS Contributors",
        "license": "Apache-2.0",
        "repository": {
            "type": "git",
            "url": "git+https://github.com/dittlihq/dittli-tts.git",
            "directory": f"packages/tts-{code}",
        },
        "bugs": {"url": "https://github.com/dittlihq/dittli-tts/issues"},
        "homepage": "https://github.com/dittlihq/dittli-tts#readme",
        "peerDependencies": {"@dittli/tts-core": "^0.5.0"},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("code", help="2-letter language code, e.g. fr")
    ap.add_argument("--name", required=True, help='display name, e.g. "French"')
    args = ap.parse_args()

    code = args.code.lower()
    pkg = REPO_ROOT / "packages" / f"tts-{code}"
    if pkg.exists():
        print(f"refusing to overwrite existing {pkg}")
        return 1

    subs = {"code": code, "CODE": code.upper(), "name": args.name}
    (pkg / "src").mkdir(parents=True)
    (pkg / "assets" / code).mkdir(parents=True)
    (pkg / "package.json").write_text(json.dumps(_package_json(code, args.name), indent=2) + "\n")
    (pkg / "src" / "index.js").write_text(INDEX_JS.format(**subs))
    (pkg / "src" / "index.d.ts").write_text(INDEX_DTS.format(**subs))

    print(f"scaffolded packages/tts-{code}/  (package.json, src/index.js, src/index.d.ts, assets/{code}/)")
    print("\nNext (see docs/ADDING_A_LANGUAGE.md):")
    print("  1. symbols + tone + id in src/dittli_tts/text/symbols.py; npm run g2p:metadata")
    print(f"  2. provide the G2P (rule-based g2p_{code}.js, or neural via scripts/export_g2p_onnx.py)")
    print("  3. fine-tune a checkpoint, warm-started from checkpoints/G.pth")
    print(f"  4. export: python -m dittli_tts.inference.export --lang {code.upper()} ...")
    print(f"  5. node scripts/check-publish-assets.js tts-{code} && npm run test:js")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
