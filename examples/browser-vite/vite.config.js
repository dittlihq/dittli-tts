import { defineConfig } from "vite";

export default defineConfig({
  // The smoke app keeps a vanilla Vite config — the only setup is the
  // pre-build `copy-assets.mjs` step that copies per-language assets
  // and the ORT WASMs into public/tts/. That's the same one-shot
  // copy a real consumer does.
  server: {
    fs: { strict: false },
  },
  optimizeDeps: {
    // Skip dep pre-bundling for the workspace packages — they're already
    // ESM and Vite would otherwise yank them out of the workspace tree
    // and break the sibling `assets/` lookup.
    exclude: ["@dittli/tts-core", "@dittli/tts-en", "@dittli/tts-de"],
  },
});
