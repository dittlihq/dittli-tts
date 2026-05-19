import { DittliTTS } from "@dittli/tts-core";
import "@dittli/tts-en";
import "@dittli/tts-de";

const ASSET_BASE = "/tts/";

const statusEl = document.getElementById("status");
const langEl = document.getElementById("lang");
const textEl = document.getElementById("text");
const playEl = document.getElementById("play");
const stopEl = document.getElementById("stop");

function log(msg) {
  statusEl.textContent = msg;
  console.log(`[smoke] ${msg}`);
}

let tts = null;
let currentLang = null;

async function getInstance(lang) {
  if (tts && currentLang === lang) return tts;
  if (tts) await tts.dispose();
  log(`init(${lang})…`);
  tts = new DittliTTS({
    language: lang,
    assetBase: ASSET_BASE,
    onProgress: (e) => log(`fetched ${e.asset} (${e.language ?? "-"})`),
  });
  await tts.init();
  currentLang = lang;
  log(`ready (${lang})`);
  return tts;
}

playEl.addEventListener("click", async () => {
  try {
    const inst = await getInstance(langEl.value);
    log(`speaking: "${textEl.value}"`);
    await inst.play(textEl.value);
    log("done");
  } catch (e) {
    log(`error: ${e?.message ?? e}`);
    console.error(e);
  }
});

stopEl.addEventListener("click", () => {
  tts?.stop();
  log("stopped");
});
