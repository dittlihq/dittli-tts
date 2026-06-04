/**
 * Generic neural-G2P runner: an `encoder.onnx` + `decoder_step.onnx` pair run
 * on the shared ORT runtime, with the greedy autoregressive loop in host code.
 *
 * Language-agnostic — parameterised entirely by the pack's `g2p_vocab.json`
 * (grapheme→id input table, id→phoneme output table, start/eos ids). A pack
 * fetches its three assets and hands the bytes here; the engine injects the ORT
 * primitives (`createSession`/`runSession`/`tensor`) so this module never
 * imports `onnxruntime-web` directly.
 *
 * Replaces per-language hand-rolled GRU ports (e.g. the old
 * `@dittli/tts-en/g2p_predict.js`): a new language with a neural G2P ships only
 * its weights + vocab and reuses this loop.
 */

const UNK_GRAPHEME = "<unk>";
const EOS_GRAPHEME = "</s>";

/**
 * @param {object} opts
 * @param {{createSession,runSession,tensor}} opts.ort  injected ORT primitives
 * @param {Uint8Array} opts.encoderBytes   char_ids[1,T] -> h_enc[1,H]
 * @param {Uint8Array} opts.decoderBytes   prev_id[1], h[1,H] -> logits[1,V], h_out[1,H]
 * @param {object} opts.vocab              parsed g2p_vocab.json
 * @param {string[]} [opts.executionProviders]
 * @returns {Promise<(word: string) => Promise<string[]>>}
 */
export async function createOnnxG2p({
  ort,
  encoderBytes,
  decoderBytes,
  vocab,
  executionProviders,
}) {
  const opts = executionProviders ? { executionProviders } : {};
  const [encoder, decoder] = await Promise.all([
    ort.createSession(encoderBytes, opts),
    ort.createSession(decoderBytes, opts),
  ]);

  const g2idx = {};
  for (let i = 0; i < vocab.graphemes.length; i++) g2idx[vocab.graphemes[i]] = i;
  const unkId = g2idx[UNK_GRAPHEME] ?? 1;
  const phonemes = vocab.phonemes;
  const startId = vocab.start_id;
  const eosId = vocab.eos_id;
  const maxDecode = vocab.max_decode ?? 20;

  return async function predict(word) {
    const chars = [...word, EOS_GRAPHEME];
    const charIds = BigInt64Array.from(chars, (c) => BigInt(g2idx[c] ?? unkId));
    const encOut = await ort.runSession(encoder, {
      char_ids: ort.tensor("int64", charIds, [1, chars.length]),
    });
    let h = encOut.h_enc;

    const out = [];
    let prevId = startId;
    for (let step = 0; step < maxDecode; step++) {
      const decOut = await ort.runSession(decoder, {
        prev_id: ort.tensor("int64", BigInt64Array.from([BigInt(prevId)]), [1]),
        h,
      });
      const logits = decOut.logits.data;
      let best = 0;
      for (let j = 1; j < logits.length; j++) {
        if (logits[j] > logits[best]) best = j;
      }
      if (best === eosId) break;
      out.push(phonemes[best]);
      prevId = best;
      h = decOut.h_out;
    }
    return out;
  };
}
