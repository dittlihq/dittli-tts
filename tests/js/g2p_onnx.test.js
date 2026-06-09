/**
 * Unit tests for the generic ONNX-G2P host loop (createOnnxG2p) with a mocked
 * ORT runtime — the greedy-decode logic that replaced the hand-rolled JS GRU.
 */

import { describe, expect, it, vi } from "vitest";

import { createOnnxG2p } from "../../packages/tts-core/src/g2p_onnx.js";

const VOCAB = {
  graphemes: ["<pad>", "<unk>", "</s>", "a", "b"],
  phonemes: ["<pad>", "<unk>", "<s>", "</s>", "AA", "B"],
  start_id: 2,
  eos_id: 3,
  max_decode: 20,
};

// A mock ORT whose decoder emits a scripted argmax sequence, ending in EOS.
function mockOrt(decoderArgmaxSequence) {
  const ENC = { kind: "enc" };
  const DEC = { kind: "dec" };
  let step = 0;
  const runSession = vi.fn(async (session, feeds) => {
    if (session === ENC) {
      expect(feeds.char_ids).toBeDefined();
      return { h_enc: { data: new Float32Array([0, 0, 0]) } };
    }
    // decoder: emit logits that argmax to the scripted id for this step
    const id = decoderArgmaxSequence[step++];
    const logits = new Float32Array(VOCAB.phonemes.length).fill(-9);
    logits[id] = 9;
    return { logits: { data: logits }, h_out: { data: new Float32Array([0, 0, 0]) } };
  });
  const createSession = vi.fn(async (bytes) => (bytes[0] === 1 ? ENC : DEC));
  const tensor = vi.fn((type, data, shape) => ({ type, data, shape }));
  return { ort: { createSession, runSession, tensor }, runSession };
}

describe("createOnnxG2p", () => {
  it("greedy-decodes to the phoneme sequence and stops at EOS", async () => {
    // ids 4,5 -> "AA","B", then 3 (</s>) stops.
    const { ort } = mockOrt([4, 5, 3]);
    const predict = await createOnnxG2p({
      ort,
      encoderBytes: new Uint8Array([1]), // -> ENC
      decoderBytes: new Uint8Array([2]), // -> DEC
      vocab: VOCAB,
    });
    expect(await predict("ab")).toEqual(["AA", "B"]);
  });

  it("maps unknown graphemes to <unk> and appends </s> to the input", async () => {
    const { ort, runSession } = mockOrt([4, 3]);
    const predict = await createOnnxG2p({
      ort,
      encoderBytes: new Uint8Array([1]),
      decoderBytes: new Uint8Array([2]),
      vocab: VOCAB,
    });
    await predict("aZ"); // "Z" is OOV -> <unk> (1); "</s>" appended (2)
    const encCall = runSession.mock.calls.find((c) => c[0].kind === "enc");
    expect(Array.from(encCall[1].char_ids.data)).toEqual([3n, 1n, 2n]); // a, <unk>, </s>
  });

  it("respects max_decode (never loops forever)", async () => {
    const { ort } = mockOrt(Array(50).fill(4)); // never emits EOS
    const predict = await createOnnxG2p({
      ort,
      encoderBytes: new Uint8Array([1]),
      decoderBytes: new Uint8Array([2]),
      vocab: { ...VOCAB, max_decode: 5 },
    });
    expect(await predict("ab")).toHaveLength(5);
  });
});
