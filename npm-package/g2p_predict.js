/**
 * g2p_en neural G2P predictor — pure JS port of Python g2p_en.G2p.predict()
 *
 * Implements a GRU encoder-decoder seq2seq model for grapheme-to-phoneme
 * conversion, matching the exact NumPy logic from the Python g2p_en package.
 */

const fs = require("node:fs");
const path = require("node:path");

let _model = null;

/**
 * Decode base64 string to Float32Array
 */
function b64ToFloat32(b64str) {
  const buf = Buffer.from(b64str, "base64");
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

/**
 * Reshape flat Float32Array into 2D array [rows][cols]
 */
function reshape2D(flat, rows, cols) {
  const result = [];
  for (let i = 0; i < rows; i++) {
    result.push(flat.subarray(i * cols, (i + 1) * cols));
  }
  return result;
}

/**
 * Load model weights from g2p_model.json
 */
function loadModel() {
  if (_model) return _model;

  const modelPath = path.join(__dirname, "g2p_model.json");
  if (!fs.existsSync(modelPath)) {
    console.warn("[g2p_predict] g2p_model.json not found, neural G2P unavailable");
    return null;
  }

  const raw = JSON.parse(fs.readFileSync(modelPath, "utf-8"));

  const m = {};
  // Decode weight tensors
  for (const name of [
    "enc_emb",
    "enc_w_ih",
    "enc_w_hh",
    "enc_b_ih",
    "enc_b_hh",
    "dec_emb",
    "dec_w_ih",
    "dec_w_hh",
    "dec_b_ih",
    "dec_b_hh",
    "fc_w",
    "fc_b",
  ]) {
    const { shape, data } = raw[name];
    const flat = b64ToFloat32(data);
    if (shape.length === 2) {
      m[name] = reshape2D(flat, shape[0], shape[1]);
      m[`${name}_shape`] = shape;
    } else {
      m[name] = flat;
    }
  }

  m.graphemes = raw.graphemes;
  m.phonemes = raw.phonemes;
  m.g2idx = {};
  for (let i = 0; i < raw.graphemes.length; i++) m.g2idx[raw.graphemes[i]] = i;
  m.idx2p = {};
  for (let i = 0; i < raw.phonemes.length; i++) m.idx2p[i] = raw.phonemes[i];
  m.homograph2features = raw.homograph2features || {};

  _model = m;
  return m;
}

/**
 * Sigmoid activation
 */
function _sigmoid(x) {
  const result = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    result[i] = 1 / (1 + Math.exp(-x[i]));
  }
  return result;
}

/**
 * GRU cell — single step, matching Python g2p_en grucell exactly
 * x: Float32Array (1, input_dim) flattened
 * h: Float32Array (1, hidden_dim) flattened
 * Returns new h
 */
function gruCell(x, h, w_ih, w_hh, b_ih, b_hh, hiddenDim) {
  const dim3 = hiddenDim * 3;

  // rzn_ih = x @ w_ih.T + b_ih
  const rzn_ih = new Float32Array(dim3);
  for (let i = 0; i < dim3; i++) {
    let sum = b_ih[i];
    const row = w_ih[i]; // w_ih[i] is already a row (transposed access pattern)
    for (let j = 0; j < x.length; j++) {
      sum += x[j] * row[j];
    }
    rzn_ih[i] = sum;
  }

  // rzn_hh = h @ w_hh.T + b_hh
  const rzn_hh = new Float32Array(dim3);
  for (let i = 0; i < dim3; i++) {
    let sum = b_hh[i];
    const row = w_hh[i];
    for (let j = 0; j < h.length; j++) {
      sum += h[j] * row[j];
    }
    rzn_hh[i] = sum;
  }

  const dim2 = hiddenDim * 2;

  // rz = sigmoid(rz_ih + rz_hh)
  const rz = new Float32Array(dim2);
  for (let i = 0; i < dim2; i++) {
    rz[i] = 1 / (1 + Math.exp(-(rzn_ih[i] + rzn_hh[i])));
  }

  // r = rz[:hidden], z = rz[hidden:]
  const r = rz.subarray(0, hiddenDim);
  const z = rz.subarray(hiddenDim, dim2);

  // n_ih = rzn_ih[dim2:]
  // n_hh = rzn_hh[dim2:]
  // n = tanh(n_ih + r * n_hh)
  const newH = new Float32Array(hiddenDim);
  for (let i = 0; i < hiddenDim; i++) {
    const n = Math.tanh(rzn_ih[dim2 + i] + r[i] * rzn_hh[dim2 + i]);
    newH[i] = (1 - z[i]) * n + z[i] * h[i];
  }

  return newH;
}

/**
 * GRU encoder — process full sequence
 */
function gruEncode(embeds, steps, w_ih, w_hh, b_ih, b_hh, hiddenDim) {
  let h = new Float32Array(hiddenDim); // zero init
  for (let t = 0; t < steps; t++) {
    h = gruCell(embeds[t], h, w_ih, w_hh, b_ih, b_hh, hiddenDim);
  }
  return h; // last hidden state
}

/**
 * Predict phonemes for a single word using the neural G2P model
 * Matches Python g2p_en.G2p.predict() exactly
 */
function predict(word) {
  const m = loadModel();
  if (!m) return null;

  const hiddenDim = m.enc_w_hh_shape[1];

  // Encode: chars + </s>
  const chars = word.split("").concat(["</s>"]);
  const encInput = chars.map((ch) => {
    const idx = m.g2idx[ch] !== undefined ? m.g2idx[ch] : m.g2idx["<unk>"];
    return m.enc_emb[idx]; // embedding vector
  });

  // Run encoder GRU
  const lastHidden = gruEncode(
    encInput,
    chars.length,
    m.enc_w_ih,
    m.enc_w_hh,
    m.enc_b_ih,
    m.enc_b_hh,
    hiddenDim,
  );

  // Decoder: start with <s> token (idx=2)
  let dec = m.dec_emb[2]; // <s> embedding
  let h = lastHidden;

  const preds = [];
  for (let i = 0; i < 20; i++) {
    h = gruCell(dec, h, m.dec_w_ih, m.dec_w_hh, m.dec_b_ih, m.dec_b_hh, hiddenDim);

    // logits = h @ fc_w.T + fc_b
    let maxVal = -Infinity,
      maxIdx = 0;
    for (let j = 0; j < m.fc_w.length; j++) {
      let logit = m.fc_b[j];
      const row = m.fc_w[j];
      for (let k = 0; k < hiddenDim; k++) {
        logit += h[k] * row[k];
      }
      if (logit > maxVal) {
        maxVal = logit;
        maxIdx = j;
      }
    }

    if (maxIdx === 3) break; // </s>
    preds.push(m.idx2p[maxIdx] || "<unk>");
    dec = m.dec_emb[maxIdx];
  }

  return preds;
}

/**
 * Get homograph features if available
 */
function getHomographFeatures(word) {
  const m = loadModel();
  if (!m?.homograph2features[word]) return null;
  return m.homograph2features[word];
}

module.exports = { predict, loadModel, getHomographFeatures };
