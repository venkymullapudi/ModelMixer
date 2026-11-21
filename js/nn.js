/* ModelMixer — tiny neural-network engine (MLP, Adam, Fisher, Gram matrices).
 * Everything runs in the browser on Float32Arrays; no dependencies. */
(function () {
  const MM = (window.MM = window.MM || {});

  // ---------- RNG ----------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0;
    while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }
  function hashStr(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }

  // ---------- Layout ----------
  // Parameters of an MLP are stored flat: W1,b1,W2,b2,... ; W is (out x in) row-major.
  // Inputs (x, y) are first lifted by a fixed, non-trainable radial-basis encoder:
  // a G×G grid of Gaussian bumps over the plane. This makes features local, so a
  // skill learned in one region lives mostly in its own weights — like real nets.
  function makeEncoder(G, extent, sigmaFactor) {
    const centers = new Float32Array(G * G * 2);
    const step = (2 * extent) / (G - 1);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      centers[2 * (i * G + j)] = -extent + j * step;
      centers[2 * (i * G + j) + 1] = -extent + i * step;
    }
    const sigma = step * (sigmaFactor || 0.5);
    return { G, centers, n: G * G, inv2s2: 1 / (2 * sigma * sigma) };
  }
  function encode(enc, X, n) {
    const C = enc.n, F = new Float32Array(n * C), c = enc.centers, k = enc.inv2s2;
    for (let s = 0; s < n; s++) {
      const x = X[2 * s], y = X[2 * s + 1];
      for (let j = 0; j < C; j++) {
        const dx = x - c[2 * j], dy = y - c[2 * j + 1];
        F[s * C + j] = Math.exp(-(dx * dx + dy * dy) * k);
      }
    }
    return F;
  }

  function makeLayout(sizes, enc) {
    if (enc) sizes = [enc.n].concat(sizes.slice(1));
    const tensors = [];
    let off = 0;
    for (let l = 0; l < sizes.length - 1; l++) {
      const inn = sizes[l], out = sizes[l + 1];
      tensors.push({ name: `W${l + 1}`, kind: 'w', layer: l, off, rows: out, cols: inn, size: out * inn });
      off += out * inn;
      tensors.push({ name: `b${l + 1}`, kind: 'b', layer: l, off, rows: out, cols: 1, size: out });
      off += out;
    }
    // inDim = width of one raw input row (2 for plane points fed through the RBF encoder)
    return { sizes, tensors, total: off, nLayers: sizes.length - 1, enc: enc || null, inDim: enc ? 2 : sizes[0] };
  }

  function initParams(layout, seed) {
    const rng = mulberry32(seed);
    const P = new Float32Array(layout.total);
    for (const t of layout.tensors) {
      if (t.kind !== 'w') continue;
      const std = Math.sqrt(2 / (t.rows + t.cols));
      for (let i = 0; i < t.size; i++) P[t.off + i] = gauss(rng) * std;
    }
    return P;
  }

  // ---------- Forward / backward ----------
  // Hidden layers use tanh; last layer is linear (logits).
  function forward(layout, P, X, n, keep) {
    const S = layout.sizes, L = S.length - 1;
    let cur = layout.enc ? encode(layout.enc, X, n) : X;
    const acts = keep ? [cur] : null;
    for (let l = 0; l < L; l++) {
      const inn = S[l], out = S[l + 1];
      const Wo = layout.tensors[2 * l].off, bo = layout.tensors[2 * l + 1].off;
      const nx = new Float32Array(n * out);
      const last = l === L - 1;
      for (let s = 0; s < n; s++) {
        const xo = s * inn, yo = s * out;
        for (let i = 0; i < out; i++) {
          let sum = P[bo + i];
          const wi = Wo + i * inn;
          for (let k = 0; k < inn; k++) sum += P[wi + k] * cur[xo + k];
          nx[yo + i] = last ? sum : Math.tanh(sum);
        }
      }
      cur = nx;
      if (keep && !last) acts.push(nx);
    }
    return { out: cur, acts };
  }

  function backward(layout, P, acts, dz, n) {
    const S = layout.sizes, L = S.length - 1;
    const g = new Float32Array(layout.total);
    let delta = dz;
    for (let l = L - 1; l >= 0; l--) {
      const inn = S[l], out = S[l + 1];
      const Wo = layout.tensors[2 * l].off, bo = layout.tensors[2 * l + 1].off;
      const a = acts[l];
      for (let s = 0; s < n; s++) {
        for (let i = 0; i < out; i++) {
          const d = delta[s * out + i];
          if (d === 0) continue;
          g[bo + i] += d;
          const wi = Wo + i * inn, ao = s * inn;
          for (let k = 0; k < inn; k++) g[wi + k] += d * a[ao + k];
        }
      }
      if (l > 0) {
        const nd = new Float32Array(n * inn);
        for (let s = 0; s < n; s++) {
          for (let i = 0; i < out; i++) {
            const d = delta[s * out + i];
            if (d === 0) continue;
            const wi = Wo + i * inn, no = s * inn;
            for (let k = 0; k < inn; k++) nd[no + k] += P[wi + k] * d;
          }
        }
        for (let j = 0; j < nd.length; j++) nd[j] *= 1 - a[j] * a[j];
        delta = nd;
      }
    }
    return g;
  }

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  // Binary cross-entropy (soft targets allowed)
  function bceGrad(layout, P, X, Y, n) {
    const { out, acts } = forward(layout, P, X, n, true);
    const dz = new Float32Array(n);
    let loss = 0;
    for (let s = 0; s < n; s++) {
      const p = sigmoid(out[s]);
      const y = Y[s];
      loss -= y * Math.log(p + 1e-7) + (1 - y) * Math.log(1 - p + 1e-7);
      dz[s] = (p - y) / n;
    }
    return { loss: loss / n, grad: backward(layout, P, acts, dz, n) };
  }

  // Softmax cross-entropy over K outputs (used by the MoE router)
  function softmaxGrad(layout, P, X, labels, n) {
    const K = layout.sizes[layout.sizes.length - 1];
    const { out, acts } = forward(layout, P, X, n, true);
    const dz = new Float32Array(n * K);
    let loss = 0;
    for (let s = 0; s < n; s++) {
      let m = -Infinity;
      for (let k = 0; k < K; k++) m = Math.max(m, out[s * K + k]);
      let z = 0;
      for (let k = 0; k < K; k++) z += Math.exp(out[s * K + k] - m);
      for (let k = 0; k < K; k++) {
        const p = Math.exp(out[s * K + k] - m) / z;
        dz[s * K + k] = (p - (labels[s] === k ? 1 : 0)) / n;
        if (labels[s] === k) loss -= Math.log(p + 1e-7);
      }
    }
    return { loss: loss / n, grad: backward(layout, P, acts, dz, n) };
  }

  function softmaxRows(out, n, K) {
    const r = new Float32Array(n * K);
    for (let s = 0; s < n; s++) {
      let m = -Infinity;
      for (let k = 0; k < K; k++) m = Math.max(m, out[s * K + k]);
      let z = 0;
      for (let k = 0; k < K; k++) z += (r[s * K + k] = Math.exp(out[s * K + k] - m));
      for (let k = 0; k < K; k++) r[s * K + k] /= z;
    }
    return r;
  }

  function makeAdam(P, lr, eps) {
    eps = eps || 1e-8;
    const m = new Float32Array(P.length), v = new Float32Array(P.length);
    let t = 0;
    return function step(g) {
      t++;
      const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
      for (let i = 0; i < P.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        P[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
    };
  }

  // Generator so callers can train in animation-frame-sized chunks.
  function* trainGen(layout, P, X, Y, n, steps, lr, lossFn, eps) {
    const step = makeAdam(P, lr, eps);
    const fn = lossFn || bceGrad;
    for (let i = 0; i < steps; i++) {
      const { loss, grad } = fn(layout, P, X, Y, n);
      step(grad);
      yield { step: i + 1, loss };
    }
  }
  // SGD with momentum (used for fine-tuning: updates scale with the gradient, so
  // weights a task barely touches stay put — keeping task vectors local).
  function* sgdGen(layout, P, X, Y, n, steps, lr, momentum, clip) {
    const v = new Float32Array(P.length);
    const mom = momentum == null ? 0.9 : momentum, cl = clip || 5;
    for (let i = 0; i < steps; i++) {
      const { loss, grad } = bceGrad(layout, P, X, Y, n);
      let nr = 0;
      for (let j = 0; j < grad.length; j++) nr += grad[j] * grad[j];
      nr = Math.sqrt(nr);
      const sc = nr > cl ? cl / nr : 1;
      for (let j = 0; j < P.length; j++) { v[j] = mom * v[j] + grad[j] * sc; P[j] -= lr * v[j]; }
      yield { step: i + 1, loss };
    }
  }

  function trainSync(layout, P, X, Y, n, steps, lr, lossFn, eps) {
    let last = null;
    for (const s of trainGen(layout, P, X, Y, n, steps, lr, lossFn, eps)) last = s;
    return last;
  }

  function predictProb(layout, P, X, n) {
    const { out } = forward(layout, P, X, n, false);
    const p = new Float32Array(n);
    for (let i = 0; i < n; i++) p[i] = sigmoid(out[i]);
    return p;
  }

  // Expected diagonal Fisher for a Bernoulli output: E[p(1-p) (dz/dθ)^2]
  function fisherDiag(layout, P, X, n) {
    const F = new Float32Array(layout.total);
    const one = new Float32Array([1]);
    for (let s = 0; s < n; s++) {
      const d = layout.inDim;
      const x = X.subarray(s * d, s * d + d);
      const { out, acts } = forward(layout, P, x, 1, true);
      const p = sigmoid(out[0]);
      const g = backward(layout, P, acts, one, 1);
      const w = p * (1 - p);
      for (let i = 0; i < F.length; i++) F[i] += w * g[i] * g[i];
    }
    for (let i = 0; i < F.length; i++) F[i] /= n;
    return F;
  }

  // Gram matrices of (bias-augmented) layer inputs, one per layer — used by RegMean.
  function layerGrams(layout, P, X, n) {
    const { acts } = forward(layout, P, X, n, true);
    const grams = [];
    for (let l = 0; l < layout.nLayers; l++) {
      const d = layout.sizes[l] + 1;
      const G = new Float64Array(d * d);
      const a = acts[l], inn = d - 1;
      const row = new Float64Array(d);
      for (let s = 0; s < n; s++) {
        for (let k = 0; k < inn; k++) row[k] = a[s * inn + k];
        row[inn] = 1;
        for (let i = 0; i < d; i++) {
          const ri = row[i];
          for (let j = 0; j < d; j++) G[i * d + j] += ri * row[j];
        }
      }
      for (let i = 0; i < G.length; i++) G[i] /= n;
      grams.push(G);
    }
    return grams;
  }

  Object.assign(MM, {
    mulberry32, gauss, hashStr, makeLayout, makeEncoder, encode, initParams, forward, backward, sigmoid,
    bceGrad, softmaxGrad, softmaxRows, makeAdam, trainGen, sgdGen, trainSync, predictProb,
    fisherDiag, layerGrams,
  });
})();
