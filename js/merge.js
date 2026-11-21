/* ModelMixer — merge engine. Implements weight-space, task-vector, data-informed,
 * layer-surgery, output-space and distillation mixing on the MLP experts, plus
 * Git Re-Basin weight matching (Hungarian algorithm) for neuron alignment. */
(function () {
  const MM = (window.MM = window.MM || {});

  // ---------- small vector helpers ----------
  const sub = (a, b) => { const r = new Float32Array(a.length); for (let i = 0; i < a.length; i++) r[i] = a[i] - b[i]; return r; };
  const zerosLike = (a) => new Float32Array(a.length);
  function dot(a, b, o, n) { let s = 0; for (let i = o; i < o + n; i++) s += a[i] * b[i]; return s; }
  function norm(a, o, n) { return Math.sqrt(dot(a, a, o, n)); }
  function weightsFor(p, K) {
    const w = (p.weights || []).slice(0, K).map(Number);
    while (w.length < K) w.push(1);
    return w;
  }
  function normalized(w) {
    const s = w.reduce((a, b) => a + b, 0);
    return s > 1e-9 ? w.map((x) => x / s) : w.map(() => 1 / w.length);
  }
  // magnitude threshold that keeps the top `frac` of |x| in a slice
  function topThreshold(x, o, n, frac) {
    if (frac >= 1) return 0;
    if (frac <= 0) return Infinity;
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = Math.abs(x[o + i]);
    a.sort();
    const keep = Math.max(1, Math.round(frac * n));
    return a[n - keep];
  }

  function taskVectors(ctx) {
    if (!ctx.base) return null;
    if (!ctx.cache.taus) ctx.cache.taus = ctx.experts.map((e) => sub(e, ctx.base));
    return ctx.cache.taus;
  }

  function weightResult(ctx, P, extra) {
    return Object.assign({
      kind: 'weights', params: P,
      predict: (X, n) => MM.predictProb(ctx.layout, P, X, n),
      cost: 1, storage: 1, info: [], warnings: [],
    }, extra || {});
  }
  const pct = (x) => `${(x * 100).toFixed(1)}%`;

  // ---------- Hungarian algorithm (min-cost assignment) ----------
  function hungarian(cost, n) {
    const INF = 1e18;
    const u = new Float64Array(n + 1), v = new Float64Array(n + 1);
    const p = new Int32Array(n + 1), way = new Int32Array(n + 1);
    for (let i = 1; i <= n; i++) {
      p[0] = i;
      let j0 = 0;
      const minv = new Float64Array(n + 1).fill(INF);
      const used = new Uint8Array(n + 1);
      do {
        used[j0] = 1;
        const i0 = p[j0];
        let delta = INF, j1 = 0;
        for (let j = 1; j <= n; j++) {
          if (used[j]) continue;
          const cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
        for (let j = 0; j <= n; j++) {
          if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
        }
        j0 = j1;
      } while (p[j0] !== 0);
      do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    const ans = new Int32Array(n);
    for (let j = 1; j <= n; j++) ans[p[j] - 1] = j - 1;
    return ans;
  }

  const identity = (n) => Int32Array.from({ length: n }, (_, i) => i);

  function applyPerms(layout, P, perms) {
    const L = layout.nLayers, out = new Float32Array(P.length);
    for (let l = 0; l < L; l++) {
      const Wt = layout.tensors[2 * l], bt = layout.tensors[2 * l + 1];
      const rp = l < L - 1 ? perms[l] : identity(Wt.rows);
      const cp = l > 0 ? perms[l - 1] : identity(Wt.cols);
      for (let i = 0; i < Wt.rows; i++) {
        for (let k = 0; k < Wt.cols; k++) out[Wt.off + i * Wt.cols + k] = P[Wt.off + rp[i] * Wt.cols + cp[k]];
        out[bt.off + i] = P[bt.off + rp[i]];
      }
    }
    return out;
  }

  // Git Re-Basin "weight matching": permute hidden units of B to best match A.
  function weightMatch(layout, A, B, rounds) {
    const L = layout.nLayers, S = layout.sizes;
    const perms = [];
    for (let l = 0; l < L - 1; l++) perms.push(identity(S[l + 1]));
    for (let r = 0; r < (rounds || 8); r++) {
      let changed = false;
      for (let l = 0; l < L - 1; l++) {
        const n = S[l + 1];
        const W = layout.tensors[2 * l], b = layout.tensors[2 * l + 1], Wn = layout.tensors[2 * l + 2];
        const cp = l > 0 ? perms[l - 1] : identity(W.cols);
        const np = l + 1 < L - 1 ? perms[l + 1] : identity(Wn.rows);
        const C = new Float64Array(n * n);
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            let s = A[b.off + i] * B[b.off + j];
            for (let k = 0; k < W.cols; k++) s += A[W.off + i * W.cols + k] * B[W.off + j * W.cols + cp[k]];
            for (let m = 0; m < Wn.rows; m++) s += A[Wn.off + m * Wn.cols + i] * B[Wn.off + np[m] * Wn.cols + j];
            C[i * n + j] = -s;
          }
        }
        const np2 = hungarian(C, n);
        for (let i = 0; i < n; i++) if (np2[i] !== perms[l][i]) { changed = true; break; }
        perms[l] = np2;
      }
      if (!changed) break;
    }
    let moved = 0, total = 0;
    perms.forEach((p) => p.forEach((v, i) => { total++; if (v !== i) moved++; }));
    return { aligned: applyPerms(layout, B, perms), perms, movedFrac: moved / total };
  }

  function alignExperts(layout, base, experts) {
    const anchor = base || experts[0];
    const out = [], moved = [];
    experts.forEach((e, i) => {
      if (!base && i === 0) { out.push(e); moved.push(0); return; }
      const r = weightMatch(layout, anchor, e);
      out.push(r.aligned); moved.push(r.movedFrac);
    });
    return { experts: out, moved };
  }

  // ---------- Router for Mixture-of-Experts ----------
  function getRouter(ctx) {
    if (ctx.cache.router) return ctx.cache.router;
    const K = ctx.K;
    const layout = MM.makeLayout([ctx.layout.inDim, 16, K]); // router reads raw inputs (2 coords or 64 pixels)
    const P = MM.initParams(layout, ctx.seed + 4242);
    const all = MM.concat(ctx.tasks.map((t) => t.train));
    const labels = new Int32Array(all.n);
    let o = 0;
    ctx.tasks.forEach((t, k) => { for (let i = 0; i < t.train.n; i++) labels[o++] = k; });
    const last = MM.trainSync(layout, P, all.X, labels, all.n, 250, 0.02, MM.softmaxGrad);
    const gates = (X, n) => MM.softmaxRows(MM.forward(layout, P, X, n, false).out, n, K);
    const g = gates(all.X, all.n);
    let correct = 0;
    for (let s = 0; s < all.n; s++) {
      let best = 0;
      for (let k = 1; k < K; k++) if (g[s * K + k] > g[s * K + best]) best = k;
      if (best === labels[s]) correct++;
    }
    ctx.cache.router = { gates, acc: correct / all.n, loss: last.loss, params: P.length };
    return ctx.cache.router;
  }

  function expertProbs(ctx, X, n) { return ctx.experts.map((e) => MM.predictProb(ctx.layout, e, X, n)); }

  function ensemblePredictor(ctx, w) {
    const wn = normalized(w);
    return (X, n) => {
      const ps = expertProbs(ctx, X, n), r = new Float32Array(n);
      for (let s = 0; s < n; s++) { let v = 0; for (let i = 0; i < ps.length; i++) v += wn[i] * ps[i][s]; r[s] = v; }
      return r;
    };
  }

  function moePredictor(ctx, mode) {
    const router = getRouter(ctx), K = ctx.K;
    return (X, n) => {
      const g = router.gates(X, n), ps = expertProbs(ctx, X, n), r = new Float32Array(n);
      for (let s = 0; s < n; s++) {
        if (mode === 'top1') {
          let best = 0;
          for (let k = 1; k < K; k++) if (g[s * K + k] > g[s * K + best]) best = k;
          r[s] = ps[best][s];
        } else {
          let v = 0;
          for (let k = 0; k < K; k++) v += g[s * K + k] * ps[k][s];
          r[s] = v;
        }
      }
      return r;
    };
  }

  // ---------- Methods ----------
  const IMPL = {
    linear(ctx, p) {
      if (p._lw) {
        const [A, B] = ctx.experts, P = zerosLike(A);
        ctx.layout.tensors.forEach((T) => {
          const v = +p.grad[T.layer];
          for (let i = T.off; i < T.off + T.size; i++) P[i] = (1 - v) * A[i] + v * B[i];
        });
        return weightResult(ctx, P, { info: p.grad.map((v, l) => ({ label: `${MM.LAYER_NAMES[l]}: A/B`, value: `${Math.round((1 - v) * 100)}/${Math.round(v * 100)}` })) });
      }
      const w = normalized(weightsFor(p, ctx.K));
      const P = zerosLike(ctx.experts[0]);
      ctx.experts.forEach((e, i) => { for (let j = 0; j < P.length; j++) P[j] += w[i] * e[j]; });
      return weightResult(ctx, P, { info: w.map((x, i) => ({ label: `Weight ${'ABC'[i]}`, value: x.toFixed(2) })) });
    },

    slerp(ctx, p) {
      const [A, B] = ctx.experts, P = zerosLike(A);
      const angles = [];
      for (const T of ctx.layout.tensors) {
        const t = p._lw ? +p.grad[T.layer] : +p.t;
        const na = norm(A, T.off, T.size), nb = norm(B, T.off, T.size);
        let c = na * nb > 1e-12 ? dot(A, B, T.off, T.size) / (na * nb) : 1;
        c = Math.max(-1, Math.min(1, c));
        const th = Math.acos(c);
        angles.push({ label: `∠ ${T.name}`, value: `${((th * 180) / Math.PI).toFixed(1)}°` });
        let s0 = 1 - t, s1 = t;
        if (Math.abs(c) < 0.9995) { s0 = Math.sin((1 - t) * th) / Math.sin(th); s1 = Math.sin(t * th) / Math.sin(th); }
        for (let i = T.off; i < T.off + T.size; i++) P[i] = s0 * A[i] + s1 * B[i];
      }
      const warnings = ctx.K > 2 ? ['SLERP interpolates exactly two models — Expert C is ignored.'] : [];
      const info = angles.filter((_, i) => i % 2 === 0);
      if (p._lw) info.push({ label: 't per layer', value: p.grad.map((v) => (+v).toFixed(2)).join(' / ') });
      return weightResult(ctx, P, { info, warnings });
    },

    model_stock(ctx) {
      const taus = taskVectors(ctx), K = ctx.K, P = new Float32Array(ctx.base);
      const info = [];
      for (const T of ctx.layout.tensors) {
        let cs = 0, cnt = 0;
        for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
          const d = norm(taus[i], T.off, T.size) * norm(taus[j], T.off, T.size);
          cs += d > 1e-12 ? dot(taus[i], taus[j], T.off, T.size) / d : 0; cnt++;
        }
        const c = Math.max(0, cs / cnt);
        const t = (K * c) / (1 + (K - 1) * c);
        for (let i = T.off; i < T.off + T.size; i++) {
          let avg = 0;
          for (let k = 0; k < K; k++) avg += ctx.experts[k][i];
          avg /= K;
          P[i] = t * avg + (1 - t) * ctx.base[i];
        }
        if (T.kind === 'w') info.push({ label: `t(${T.name}) · cos ${c.toFixed(2)}`, value: t.toFixed(2) });
      }
      return weightResult(ctx, P, { info });
    },

    task_arithmetic(ctx, p) {
      const taus = taskVectors(ctx), w = weightsFor(p, ctx.K), lam = +p.lambda;
      const P = new Float32Array(ctx.base);
      taus.forEach((t, i) => { for (let j = 0; j < P.length; j++) P[j] += lam * w[i] * t[j]; });
      return weightResult(ctx, P, { info: [{ label: 'Scaling λ', value: lam.toFixed(2) }] });
    },

    ties(ctx, p) {
      const taus = taskVectors(ctx), K = ctx.K, dens = +p.density, lam = +p.lambda;
      const P = new Float32Array(ctx.base);
      let conflicts = 0, overlaps = 0, kept = 0, total = 0;
      for (const T of ctx.layout.tensors) {
        const thr = taus.map((t) => topThreshold(t, T.off, T.size, dens));
        for (let j = T.off; j < T.off + T.size; j++) {
          let sum = 0, nz = 0, pos = 0, neg = 0;
          const v = [];
          for (let i = 0; i < K; i++) {
            const x = Math.abs(taus[i][j]) >= thr[i] ? taus[i][j] : 0;
            v.push(x); sum += x;
            if (x !== 0) { nz++; kept++; if (x > 0) pos++; else neg++; }
          }
          total += K;
          if (nz > 1) { overlaps++; if (pos && neg) conflicts++; }
          const sg = Math.sign(sum);
          let acc = 0, cnt = 0;
          for (let i = 0; i < K; i++) if (v[i] !== 0 && Math.sign(v[i]) === sg) { acc += v[i]; cnt++; }
          P[j] += lam * (cnt ? acc / cnt : 0);
        }
      }
      return weightResult(ctx, P, {
        info: [
          { label: 'Task-vector entries kept', value: pct(kept / total) },
          { label: 'Overlapping entries', value: String(overlaps) },
          { label: 'Sign conflicts resolved', value: overlaps ? pct(conflicts / overlaps) : '—' },
        ],
      });
    },

    dare(ctx, p) {
      const taus = taskVectors(ctx), K = ctx.K, drop = +p.drop, lam = +p.lambda;
      const w = weightsFor(p, K);
      const rng = MM.mulberry32(1000 + (+p.seed || 0));
      const keep = 1 - drop, scale = keep > 0 ? 1 / keep : 0;
      const sparse = taus.map((t) => { const r = zerosLike(t); for (let j = 0; j < t.length; j++) if (rng() < keep) r[j] = t[j] * scale; return r; });
      const P = new Float32Array(ctx.base);
      let conflicts = 0, overlaps = 0;
      for (let j = 0; j < P.length; j++) {
        if (p.mode === 'ties') {
          let sum = 0, pos = 0, neg = 0;
          for (let i = 0; i < K; i++) { const x = w[i] * sparse[i][j]; sum += x; if (x > 0) pos++; else if (x < 0) neg++; }
          if (pos + neg > 1) { overlaps++; if (pos && neg) conflicts++; }
          const sg = Math.sign(sum);
          let acc = 0, cnt = 0;
          for (let i = 0; i < K; i++) { const x = w[i] * sparse[i][j]; if (x !== 0 && Math.sign(x) === sg) { acc += x; cnt++; } }
          P[j] += lam * (cnt ? acc / cnt : 0);
        } else {
          for (let i = 0; i < K; i++) P[j] += lam * w[i] * sparse[i][j];
        }
      }
      const info = [
        { label: 'Dropped per task vector', value: pct(drop) },
        { label: 'Survivor rescale', value: `×${scale.toFixed(2)}` },
      ];
      if (p.mode === 'ties') info.push({ label: 'Sign conflicts resolved', value: overlaps ? pct(conflicts / overlaps) : '—' });
      return weightResult(ctx, P, { info });
    },

    breadcrumbs(ctx, p) {
      const taus = taskVectors(ctx), K = ctx.K, dens = +p.density, gamma = +p.gamma, lam = +p.lambda;
      const P = new Float32Array(ctx.base);
      let kept = 0, total = 0;
      for (const T of ctx.layout.tensors) {
        const hi = taus.map((t) => topThreshold(t, T.off, T.size, gamma));
        const lo = taus.map((t) => topThreshold(t, T.off, T.size, Math.min(1, gamma + dens)));
        for (let j = T.off; j < T.off + T.size; j++) {
          for (let i = 0; i < K; i++) {
            const a = Math.abs(taus[i][j]);
            total++;
            const keepIt = a >= lo[i] && (gamma <= 0 || a < hi[i]);
            if (keepIt) { kept++; P[j] += lam * taus[i][j]; }
          }
        }
      }
      return weightResult(ctx, P, {
        info: [
          { label: 'Outliers removed (top)', value: pct(gamma) },
          { label: 'Entries kept', value: pct(kept / total) },
        ],
      });
    },

    fisher(ctx, p) {
      const F = getFishers(ctx), K = ctx.K, eps = +p.eps;
      const means = F.map((f) => f.reduce((a, b) => a + b, 0) / f.length);
      const P = zerosLike(ctx.experts[0]);
      const dom = new Array(K).fill(0);
      for (let j = 0; j < P.length; j++) {
        let num = 0, den = 0, best = 0, bv = -1;
        for (let i = 0; i < K; i++) {
          const f = F[i][j] + eps * means[i] + 1e-12;
          num += f * ctx.experts[i][j]; den += f;
          if (f > bv) { bv = f; best = i; }
        }
        dom[best]++;
        P[j] = num / den;
      }
      return weightResult(ctx, P, {
        info: dom.map((d, i) => ({ label: `Params led by ${'ABC'[i]}`, value: pct(d / P.length) })),
      });
    },

    regmean(ctx, p) {
      const grams = getGrams(ctx), K = ctx.K, alpha = +p.alpha, L = ctx.layout.nLayers;
      const P = zerosLike(ctx.experts[0]);
      for (let l = 0; l < L; l++) {
        const Wt = ctx.layout.tensors[2 * l], bt = ctx.layout.tensors[2 * l + 1];
        const d = Wt.cols + 1, out = Wt.rows;
        const Gsum = new Float64Array(d * d), WG = new Float64Array(out * d);
        for (let i = 0; i < K; i++) {
          const G = grams[i][l], E = ctx.experts[i];
          const Gt = new Float64Array(d * d);
          for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) Gt[a * d + b] = a === b ? G[a * d + b] : alpha * G[a * d + b];
          for (let a = 0; a < d * d; a++) Gsum[a] += Gt[a];
          for (let r = 0; r < out; r++) {
            for (let b = 0; b < d; b++) {
              let s = 0;
              for (let a = 0; a < d; a++) {
                const wa = a < d - 1 ? E[Wt.off + r * Wt.cols + a] : E[bt.off + r];
                s += wa * Gt[a * d + b];
              }
              WG[r * d + b] += s;
            }
          }
        }
        for (let a = 0; a < d; a++) Gsum[a * d + a] += 1e-4;
        const inv = invert(Gsum, d);
        for (let r = 0; r < out; r++) {
          for (let b = 0; b < d; b++) {
            let s = 0;
            for (let a = 0; a < d; a++) s += WG[r * d + a] * inv[a * d + b];
            if (b < d - 1) P[Wt.off + r * Wt.cols + b] = s; else P[bt.off + r] = s;
          }
        }
      }
      return weightResult(ctx, P, { info: [{ label: 'Off-diagonal scale α', value: (+p.alpha).toFixed(2) }, { label: 'Layers solved in closed form', value: String(L) }] });
    },

    stitch(ctx, p) {
      const L = ctx.layout.nLayers, P = zerosLike(ctx.experts[0]);
      const src = [];
      for (let l = 0; l < L; l++) {
        const s = Math.min(ctx.K - 1, +((p.layers || [])[l] ?? Math.min(l, ctx.K - 1)));
        src.push(s);
        for (const T of [ctx.layout.tensors[2 * l], ctx.layout.tensors[2 * l + 1]]) {
          for (let i = T.off; i < T.off + T.size; i++) P[i] = ctx.experts[s][i];
        }
      }
      return weightResult(ctx, P, { info: src.map((s, l) => ({ label: `Layer ${l + 1}`, value: `from ${'ABC'[s]}` })) });
    },

    ensemble(ctx, p) {
      const w = weightsFor(p, ctx.K);
      return {
        kind: 'outputs', predict: ensemblePredictor(ctx, w), cost: ctx.K, storage: ctx.K,
        info: [{ label: 'Forward passes / input', value: String(ctx.K) }], warnings: [],
      };
    },

    moe(ctx, p) {
      const router = getRouter(ctx);
      const mode = p.mode || 'top1';
      return {
        kind: 'outputs', predict: moePredictor(ctx, mode), router,
        cost: (mode === 'top1' ? 1 : ctx.K) + 0.1, storage: ctx.K + 0.05,
        info: [
          { label: 'Router accuracy (which expert?)', value: pct(router.acc) },
          { label: 'Experts run / input', value: mode === 'top1' ? '1' : String(ctx.K) },
        ],
        warnings: router.acc < 0.8 ? ['The router cannot tell these inputs apart — tasks overlap in input space, so routing is close to a coin flip.'] : [],
      };
    },

    distill(ctx, p) {
      const key = `distill:${p.teacher}:${p.steps}`;
      if (!ctx.cache[key]) {
        const teacher = p.teacher === 'ensemble' ? ensemblePredictor(ctx, weightsFor({}, ctx.K)) : moePredictor(ctx, 'soft');
        const pool = MM.concat(ctx.tasks.map((t) => t.train));
        const rng = MM.mulberry32(ctx.seed + 77);
        // transfer set: every task input, half of them jittered (no labels used)
        const n = pool.n, X = new Float32Array(pool.X);
        const d = ctx.layout.inDim;
        for (let i = 0; i < n; i += 2) for (let k = 0; k < d; k++) X[d * i + k] += MM.gauss(rng) * 0.15;
        const Y = teacher(X, n);
        const S = ctx.base ? new Float32Array(ctx.base) : MM.initParams(ctx.layout, ctx.seed + 99);
        const last = MM.trainSync(ctx.layout, S, X, Y, n, +p.steps, 0.02);
        ctx.cache[key] = { S, loss: last ? last.loss : 0, n };
      }
      const r = ctx.cache[key];
      return weightResult(ctx, r.S, {
        info: [
          { label: 'Unlabeled transfer points', value: String(r.n) },
          { label: 'Final distillation loss', value: r.loss.toFixed(3) },
          { label: 'Teacher', value: p.teacher === 'ensemble' ? 'Ensemble' : 'MoE (soft)' },
        ],
      });
    },
  };

  function getFishers(ctx) {
    if (!ctx.cache.fisher) ctx.cache.fisher = ctx.experts.map((e, i) => MM.fisherDiag(ctx.layout, e, ctx.tasks[i].train.X, ctx.tasks[i].train.n));
    return ctx.cache.fisher;
  }
  function getGrams(ctx) {
    if (!ctx.cache.grams) ctx.cache.grams = ctx.experts.map((e, i) => MM.layerGrams(ctx.layout, e, ctx.tasks[i].train.X, ctx.tasks[i].train.n));
    return ctx.cache.grams;
  }

  // Gauss-Jordan inverse (d is small: hidden+1)
  function invert(M, d) {
    const A = new Float64Array(M), I = new Float64Array(d * d);
    for (let i = 0; i < d; i++) I[i * d + i] = 1;
    for (let c = 0; c < d; c++) {
      let piv = c;
      for (let r = c + 1; r < d; r++) if (Math.abs(A[r * d + c]) > Math.abs(A[piv * d + c])) piv = r;
      if (piv !== c) for (let k = 0; k < d; k++) {
        [A[c * d + k], A[piv * d + k]] = [A[piv * d + k], A[c * d + k]];
        [I[c * d + k], I[piv * d + k]] = [I[piv * d + k], I[c * d + k]];
      }
      const pv = A[c * d + c] || 1e-12;
      for (let k = 0; k < d; k++) { A[c * d + k] /= pv; I[c * d + k] /= pv; }
      for (let r = 0; r < d; r++) {
        if (r === c) continue;
        const f = A[r * d + c];
        if (f === 0) continue;
        for (let k = 0; k < d; k++) { A[r * d + k] -= f * A[c * d + k]; I[r * d + k] -= f * I[c * d + k]; }
      }
    }
    return I;
  }

  /** Is a per-layer gradient active for this method / context? */
  function layerwiseOn(meta, ctx, params) {
    const g = meta.gradient;
    return !!(g && params && params.layerwise && Array.isArray(params.grad) && params.grad.length === 3 && !(g.k2 && ctx.K !== 2));
  }

  function run(id, ctx, params) {
    const meta = MM.METHOD_BY_ID[id];
    if (meta.req.base && !ctx.base) return { blocked: 'Needs a shared base model. In this scenario every expert started from its own random init, so there are no task vectors to work with.' };
    if (meta.minModels && ctx.K < meta.minModels) return { blocked: `Needs at least ${meta.minModels} experts.` };
    const lw = layerwiseOn(meta, ctx, params);
    if (lw && meta.gradient.key === 'lambda') {
      // These merges are θ = θ_base + λ·τ_merged, so merge once at λ = 1 and rescale each layer's update.
      const r = IMPL[id](ctx, Object.assign({}, params, { lambda: 1 }));
      if (r.blocked || !r.params) return r;
      const P = r.params, base = ctx.base;
      ctx.layout.tensors.forEach((T) => {
        const lam = +params.grad[T.layer];
        for (let i = T.off; i < T.off + T.size; i++) P[i] = base[i] + lam * (P[i] - base[i]);
      });
      r.info = (r.info || []).filter((x) => !/λ/.test(x.label)).concat([{ label: 'λ per layer', value: params.grad.map((v) => (+v).toFixed(2)).join(' / ') }]);
      return r;
    }
    return IMPL[id](ctx, lw ? Object.assign({}, params, { _lw: true }) : params);
  }

  function evaluate(predict, tasks) {
    return tasks.map((t) => {
      const p = predict(t.test.X, t.test.n);
      let c = 0;
      for (let i = 0; i < t.test.n; i++) if ((p[i] > 0.5 ? 1 : 0) === t.test.Y[i]) c++;
      return c / t.test.n;
    });
  }

  // Task-vector geometry: pairwise cosine + sign-conflict rate among the top-20% entries.
  function geometry(ctx) {
    // Without a base there are no task vectors: compare the raw weights instead
    // (this is where neuron alignment visibly raises the similarity).
    const ref = ctx.base || zerosLike(ctx.experts[0]);
    const taus = ctx.experts.map((e) => sub(e, ref));
    const pairs = [];
    for (let i = 0; i < ctx.K; i++) for (let j = i + 1; j < ctx.K; j++) {
      const c = dot(taus[i], taus[j], 0, ref.length) / (norm(taus[i], 0, ref.length) * norm(taus[j], 0, ref.length) + 1e-12);
      const ti = topThreshold(taus[i], 0, ref.length, 0.2), tj = topThreshold(taus[j], 0, ref.length, 0.2);
      let both = 0, conf = 0;
      for (let k = 0; k < ref.length; k++) {
        if (Math.abs(taus[i][k]) >= ti && Math.abs(taus[j][k]) >= tj) { both++; if (Math.sign(taus[i][k]) !== Math.sign(taus[j][k])) conf++; }
      }
      pairs.push({ i, j, cos: c, conflict: both ? conf / both : 0, overlap: both / (0.2 * ref.length) });
    }
    const norms = taus.map((t) => norm(t, 0, t.length));
    return { pairs, norms, hasBase: !!ctx.base };
  }

  Object.assign(MM, { runMethod: run, layerwiseOn, evaluate, alignExperts, weightMatch, geometry, hungarian, getRouter });
})();
