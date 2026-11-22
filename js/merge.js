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

    // Karcher (Fréchet) mean on the unit sphere, per tensor: SLERP generalised to any number of models.
    karcher(ctx, p) {
      const K = ctx.K, E = ctx.experts, w = normalized(weightsFor(p, K)), P = zerosLike(E[0]);
      const MAX_IT = 20;
      let itSum = 0, angSum = 0, nT = 0;
      for (const T of ctx.layout.tensors) {
        const o = T.off, n = T.size;
        const norms = E.map((e) => norm(e, o, n));
        const U = E.map((e, i) => { const u = new Float64Array(n), s = norms[i] > 1e-12 ? 1 / norms[i] : 0; for (let k = 0; k < n; k++) u[k] = e[o + k] * s; return u; });
        const m = new Float64Array(n);
        U.forEach((u, i) => { for (let k = 0; k < n; k++) m[k] += w[i] * u[k]; });
        const unit = (v) => { let s = 0; for (let k = 0; k < n; k++) s += v[k] * v[k]; s = Math.sqrt(s); if (s > 1e-12) for (let k = 0; k < n; k++) v[k] /= s; return s; };
        if (unit(m) <= 1e-12) m.set(U[0]);
        let it = 0;
        for (; it < MAX_IT; it++) {
          // gradient step: average of log maps log_m(u_i), then move along the geodesic (exp map)
          const g = new Float64Array(n);
          for (let i = 0; i < K; i++) {
            let c = 0;
            for (let k = 0; k < n; k++) c += m[k] * U[i][k];
            c = Math.max(-1, Math.min(1, c));
            const th = Math.acos(c);
            if (th < 1e-9) continue;
            const f = (w[i] * th) / Math.sin(th);
            for (let k = 0; k < n; k++) g[k] += f * (U[i][k] - c * m[k]);
          }
          let gn = 0;
          for (let k = 0; k < n; k++) gn += g[k] * g[k];
          gn = Math.sqrt(gn);
          if (gn < 1e-9) break;
          const cg = Math.cos(gn), sg = Math.sin(gn) / gn;
          for (let k = 0; k < n; k++) m[k] = cg * m[k] + sg * g[k];
          unit(m);
        }
        const rad = norms.reduce((a, r, i) => a + w[i] * r, 0);
        for (let k = 0; k < n; k++) P[o + k] = rad * m[k];
        if (T.kind === 'w') {
          let a = 0;
          for (let i = 0; i < K; i++) { let c = 0; for (let k = 0; k < n; k++) c += m[k] * U[i][k]; a += w[i] * Math.acos(Math.max(-1, Math.min(1, c))); }
          angSum += a; itSum += it; nT++;
        }
      }
      return weightResult(ctx, P, {
        info: [
          { label: 'Mean angle to experts', value: `${((angSum / nT) * 180 / Math.PI).toFixed(1)}°` },
          { label: 'Iterations per tensor', value: itSum ? (itSum / nT).toFixed(1) : 'none (start was exact)' },
        ],
      });
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

    // SCE (FuseChat): Select high-variance entries, Calculate energy-based model weights, Erase minority signs.
    sce(ctx, p) {
      const taus = taskVectors(ctx), K = ctx.K, top = +p.topk, lam = +p.lambda;
      const P = new Float32Array(ctx.base);
      let kept = 0, total = 0, erased = 0, nz = 0;
      const etaShow = new Array(K).fill(0);
      let wTensors = 0;
      for (const T of ctx.layout.tensors) {
        const o = T.off, n = T.size;
        const vr = new Float32Array(n);
        for (let k = 0; k < n; k++) {
          let m = 0;
          for (let i = 0; i < K; i++) m += taus[i][o + k];
          m /= K;
          let v = 0;
          for (let i = 0; i < K; i++) v += (taus[i][o + k] - m) ** 2;
          vr[k] = v / K;
        }
        const thr = topThreshold(vr, 0, n, top);
        const sel = (k) => vr[k] > 0 && vr[k] >= thr;
        const en = taus.map((t) => { let s = 0; for (let k = 0; k < n; k++) if (sel(k)) s += t[o + k] * t[o + k]; return s; });
        const es = en.reduce((a, b) => a + b, 0);
        const eta = en.map((e) => (es > 0 ? e / es : 1 / K));
        if (T.kind === 'w') { eta.forEach((e, i) => (etaShow[i] += e)); wTensors++; }
        for (let k = 0; k < n; k++) {
          total++;
          if (!sel(k)) continue;
          kept++;
          let sum = 0;
          for (let i = 0; i < K; i++) sum += eta[i] * taus[i][o + k];
          const sg = Math.sign(sum);
          let acc = 0, ws = 0;
          for (let i = 0; i < K; i++) {
            const x = taus[i][o + k];
            if (x === 0) continue;
            nz++;
            if (Math.sign(x) === sg) { acc += eta[i] * x; ws += eta[i]; } else erased++;
          }
          P[o + k] += lam * (ws > 0 ? acc / ws : 0);
        }
      }
      return weightResult(ctx, P, {
        info: [
          { label: 'Entries selected (high variance)', value: pct(kept / total) },
          { label: 'Minority signs erased', value: nz ? pct(erased / nz) : '—' },
          ...etaShow.map((e, i) => ({ label: `Energy weight ${'ABC'[i]}`, value: (e / Math.max(1, wTensors)).toFixed(2) })),
        ],
      });
    },

    // AdaMerging: learn task-wise or layer-wise coefficients λ by minimising prediction entropy on
    // unlabeled inputs (test-time adaptation; no labels). θ = θ_base + Σ_k λ_k(,l) · τ_k
    adamerging(ctx, p) {
      const mode = p.mode === 'task' ? 'task' : 'layer';
      const objective = ['ensemble', 'anchored'].includes(p.objective) ? p.objective : 'entropy';
      const key = `ada:${mode}:${objective}`;
      const taus = taskVectors(ctx), K = ctx.K, layout = ctx.layout, base = ctx.base;
      const nL = mode === 'layer' ? layout.nLayers : 1;
      const layerOf = new Int8Array(base.length);
      if (nL > 1) for (const T of layout.tensors) layerOf.fill(T.layer, T.off, T.off + T.size);
      const build = (lm, P) => {
        for (let j = 0; j < base.length; j++) {
          let v = base[j];
          const l = nL > 1 ? layerOf[j] : 0;
          for (let k = 0; k < K; k++) v += lm[k * nL + l] * taus[k][j];
          P[j] = v;
        }
        return P;
      };
      if (!ctx.cache[key]) {
        // unlabeled pool: an even sample of every task's inputs (labels never used)
        const all = MM.concat(ctx.tasks.map((t) => t.train));
        const d = all.d || 2, m = Math.min(240, all.n), stride = all.n / m;
        const X = new Float32Array(m * d);
        for (let i = 0; i < m; i++) { const s = Math.floor(i * stride); X.set(all.X.subarray(s * d, s * d + d), i * d); }
        const P = new Float32Array(base.length);
        // objectives (picked on 12 toy bundles, see IMPROVEMENTS.md): 'entropy' (the paper) · 'anchored'
        // (entropy + a pull back toward the initial λ so coefficients can't run away; the default) ·
        // 'ensemble' (cross-entropy to the experts' average: steadiest on the plane toys, weaker on images).
        // Most-confident-expert labels and entropy-only-where-experts-agree were measured and dropped.
        let target = null;
        if (objective === 'ensemble') {
          const probs = ctx.experts.map((e) => MM.predictProb(layout, e, X, m));
          target = new Float32Array(m);
          for (let i = 0; i < m; i++) { let s2 = 0; for (let k = 0; k < K; k++) s2 += probs[k][i]; target[i] = s2 / K; }
        }
        const entropy = (lm) => {
          const out = MM.forward(layout, build(lm, P), X, m, false).out;
          let e = 0;
          for (let i = 0; i < m; i++) {
            const q = 1 / (1 + Math.exp(-out[i])), y = target ? target[i] : q;
            e -= y * Math.log(q + 1e-9) + (1 - y) * Math.log(1 - q + 1e-9);
          }
          e /= m;
          if (objective === 'anchored') { let r2 = 0; for (let i = 0; i < lm.length; i++) r2 += (lm[i] - 0.3) ** 2; e += 0.05 * r2 / lm.length; }
          return e;
        };
        const lam = new Float64Array(K * nL).fill(0.3); // the paper's initial coefficient
        const mA = new Float64Array(lam.length), vA = new Float64Array(lam.length);
        const lr = 0.05, h = 1e-2, STEPS = 40;
        const e0 = entropy(lam);
        const trace = [e0];
        for (let s = 1; s <= STEPS; s++) {
          const g = new Float64Array(lam.length);
          for (let i = 0; i < lam.length; i++) {
            const keep = lam[i];
            lam[i] = keep + h; const ep = entropy(lam);
            lam[i] = keep - h; const em = entropy(lam);
            lam[i] = keep;
            g[i] = (ep - em) / (2 * h);
          }
          for (let i = 0; i < lam.length; i++) {
            mA[i] = 0.9 * mA[i] + 0.1 * g[i];
            vA[i] = 0.999 * vA[i] + 0.001 * g[i] * g[i];
            const mh = mA[i] / (1 - 0.9 ** s), vh = vA[i] / (1 - 0.999 ** s);
            lam[i] = Math.min(1.5, Math.max(0, lam[i] - (lr * mh) / (Math.sqrt(vh) + 1e-8)));
          }
          if (s % 5 === 0) trace.push(entropy(lam));
        }
        ctx.cache[key] = { lam: Array.from(lam), e0, e1: entropy(lam), trace, n: m, nL };
      }
      const c = ctx.cache[key];
      const P = build(c.lam, new Float32Array(base.length));
      const fmt = (v) => (+v).toFixed(2);
      const info = [{ label: { ensemble: 'Loss vs expert average', anchored: 'Entropy + anchor to λ₀', entropy: 'Prediction entropy (unlabeled)' }[objective], value: `${c.e0.toFixed(3)} → ${c.e1.toFixed(3)}` }];
      for (let k = 0; k < K; k++) {
        const row = c.lam.slice(k * c.nL, (k + 1) * c.nL);
        info.push({ label: `λ ${'ABC'[k]}${c.nL > 1 ? ' per layer' : ''}`, value: row.map(fmt).join(' / ') });
      }
      return weightResult(ctx, P, { info, ada: { lam: c.lam, nL: c.nL, K } });
    },

    // Localize-and-Stitch: a sparse mask per task (learned on that task's data, or top-|τ| when dataless),
    // then add the masked task vectors to the base, averaging wherever masks overlap.
    localize_stitch(ctx, p) {
      const taus = taskVectors(ctx), K = ctx.K, dens = +p.density, mode = p.mode === 'dataless' ? 'dataless' : 'data';
      const layout = ctx.layout, base = ctx.base, n = base.length;
      const key = `ls:${mode}:${dens}`;
      if (!ctx.cache[key]) {
        const masks = taus.map((t, i) => {
          const thr0 = topThreshold(t, 0, n, dens);
          const init = new Uint8Array(n);
          for (let j = 0; j < n; j++) init[j] = t[j] !== 0 && Math.abs(t[j]) >= thr0 ? 1 : 0;
          if (mode === 'dataless') return init;
          // learn mask logits s on task i's labelled data: θ = θ_base + σ(s) ⊙ τ_i, BCE loss, Adam on s
          const d = ctx.tasks[i].train, dd = d.d || 2, m = Math.min(200, d.n), stride = d.n / m;
          const X = new Float32Array(m * dd), Y = new Float32Array(m);
          for (let a = 0; a < m; a++) { const sIdx = Math.floor(a * stride); X.set(d.X.subarray(sIdx * dd, sIdx * dd + dd), a * dd); Y[a] = d.Y[sIdx]; }
          const s = new Float32Array(n), P = new Float32Array(n);
          for (let j = 0; j < n; j++) s[j] = init[j] ? 3 : -3;
          const mA = new Float32Array(n), vA = new Float32Array(n), lr = 0.1, STEPS = 30;
          for (let step = 1; step <= STEPS; step++) {
            for (let j = 0; j < n; j++) P[j] = base[j] + t[j] / (1 + Math.exp(-s[j]));
            const { grad } = MM.bceGrad(layout, P, X, Y, m);
            for (let j = 0; j < n; j++) {
              const sg = 1 / (1 + Math.exp(-s[j]));
              const g = grad[j] * t[j] * sg * (1 - sg);
              mA[j] = 0.9 * mA[j] + 0.1 * g;
              vA[j] = 0.999 * vA[j] + 0.001 * g * g;
              s[j] -= (lr * (mA[j] / (1 - 0.9 ** step))) / (Math.sqrt(vA[j] / (1 - 0.999 ** step)) + 1e-8);
            }
          }
          // binarise: keep the top `density` fraction by learned logit (sparsity is enforced here)
          const thr = topThreshold(s.map((x, j) => (t[j] !== 0 ? x + 10 : 0)), 0, n, dens);
          const mask = new Uint8Array(n);
          for (let j = 0; j < n; j++) mask[j] = t[j] !== 0 && s[j] + 10 >= thr ? 1 : 0;
          return mask;
        });
        // how much of each skill survives on its own (base + its masked vector)
        const own = masks.map((mk, i) => {
          const P = new Float32Array(base);
          for (let j = 0; j < n; j++) if (mk[j]) P[j] += taus[i][j];
          return MM.evaluate((X, nn) => MM.predictProb(layout, P, X, nn), [ctx.tasks[i]])[0];
        });
        ctx.cache[key] = { masks, own };
      }
      const { masks, own } = ctx.cache[key];
      const P = new Float32Array(base);
      let overlap = 0, any = 0;
      for (let j = 0; j < n; j++) {
        let cnt = 0, acc = 0;
        for (let i = 0; i < K; i++) if (masks[i][j]) { cnt++; acc += taus[i][j]; }
        if (cnt) { any++; if (cnt > 1) overlap++; P[j] += acc / cnt; }
      }
      return weightResult(ctx, P, {
        info: [
          { label: `Mask density (${mode === 'data' ? 'learned' : 'by |τ|'})`, value: pct(masks[0].reduce((a, b) => a + b, 0) / n) },
          { label: 'Masked entries that overlap', value: any ? pct(overlap / any) : '—' },
          ...own.map((a, i) => ({ label: `Localized ${'ABC'[i]} alone`, value: pct(a) })),
        ],
      });
    },

    // DELLA: magnitude-ranked drop probabilities (MagPrune), rescale, then TIES-style sign election.
    della(ctx, p) {
      const taus = taskVectors(ctx), K = ctx.K, dens = +p.density, eps = +p.epsilon, lam = +p.lambda;
      const rng = MM.mulberry32(2000 + (+p.seed || 0));
      let kept = 0, total = 0;
      const sparse = taus.map((t) => {
        const r = zerosLike(t);
        for (const T of ctx.layout.tensors) {
          const rows = T.kind === 'w' ? T.rows : 1, cols = T.kind === 'w' ? T.cols : T.size;
          for (let row = 0; row < rows; row++) {
            const o = T.off + row * cols;
            const idx = Array.from({ length: cols }, (_, k) => k).sort((a, b) => Math.abs(t[o + a]) - Math.abs(t[o + b]));
            for (let rank = 0; rank < cols; rank++) {
              const k = idx[rank];
              const kp = Math.min(1, Math.max(0.01, dens - eps / 2 + (cols > 1 ? (eps * rank) / (cols - 1) : eps / 2)));
              total++;
              if (rng() < kp) { r[o + k] = t[o + k] / kp; kept++; }
            }
          }
        }
        return r;
      });
      const P = new Float32Array(ctx.base);
      let conflicts = 0, overlaps = 0;
      for (let j = 0; j < P.length; j++) {
        let sum = 0, pos = 0, neg = 0;
        for (let i = 0; i < K; i++) { const x = sparse[i][j]; sum += x; if (x > 0) pos++; else if (x < 0) neg++; }
        if (pos + neg > 1) { overlaps++; if (pos && neg) conflicts++; }
        const sg = Math.sign(sum);
        let acc = 0, cnt = 0;
        for (let i = 0; i < K; i++) { const x = sparse[i][j]; if (x !== 0 && Math.sign(x) === sg) { acc += x; cnt++; } }
        P[j] += lam * (cnt ? acc / cnt : 0);
      }
      return weightResult(ctx, P, {
        info: [
          { label: 'Entries kept (magnitude-weighted)', value: pct(kept / total) },
          { label: 'Sign conflicts resolved', value: overlaps ? pct(conflicts / overlaps) : '—' },
        ],
      });
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
