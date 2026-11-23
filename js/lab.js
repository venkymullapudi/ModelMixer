/* ModelMixer — headless lab core: train a set of experts, build merge contexts,
 * score methods and summarise multi-seed robustness. No DOM access, so the same
 * code drives the UI, the Node test-suite and the robustness runner. */
(function () {
  const root = typeof window !== 'undefined' ? window : globalThis;
  const MM = (root.MM = root.MM || {});

  const BASE_STEPS = 200, LR = 0.01, FT_LR = 0.1;

  /** Generator: yields {type:'start'|'step'|'done', ...} events and returns the trained bundle.
   *  cfg = { experts:[spec], init:'shared'|'independent', hidden, ftSteps, seed } */
  /** Layout and task data for a training config (cheap; no training). */
  function bundleShape(cfg) {
    const { experts: specs, hidden, seed } = cfg;
    const digits = MM.taskKind(specs[0].task) === 'digits';
    const layout = digits ? MM.makeLayout([MM.DIGIT_D, hidden, hidden, 1]) : MM.makeLayout([2, hidden, hidden, 1], MM.makeEncoder(9, MM.EXTENT, 0.5));
    return { layout, tasks: specs.map((e, i) => MM.makeTaskData(e, i, seed)), digits };
  }

  function* trainBundle(cfg) {
    const { experts: specs, init, ftSteps, seed } = cfg;
    const K = specs.length;
    const { layout, tasks, digits } = bundleShape(cfg);
    const shared = init === 'shared';
    const total = shared ? BASE_STEPS + K * ftSteps : K * (BASE_STEPS + ftSteps);
    const losses = [];
    const live = { layout, tasks, base: null, experts: [], losses };
    yield { type: 'start', live, total };

    function* drive(gen, which, P) {
      let i = 0;
      for (const s of gen) {
        if (i++ % 3 === 0) losses.push(s.loss);
        yield { type: 'step', which, P, loss: s.loss };
      }
      yield { type: 'done', which, P };
    }

    let base = null;
    const experts = [];
    if (shared) {
      base = MM.initParams(layout, seed);
      live.base = base;
      const pre = MM.concat([digits ? MM.genericDigitSet(seed, 900) : MM.genericSet(seed, 900), MM.pretrainSet(tasks, 0.05, seed)]);
      yield* drive(MM.trainGen(layout, base, pre.X, pre.Y, pre.n, BASE_STEPS, LR), 'base', base);
      for (let i = 0; i < K; i++) {
        const e = new Float32Array(base);
        live.experts[i] = e;
        const d = tasks[i].train;
        yield* drive(MM.sgdGen(layout, e, d.X, d.Y, d.n, ftSteps, FT_LR, 0.9), i, e);
        experts.push(e);
      }
    } else {
      for (let i = 0; i < K; i++) {
        const e = MM.initParams(layout, seed * 13 + 101 * (i + 1));
        live.experts[i] = e;
        const d = tasks[i].train;
        yield* drive(MM.trainGen(layout, e, d.X, d.Y, d.n, BASE_STEPS + ftSteps, LR), i, e);
        experts.push(e);
      }
    }
    const pred = (P) => (X, n) => MM.predictProb(layout, P, X, n);
    return {
      layout, base, experts, tasks, K, losses, seed,
      expertAccs: experts.map((e) => MM.evaluate(pred(e), tasks)),
      baseAccs: base ? MM.evaluate(pred(base), tasks) : null,
      ctx: {},
    };
  }

  function trainBundleSync(cfg) {
    const it = trainBundle(cfg);
    for (;;) { const r = it.next(); if (r.done) return r.value; }
  }

  /** Merge context (raw or neuron-aligned), cached on the bundle. */
  function bundleCtx(T, align) {
    const key = align ? 'aligned' : 'raw';
    if (!T.ctx[key]) {
      let experts = T.experts, moved = null;
      if (align) { const r = MM.alignExperts(T.layout, T.base, T.experts); experts = r.experts; moved = r.moved; }
      T.ctx[key] = { layout: T.layout, base: T.base, experts, K: experts.length, tasks: T.tasks, cache: {}, seed: T.seed, moved };
    }
    return T.ctx[key];
  }

  /** Run one method and attach per-task accuracy, average and skill retention. */
  function scoreMethod(T, ctx, id, params) {
    const r = MM.runMethod(id, ctx, params);
    if (r.blocked) return r;
    r.accs = MM.evaluate(r.predict, ctx.tasks);
    r.avg = r.accs.reduce((a, b) => a + b, 0) / r.accs.length;
    r.retention = r.accs.reduce((a, acc, i) => a + acc / Math.max(1e-6, T.expertAccs[i][i]), 0) / r.accs.length;
    return r;
  }

  /** Summarise per-seed results: runs[s] = { [methodId]: {avg, retention, accs} | {blocked} } */
  function summarizeSeeds(runs, ids) {
    const S = runs.length;
    const winners = runs.map((run) => {
      let best = -1, who = [];
      for (const id of ids) {
        const r = run[id];
        if (!r || r.blocked) continue;
        if (r.avg > best + 1e-9) { best = r.avg; who = [id]; } else if (Math.abs(r.avg - best) <= 1e-9) who.push(id);
      }
      return who;
    });
    return ids.map((id) => {
      const vals = runs.map((run) => run[id]).filter((r) => r && !r.blocked);
      if (!vals.length) return { id, blocked: runs[0][id] && runs[0][id].blocked };
      const avgs = vals.map((r) => r.avg);
      const mean = avgs.reduce((a, b) => a + b, 0) / avgs.length;
      const sd = Math.sqrt(avgs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, avgs.length - 1));
      const ret = vals.reduce((a, r) => a + r.retention, 0) / vals.length;
      const worstTask = Math.min(...vals.map((r) => Math.min(...r.accs)));
      const wins = winners.reduce((a, w) => a + (w.includes(id) ? 1 / w.length : 0), 0);
      return { id, avgs, mean, sd, min: Math.min(...avgs), max: Math.max(...avgs), retention: ret, worstTask, wins, seeds: S };
    });
  }

  /** Short human label for a recipe, e.g. "TIES · k 0.20 · λ 1.00 · aligned". */
  const ABBR = { density: 'k', lambda: 'λ', drop: 'p', gamma: 'γ', t: 't', eps: 'ε', alpha: 'α', seed: 'seed' };
  /**
   * What differs between two recipes A and B ({ method, params, align }), in the order a practitioner reads it:
   * [{ what, a, b, apply }]. Knobs are compared only when the method is the same (different methods have different
   * knobs). Values are formatted like the controls; knobs replaced by per-layer values are skipped.
   * apply(r) returns a copy of recipe r with just this one change taken from B (used for attribution).
   */
  function recipeDiff(A, B, K) {
    const out = [];
    const clone = (r) => ({ method: r.method, align: !!r.align, params: JSON.parse(JSON.stringify(r.params)) });
    const setP = (fn) => (r) => { const c = clone(r); fn(c.params); return c; };
    const ma = MM.METHOD_BY_ID[A.method], mb = MM.METHOD_BY_ID[B.method];
    if (A.method !== B.method) out.push({ what: 'Method', a: ma.name, b: mb.name, apply: (r) => ({ method: B.method, align: !!r.align, params: JSON.parse(JSON.stringify(B.params)) }) });
    if (!!A.align !== !!B.align) out.push({ what: 'Neuron alignment', a: A.align ? 'on' : 'off', b: B.align ? 'on' : 'off', apply: (r) => Object.assign(clone(r), { align: !!B.align }) });
    if (A.method !== B.method) return out;
    const m = ma, pa = A.params, pb = B.params;
    const g = m.gradient && !(m.gradient.k2 && K !== 2) ? m.gradient : null;
    const lwA = !!(g && pa.layerwise), lwB = !!(g && pb.layerwise);
    const name = (d) => d.label.replace(/\s*\(.*\)\s*$/, '').trim();
    const num = (v, step) => (step >= 1 ? String(Math.round(v)) : (+v).toFixed(2));
    for (const d of m.params) {
      const perLayer = g && (d.key === g.key || (g.key === 'balance' && d.type === 'weights'));
      if (perLayer && (lwA || lwB)) continue; // shown as per-layer values below
      if (d.type === 'range') {
        if (Math.abs(+pa[d.key] - +pb[d.key]) > 1e-9) out.push({ what: name(d), a: num(pa[d.key], d.step), b: num(pb[d.key], d.step), apply: setP((p) => { p[d.key] = pb[d.key]; }) });
      } else if (d.type === 'select') {
        const lbl = (v) => { const o = d.options.find(([k]) => k === String(v)); return o ? o[1] : String(v); };
        if (String(pa[d.key]) !== String(pb[d.key])) out.push({ what: name(d), a: lbl(pa[d.key]), b: lbl(pb[d.key]), apply: setP((p) => { p[d.key] = pb[d.key]; }) });
      } else if (d.type === 'weights') {
        const wa = pa.weights.slice(0, K), wb = pb.weights.slice(0, K);
        if (wa.some((x, i) => Math.abs(x - wb[i]) > 1e-9)) out.push({ what: 'Expert weights', a: wa.map((x) => (+x).toFixed(2)).join(' / '), b: wb.map((x) => (+x).toFixed(2)).join(' / '), apply: setP((p) => { p.weights = pb.weights.slice(); }) });
      } else if (d.type === 'layers') {
        (MM.LAYER_NAMES || ['Hidden 1', 'Hidden 2', 'Output']).forEach((ln, l) => {
          if (+pa.layers[l] !== +pb.layers[l]) out.push({ what: `${ln} from`, a: 'ABC'[pa.layers[l]], b: 'ABC'[pb.layers[l]], apply: setP((p) => { p.layers[l] = pb.layers[l]; }) });
        });
      }
    }
    if (g) {
      const gl = g.label;
      const gv = (p) => p.grad.map((v) => (+v).toFixed(2)).join(' → ');
      // the per-layer change also carries B's scalar knob, which the per-layer values replace
      const takeLayers = setP((p) => {
        p.layerwise = pb.layerwise; p.grad = pb.grad.slice();
        if (g.key in pb) p[g.key] = pb[g.key];
        if (g.key === 'balance' && pb.weights) p.weights = pb.weights.slice();
      });
      if (lwA !== lwB) out.push({ what: `${gl} per layer`, a: lwA ? gv(pa) : 'off', b: lwB ? gv(pb) : 'off', apply: takeLayers });
      else if (lwA && pa.grad.some((v, i) => Math.abs(v - pb.grad[i]) > 1e-9)) out.push({ what: `${gl} per layer`, a: gv(pa), b: gv(pb), apply: takeLayers });
    }
    return out;
  }

  /**
   * Which change mattered? Starting from A, apply each change in recipeDiff(A, B) on its own and score it.
   * score(recipe) -> average accuracy (or null if the recipe can't run). Returns
   * { a, b, total, parts: [{ what, a, b, delta }], interaction } where interaction = total − Σ deltas
   * (what the changes do together beyond their separate effects). null if fewer than 2 changes.
   */
  function attributeChanges(A, B, K, score) {
    const diff = recipeDiff(A, B, K);
    if (diff.length < 2) return null;
    const sa = score(A), sb = score(B);
    if (sa == null || sb == null) return null;
    const parts = diff.map((x) => {
      const s1 = score(x.apply(A));
      return { what: x.what, a: x.a, b: x.b, delta: s1 == null ? null : s1 - sa };
    });
    const known = parts.filter((p) => p.delta != null).reduce((t, p) => t + p.delta, 0);
    return { a: sa, b: sb, total: sb - sa, parts, interaction: (sb - sa) - known };
  }

  /**
   * Summarise attributeChanges results from several seeds (same A and B, so the same parts in the same order).
   * Per change: mean / sd of the effect, and how many seeds move it up, down or barely (|Δ| < flat, default 0.2 pts).
   * sign: 'up' | 'down' when every seed agrees, 'flat' when every seed is flat, else 'mixed'.
   */
  function summarizeAttributions(list, flat) {
    flat = flat == null ? 0.002 : flat;
    list = list.filter(Boolean);
    const n = list.length;
    const ms = (xs) => { const m = xs.reduce((a, b) => a + b, 0) / xs.length; return { mean: m, sd: xs.length > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)) : 0 }; };
    if (!n) return null;
    const parts = list[0].parts.map((p0, i) => {
      const ds = list.map((at) => at.parts[i].delta).filter((d) => d != null);
      if (!ds.length) return { what: p0.what, a: p0.a, b: p0.b, n: 0, mean: null, sd: null, up: 0, down: 0, flat: 0, sign: 'none' };
      const up = ds.filter((d) => d > flat).length, down = ds.filter((d) => d < -flat).length, fl = ds.length - up - down;
      const sign = up === ds.length ? 'up' : down === ds.length ? 'down' : fl === ds.length ? 'flat' : 'mixed';
      return Object.assign({ what: p0.what, a: p0.a, b: p0.b, n: ds.length, up, down, flat: fl, sign }, ms(ds));
    });
    return { n, parts, total: ms(list.map((at) => at.total)), interaction: ms(list.map((at) => at.interaction)) };
  }

  function recipeLabel(id, params, align, K) {
    const m = MM.METHOD_BY_ID[id];
    const parts = [m.short];
    for (const d of m.params) {
      const v = params[d.key];
      if (d.type === 'range') {
        if (d.key === 'seed' && +v === d.default) continue;
        parts.push(`${ABBR[d.key] || d.key} ${d.step >= 1 ? v : (+v).toFixed(2)}`);
      } else if (d.type === 'select') {
        const o = d.options.find(([k]) => k === String(v));
        parts.push(d.key === 'steps' ? `${v} steps` : (o ? o[1] : String(v)).replace(/ \(.*\)$/, ''));
      } else if (d.type === 'weights') {
        const w = params.weights.slice(0, K);
        if (w.some((x) => x !== w[0])) parts.push(`w ${w.map((x) => (+x).toFixed(2)).join('/')}`);
      } else if (d.type === 'layers') {
        parts.push(params.layers.map((i) => 'ABC'[i]).join('·'));
      }
    }
    if (m.gradient && params.layerwise && !(m.gradient.k2 && K !== 2)) {
      // per-layer values replace the scalar knob in the label
      const g = m.gradient, abbr = g.key === 'balance' ? 'B share' : g.key === 'lambda' ? 'λ' : g.key;
      const i = parts.findIndex((x) => x.startsWith(`${ABBR[g.key] || g.key} `));
      const lbl = `${abbr} ${params.grad.map((v) => (+v).toFixed(2)).join('→')}`;
      if (i >= 0) parts[i] = lbl; else parts.splice(1, 0, lbl);
      if (g.key === 'balance') { const wi = parts.findIndex((x) => x.startsWith('w ')); if (wi >= 0) parts.splice(wi, 1); }
    }
    if (align) parts.push('aligned');
    return parts.join(' · ');
  }

  /** Compare two scored results on the same tasks: accuracy deltas (b − a) and how often they disagree. */
  function compareResults(tasks, a, b) {
    let dis = 0, tot = 0;
    const perTask = tasks.map((t, i) => {
      const pa = a.predict(t.test.X, t.test.n), pb = b.predict(t.test.X, t.test.n);
      let d = 0;
      for (let j = 0; j < t.test.n; j++) if ((pa[j] > 0.5) !== (pb[j] > 0.5)) d++;
      dis += d; tot += t.test.n;
      return { dAcc: b.accs[i] - a.accs[i], disagree: d / t.test.n };
    });
    return { perTask, disagree: dis / tot, dAvg: b.avg - a.avg, dRetention: b.retention - a.retention };
  }

  /** Training config for a built-in scenario at the toy defaults (independent of the user's lab tweaks). */
  function toyConfig(scenarioId, seed) {
    const sc = MM.SCENARIOS.find((x) => x.id === scenarioId);
    return { experts: sc.experts.map((e) => ({ ...e })), init: sc.init, hidden: 32, ftSteps: 250, seed };
  }

  /** Score `ids` on every bundle (one per seed). Returns runs in the shape summarizeSeeds() expects. */
  function seedRuns(bundles, ids, align, paramsFor) {
    return bundles.map((T) => {
      const ctx = bundleCtx(T, align);
      const run = {};
      for (const id of ids) {
        const r = scoreMethod(T, ctx, id, paramsFor(id, T.K));
        run[id] = r.blocked ? { blocked: r.blocked } : { avg: r.avg, retention: r.retention, accs: r.accs };
      }
      return run;
    });
  }

  // ---------- knob sensitivity ----------
  /** Continuous knobs worth sweeping for a method (the DARE random seed is not a knob). */
  function sweepKnobs(id, K, params) {
    const m = MM.METHOD_BY_ID[id];
    // with per-layer values on, the scalar they replace has no effect, so don't offer to sweep it
    const g = m.gradient && params && params.layerwise && !(m.gradient.k2 && K !== 2) ? m.gradient.key : null;
    const out = m.params.filter((d) => d.type === 'range' && d.key !== 'seed' && d.key !== g)
      .map((d) => ({ key: d.key, label: d.label, min: d.min, max: d.max, step: d.step }));
    if (K === 2 && g !== 'balance' && m.params.some((d) => d.type === 'weights')) out.push({ key: 'balance', label: 'Balance A ↔ B', min: 0, max: 1, step: 0.05 });
    return out;
  }
  function knobValue(params, knob) {
    if (knob.key !== 'balance') return +params[knob.key];
    const [a, b] = params.weights;
    return a + b > 0 ? b / (a + b) : 0.5;
  }
  function withKnob(params, knob, v) {
    const p = JSON.parse(JSON.stringify(params));
    if (knob.key === 'balance') p.weights = [1 - v, v, p.weights[2]];
    else p[knob.key] = v;
    return p;
  }
  function sweepValues(knob, n) {
    return Array.from({ length: n }, (_, i) => +(knob.min + ((knob.max - knob.min) * i) / (n - 1)).toFixed(4));
  }
  /** Score a method at each knob value on one bundle. */
  function sweepMethod(T, ctx, id, params, knob, xs) {
    return xs.map((x) => {
      const r = scoreMethod(T, ctx, id, withKnob(params, knob, x));
      return r.blocked ? { x, blocked: r.blocked } : { x, avg: r.avg, accs: r.accs };
    });
  }
  /** From mean (and optional sd) curves: best setting, safest setting and the forgiving range. */
  function pickSettings(xs, mean, sd, tol) {
    tol = tol == null ? 0.01 : tol;
    let bi = 0;
    mean.forEach((v, i) => { if (v > mean[bi] + 1e-12) bi = i; });
    // safe = best lower bound: mean - sd with seeds, otherwise the worst of each point and its neighbours
    const lower = mean.map((v, i) => (sd ? v - sd[i] : Math.min(v, mean[Math.max(0, i - 1)], mean[Math.min(mean.length - 1, i + 1)])));
    let si = 0;
    lower.forEach((v, i) => { if (v > lower[si] + 1e-12) si = i; });
    let lo = bi, hi = bi;
    while (lo > 0 && mean[lo - 1] >= mean[bi] - tol) lo--;
    while (hi < mean.length - 1 && mean[hi + 1] >= mean[bi] - tol) hi++;
    return { best: xs[bi], bestVal: mean[bi], safe: xs[si], safeVal: mean[si], safeLower: lower[si], range: [xs[lo], xs[hi]], rangeFrac: (xs[hi] - xs[lo]) / (xs[xs.length - 1] - xs[0] || 1) };
  }

  /**
   * Does the best setting transfer across seeds? curves = one array of average accuracies per seed, over xs.
   * pickX = the value the panel recommends (e.g. the best of the mean curve).
   * Returns { bests: [{ x, val }], spread (fraction of the knob range between the lowest and highest per-seed best),
   *   regret (worst over seeds of: that seed's best − its accuracy at pickX), moves (spread > 0.15) }.
   */
  function seedBests(xs, curves, pickX) {
    const bests = curves.map((c) => { let bi = 0; c.forEach((v, i) => { if (v > c[bi] + 1e-12) bi = i; }); return { x: xs[bi], val: c[bi] }; });
    const span = xs[xs.length - 1] - xs[0] || 1;
    const bx = bests.map((b) => b.x);
    const spread = (Math.max(...bx) - Math.min(...bx)) / span;
    let pi = 0;
    xs.forEach((x, i) => { if (Math.abs(x - pickX) < Math.abs(xs[pi] - pickX)) pi = i; });
    const regret = Math.max(...curves.map((c, k) => bests[k].val - c[pi]));
    return { bests, spread, regret, moves: spread > 0.15 };
  }

  /**
   * One verdict for "does this knob's best setting transfer across seeds?", shared by the sensitivity panel and the
   * exported config. curves = per-seed average-accuracy arrays over xs (2+ seeds).
   * kind: 'moves' (per-seed bests spread > 15% of the range AND the mean-best costs > 1 pt on some seed),
   * 'transfers' (bests spread out but the curve is flat enough), 'holds' (bests agree and it costs ≤ 1 pt),
   * 'narrow' (bests sit in the same spot but the peak is so sharp that the small shift costs > 1 pt somewhere).
   */
  function knobTransfer(xs, curves, tol) {
    const n = curves.length;
    const mean = xs.map((_, i) => curves.reduce((a, c) => a + c[i], 0) / n);
    const sd = n > 1 ? xs.map((_, i) => Math.sqrt(curves.reduce((a, c) => a + (c[i] - mean[i]) ** 2, 0) / (n - 1))) : null;
    const pick = pickSettings(xs, mean, sd, tol == null ? 0.02 : tol);
    const sb = seedBests(xs, curves, pick.best);
    const kind = sb.regret > 0.01 ? (sb.moves ? 'moves' : 'narrow') : sb.moves ? 'transfers' : 'holds';
    return { kind, n, best: pick.best, bestVal: pick.bestVal, bests: sb.bests.map((b) => b.x), regret: sb.regret, spread: sb.spread };
  }

  // ---------- paired comparisons across seeds ----------
  // two-sided 95% critical t values by degrees of freedom (n − 1)
  const T95 = [Infinity, 12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23];
  /** Per-seed difference a − b in average accuracy, with a paired-t / sign-consistency verdict. */
  function pairedDiff(runs, a, b) {
    const diffs = runs.map((r) => (r[a] && r[b] && !r[a].blocked && !r[b].blocked ? r[a].avg - r[b].avg : null)).filter((x) => x !== null);
    const n = diffs.length;
    const mean = n ? diffs.reduce((x, y) => x + y, 0) / n : 0;
    const sd = n > 1 ? Math.sqrt(diffs.reduce((x, d) => x + (d - mean) ** 2, 0) / (n - 1)) : 0;
    const se = n > 1 ? sd / Math.sqrt(n) : 0;
    const t = se > 1e-12 ? mean / se : mean === 0 ? 0 : Math.sign(mean) * Infinity;
    const eps = 5e-4; // differences under 0.05 pts count as ties
    const wins = diffs.filter((d) => d > eps).length, losses = diffs.filter((d) => d < -eps).length;
    const crit = T95[Math.min(n - 1, T95.length - 1)] || Infinity;
    let verdict;
    if (n < 2) verdict = 'too-few';
    else if (Math.abs(mean) < eps && Math.abs(t) < crit) verdict = 'same';
    else if (Math.abs(t) >= crit || ((wins === n || losses === n) && n >= 3)) verdict = mean > 0 ? 'a' : 'b';
    else if (wins > 0 && losses === 0) verdict = 'lean-a'; // never loses, but the gap is too variable to call yet
    else if (losses > 0 && wins === 0) verdict = 'lean-b';
    else verdict = 'noise';
    return { diffs, n, mean, sd, se, t, wins, losses, ties: n - wins - losses, crit, verdict };
  }

  /** CSV of every per-seed result: one row per (seed, method). */
  function seedRunsCSV(runs, ids, seeds, taskNames) {
    const q = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    const f = (x) => (x == null ? '' : (+x).toFixed(4));
    const head = ['seed', 'method', 'average', 'retention', ...taskNames, 'blocked'];
    const rows = [head.map(q).join(',')];
    runs.forEach((run, s) => ids.forEach((id) => {
      const r = run[id] || {};
      rows.push([seeds[s], id, f(r.avg), f(r.retention), ...taskNames.map((_, k) => f(r.accs ? r.accs[k] : null)), r.blocked || ''].map(q).join(','));
    }));
    return rows.join('\n') + '\n';
  }

  /** Score two recipes ({method, params, align}) on each bundle: runs = [{ A: {...}, B: {...} }, …] for pairedDiff. */
  function recipeRuns(bundles, recA, recB) {
    return bundles.map((T) => {
      const one = (rec) => {
        const r = scoreMethod(T, bundleCtx(T, rec.align), rec.method, rec.params);
        return r.blocked ? { blocked: r.blocked } : { avg: r.avg, retention: r.retention, accs: r.accs };
      };
      return { A: one(recA), B: one(recB) };
    });
  }

  // ---------- per-layer sensitivity (two layers' gradient values, third held fixed) ----------
  function layerGridParams(params, la, lb, va, vb) {
    const p = JSON.parse(JSON.stringify(params));
    p.layerwise = true;
    p.grad[la] = va; p.grad[lb] = vb;
    return p;
  }
  /** Average accuracy over a vals × vals grid: rows = layer la, cols = layer lb. Blocked cells are null. */
  function layerGrid(T, ctx, id, params, la, lb, vals) {
    return vals.map((va) => vals.map((vb) => {
      const r = scoreMethod(T, ctx, id, layerGridParams(params, la, lb, va, vb));
      return r.blocked ? null : r.avg;
    }));
  }
  /** Best cell, safest cell (best worst-case over its 3×3 neighbourhood), forgiving fraction and per-axis sensitivity. */
  function pickGrid(M, tol) {
    tol = tol == null ? 0.02 : tol;
    const R = M.length, C = M[0].length;
    // "safe" = good in every direction, so only interior cells qualify (edge cells have fewer neighbours to fail)
    const interior = R >= 3 && C >= 3;
    let best = null, safe = null;
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
      const v = M[i][j];
      if (v == null) continue;
      if (!best || v > best.v + 1e-12) best = { i, j, v };
      if (interior && (i === 0 || j === 0 || i === R - 1 || j === C - 1)) continue;
      let lower = Infinity;
      for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
        const a = M[i + di] && M[i + di][j + dj];
        if (a != null) lower = Math.min(lower, a);
      }
      if (!safe || lower > safe.lower + 1e-12) safe = { i, j, v, lower };
    }
    let near = 0, cnt = 0, sensA = 0, sensB = 0;
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) if (M[i][j] != null) { cnt++; if (M[i][j] >= best.v - tol) near++; }
    for (let j = 0; j < C; j++) { const col = M.map((row) => row[j]).filter((x) => x != null); if (col.length) sensA += Math.max(...col) - Math.min(...col); }
    for (let i = 0; i < R; i++) { const row = M[i].filter((x) => x != null); if (row.length) sensB += Math.max(...row) - Math.min(...row); }
    return { best, safe, near: cnt ? near / cnt : 0, sensA: sensA / C, sensB: sensB / R };
  }

  /**
   * Per-layer grid over several seeds: grids = one vals × vals matrix of average accuracy per seed.
   * Returns { mean (cell-wise mean grid), pick (pickGrid of the mean), bests (each seed's best cell), spread (largest
   * distance between two seeds' best cells, as a fraction of the grid width), regret (worst seed's loss at the mean's
   * best cell), kind ('moves' | 'narrow' | 'transfers' | 'holds', same thresholds as knobTransfer), layer (per seed,
   * which axis the recipe is most sensitive to: 'a' rows, 'b' columns, 'both') }.
   */
  function gridTransfer(grids) {
    const n = grids.length, R = grids[0].length, C = grids[0][0].length;
    const mean = grids[0].map((row, i) => row.map((_, j) => {
      const vs = grids.map((g) => g[i][j]);
      return vs.some((v) => v == null) ? null : vs.reduce((a, b) => a + b, 0) / n;
    }));
    const pick = pickGrid(mean);
    const per = grids.map((g) => pickGrid(g));
    const bests = per.map((q) => ({ i: q.best.i, j: q.best.j, v: q.best.v }));
    const span = Math.max(R - 1, C - 1) || 1;
    let spread = 0;
    for (const a of bests) for (const b of bests) spread = Math.max(spread, Math.max(Math.abs(a.i - b.i), Math.abs(a.j - b.j)) / span);
    const regret = Math.max(...grids.map((g, k) => per[k].best.v - g[pick.best.i][pick.best.j]));
    const kind = regret > 0.01 ? (spread > 0.15 ? 'moves' : 'narrow') : spread > 0.15 ? 'transfers' : 'holds';
    const layer = per.map((q) => (q.sensA > q.sensB * 1.5 ? 'a' : q.sensB > q.sensA * 1.5 ? 'b' : 'both'));
    return { n, mean, pick, bests, spread, regret, kind, layer };
  }

  // ---------- pixel saliency for digit tasks ----------
  /** Mean |Δp| / δ when each of the 64 pixels is nudged by δ, over the first m test digits of a task. */
  function digitSaliency(predict, task, m, delta) {
    const D = MM.DIGIT_D;
    m = Math.min(m || 80, task.test.n); delta = delta || 0.25;
    const X = task.test.X.slice(0, m * D), X2 = new Float32Array(X);
    const p0 = predict(X, m), sal = new Float32Array(D);
    for (let q = 0; q < D; q++) {
      for (let s = 0; s < m; s++) X2[s * D + q] = X[s * D + q] + delta;
      const p = predict(X2, m);
      let a = 0;
      for (let s = 0; s < m; s++) a += Math.abs(p[s] - p0[s]);
      sal[q] = a / (m * delta);
      for (let s = 0; s < m; s++) X2[s * D + q] = X[s * D + q];
    }
    return sal;
  }
  function mapCosine(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return na > 0 && nb > 0 ? d / Math.sqrt(na * nb) : 0;
  }

  /** Per-task saliency maps of each expert on its own task (the reference patterns). */
  function expertSaliency(ctx) {
    return ctx.tasks.map((t, k) => digitSaliency((X, n) => MM.predictProb(ctx.layout, ctx.experts[k], X, n), t));
  }
  /** A merge's saliency on every task, its pattern match with each expert, and the mean match. */
  function saliencyRow(ctx, predict, expertMaps) {
    const maps = ctx.tasks.map((t) => digitSaliency(predict, t));
    const match = maps.map((m, k) => mapCosine(expertMaps[k], m));
    return { maps, match, mean: match.reduce((a, b) => a + b, 0) / match.length };
  }

  /** One seed: score a recipe on bundle T and measure its saliency match per task. null if the method is blocked. */
  function saliencyRun(T, method, params, align) {
    const ctx = MM.bundleCtx(T, align);
    const r = MM.scoreMethod(T, ctx, method, params);
    if (r.blocked) return null;
    const row = saliencyRow(ctx, r.predict, expertSaliency(ctx));
    return { match: row.match, merged: r.accs.slice(), expert: T.expertAccs.map((e, i) => e[i]), router: !!r.router };
  }
  /**
   * Summarise saliency over seeds. runs = [{ match:[K], merged:[K], expert:[K], router }].
   * Per task: match mean / sd / min / max, how many seeds give each reading kind, the most common kind and its share.
   */
  function saliencySeedSummary(runs) {
    runs = runs.filter(Boolean);
    const n = runs.length, K = n ? runs[0].match.length : 0;
    return Array.from({ length: K }, (_, k) => {
      const ms = runs.map((r) => r.match[k]);
      const mean = ms.reduce((a, b) => a + b, 0) / n;
      const sd = n > 1 ? Math.sqrt(ms.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : 0;
      const counts = {};
      runs.forEach((r) => { const kind = MM.saliencyReading(r.match[k], r.merged[k], r.expert[k], { router: r.router }).kind; counts[kind] = (counts[kind] || 0) + 1; });
      const top = Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0];
      return { n, mean, sd, min: Math.min(...ms), max: Math.max(...ms), counts, top, agree: counts[top] / n };
    });
  }

  // ---------- undo / redo ----------
  /**
   * Snapshot history. record(snap, t, key) adds a snapshot (strings compared for equality); if the previous
   * record had the same coalescing key and came within `windowMs`, it replaces the top instead (one undo step
   * per slider drag). undo()/redo() return the snapshot to restore, or null.
   */
  /**
   * Cross-toy summary of knob-transfer tables. entries = [{ label, rows: [{ method, knob, kind }] }].
   * Returns { keys (method|knob, first-seen order), everywhere (carries over on every toy that measured it, and on
   * every toy), split (carries over on some toys, not on others) }. "Carries over" = kind 'holds' or 'transfers'.
   */
  function transferSummary(entries) {
    const keys = [];
    entries.forEach((e) => e.rows.forEach((r) => { const k = `${r.method}|${r.knob}`; if (!keys.includes(k)) keys.push(k); }));
    const find = (e, k) => e.rows.find((r) => `${r.method}|${r.knob}` === k);
    const carry = (r) => !!r && (r.kind === 'holds' || r.kind === 'transfers');
    const everywhere = keys.filter((k) => entries.every((e) => carry(find(e, k))));
    const split = keys.filter((k) => { const rs = entries.map((e) => find(e, k)).filter(Boolean); return rs.length > 1 && rs.some(carry) && !rs.every(carry); });
    return { keys, everywhere, split };
  }

  /**
   * A least-recently-used map with a size cap. keep(key) marks entries that must never be evicted (e.g. the
   * Lab's current models); if only kept entries are left, the cache may grow past the cap rather than drop them.
   */
  function makeLRU(limit, keep) {
    const m = new Map();
    let evicted = 0;
    return {
      get(k) { if (!m.has(k)) return undefined; const v = m.get(k); m.delete(k); m.set(k, v); return v; },
      has: (k) => m.has(k),
      set(k, v) {
        m.delete(k); m.set(k, v);
        while (m.size > limit) {
          let victim;
          for (const key of m.keys()) if (!(keep && keep(key))) { victim = key; break; }
          if (victim === undefined) break;
          m.delete(victim); evicted++;
        }
      },
      delete: (k) => m.delete(k),
      get size() { return m.size; },
      get evicted() { return evicted; },
      keys: () => Array.from(m.keys()),
    };
  }

  /**
   * Undo coalescing keys. Knob edits get a key (method, alignment, pin), so a quick run of them merges into one step.
   * An edit that changes the method, alignment or pin gets null, so it is always its own step and the knob edit
   * after it can't overwrite it.
   */
  function editKeyTracker() {
    let last;
    return (method, align, pin) => {
      const k = JSON.stringify([method, !!align, pin || null]);
      const out = last === undefined || k === last ? k : null;
      last = k;
      return out;
    };
  }
  function makeHistory(limit, windowMs) {
    limit = limit || 100; windowMs = windowMs == null ? 700 : windowMs;
    const past = [], future = [];
    let lastT = -Infinity, lastKey = null;
    return {
      record(snap, t, key) {
        if (past.length && past[past.length - 1] === snap) return false;
        if (past.length > 1 && key != null && key === lastKey && t - lastT < windowMs) past[past.length - 1] = snap;
        else { past.push(snap); if (past.length > limit) past.shift(); }
        future.length = 0; lastT = t; lastKey = key;
        return true;
      },
      undo() { if (past.length < 2) return null; future.push(past.pop()); lastKey = null; return past[past.length - 1]; },
      redo() { if (!future.length) return null; past.push(future.pop()); lastKey = null; return past[past.length - 1]; },
      canUndo: () => past.length > 1,
      canRedo: () => future.length > 0,
      size: () => past.length,
    };
  }

  Object.assign(MM, { gridTransfer, transferSummary, makeLRU, seedBests, knobTransfer, recipeDiff, attributeChanges, summarizeAttributions, saliencyRun, saliencySeedSummary, makeHistory, editKeyTracker, digitSaliency, mapCosine, expertSaliency, saliencyRow, layerGridParams, layerGrid, pickGrid, recipeRuns, pairedDiff, seedRunsCSV, sweepKnobs, knobValue, withKnob, sweepValues, sweepMethod, pickSettings, toyConfig, seedRuns, recipeLabel, compareResults, bundleShape, BASE_STEPS, LR, FT_LR, trainBundle, trainBundleSync, bundleCtx, scoreMethod, summarizeSeeds });
})();
