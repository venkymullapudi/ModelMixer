/* ModelMixer — headless lab core: train a set of experts, build merge contexts,
 * score methods and summarise multi-seed robustness. No DOM access, so the same
 * code drives the UI, the Node test-suite and the robustness runner. */
(function () {
  const root = typeof window !== 'undefined' ? window : globalThis;
  const MM = (root.MM = root.MM || {});

  const BASE_STEPS = 200, LR = 0.01, FT_LR = 0.1;

  /** Generator: yields {type:'start'|'step'|'done', ...} events and returns the trained bundle.
   *  cfg = { experts:[spec], init:'shared'|'independent', hidden, ftSteps, seed } */
  function* trainBundle(cfg) {
    const { experts: specs, init, hidden, ftSteps, seed } = cfg;
    const K = specs.length;
    const layout = MM.makeLayout([2, hidden, hidden, 1], MM.makeEncoder(9, MM.EXTENT, 0.5));
    const tasks = specs.map((e, i) => MM.makeTaskData(e, i, seed));
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
      const pre = MM.concat([MM.genericSet(seed, 900), MM.pretrainSet(tasks, 0.05, seed)]);
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

  Object.assign(MM, { sweepKnobs, knobValue, withKnob, sweepValues, sweepMethod, pickSettings, toyConfig, seedRuns, recipeLabel, compareResults, BASE_STEPS, LR, FT_LR, trainBundle, trainBundleSync, bundleCtx, scoreMethod, summarizeSeeds });
})();
