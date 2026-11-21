/* ModelMixer test-suite. Pure: takes the MM namespace, returns [{name, ok, detail}].
 * Runs in Node (tests/run.js, used by CI) and in a browser (window.MMTests.run(MM)). */
(function () {
  function suite(MM) {
    const results = [];
    const test = (name, fn) => {
      try {
        const detail = fn();
        results.push({ name, ok: true, detail: detail || '' });
      } catch (e) {
        results.push({ name, ok: false, detail: e && e.message ? e.message : String(e) });
      }
    };
    const assert = (cond, msg) => { if (!cond) throw new Error(msg); };
    const pct = (x) => `${(x * 100).toFixed(1)}%`;
    const params = (id, K) => MM.defaultParams(MM.METHOD_BY_ID[id], K);
    const scenario = (id, over) => {
      const sc = MM.SCENARIOS.find((s) => s.id === id);
      return Object.assign({ experts: sc.experts.map((e) => ({ ...e })), init: sc.init, hidden: 32, ftSteps: 250, seed: 7 }, over || {});
    };

    // ---------- primitives ----------
    test('Hungarian finds the optimal assignment', () => {
      // cost matrix whose unique optimum is 0->2, 1->0, 2->1 (total 3)
      const C = [9, 9, 1, 1, 9, 9, 9, 1, 9];
      const a = Array.from(MM.hungarian(C, 3));
      assert(a.join() === '2,0,1', `got ${a.join()}`);
      // brute-force check on a random 6x6
      const rng = MM.mulberry32(3), n = 6, R = Array.from({ length: n * n }, () => rng());
      const h = MM.hungarian(R, n);
      const cost = (p) => p.reduce((s, j, i) => s + R[i * n + j], 0);
      let best = Infinity;
      const perm = (arr, k) => {
        if (k === n) { best = Math.min(best, cost(arr)); return; }
        for (let i = k; i < n; i++) { [arr[k], arr[i]] = [arr[i], arr[k]]; perm(arr, k + 1); [arr[k], arr[i]] = [arr[i], arr[k]]; }
      };
      perm(Array.from({ length: n }, (_, i) => i), 0);
      assert(Math.abs(cost(Array.from(h)) - best) < 1e-9, `hungarian ${cost(Array.from(h))} vs brute ${best}`);
    });

    test('Weight matching undoes a hidden-unit shuffle exactly', () => {
      const layout = MM.makeLayout([2, 12, 12, 1], MM.makeEncoder(5, MM.EXTENT, 0.5));
      const A = MM.initParams(layout, 11);
      // build B = A with hidden units shuffled (same function, different weights)
      const rng = MM.mulberry32(5);
      const shuffle = (n) => { const p = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p; };
      const perms = [shuffle(12), shuffle(12)];
      const B = new Float32Array(A.length);
      const L = layout.nLayers;
      for (let l = 0; l < L; l++) {
        const W = layout.tensors[2 * l], b = layout.tensors[2 * l + 1];
        const rp = l < L - 1 ? perms[l] : [0];
        const cp = l > 0 ? perms[l - 1] : Array.from({ length: W.cols }, (_, i) => i);
        for (let i = 0; i < W.rows; i++) {
          for (let k = 0; k < W.cols; k++) B[W.off + i * W.cols + k] = A[W.off + rp[i] * W.cols + cp[k]];
          B[b.off + i] = A[b.off + rp[i]];
        }
      }
      const X = new Float32Array([0.3, -0.2, 1.1, 0.4, -0.9, 0.8]);
      const pa = MM.predictProb(layout, A, X, 3), pb = MM.predictProb(layout, B, X, 3);
      for (let i = 0; i < 3; i++) assert(Math.abs(pa[i] - pb[i]) < 1e-5, 'shuffle changed the function');
      const r = MM.weightMatch(layout, A, B);
      let maxDiff = 0;
      for (let i = 0; i < A.length; i++) maxDiff = Math.max(maxDiff, Math.abs(A[i] - r.aligned[i]));
      assert(maxDiff < 1e-6, `aligned weights differ by ${maxDiff}`);
      return `moved ${pct(r.movedFrac)} of units back`;
    });

    test('summarizeSeeds: mean, sd and split wins', () => {
      const runs = [
        { a: { avg: 0.9, retention: 1, accs: [0.9, 0.9] }, b: { avg: 0.9, retention: 1, accs: [0.8, 1] }, c: { blocked: 'x' } },
        { a: { avg: 0.8, retention: 1, accs: [0.8, 0.8] }, b: { avg: 0.7, retention: 1, accs: [0.7, 0.7] }, c: { blocked: 'x' } },
      ];
      const [a, b, c] = MM.summarizeSeeds(runs, ['a', 'b', 'c']);
      assert(Math.abs(a.mean - 0.85) < 1e-9 && Math.abs(a.sd - Math.sqrt(0.005)) < 1e-9, `a mean/sd ${a.mean}/${a.sd}`);
      assert(a.wins === 1.5 && b.wins === 0.5, `wins ${a.wins}/${b.wins}`);
      assert(b.worstTask === 0.7, `worst ${b.worstTask}`);
      assert(c.blocked === 'x', 'blocked passthrough');
    });

    // ---------- shareable recipe links ----------
    test('Recipe links round-trip experts, settings, knobs and the pin', () => {
      const r = {
        experts: [{ task: 'moons', region: 'left' }, { task: 'circles', region: 'right', flip: true }, { task: 'xor', region: 'top' }],
        init: 'independent', hidden: 64, ftSteps: 375, seed: 4242, align: true,
        method: 'dare', params: { drop: 0.73, lambda: 1.35, mode: 'linear', seed: 3, weights: [1, 0.5, 0.25] },
        pin: { method: 'stitch', params: { layers: [2, 0, 1] }, align: false },
      };
      const code = MM.encodeRecipe(r);
      assert(/^[A-Za-z0-9_-]+$/.test(code), 'not URL-safe');
      const d = MM.decodeRecipe(code);
      assert(d.ok, d.error);
      const x = d.recipe;
      assert(JSON.stringify(x.experts) === JSON.stringify(r.experts), JSON.stringify(x.experts));
      assert(x.init === 'independent' && x.hidden === 64 && x.ftSteps === 375 && x.seed === 4242 && x.align === true, 'settings');
      assert(x.method === 'dare' && x.params.drop === 0.73 && x.params.lambda === 1.35 && x.params.mode === 'linear' && x.params.seed === 3, JSON.stringify(x.params));
      assert(x.params.weights.join() === '1,0.5,0.25', 'weights');
      assert(x.pin && x.pin.method === 'stitch' && x.pin.params.layers.join() === '2,0,1', 'pin');
      assert(MM.recipeFromHash(`https://x.io/modelmixer.html#r=${code}`) === code && MM.recipeFromHash('#other') === null, 'hash parse');
      return `${code.length} chars`;
    });
    test('Recipe links reject or repair bad input', () => {
      assert(!MM.decodeRecipe('not-a-recipe!!').ok, 'garbage accepted');
      assert(!MM.decodeRecipe('').ok, 'empty accepted');
      const enc = (o) => btoa(JSON.stringify(o)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
      const base = { v: 1, e: [['moons', 'left', 0], ['circles', 'right', 0]], i: 's', h: 32, f: 250, s: 7, a: 0, m: 'ties', p: {} };
      assert(!MM.decodeRecipe(enc({ ...base, m: 'evil' })).ok, 'unknown method accepted');
      assert(!MM.decodeRecipe(enc({ ...base, e: [['moons', 'mars', 0], ['circles', 'right', 0]] })).ok, 'unknown region accepted');
      assert(!MM.decodeRecipe(enc({ ...base, e: [['moons', 'left', 0]] })).ok, 'single expert accepted');
      assert(!MM.decodeRecipe(enc({ ...base, v: 9 })).ok, 'future version accepted');
      const d = MM.decodeRecipe(enc({ ...base, h: 999, f: 99999, s: -5, p: { density: 7, lambda: 'x', bogus: 1 }, pn: { m: 'nope' } }));
      assert(d.ok, d.error);
      const x = d.recipe;
      assert(x.hidden === 32 && x.ftSteps === 800 && x.seed === 1, `settings ${x.hidden}/${x.ftSteps}/${x.seed}`);
      assert(x.params.density === 1 && x.params.lambda === 1 && !('bogus' in x.params), JSON.stringify(x.params));
      assert(x.pin === null, 'unknown pin method should be dropped');
      const st = MM.decodeRecipe(enc({ ...base, m: 'stitch', p: { layers: [2, 9, -1] } }));
      assert(st.recipe.params.layers.join() === '1,1,0', `layers clamped to K=2: ${st.recipe.params.layers}`);
    });

    // ---------- recipe advisor ----------
    const advise = (o) => MM.adviseRecipe(o);
    const top = (r, n) => r.recs.slice(0, n || 3).map((x) => x.id);
    test('Advisor default (shared base, no data, one model) recommends task-vector merges', () => {
      const r = advise({});
      assert(['ties', 'dare', 'task_arithmetic'].includes(r.recs[0].id), `top is ${r.recs[0].id}`);
      const out = r.excluded.map((x) => x.id);
      ['fisher', 'regmean', 'distill', 'ensemble', 'moe'].forEach((id) => assert(out.includes(id), `${id} should be ruled out`));
      assert(r.scenario === 'two' && r.align === false, `${r.scenario}/${r.align}`);
      return top(r).join(', ');
    });
    test('Advisor: no shared base rules out task vectors and turns on alignment', () => {
      const r = advise({ base: 'no' });
      const out = r.excluded.map((x) => x.id);
      ['task_arithmetic', 'ties', 'dare', 'breadcrumbs'].forEach((id) => assert(out.includes(id), `${id} not ruled out`));
      assert(r.scenario === 'strangers' && r.align === true, `${r.scenario}/${r.align}`);
      assert(r.recs.find((x) => x.id === 'linear').caveats.some((c) => /align/i.test(c)), 'linear should carry an alignment caveat');
      const r2 = advise({ base: 'no', deploy: 'any', train: 'yes', data: 'yes' });
      assert(['ensemble', 'moe', 'distill'].includes(r2.recs[0].id), `top is ${r2.recs[0].id}`);
    });
    test('Advisor: hard constraints are respected', () => {
      const r = advise({ count: '3+' });
      assert(r.excluded.some((x) => x.id === 'slerp'), 'SLERP should be out for 3+ models');
      assert(r.scenario === 'three', r.scenario);
      const rr = advise({ deploy: 'router', data: 'yes', train: 'yes' });
      assert(rr.excluded.some((x) => x.id === 'ensemble') && !rr.excluded.some((x) => x.id === 'moe'), 'router: ensemble out, MoE in');
      for (const combo of [{}, { base: 'no' }, { deploy: 'any', data: 'yes', train: 'yes' }, { goal: 'same', count: '3+' }]) {
        const x = advise(combo);
        x.recs.forEach((rec) => {
          const m = MM.METHOD_BY_ID[rec.id];
          assert(!(m.req.data && x.answers.data === 'no') && !(m.req.training && x.answers.train === 'no'), `${rec.id} violates constraints`);
        });
        assert(x.recs.length + x.excluded.length === MM.METHODS.length, 'every method accounted for');
      }
    });
    test('Advisor: data → RegMean; same-skill runs → Model Stock; conflict → note', () => {
      assert(advise({ data: 'yes' }).recs[0].id === 'regmean', `data top is ${advise({ data: 'yes' }).recs[0].id}`);
      assert(advise({ goal: 'same' }).recs[0].id === 'model_stock', `same top is ${advise({ goal: 'same' }).recs[0].id}`);
      const c = advise({ conflict: 'yes', deploy: 'router', data: 'yes', train: 'yes' });
      assert(c.notes.some((n) => /contradictory/.test(n)) && c.scenario === 'overlap', 'conflict note/scenario');
      assert(c.recs.find((x) => x.id === 'moe').caveats.some((t) => /router/i.test(t)), 'MoE should warn about routing identical inputs');
    });

    test('pairedDiff: consistent, noisy and identical comparisons', () => {
      const mk = (a, b) => a.map((x, i) => ({ X: { avg: x }, Y: { avg: b[i] } }));
      const c = MM.pairedDiff(mk([0.9, 0.91, 0.88, 0.92, 0.9], [0.85, 0.86, 0.84, 0.87, 0.86]), 'X', 'Y');
      assert(c.verdict === 'a' && c.wins === 5 && Math.abs(c.mean - 0.046) < 1e-9, JSON.stringify(c));
      const nz = MM.pairedDiff(mk([0.9, 0.8, 0.92, 0.81, 0.9], [0.85, 0.88, 0.86, 0.86, 0.87]), 'X', 'Y');
      assert(nz.verdict === 'noise', `noisy pair: ${nz.verdict} (t ${nz.t.toFixed(2)})`);
      const lean = MM.pairedDiff(mk([0.9, 0.95, 0.9, 0.86, 0.9], [0.9, 0.85, 0.89, 0.85, 0.8]), 'X', 'Y');
      assert(lean.verdict === 'lean-a' && lean.losses === 0, `lean: ${lean.verdict} t ${lean.t.toFixed(2)}`);
      const same = MM.pairedDiff(mk([0.9, 0.8, 0.7], [0.9, 0.8, 0.7]), 'X', 'Y');
      assert(same.verdict === 'same' && same.ties === 3, JSON.stringify(same));
      const blocked = MM.pairedDiff([{ X: { avg: 0.9 }, Y: { blocked: 'no' } }], 'X', 'Y');
      assert(blocked.n === 0 && blocked.verdict === 'too-few', 'blocked runs are skipped');
    });
    test('seedRunsCSV: header, one row per seed × method, quoting', () => {
      const runs = [{ ties: { avg: 0.9, retention: 0.95, accs: [0.8, 1] }, moe: { blocked: 'needs data, "really"' } }];
      const csv = MM.seedRunsCSV(runs, ['ties', 'moe'], [7], ['Moons', 'Digits 0 vs 1']).trim().split('\n');
      assert(csv.length === 3, csv.join(' | '));
      assert(csv[0] === 'seed,method,average,retention,Moons,Digits 0 vs 1,blocked', csv[0]);
      assert(csv[1] === '7,ties,0.9000,0.9500,0.8000,1.0000,', csv[1]);
      assert(csv[2] === '7,moe,,,,,"needs data, ""really"""', csv[2]);
    });

    test('blendRanking: promotes a pick that beats the rules\' #1 on every seed, otherwise keeps rule order', () => {
      const r = (xs) => xs.map((avg) => ({ avg }));
      const win = MM.blendRanking(['a', 'b', 'c'], { a: r([0.80, 0.82, 0.81]), b: r([0.86, 0.88, 0.85]), c: r([0.79, 0.90, 0.70]) });
      assert(win.order.join() === 'b,a,c' && win.promoted.id === 'b' && win.promoted.over === 'a', JSON.stringify(win.order));
      const noisy = MM.blendRanking(['a', 'b', 'c'], { a: r([0.80, 0.82, 0.81]), b: r([0.86, 0.78, 0.85]), c: r([0.7, 0.7, 0.7]) });
      assert(noisy.order.join() === 'a,b,c' && !noisy.promoted, `noisy: ${noisy.order}`);
      const two = MM.blendRanking(['a', 'b', 'c'], { a: r([0.5, 0.5, 0.5]), b: r([0.6, 0.6, 0.6]), c: r([0.9, 0.9, 0.9]) });
      assert(two.order[0] === 'c', 'the bigger consistent winner is promoted');
      const blocked = MM.blendRanking(['a', 'b'], { a: r([0.8, 0.8, 0.8]), b: [{ blocked: 'x' }, { blocked: 'x' }, { blocked: 'x' }] });
      assert(blocked.order.join() === 'a,b', 'a blocked pick is never promoted');
    });

    // ---------- training ----------
    const cfgTwo = scenario('two');
    const T = MM.trainBundleSync(cfgTwo);

    test('Training is deterministic for a fixed seed', () => {
      const T2 = MM.trainBundleSync(cfgTwo);
      for (let k = 0; k < T.K; k++) for (let i = 0; i < T.experts[k].length; i++) assert(T.experts[k][i] === T2.experts[k][i], `expert ${k} differs at ${i}`);
    });

    test('Two skills: each expert masters its own task, base does not', () => {
      T.expertAccs.forEach((a, i) => assert(a[i] > 0.9, `expert ${i} only ${pct(a[i])} on own task`));
      const baseAvg = T.baseAccs.reduce((a, b) => a + b, 0) / T.K;
      assert(baseAvg < 0.8, `base already at ${pct(baseAvg)} — merging would be trivial`);
      return `experts ${T.expertAccs.map((a, i) => pct(a[i])).join(' / ')}, base ${pct(baseAvg)}`;
    });

    // ---------- every method ----------
    const ctx = MM.bundleCtx(T, false);
    const floors = { regmean: 0.95, moe: 0.95, distill: 0.9, ensemble: 0.6 };
    const scores = {};
    MM.METHODS.forEach((m) => {
      test(`${m.name} runs and clears its floor`, () => {
        const r = MM.scoreMethod(T, ctx, m.id, params(m.id, T.K));
        assert(!r.blocked, `blocked: ${r.blocked}`);
        r.accs.forEach((a) => assert(a >= 0 && a <= 1 && Number.isFinite(a), `bad accuracy ${a}`));
        assert(Number.isFinite(r.cost) && Number.isFinite(r.storage), 'cost/storage missing');
        const floor = floors[m.id] || 0.65;
        assert(r.avg >= floor, `avg ${pct(r.avg)} < floor ${pct(floor)}`);
        scores[m.id] = r.avg;
        return pct(r.avg);
      });
    });

    test('Task-vector tricks beat plain averaging on two skills', () => {
      const best = Math.max(scores.ties || 0, scores.dare || 0, scores.task_arithmetic || 0);
      assert(best >= (scores.linear || 0) - 0.005, `best TV ${pct(best)} vs linear ${pct(scores.linear || 0)}`);
    });

    test('seedRuns + summarizeSeeds agree with direct scoring', () => {
      const ids = ['linear', 'ties'];
      const runs = MM.seedRuns([T, T], ids, false, (id, K) => params(id, K));
      const sum = MM.summarizeSeeds(runs, ids);
      sum.forEach((x) => {
        const direct = MM.scoreMethod(T, ctx, x.id, params(x.id, T.K)).avg;
        assert(Math.abs(x.mean - direct) < 1e-12 && x.sd === 0, `${x.id}: ${x.mean} vs ${direct}, sd ${x.sd}`);
      });
      const cfg = MM.toyConfig('three', 5);
      assert(cfg.experts.length === 3 && cfg.seed === 5 && cfg.hidden === 32, JSON.stringify(cfg));
    });

    test('Knob sweep: knobs listed, balance endpoints reproduce the experts', () => {
      assert(MM.sweepKnobs('ties', 2).map((k) => k.key).join() === 'density,lambda', 'ties knobs');
      assert(MM.sweepKnobs('dare', 2).map((k) => k.key).join() === 'drop,lambda,balance', 'dare knobs (no seed, balance for K=2)');
      assert(MM.sweepKnobs('model_stock', 2).length === 0 && MM.sweepKnobs('linear', 3).length === 0, 'no-knob methods');
      const knob = MM.sweepKnobs('linear', 2)[0];
      const pts = MM.sweepMethod(T, ctx, 'linear', params('linear', 2), knob, [0, 1]);
      assert(Math.abs(pts[0].accs[0] - T.expertAccs[0][0]) < 1e-9 && Math.abs(pts[1].accs[1] - T.expertAccs[1][1]) < 1e-9, 'balance 0/1 should be expert A/B');
      assert(Math.abs(MM.knobValue({ weights: [1, 3, 1] }, knob) - 0.75) < 1e-12, 'knobValue balance');
      const xs = MM.sweepValues({ min: 0, max: 2, step: 0.05 }, 5);
      assert(xs.join() === '0,0.5,1,1.5,2', xs.join());
    });

    test('pickSettings: best, safe and forgiving range', () => {
      const xs = [0, 1, 2, 3, 4, 5];
      const mean = [0.5, 0.80, 0.86, 0.85, 0.95, 0.6]; // spike at 4, plateau at 1–3
      const a = MM.pickSettings(xs, mean);
      assert(a.best === 4 && a.range[0] === 4 && a.range[1] === 4, JSON.stringify(a));
      assert(a.safe >= 1 && a.safe <= 3, `safe should be on the plateau (1–3), not the spike; got ${a.safe}`);
      const b = MM.pickSettings(xs, mean, [0, 0, 0.01, 0.01, 0.2, 0]);
      assert(b.safe === 2 || b.safe === 3, `with sd the noisy spike loses: ${b.safe}`);
      const c = MM.pickSettings([0, 1, 2], [0.9, 0.905, 0.9]);
      assert(c.range[0] === 0 && c.range[1] === 2 && c.rangeFrac === 1, JSON.stringify(c));
    });

    test('Reviewed defaults are no worse than the old ones (Two skills)', () => {
      // Defaults changed in the 2026-11 review (3 scenarios × 3 seeds); guard against regressions on the test bundle.
      const cmp = (id, oldOver) => {
        const now = MM.scoreMethod(T, ctx, id, params(id, T.K)).avg;
        const old = MM.scoreMethod(T, ctx, id, Object.assign(params(id, T.K), oldOver)).avg;
        assert(now >= old - 0.01, `${id}: new ${pct(now)} < old ${pct(old)}`);
        return `${id} ${pct(old)}→${pct(now)}`;
      };
      // TIES density 0.3 → 0.2 is not checked here: it wins on mean, worst case and spread across 9 bundles,
      // but a single bundle can go either way (this one favours 0.3 by ~1 pt), which is exactly seed noise.
      return [cmp('breadcrumbs', { density: 0.6, gamma: 0.02 }), cmp('fisher', { eps: 0.05 })].join(', ');
    });

    test('Per-layer values: flat gradient equals the scalar knob', () => {
      const same = (a, b, what) => { for (let i = 0; i < a.length; i++) assert(Math.abs(a[i] - b[i]) < 1e-6, `${what} differs at ${i}`); };
      const sl = params('slerp', 2); sl.t = 0.3;
      same(MM.runMethod('slerp', ctx, sl).params, MM.runMethod('slerp', ctx, Object.assign({}, sl, { layerwise: true, grad: [0.3, 0.3, 0.3] })).params, 'slerp');
      for (const id of ['task_arithmetic', 'ties', 'dare', 'breadcrumbs']) {
        const p0 = params(id, 2); p0.lambda = 0.7;
        same(MM.runMethod(id, ctx, p0).params, MM.runMethod(id, ctx, Object.assign({}, p0, { layerwise: true, grad: [0.7, 0.7, 0.7] })).params, id);
      }
    });

    test('Per-layer values act layer by layer', () => {
      const L = T.layout;
      const out = L.tensors.filter((t) => t.layer === 2);
      // λ = 0 on the output layer leaves it at the base
      const r = MM.runMethod('task_arithmetic', ctx, Object.assign(params('task_arithmetic', 2), { layerwise: true, grad: [1, 1, 0] }));
      out.forEach((t) => { for (let i = t.off; i < t.off + t.size; i++) assert(Math.abs(r.params[i] - T.base[i]) < 1e-6, 'output layer should equal base'); });
      // Linear balance [0, 0, 1] = Layer Stitch with sources A·A·B
      const lin = MM.runMethod('linear', ctx, Object.assign(params('linear', 2), { layerwise: true, grad: [0, 0, 1] })).params;
      const st = MM.runMethod('stitch', ctx, Object.assign(params('stitch', 2), { layers: [0, 0, 1] })).params;
      for (let i = 0; i < lin.length; i++) assert(Math.abs(lin[i] - st[i]) < 1e-6, `linear gradient ≠ stitch at ${i}`);
      // 3 experts: the balance gradient is ignored (needs exactly 2)
      assert(!MM.layerwiseOn(MM.METHOD_BY_ID.linear, { K: 3 }, { layerwise: true, grad: [0, 0, 1] }), 'balance gradient needs K=2');
      // label, sweep knobs and link round-trip
      const pl = Object.assign(params('slerp', 2), { layerwise: true, grad: [0.2, 0.5, 0.8] });
      assert(MM.recipeLabel('slerp', pl, false, 2) === 'SLERP · t 0.20→0.50→0.80', MM.recipeLabel('slerp', pl, false, 2));
      assert(MM.sweepKnobs('slerp', 2, pl).length === 0 && MM.sweepKnobs('ties', 2, Object.assign(params('ties', 2), { layerwise: true })).map((k) => k.key).join() === 'density', 'sweep knobs with gradient');
      const code = MM.encodeRecipe({ experts: [{ task: 'moons', region: 'left' }, { task: 'circles', region: 'right' }], init: 'shared', hidden: 32, ftSteps: 250, seed: 7, align: false, method: 'slerp', params: pl, pin: null });
      const back = MM.decodeRecipe(code).recipe.params;
      assert(back.layerwise === true && back.grad.join() === '0.2,0.5,0.8', JSON.stringify(back));
    });

    test('Karcher mean of two models = normalised-direction midpoint at the average norm', () => {
      // For two equal-weight models the Karcher mean has a closed form: direction ∝ uA + uB, norm = mean norm.
      // (SLERP interpolates the raw vectors, so its midpoint points along A + B and differs when norms differ.)
      const k = MM.runMethod('karcher', ctx, params('karcher', 2)).params;
      const [A, B] = T.experts;
      T.layout.tensors.forEach((t) => {
        let na = 0, nb = 0;
        for (let i = t.off; i < t.off + t.size; i++) { na += A[i] * A[i]; nb += B[i] * B[i]; }
        na = Math.sqrt(na); nb = Math.sqrt(nb);
        const mid = []; let nm = 0;
        for (let i = t.off; i < t.off + t.size; i++) { const v = A[i] / na + B[i] / nb; mid.push(v); nm += v * v; }
        nm = Math.sqrt(nm);
        const R = (na + nb) / 2;
        let err = 0;
        for (let j = 0; j < mid.length; j++) err = Math.max(err, Math.abs(k[t.off + j] - (R * mid[j]) / nm));
        assert(err < 1e-4 * R + 1e-6, `${t.name}: max error ${err}`);
      });
    });

    test('DELLA with ε = 0 and density 1 equals TIES at density 1', () => {
      const d = MM.runMethod('della', ctx, Object.assign(params('della', 2), { density: 1, epsilon: 0 })).params;
      const t = MM.runMethod('ties', ctx, Object.assign(params('ties', 2), { density: 1 })).params;
      for (let i = 0; i < d.length; i++) assert(Math.abs(d[i] - t[i]) < 1e-6, `differs at ${i}`);
    });

    test('AdaMerging lowers prediction entropy; task-wise result = Task Arithmetic with the learned λs', () => {
      const rl = MM.runMethod('adamerging', ctx, Object.assign(params('adamerging', 2), { mode: 'layer' }));
      const cl = ctx.cache['ada:layer'];
      assert(cl.e1 < cl.e0 && cl.lam.length === 2 * T.layout.nLayers, `layer-wise entropy ${cl.e0} → ${cl.e1}`);
      assert(cl.lam.every((v) => v >= 0 && v <= 1.5), 'λ out of bounds');
      const rt = MM.runMethod('adamerging', ctx, Object.assign(params('adamerging', 2), { mode: 'task' }));
      const ct = ctx.cache['ada:task'];
      assert(ct.e1 < ct.e0 && ct.lam.length === 2, `task-wise entropy ${ct.e0} → ${ct.e1}`);
      const ta = MM.runMethod('task_arithmetic', ctx, Object.assign(params('task_arithmetic', 2), { lambda: 1, weights: [ct.lam[0], ct.lam[1], 1] })).params;
      for (let i = 0; i < ta.length; i++) assert(Math.abs(ta[i] - rt.params[i]) < 1e-5, `differs from task arithmetic at ${i}`);
      return `entropy ${cl.e0.toFixed(3)} → ${cl.e1.toFixed(3)}; λ ${cl.lam.map((v) => v.toFixed(2)).join(' ')}; ${pct(MM.scoreMethod(T, ctx, 'adamerging', params('adamerging', 2)).avg)}`;
    });

    test('SCE selects the requested fraction of entries', () => {
      const r = MM.runMethod('sce', ctx, Object.assign(params('sce', 2), { topk: 0.2 }));
      const sel = parseFloat(r.info.find((x) => /selected/.test(x.label)).value) / 100;
      assert(Math.abs(sel - 0.2) < 0.03, `selected ${sel}`);
      // top-k = 1 selects every entry with non-zero variance: entries where the experts agree exactly
      // (often both 0) carry no disagreement and are skipped, which is SCE's documented trade-off
      const all = MM.runMethod('sce', ctx, Object.assign(params('sce', 2), { topk: 1 }));
      const selAll = parseFloat(all.info.find((x) => /selected/.test(x.label)).value) / 100;
      const [ta, tb] = T.experts.map((e) => e.map((v, i) => v - T.base[i]));
      let varied = 0;
      for (let i = 0; i < ta.length; i++) if (ta[i] !== tb[i]) varied++;
      assert(Math.abs(selAll - varied / ta.length) < 0.002, `topk 1 selected ${selAll}, entries with variance ${varied / ta.length}`);
      return `top-k 0.2 → ${pct(sel)}; top-k 1 → ${pct(selAll)} (rest: experts identical)`;
    });

    test('recipeRuns + pairedDiff: a recipe against itself is "same"; different recipes pair up per seed', () => {
      const rec = { method: 'ties', params: params('ties', 2), align: false };
      const same = MM.pairedDiff(MM.recipeRuns([T, T], rec, rec), 'B', 'A');
      assert(same.verdict === 'same' && same.n === 2, JSON.stringify(same));
      const runs = MM.recipeRuns([T], rec, { method: 'regmean', params: params('regmean', 2), align: false });
      const direct = MM.scoreMethod(T, ctx, 'regmean', params('regmean', 2)).avg - MM.scoreMethod(T, ctx, 'ties', params('ties', 2)).avg;
      assert(Math.abs(runs[0].B.avg - runs[0].A.avg - direct) < 1e-12, 'paired difference mismatch');
    });

    test('Linear with weights [1, 0] reproduces Expert A', () => {
      const p = params('linear', T.K);
      p.weights = [1, 0];
      const r = MM.scoreMethod(T, ctx, 'linear', p);
      assert(Math.abs(r.accs[0] - T.expertAccs[0][0]) < 1e-9, `${pct(r.accs[0])} vs ${pct(T.expertAccs[0][0])}`);
    });

    test('compareResults: identical recipes agree everywhere', () => {
      const a = MM.scoreMethod(T, ctx, 'ties', params('ties', T.K));
      const b = MM.scoreMethod(T, ctx, 'ties', params('ties', T.K));
      const c = MM.compareResults(T.tasks, a, b);
      assert(c.disagree === 0 && c.dAvg === 0 && c.perTask.every((t) => t.dAcc === 0), JSON.stringify(c));
    });

    test('compareResults: deltas match accuracies, disagreement bounds the gap', () => {
      const a = MM.scoreMethod(T, ctx, 'linear', params('linear', T.K));
      const b = MM.scoreMethod(T, ctx, 'regmean', params('regmean', T.K));
      const c = MM.compareResults(T.tasks, a, b);
      assert(Math.abs(c.dAvg - (b.avg - a.avg)) < 1e-12, 'dAvg mismatch');
      c.perTask.forEach((t, i) => {
        assert(Math.abs(t.dAcc - (b.accs[i] - a.accs[i])) < 1e-12, `task ${i} dAcc mismatch`);
        // a point can only change correctness if the two models disagree on it
        assert(t.disagree + 1e-12 >= Math.abs(t.dAcc), `task ${i}: disagree ${t.disagree} < |Δacc| ${Math.abs(t.dAcc)}`);
      });
      return `Linear → RegMean ${pct(c.dAvg)} avg, differ on ${pct(c.disagree)}`;
    });

    test('recipeLabel names the knobs', () => {
      const p = params('ties', 2);
      const l = MM.recipeLabel('ties', p, true, 2);
      assert(l === 'TIES · k 0.20 · λ 1.00 · aligned', l);
      const w = params('linear', 2); w.weights = [1, 0.5, 1];
      assert(MM.recipeLabel('linear', w, false, 2) === 'Linear · w 1.00/0.50', MM.recipeLabel('linear', w, false, 2));
      assert(MM.recipeLabel('linear', params('linear', 2), false, 2) === 'Linear', 'equal weights should be omitted');
      const d = MM.recipeLabel('dare', params('dare', 2), false, 2);
      assert(d === 'DARE · p 0.50 · λ 1.00 · DARE-TIES', d);
    });

    // ---------- digits (image-like task) ----------
    const Td = MM.trainBundleSync(scenario('digits'));
    test('Digits: 64-pixel experts master their pairs; the base does not', () => {
      assert(Td.layout.inDim === 64 && Td.layout.sizes[0] === 64 && Td.tasks[0].train.d === 64, 'input size');
      Td.expertAccs.forEach((a, i) => assert(a[i] > 0.95, `expert ${i} only ${pct(a[i])}`));
      const baseAvg = Td.baseAccs.reduce((a, b) => a + b, 0) / Td.K;
      assert(baseAvg < 0.9, `base already ${pct(baseAvg)}`);
      return `experts ${Td.expertAccs.map((a, i) => pct(a[i])).join(' / ')}, base ${pct(baseAvg)}`;
    });
    test('Digits: every method runs; router and data-informed merges stay strong', () => {
      const c = MM.bundleCtx(Td, false);
      const got = {};
      MM.METHODS.forEach((m) => {
        const r = MM.scoreMethod(Td, c, m.id, params(m.id, Td.K));
        assert(!r.blocked && Number.isFinite(r.avg), `${m.id} failed`);
        got[m.id] = r.avg;
      });
      assert(got.moe > 0.95 && got.distill > 0.95 && got.regmean > 0.9, `moe ${pct(got.moe)}, distill ${pct(got.distill)}, regmean ${pct(got.regmean)}`);
      assert(MM.getRouter(c).acc > 0.95, 'router should tell 0/1 digits from 2/3 digits');
      return `RegMean ${pct(got.regmean)}, TIES ${pct(got.ties)}, Linear ${pct(got.linear)}`;
    });
    test('Digits: links round-trip and mixed kinds are rejected', () => {
      const base = { init: 'shared', hidden: 32, ftSteps: 250, seed: 7, align: false, method: 'ties', params: params('ties', 2), pin: null };
      const d = MM.decodeRecipe(MM.encodeRecipe(Object.assign({ experts: [{ task: 'd01' }, { task: 'd45', flip: true }] }, base)));
      assert(d.ok && JSON.stringify(d.recipe.experts) === '[{"task":"d01"},{"task":"d45","flip":true}]', JSON.stringify(d));
      const mixed = MM.decodeRecipe(MM.encodeRecipe(Object.assign({ experts: [{ task: 'd01' }, { task: 'moons', region: 'left' }] }, base)));
      assert(!mixed.ok, 'mixed digit/plane experts should be rejected');
    });

    // ---------- strangers ----------
    const Ts = MM.trainBundleSync(scenario('strangers'));
    test('Strangers: task-vector methods are blocked without a base', () => {
      const c = MM.bundleCtx(Ts, false);
      ['task_arithmetic', 'ties', 'dare', 'breadcrumbs'].forEach((id) => assert(MM.scoreMethod(Ts, c, id, params(id, Ts.K)).blocked, `${id} not blocked`));
    });
    test('Strangers: neuron alignment improves plain averaging', () => {
      const raw = MM.scoreMethod(Ts, MM.bundleCtx(Ts, false), 'linear', params('linear', Ts.K));
      const al = MM.scoreMethod(Ts, MM.bundleCtx(Ts, true), 'linear', params('linear', Ts.K));
      assert(al.avg > raw.avg + 0.02, `aligned ${pct(al.avg)} vs raw ${pct(raw.avg)}`);
      return `${pct(raw.avg)} → ${pct(al.avg)}`;
    });

    return results;
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = suite;
  else (window.MMTests = window.MMTests || {}).run = suite;
})();
