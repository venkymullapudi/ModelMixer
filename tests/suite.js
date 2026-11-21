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
      assert(l === 'TIES · k 0.30 · λ 1.00 · aligned', l);
      const w = params('linear', 2); w.weights = [1, 0.5, 1];
      assert(MM.recipeLabel('linear', w, false, 2) === 'Linear · w 1.00/0.50', MM.recipeLabel('linear', w, false, 2));
      assert(MM.recipeLabel('linear', params('linear', 2), false, 2) === 'Linear', 'equal weights should be omitted');
      const d = MM.recipeLabel('dare', params('dare', 2), false, 2);
      assert(d === 'DARE · p 0.50 · λ 1.00 · DARE-TIES', d);
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
