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

    // ---------- mergekit import ----------
    test('parseYAML: maps, lists of maps, inline lists, comments, quotes', () => {
      const y = MM.parseYAML(`# comment
merge_method: ties   # trailing comment
base_model: "org/base#1"
models:
  - model: a/one
    parameters:
      density: 0.5
      weight: [0, 0.5, 1]
  - model: b/two
parameters: {normalize: false, int8_mask: true}
dtype: bfloat16`);
      assert(y.merge_method === 'ties' && y.base_model === 'org/base#1', JSON.stringify(y));
      assert(y.models.length === 2 && y.models[0].parameters.density === 0.5 && y.models[0].parameters.weight.join() === '0,0.5,1' && y.models[1].model === 'b/two', JSON.stringify(y.models));
      assert(y.parameters.normalize === false && y.parameters.int8_mask === true && y.dtype === 'bfloat16', JSON.stringify(y.parameters));
    });
    test('importMergekit: canonical SLERP gradient, filters, DARE, TIES, passthrough, MoE, errors', () => {
      const sl = MM.importMergekit(`slices:
merge_method: slerp
base_model: a/one
models:
  - model: a/one
  - model: b/two
parameters:
  t: [0, 0.5, 0.3, 0.7, 1]
dtype: bfloat16`);
      assert(sl.ok && sl.method === 'slerp' && sl.params.layerwise && sl.params.grad.join() === '0,0.3,1', JSON.stringify(sl));
      const flt = MM.importMergekit(`merge_method: slerp
models:
  - model: a
  - model: b
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5`);
      assert(flt.ok && flt.params.t === 0.5 && !flt.params.layerwise && flt.warnings.some((w) => /filters/.test(w)), JSON.stringify(flt));
      const da = MM.importMergekit(`merge_method: dare_ties
base_model: base
models:
  - model: base
  - model: x
    parameters: {density: 0.53, weight: 0.4}
  - model: y
    parameters: {density: 0.53, weight: 0.4}
parameters:
  int8_mask: true`);
      assert(da.ok && da.method === 'dare' && da.K === 2 && Math.abs(da.params.drop - 0.47) < 1e-9 && da.params.lambda === 0.4 && da.params.mode === 'ties', JSON.stringify(da));
      assert(da.notes.some((n) => /base model listed/.test(n)), 'base listed under models should be dropped with a note');
      const ti = MM.importMergekit(`merge_method: ties
base_model: base
models:
  - model: x
    parameters: {density: 0.2, weight: 1.0}
  - model: y
    parameters: {density: 0.4, weight: 1.0}
parameters:
  normalize: true`);
      assert(ti.ok && Math.abs(ti.params.density - 0.3) < 1e-9 && ti.params.lambda === 1 && ti.warnings.some((w) => /density differs/.test(w)), JSON.stringify(ti));
      const pt = MM.importMergekit(`merge_method: passthrough
slices:
  - sources:
      - model: a
        layer_range: [0, 24]
  - sources:
      - model: b
        layer_range: [8, 32]`);
      assert(pt.ok && pt.method === 'stitch' && pt.params.layers.join() === '0,0,1', JSON.stringify(pt));
      const moe = MM.importMergekit(`base_model: a
gate_mode: hidden
experts_per_token: 2
experts:
  - source_model: a
    positive_prompts: ["chat"]
  - source_model: b
    positive_prompts: ["math"]`);
      assert(moe.ok && moe.method === 'moe' && moe.params.mode === 'soft' && moe.models.join() === 'a,b', JSON.stringify(moe));
      // base_model only counts for base-relative methods: SLERP's is an endpoint, mergekit-moe's is usually an expert
      assert(sl.base === null && moe.base === null && da.base === 'base' && ti.base === 'base', `bases: ${sl.base} / ${moe.base} / ${da.base} / ${ti.base}`);
      // passthrough keeps a known model order so Export → Import round-trips
      const ord = MM.importMergekit('merge_method: passthrough\nslices:\n  - sources:\n      - model: b\n        layer_range: [0, 11]\n  - sources:\n      - model: a\n        layer_range: [11, 22]\n  - sources:\n      - model: b\n        layer_range: [22, 32]', { order: ['a', 'b'] });
      assert(ord.models.join() === 'a,b' && ord.params.layers.join() === '1,0,1', JSON.stringify(ord));
      assert(!MM.importMergekit('merge_method: frankenstein\nmodels:\n  - model: a\n  - model: b').ok, 'unknown method accepted');
      assert(!MM.importMergekit('merge_method: linear\nmodels:\n  - model: a').ok, 'one model accepted');
      assert(!MM.importMergekit('just some text').ok, 'non-mapping accepted');
      const big = MM.importMergekit(`merge_method: ties
base_model: base
models:
  - model: x
    parameters: {density: 0.5, weight: 5}
  - model: y
    parameters: {density: 0.5, weight: 5}
parameters:
  normalize: false`);
      assert(big.ok && big.params.lambda === 2 && big.warnings.some((w) => /clamped/.test(w)), `λ should clamp to the lab range: ${JSON.stringify(big.params)}`);
    });

    // ---------- undo / redo ----------
    test('Undo key: pinning then nudging a knob within the window stays two undo steps', () => {
      const h = MM.makeHistory(100, 700), key = MM.editKeyTracker(), pin = { method: 'ties', params: { density: 0.2 }, align: false };
      const snap = (d, p, al) => JSON.stringify({ method: 'ties', density: d, pin: p, align: !!al });
      h.record(snap(0.2, null), 0, key('ties', false, null));
      h.record(snap(0.2, pin), 100, key('ties', false, pin));      // pin
      h.record(snap(0.23, pin), 300, key('ties', false, pin));     // knob, 200 ms later
      h.record(snap(0.25, pin), 400, key('ties', false, pin));     // same drag: coalesces
      h.record(snap(0.25, pin, true), 500, key('ties', true, pin)); // alignment toggled right after
      assert(h.size() === 4, `pin, drag and alignment are three steps: size ${h.size()}`);
      assert(h.undo() === snap(0.25, pin), 'undo removes only the alignment change');
      assert(h.undo() === snap(0.2, pin), 'next undo removes the knob drag and keeps the pin');
      assert(h.undo() === snap(0.2, null), 'last undo removes the pin');
    });
    test('recipeDiff: method, alignment, knobs, selects, weights, layers and per-layer values', () => {
      const P = (id, over) => Object.assign(params(id, 2), over || {});
      const R = (method, p, align) => ({ method, params: p, align: !!align });
      assert(MM.recipeDiff(R('ties', P('ties')), R('ties', P('ties')), 2).length === 0, 'identical recipes: no changes');
      const m = MM.recipeDiff(R('ties', P('ties')), R('dare', P('dare'), true), 2);
      assert(m.length === 2 && m[0].what === 'Method' && m[0].b === MM.METHOD_BY_ID.dare.name && m[1].what === 'Neuron alignment' && m[1].b === 'on', JSON.stringify(m));
      const k = MM.recipeDiff(R('ties', P('ties', { density: 0.2 })), R('ties', P('ties', { density: 0.35 })), 2);
      assert(k.length === 1 && k[0].a === '0.20' && k[0].b === '0.35' && !/\(/.test(k[0].what), JSON.stringify(k));
      const sel = MM.recipeDiff(R('dare', P('dare', { mode: 'ties' })), R('dare', P('dare', { mode: 'linear' })), 2);
      assert(sel.length === 1 && sel[0].b === 'DARE-Linear (sum)', JSON.stringify(sel));
      const w = MM.recipeDiff(R('linear', P('linear')), R('linear', P('linear', { weights: [1, 0.5, 1] })), 2);
      assert(w.length === 1 && w[0].what === 'Expert weights' && w[0].b === '1.00 / 0.50', JSON.stringify(w));
      const lw = MM.recipeDiff(R('ties', P('ties', { lambda: 1 })), R('ties', P('ties', { lambda: 1.5, layerwise: true, grad: [0.5, 1, 1.5] })), 2);
      assert(lw.length === 1 && lw[0].what === 'λ per layer' && lw[0].a === 'off' && lw[0].b === '0.50 → 1.00 → 1.50', `scalar λ hidden when per-layer: ${JSON.stringify(lw)}`);
      return `${k[0].what} ${k[0].a} → ${k[0].b}`;
    });
    test('Transfer snapshot: well-formed, every toy measured, and its headline still follows from its rows', () => {
      const snap = MM.TRANSFER_SNAPSHOT;
      assert(snap && snap.seeds === 5 && /^[0-9a-f]{64}$/.test(snap.engineHash), 'snapshot header');
      assert(snap.entries.length === MM.SCENARIOS.length && MM.SCENARIOS.every((sc) => snap.entries.some((e) => e.scenario === sc.id)), 'one entry per scenario');
      const kinds = new Set(['moves', 'narrow', 'transfers', 'holds']);
      snap.entries.forEach((e) => { assert(e.rows.length > 0, `${e.label}: rows`); e.rows.forEach((r) => assert(kinds.has(r.kind) && MM.METHODS.some((m) => m.short === r.method), `${e.label}: ${r.method} ${r.knob}`)); });
      const sum = MM.transferSummary(snap.entries);
      // the claim the Atlas leads with
      assert(sum.everywhere.length === 0, `expected no knob value to carry over on every toy, got ${sum.everywhere.join(', ')}`);
      assert(sum.split.length >= 10, `expected many knobs to carry over on some toys only, got ${sum.split.length}`);
      return `${snap.entries.length} toys, ${sum.keys.length} knobs: 0 carry over everywhere, ${sum.split.length} depend on the toy`;
    });
    test('gridTransfer: mean grid, per-seed best cells, moves vs holds, and which layer matters per seed', () => {
      const G = (bi, bj, rowSens) => [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === bi && j === bj ? 0.95 : 0.8 - (rowSens ? 0.05 * Math.abs(i - bi) : 0.05 * Math.abs(j - bj)))));
      const same = MM.gridTransfer([G(1, 1, true), G(1, 1, true), G(1, 1, true)]);
      assert(same.kind === 'holds' && same.regret === 0 && same.pick.best.i === 1 && same.pick.best.j === 1 && same.layer.every((x) => x === 'a'), JSON.stringify({ k: same.kind, l: same.layer }));
      const mv = MM.gridTransfer([G(0, 0, true), G(2, 2, false), G(0, 0, true)]);
      assert(mv.kind === 'moves' && mv.spread === 1 && mv.regret > 0.1, `${mv.kind} ${mv.spread} ${mv.regret}`);
      assert(mv.layer.join() === 'a,b,a', `layer per seed: ${mv.layer.join()}`);
      assert(Math.abs(mv.mean[0][0] - (0.95 + 0.7 + 0.95) / 3) < 1e-12, 'mean grid');
    });
    test('transferSummary: everywhere vs split', () => {
      const e = (label, kinds) => ({ label, rows: Object.entries(kinds).map(([knob, kind]) => ({ method: 'TIES', knob, kind })) });
      const s = MM.transferSummary([e('A', { λ: 'transfers', k: 'moves', ε: 'holds' }), e('B', { λ: 'holds', k: 'narrow', ε: 'moves' })]);
      assert(s.everywhere.join() === 'TIES|λ' && s.split.join() === 'TIES|ε' && s.keys.length === 3, JSON.stringify(s));
    });
    test('makeLRU: caps size, evicts least recently used, never evicts kept keys', () => {
      const c = MM.makeLRU(3, (k) => k === 'lab');
      ['lab', 'a', 'b'].forEach((k) => c.set(k, k));
      c.get('a'); // a is now most recent: b is the oldest evictable
      c.set('d', 'd');
      assert(c.size === 3 && c.has('lab') && c.has('a') && c.has('d') && !c.has('b') && c.evicted === 1, c.keys().join());
      c.set('e', 'e'); // oldest evictable is a (lab is kept even though it is older)
      assert(!c.has('a') && c.has('lab') && c.keys().join() === 'lab,d,e', c.keys().join());
      const k = MM.makeLRU(1, () => true);
      k.set('x', 1); k.set('y', 2);
      assert(k.size === 2, 'when everything is kept the cache grows instead of dropping kept entries');
    });
    test('makeHistory: push, coalesce drags, undo/redo, redo cleared by a new change, limit', () => {
      const h = MM.makeHistory(4, 500);
      h.record('a', 0, null);
      assert(!h.canUndo() && !h.record('a', 10, null), 'initial state; identical snapshot ignored');
      h.record('b1', 1000, 'k'); h.record('b2', 1200, 'k'); h.record('b3', 1400, 'k'); // one drag → one step
      assert(h.size() === 2, `drag should coalesce: size ${h.size()}`);
      h.record('c', 3000, 'k'); // same key but later → new step
      assert(h.undo() === 'b3' && h.undo() === 'a' && h.undo() === null, 'undo walks back');
      assert(h.redo() === 'b3' && h.canRedo(), 'redo walks forward');
      h.record('d', 5000, 'x');
      assert(!h.canRedo(), 'a new change clears redo');
      for (const s of ['e', 'f', 'g', 'h']) h.record(s, 9000 + s.charCodeAt(0) * 1000, s);
      assert(h.size() === 4, `limit: ${h.size()}`);
    });

    // ---------- recipe report ----------
    test('buildReport: sections, escaping, and nothing invented for checks that were not run', () => {
      const base = {
        title: 'TIES-Merging recipe report', date: '2026-11-22', link: 'https://x.io/#r=abc',
        recipe: { method: 'TIES-Merging', family: 'Task vectors', label: 'TIES · k 0.20 · λ 1.00', scenario: 'Two skills', experts: ['A: Moons', 'B: <script>alert(1)<\/script>'], lineage: 'shared base', aligned: false, width: 32, ftSteps: 250, seed: 7 },
        result: { tasks: [{ name: 'A · Moons', merged: 0.792, expert: 1, base: 0.794 }, { name: 'B | pipes', merged: 0.992, expert: 1, base: null }], avg: 0.892, retention: 0.892, cost: '1×', storage: '1×', warnings: [] },
        method: { tagline: 't', how: ['one', 'two'], pros: ['p'], cons: ['c'], realWorld: '', paper: 'Yadav et al.', oss: [{ name: 'mergekit', url: 'https://github.com/arcee-ai/mergekit' }] },
        export: { file: 'config.yml', yaml: 'merge_method: ties\n', cmd: 'mergekit-yaml config.yml ./merged', note: '' },
      };
      const a = MM.buildReport(base);
      assert(a.html.startsWith('<!doctype html>') && a.html.includes('79.2%') && a.html.includes('89.2%'), 'scorecard in html');
      assert(!a.html.includes('<script>alert') && a.html.includes('&lt;script&gt;'), 'html must escape user text');
      assert(a.markdown.includes('```yaml\nmerge_method: ties\n```') && a.markdown.includes('B \\| pipes'), 'markdown yaml block and escaped pipe');
      for (const t of ['How forgiving', 'Which layer', 'Is the winner real', 'Compared with the pinned', 'can be copied']) assert(!a.html.includes(t) && !a.markdown.includes(t), `"${t}" should be omitted when not run`);
      const tr = MM.buildReport(Object.assign({}, base, { transfer: { seeds: 5, knobs: 3, rows: [
        { knob: 'Density k', kind: 'moves', lo: '0.08', hi: '0.39', best: '0.33', regret: 0.024, now: '0.20', basis: 'recipe' },
        { knob: 'Scaling λ', kind: 'transfers', lo: '0.38', hi: '0.75', best: '0.63', regret: 0.007, now: '1.00', basis: 'defaults' }] } }));
      assert(tr.html.includes('Which knob values can be copied? (5 seeds)') && tr.html.includes('moves with the seed') && tr.html.includes('default settings'), 'transfer section in html');
      assert(tr.markdown.includes('| Density k | moves with the seed | 0.08 – 0.39 | 0.33 | 2.4 pts | 0.20 | this recipe |'), 'transfer row in markdown');
      assert(tr.markdown.includes('1 of 2 measured knobs should be tuned') && tr.markdown.includes('1 knob was not measured'), 'transfer note');
      assert(tr.markdown.includes('> **Before a real merge** (5-seed toy evidence): Tune on your own models: Density k (moves with the seed). Can be copied: Scaling λ 0.63. 1 knob not measured.'), 'before-a-real-merge summary in markdown');
      assert(tr.html.indexOf('Before a real merge') < tr.html.indexOf('Result on the toy'), 'summary comes first');
      assert(!a.html.includes('Before a real merge'), 'no summary without evidence');
      const pre = MM.buildReport(Object.assign({}, base, { transfer: { seeds: 5, knobs: 2, rows: [
        { knob: 'Density k', kind: 'moves', lo: '0.08', hi: '0.39', best: '0.33', regret: 0.024, now: '0.20', basis: 'snapshot' }] } }));
      assert(pre.markdown.includes('pre-measured (default settings)') && pre.markdown.includes('(5-seed toy evidence, pre-measured)'), 'snapshot rows are labelled as pre-measured');
      const ly = MM.buildReport(Object.assign({}, base, { layers: [{ pair: 'Hidden 1 × Output', layer: 'Hidden 1', kind: 'moves', cost: '2.5 pts', n: 5, sentence: 'λ per layer (Hidden 1 × Output, 5 seeds): Hidden 1 is the sensitive layer on every seed, so tune it first; the best cell moves with the seed (2.5 pts on the worst seed).' }] }));
      assert(ly.markdown.includes('> **Before a real merge** (5-seed toy evidence): λ per layer (Hidden 1 × Output'), 'layer evidence alone opens the callout');
      assert(ly.html.includes('Hidden 1 is the sensitive layer on every seed, so tune it first'), 'layer sentence in html');
      const b = MM.buildReport(Object.assign({}, base, {
        sensitivity: 'Best density 0.33.', layerMap: 'Hidden 1 matters.',
        robust: { seeds: 5, rows: [{ name: 'TIES-Merging', mean: 0.913, sd: 0.032, worst: 0.756, stability: 'some spread', current: true }], verdict: 'No clear winner.', h2h: 'leans ahead' },
        compare: { pinned: 'DARE · p 0.50', now: 'TIES · k 0.20', summary: 'B is better.', edge: 'No reliable difference.' },
      }));
      for (const t of ['How forgiving', 'Which layer', 'Is the winner real? (5 seeds)', 'Compared with the pinned', 'this recipe', '91.3% ± 3.2']) assert(b.html.includes(t), `missing "${t}"`);
      assert(!a.html.includes('pixel saliency') && !a.markdown.includes('pixel saliency'), 'no saliency section for plane toys');
      const c = MM.buildReport(Object.assign({}, base, { saliency: { mean: 0.6, tasks: [
        { name: 'A · Digits 0 vs 1', match: 0.35, merged: 0.98, expert: 1, expertPng: 'data:image/png;base64,AAAA', mergedPng: 'data:image/png;base64,BBBB' },
        { name: 'B · Digits 1 vs 7', match: 0.93, merged: 0.52, expert: 1 }] } }));
      assert(c.html.includes('Where the merge looks') && c.html.includes('35% match') && c.html.includes('base64,BBBB'), 'saliency section in html');
      assert(c.markdown.includes('| A · Digits 0 vs 1 | 35% |') && c.markdown.includes('conflict signature'), 'saliency table in markdown');
      assert(c.html.includes('5-seed check') && !c.markdown.includes('Across seeds'), 'unchecked: suggest the seed check, no seed column');
      const st = (counts, n) => { const top = Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0]; return { n, counts, top, agree: counts[top] / n, mean: 0.8, sd: 0.1, min: 0.6, max: 0.9 }; };
      const s = MM.buildReport(Object.assign({}, base, { saliency: { mean: 0.6, tasks: [
        { name: 'A · Digits 0 vs 1', match: 0.87, merged: 0.82, expert: 1, seeds: st({ lost: 3, conflict: 2 }, 5) },
        { name: 'B · Digits 1 vs 7', match: 0.35, merged: 0.98, expert: 1, seeds: st({ other: 5 }, 5) }] } }));
      assert(s.markdown.includes('Across seeds |') && s.markdown.includes('unstable across 5 seeds: skill lost 3/5, conflict signature 2/5'), 'seed column in markdown');
      assert(s.html.includes('On this seed, the merge looks at the right pixels') && s.html.includes('treat that as a guess'), 'unstable reading is softened');
      assert(s.html.includes('The same reading on all 5 seeds') && s.html.includes('This merge keeps the skill but reads different pixels'), 'stable reading stated plainly');
      assert(!s.html.includes('5-seed check'), 'no nudge once checked');
      assert(MM.seedStability(st({ same: 4, other: 1 }, 5)).level === 'mostly' && MM.seedStability(st({ same: 3, other: 2 }, 5)).level === 'unstable', '4/5 mostly, 3/5 unstable');
    });
    test('combineLayerEvidence: one sentence when every map agrees on the layer, one per map otherwise', () => {
      const it = (pair, layer, kind, regret) => ({ knob: 'λ', pair, layer, kind, regret, n: 5, sentence: `${pair}: ${layer || 'mixed'} ${kind}` });
      const same = MM.combineLayerEvidence([it('Hidden 1 × Hidden 2', 'Hidden 1', 'moves', 0.015), it('Hidden 1 × Output', 'Hidden 1', 'moves', 0.025)]);
      assert(same.length === 1 && same[0] === 'λ per layer (Hidden 1 × Hidden 2, Hidden 1 × Output; 5 seeds): Hidden 1 is the sensitive layer on every seed in both maps that include it, so tune it first; the best cell moves with the seed in both (up to 2.5 pts on the worst seed).', same[0]);
      const partial = MM.combineLayerEvidence([it('Hidden 1 × Hidden 2', 'Hidden 1', 'moves', 0.015), it('Hidden 1 × Output', 'Hidden 1', 'moves', 0.025), it('Hidden 2 × Output', null, 'moves', 0.018)]);
      assert(partial.length === 2 && /both maps that include it/.test(partial[0]) && partial[1] === 'Hidden 2 × Output: mixed moves', JSON.stringify(partial));
      const mixedKind = MM.combineLayerEvidence([it('A × B', 'Hidden 1', 'moves', 0.02), it('A × C', 'Hidden 1', 'holds', 0.002)]);
      assert(mixedKind.length === 1 && /moves with the seed in A × B, holds across seeds in A × C/.test(mixedKind[0]), mixedKind[0]);
      const differ = MM.combineLayerEvidence([it('A × B', 'Hidden 1', 'moves', 0.02), it('B × C', 'Output', 'moves', 0.02)]);
      assert(differ.length === 2, 'different layers stay separate');
      assert(MM.combineLayerEvidence([it('A × B', null, 'moves', 0.02), it('A × C', null, 'moves', 0.02)]).length === 2, 'no single layer: separate');
      assert(MM.combineLayerEvidence(null).length === 0 && MM.combineLayerEvidence([it('A × B', 'Hidden 1', 'holds', 0)]).length === 1, 'empty and single');
    });
    test('saliencyReading: the four match × accuracy cases', () => {
      const r = (m, a) => MM.saliencyReading(m, a, 1).tone + ':' + MM.saliencyReading(m, a, 1).text.split(' ').slice(0, 3).join(' ');
      assert(MM.saliencyReading(0.95, 0.97, 1).tone === 'good', r(0.95, 0.97));
      assert(/another way/.test(MM.saliencyReading(0.35, 0.98, 1).text), r(0.35, 0.98));
      assert(/conflict signature/.test(MM.saliencyReading(0.93, 0.52, 1).text), r(0.93, 0.52));
      assert(/lost both/.test(MM.saliencyReading(0.4, 0.5, 1).text), r(0.4, 0.5));
      assert(/router/.test(MM.saliencyReading(0.35, 0.98, 1, { router: true }).text) && !/router/.test(MM.saliencyReading(0.35, 0.98, 1).text), 'router named only when the merge has one');
    });

    // ---------- recipe advisor ----------
    const advise = (o) => MM.adviseRecipe(o);
    const top = (r, n) => r.recs.slice(0, n || 3).map((x) => x.id);
    test('Advisor without a shared base: the alignment note states what was measured', () => {
      const r = advise({ base: 'no' });
      assert(r.scenario === 'strangers' && advise({ base: 'no', goal: 'same' }).scenario === 'twins', 'same-skill runs without a base are checked on Twin runs');
      assert(r.notes.some((t) => /hit-or-miss/.test(t) && /same functions/.test(t)), r.notes.join(' | '));
      assert(MM.SCENARIOS.find((s) => s.id === 'strangers').desc.startsWith('Two different skills'), 'Strangers describes different skills');
    });
    test('Advisor imbalance question: DELLA first, output-layer caveat, Hard + easy toy; nothing changes by default', () => {
      const d = advise({}), u = advise({ imbalance: 'unknown' }), n = advise({ imbalance: 'no' });
      assert(JSON.stringify(top(d, 5)) === JSON.stringify(top(u, 5)) && JSON.stringify(top(d, 5)) === JSON.stringify(top(n, 5)) && d.scenario === 'two', 'unchecked or balanced: same advice');
      const y = advise({ imbalance: 'yes' });
      assert(y.recs[0].id === 'della' && y.recs[0].reasons.some((t) => /9 of 10 seeds/.test(t)), `imbalance should put DELLA first, got ${top(y, 3)}`);
      assert(y.recs.find((x) => x.id === 'dare').caveats.some((t) => /hard skill suffers/.test(t)), 'DARE caveat');
      assert(y.scenario === 'hard', `evidence toy should be Hard + easy, got ${y.scenario}`);
      const ties = y.recs.find((x) => x.id === 'ties');
      assert(ties && ties.caveats.some((c) => /output-layer λ at or below 1/.test(c)) && y.notes.some((t) => /output layer is where merges break/.test(t)), 'caveat and note');
      assert(advise({ imbalance: 'yes', base: 'no' }).scenario === 'strangers' && advise({ imbalance: 'yes', count: '3+' }).scenario === 'three', 'other situations keep their toy');
    });
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

    test('Advisor evidence toy: digits map to the digit scenarios with the right init', () => {
      const d = advise({ toy: 'digits' });
      assert(d.scenario === 'digits' && d.init === 'shared' && !d.align, JSON.stringify([d.scenario, d.init, d.align]));
      const c = advise({ toy: 'digits', conflict: 'yes' });
      assert(c.scenario === 'digits-conflict', c.scenario);
      const n = advise({ toy: 'digits', base: 'no' });
      assert(n.scenario === 'digits' && n.init === 'independent' && n.align, JSON.stringify([n.scenario, n.init, n.align]));
      const p = advise({});
      assert(p.scenario === 'two' && p.init === 'shared', 'plane default unchanged');
      assert(advise({ base: 'no' }).init === 'independent' && advise({ toy: 'digits', count: '3+' }).notes.some((x) => /two experts/.test(x)), 'init / note');
      // the rules (ranking) don't depend on the evidence toy
      assert(advise({ toy: 'digits' }).recs.map((x) => x.id).join() === p.recs.map((x) => x.id).join(), 'toy must not change the rule ranking');
    });

    test('Advisor "both toys": two toys, and promotion only when both toys agree', () => {
      const b = advise({ toy: 'both' });
      assert(b.toys.length === 2 && b.toys[0].scenario === 'two' && b.toys[1].scenario === 'digits' && b.scenario === 'two', JSON.stringify(b.toys));
      const nb = advise({ toy: 'both', base: 'no' });
      assert(nb.toys.map((t) => `${t.scenario}:${t.init}`).join() === 'strangers:independent,digits:independent', JSON.stringify(nb.toys));
      const r = (xs) => xs.map((avg) => ({ avg }));
      const ids = ['a', 'b', 'c'];
      const promoteB = MM.blendRanking(ids, { a: r([0.8, 0.8, 0.8]), b: r([0.9, 0.9, 0.9]), c: r([0.7, 0.7, 0.7]) });
      const keepA = MM.blendRanking(ids, { a: r([0.9, 0.9, 0.9]), b: r([0.8, 0.8, 0.8]), c: r([0.7, 0.7, 0.7]) });
      const both = MM.combineToyBlends(ids, [promoteB, promoteB]);
      assert(both.order.join() === 'b,a,c' && both.promoted.id === 'b' && both.agree, JSON.stringify(both));
      const split = MM.combineToyBlends(ids, [promoteB, keepA]);
      assert(split.order.join() === 'a,b,c' && !split.promoted && !split.agree && split.best.join() === 'b,a', JSON.stringify(split));
      // a near-tie on one toy is not a disagreement: a is best on toy 1 and within noise of b on toy 2
      const noisyB = MM.blendRanking(ids, { a: r([0.80, 0.90, 0.85]), b: r([0.88, 0.86, 0.87]), c: r([0.5, 0.5, 0.5]) });
      const tie = MM.combineToyBlends(ids, [keepA, noisyB]);
      assert(tie.agree && tie.common === 'a' && tie.order.join() === 'a,b,c', JSON.stringify(tie));
      const one = MM.combineToyBlends(ids, [keepA]);
      assert(one.order.join() === 'a,b,c' && one.agree, 'single toy passes through');
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

    test('attributeChanges: one change at a time from A, deltas plus interaction add up to the total', () => {
      const P = (id, over) => Object.assign(params(id, 2), over || {});
      const A = { method: 'ties', params: P('ties', { density: 0.2, lambda: 1 }), align: false };
      const B = { method: 'ties', params: P('ties', { density: 0.4, lambda: 1.4 }), align: false };
      // a fake score: density helps +10 pts per 0.2, λ hurts 5 pts per 0.4, plus an interaction term
      const fake = (r) => 0.5 + (r.params.density - 0.2) * 0.5 - (r.params.lambda - 1) * 0.125 + (r.params.density > 0.3 && r.params.lambda > 1.2 ? 0.02 : 0);
      const at = MM.attributeChanges(A, B, 2, fake);
      assert(at.parts.length === 2 && at.parts[0].what === 'Density k' && Math.abs(at.parts[0].delta - 0.1) < 1e-9 && Math.abs(at.parts[1].delta + 0.05) < 1e-9, JSON.stringify(at.parts));
      assert(Math.abs(at.interaction - 0.02) < 1e-9 && Math.abs(at.parts[0].delta + at.parts[1].delta + at.interaction - at.total) < 1e-12, 'interaction closes the sum');
      assert(MM.attributeChanges(A, { ...A, params: P('ties', { density: 0.4, lambda: 1 }) }, 2, fake) === null, 'a single change needs no attribution');
      // real scores on the trained toy: method + alignment
      const real = (r) => { const x = MM.scoreMethod(T, MM.bundleCtx(T, r.align), r.method, r.params); return x.blocked ? null : x.avg; };
      const rt = MM.attributeChanges({ method: 'linear', params: P('linear'), align: false }, { method: 'ties', params: P('ties'), align: true }, 2, real);
      assert(rt && rt.parts.length === 2 && rt.parts.every((p) => Number.isFinite(p.delta)), JSON.stringify(rt));
      // the per-layer change carries the scalar knob too, so it isn't lost to "interaction"
      const lwB = { method: 'ties', params: P('ties', { lambda: 1.5, layerwise: true, grad: [0.5, 1, 1.5] }), align: true };
      const step = MM.recipeDiff(A, lwB, 2).find((x) => x.what === 'λ per layer').apply(A);
      assert(step.params.layerwise === true && step.params.lambda === 1.5 && step.params.grad.join() === '0.5,1,1.5' && step.align === false, JSON.stringify(step));
      return `Linear → TIES+aligned: method ${(rt.parts[0].delta * 100).toFixed(1)}, alignment ${(rt.parts[1].delta * 100).toFixed(1)}, interaction ${(rt.interaction * 100).toFixed(1)} pts`;
    });
    test('summarizeAttributions: mean ± sd per change, sign agreement, flat and mixed', () => {
      const mk = (d1, d2, total) => ({ total, interaction: total - d1 - (d2 == null ? 0 : d2), parts: [{ what: 'k', a: '0.2', b: '0.3', delta: d1 }, { what: 'λ', a: '1', b: '1.3', delta: d2 }] });
      const s = MM.summarizeAttributions([mk(0.01, -0.02, -0.01), mk(0.02, 0.001, 0.02), mk(0.03, -0.001, 0.03), null]);
      assert(s.n === 3 && Math.abs(s.parts[0].mean - 0.02) < 1e-12 && Math.abs(s.parts[0].sd - 0.01) < 1e-12 && s.parts[0].sign === 'up' && s.parts[0].up === 3, JSON.stringify(s.parts[0]));
      assert(s.parts[1].sign === 'mixed' && s.parts[1].down === 1 && s.parts[1].flat === 2, JSON.stringify(s.parts[1]));
      assert(MM.summarizeAttributions([mk(0.001, null, 0), mk(-0.001, null, 0)]).parts[0].sign === 'flat', 'all within 0.2 pts is flat');
      assert(MM.summarizeAttributions([mk(0.01, null, 0)]).parts[1].sign === 'none', 'never runnable alone');
      assert(Math.abs(s.total.mean - 0.04 / 3) < 1e-12, 'total mean');
    });
    test('seedBests: per-seed best values, spread across the range, and regret of the recommended value', () => {
      const xs = [0, 0.25, 0.5, 0.75, 1];
      const same = MM.seedBests(xs, [[0.8, 0.9, 0.95, 0.9, 0.8], [0.79, 0.88, 0.93, 0.9, 0.81]], 0.5);
      assert(!same.moves && same.spread === 0 && same.bests.every((b) => b.x === 0.5) && same.regret === 0, JSON.stringify(same));
      const moving = MM.seedBests(xs, [[0.9, 0.85, 0.8, 0.8, 0.8], [0.8, 0.8, 0.82, 0.85, 0.95], [0.8, 0.9, 0.85, 0.8, 0.8]], 0.25);
      assert(moving.moves && moving.spread === 1 && moving.bests.map((b) => b.x).join() === '0,1,0.25', JSON.stringify(moving.bests));
      assert(Math.abs(moving.regret - 0.15) < 1e-12, `worst seed loses 15 pts at 0.25: ${moving.regret}`);
    });
    test('knobTransfer: moves / transfers / holds', () => {
      const xs = [0, 0.25, 0.5, 0.75, 1];
      assert(MM.knobTransfer(xs, [[0.8, 0.9, 0.95, 0.9, 0.8], [0.79, 0.88, 0.93, 0.9, 0.81]]).kind === 'holds', 'same peak');
      const flat = MM.knobTransfer(xs, [[0.9, 0.905, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9, 0.906]]);
      assert(flat.kind === 'transfers' && flat.regret < 0.01, JSON.stringify(flat));
      const mv = MM.knobTransfer(xs, [[0.95, 0.85, 0.8, 0.8, 0.8], [0.8, 0.8, 0.82, 0.85, 0.95], [0.8, 0.9, 0.85, 0.8, 0.8]]);
      assert(mv.kind === 'moves' && mv.n === 3 && mv.bests.join() === '0,1,0.25', JSON.stringify(mv));
      // same spot on every seed, but a cliff next to it: the small shift costs a lot
      const x9 = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1];
      const peak = (i) => x9.map((_, j) => (j === i ? 0.95 : 0.6));
      const nr = MM.knobTransfer(x9, [peak(4), peak(5), peak(4)]);
      assert(nr.kind === 'narrow' && nr.regret > 0.3 && nr.spread <= 0.15, JSON.stringify(nr));
    });
    test('Which layer first: Hidden 1 swings most on the test seed for every method with a per-layer tip', () => {
      const out = [];
      for (const id of Object.keys(MM.LAYER_FIRST)) {
        const m = MM.METHOD_BY_ID[id], g = m.gradient, e = MM.LAYER_FIRST[id];
        assert(g && e.h1 > e.n / 2 && e.h1 <= e.n && MM.layerFirstTip(id).includes(`${e.h1} of ${e.n}`), `${id}: table entry`);
        const vals = Array.from({ length: 7 }, (_, i) => +(g.min + ((g.max - g.min) * i) / 6).toFixed(4));
        const ctx = MM.bundleCtx(T, false), base = Object.assign(params(id, 2), { layerwise: true });
        const sw = [0, 0, 0];
        for (const [a, b] of [[0, 1], [0, 2], [1, 2]]) { const q = MM.pickGrid(MM.layerGrid(T, ctx, id, base, a, b, vals)); sw[a] += q.sensA / 2; sw[b] += q.sensB / 2; }
        assert(sw[0] > sw[1] && sw[0] > sw[2], `${id}: Hidden 1 should swing most on Two skills seed 7, got ${sw.map((v) => (v * 100).toFixed(1)).join(' / ')}`);
        out.push(`${m.short} ${(sw[0] * 100).toFixed(0)} pts`);
      }
      assert(MM.layerFirstTip('regmean') === null, 'no tip without measurements');
      return `Hidden 1 swing: ${out.join(', ')}`;
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
      const rl = MM.runMethod('adamerging', ctx, Object.assign(params('adamerging', 2), { mode: 'layer', objective: 'entropy' }));
      const cl = ctx.cache['ada:layer:entropy'];
      assert(cl.e1 < cl.e0 && cl.lam.length === 2 * T.layout.nLayers, `layer-wise entropy ${cl.e0} → ${cl.e1}`);
      assert(cl.lam.every((v) => v >= 0 && v <= 1.5), 'λ out of bounds');
      const rt = MM.runMethod('adamerging', ctx, Object.assign(params('adamerging', 2), { mode: 'task', objective: 'entropy' }));
      const ct = ctx.cache['ada:task:entropy'];
      assert(ct.e1 < ct.e0 && ct.lam.length === 2, `task-wise entropy ${ct.e0} → ${ct.e1}`);
      const ta = MM.runMethod('task_arithmetic', ctx, Object.assign(params('task_arithmetic', 2), { lambda: 1, weights: [ct.lam[0], ct.lam[1], 1] })).params;
      for (let i = 0; i < ta.length; i++) assert(Math.abs(ta[i] - rt.params[i]) < 1e-5, `differs from task arithmetic at ${i}`);
      return `entropy ${cl.e0.toFixed(3)} → ${cl.e1.toFixed(3)}; λ ${cl.lam.map((v) => v.toFixed(2)).join(' ')}; ${pct(MM.scoreMethod(T, ctx, 'adamerging', params('adamerging', 2)).avg)}`;
    });

    test('AdaMerging anchored and expert-average objectives lower their losses; anchoring keeps λ closer to λ₀', () => {
      for (const obj of ['anchored', 'ensemble']) {
        MM.runMethod('adamerging', ctx, Object.assign(params('adamerging', 2), { mode: 'layer', objective: obj }));
        const c = ctx.cache[`ada:layer:${obj}`];
        assert(c && c.e1 < c.e0 && c.lam.every((v) => v >= 0 && v <= 1.5), `${obj}: ${JSON.stringify(c && { e0: c.e0, e1: c.e1 })}`);
      }
      const dev = (k) => ctx.cache[k].lam.reduce((a, v) => a + (v - 0.3) ** 2, 0);
      assert(dev('ada:layer:anchored') <= dev('ada:layer:entropy') + 1e-9, 'anchored λ should stay at least as close to 0.3 as plain entropy');
      return `defaults: ${JSON.stringify(params('adamerging', 2))} → ${pct(MM.scoreMethod(T, ctx, 'adamerging', params('adamerging', 2)).avg)}`;
    });

    test('Localize-and-Stitch: masks keep the requested fraction; outside them θ = base; overlaps are averaged', () => {
      for (const mode of ['dataless', 'data']) {
        const pp = Object.assign(params('localize_stitch', 2), { density: 0.1, mode });
        const r = MM.runMethod('localize_stitch', ctx, pp);
        const { masks } = ctx.cache[`ls:${mode}:0.1`];
        const taus = T.experts.map((e) => e.map((v, i) => v - T.base[i]));
        masks.forEach((mk) => { const f = mk.reduce((a, b) => a + b, 0) / mk.length; assert(Math.abs(f - 0.1) < 0.01, `${mode} mask density ${f}`); });
        for (let j = 0; j < T.base.length; j++) {
          const inA = masks[0][j], inB = masks[1][j];
          const want = T.base[j] + (inA && inB ? (taus[0][j] + taus[1][j]) / 2 : inA ? taus[0][j] : inB ? taus[1][j] : 0);
          assert(Math.abs(r.params[j] - want) < 1e-5, `${mode}: entry ${j} is ${r.params[j]}, expected ${want}`);
        }
      }
      return ['data', 'dataless'].map((mode) => `${mode} ${pct(MM.scoreMethod(T, ctx, 'localize_stitch', Object.assign(params('localize_stitch', 2), { mode })).avg)}`).join(', ');
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

    test('layerGrid/pickGrid: grid corners match Layer Stitch; best and sensitivity are sane', () => {
      const p = Object.assign(params('linear', 2), { layerwise: true, grad: [0, 0, 1] });
      const vals = [0, 0.5, 1];
      const M = MM.layerGrid(T, ctx, 'linear', p, 0, 2, vals); // rows: hidden 1, cols: output; hidden 2 held at 0 (A)
      const stitch = (layers) => MM.scoreMethod(T, ctx, 'stitch', Object.assign(params('stitch', 2), { layers })).avg;
      assert(Math.abs(M[0][2] - stitch([0, 0, 1])) < 1e-9, 'corner (A, ·, B) should equal Stitch A·A·B');
      assert(Math.abs(M[2][0] - stitch([1, 0, 0])) < 1e-9, 'corner (B, ·, A) should equal Stitch B·A·A');
      const g = MM.pickGrid(M);
      assert(g.best.v === Math.max(...M.flat()) && g.safe.lower <= g.safe.v && g.near > 0 && g.near <= 1, JSON.stringify(g));
      assert(g.safe.i === 1 && g.safe.j === 1, 'on a 3×3 grid the only interior cell is the safe one');
      const toy = MM.pickGrid([[0.5, 0.9], [0.5, 0.9]]); // only the column (axis b) matters
      assert(toy.sensB > 0.3 && toy.sensA === 0, `axis sensitivity ${toy.sensA} / ${toy.sensB}`);
      return `best H1=${vals[g.best.i]} Out=${vals[g.best.j]} → ${pct(g.best.v)}`;
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
    test('Digit saliency: non-negative, matches itself, zero for a constant model; conflict scenario defined', () => {
      const t = Td.tasks[0];
      const ex = MM.digitSaliency((X, n) => MM.predictProb(Td.layout, Td.experts[0], X, n), t);
      assert(ex.length === 64 && ex.every((v) => v >= 0) && ex.reduce((a, b) => a + b, 0) > 0, 'expert saliency');
      assert(Math.abs(MM.mapCosine(ex, ex) - 1) < 1e-9, 'self match should be 1');
      const flat = MM.digitSaliency((X, n) => new Float32Array(n).fill(0.7), t);
      assert(flat.every((v) => v === 0), 'a constant model has zero saliency');
      const sc = MM.SCENARIOS.find((s) => s.id === 'digits-conflict');
      assert(sc && sc.experts.map((e) => e.task).join() === 'd01,d17', 'digits-conflict scenario');
      // the conflict is real: "1" is class 1 for A (0 vs 1) and class 0 for B (1 vs 7)
      assert(MM.TASKS.d01.pair[1] === 1 && MM.TASKS.d17.pair[0] === 1, 'both tasks contain the digit 1 on opposite sides');
      const mg = MM.digitSaliency(MM.scoreMethod(Td, MM.bundleCtx(Td, false), 'linear', params('linear', 2)).predict, t);
      return `Linear keeps ${Math.round(MM.mapCosine(ex, mg) * 100)}% of expert A's pixel pattern`;
    });
    test('Arena saliency strip: experts match themselves, merges give one finite match per task', () => {
      const c = MM.bundleCtx(Td, false), ex = MM.expertSaliency(c);
      assert(ex.length === Td.K && ex.every((m) => m.length === 64), 'one 8×8 map per task');
      const self0 = MM.saliencyRow(c, (X, n) => MM.predictProb(c.layout, c.experts[0], X, n), ex);
      assert(Math.abs(self0.match[0] - 1) < 1e-9, 'expert A matches its own map');
      const out = [];
      for (const id of ['linear', 'ties', 'ensemble']) {
        const r = MM.saliencyRow(c, MM.scoreMethod(Td, c, id, params(id, 2)).predict, ex);
        assert(r.match.length === Td.K && r.match.every((v) => Number.isFinite(v) && v >= -1 && v <= 1 + 1e-9), `${id} matches`);
        assert(Math.abs(r.mean - r.match.reduce((a, b) => a + b, 0) / Td.K) < 1e-12, `${id} mean`);
        out.push(`${id} ${Math.round(r.mean * 100)}%`);
      }
      return out.join(', ');
    });
    test('Saliency over seeds: one run per seed, and a summary that flags unstable readings', () => {
      const run = MM.saliencyRun(Td, 'moe', params('moe', 2), false);
      assert(run && run.match.length === Td.K && run.router === true && run.merged.length === Td.K, 'one seed run');
      const steady = MM.saliencySeedSummary([run, run, run]);
      assert(steady.every((t) => t.agree === 1 && t.n === 3 && t.sd === 0), 'identical seeds agree with no spread');
      const mk = (m) => ({ match: [m, 0.9], merged: [0.98, 0.5], expert: [1, 1], router: false });
      const mixed = MM.saliencySeedSummary([mk(0.95), mk(0.6), mk(0.62), null]);
      assert(mixed[0].n === 3 && mixed[0].top === 'other' && Math.abs(mixed[0].agree - 2 / 3) < 1e-9, `task A: ${JSON.stringify(mixed[0].counts)}`);
      assert(mixed[0].min === 0.6 && mixed[0].max === 0.95 && mixed[0].sd > 0.15, 'range and spread');
      assert(mixed[1].top === 'conflict' && mixed[1].agree === 1, 'task B: conflict on every seed');
      return `MoE seed-7 matches ${run.match.map((m) => Math.round(m * 100) + '%').join(' / ')}`;
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

    // ---------- recipe gallery: every lesson must still be true ----------
    test('Gallery: every card says how many seeds it was measured on, and CI checks at least one', () => {
      for (const g of MM.GALLERY) {
        assert(g.measured && g.measured.seeds >= 3, `${g.id}: measured on at least 3 seeds`);
        const b = MM.evidenceBadge(g);
        assert(b.startsWith(`Measured on ${g.measured.seeds} seeds`) && /CI re-checks (seed 7|\d+ seeds)/.test(b), `${g.id}: ${b}`);
      }
      return MM.GALLERY.map((g) => `${g.id}: ${g.measured.seeds}`).join(', ');
    });
    test('checkLessonSeeds: mean lead and two-thirds wins', () => {
      const g = { A: {}, B: {}, expect: { kind: 'b-beats-a', meanBy: 0.05 } };
      const fakeCheck = MM.checkLesson;
      const runs = [[0.8, 0.9], [0.8, 0.95], [0.85, 0.83]];
      let k = 0;
      MM.checkLesson = () => { const [a, b] = runs[k++]; return { a, b }; };
      try {
        const r = MM.checkLessonSeeds(g, [1, 2, 3]);
        assert(r.ok && r.wins === 2 && Math.abs(r.mean - (0.1 + 0.15 - 0.02) / 3) < 1e-12, JSON.stringify(r));
        k = 0; runs[1] = [0.8, 0.79];
        assert(!MM.checkLessonSeeds(g, [1, 2, 3]).ok, 'one win of three fails');
      } finally { MM.checkLesson = fakeCheck; }
    });
    test('Gallery: every card encodes to a valid link, and its lesson holds on seed 7', () => {
      const bundles = { two: T, strangers: Ts, digits: Td };
      const out = [];
      for (const g of MM.GALLERY) {
        const d = MM.decodeRecipe(MM.encodeRecipe(MM.galleryRecipe(g)));
        assert(d.ok && d.recipe.method === g.B.method && d.recipe.pin && d.recipe.pin.method === g.A.method, `${g.id}: link round-trip`);
        if (!bundles[g.scenario]) bundles[g.scenario] = MM.trainBundleSync(scenario(g.scenario));
        const c = MM.checkLesson(g, bundles[g.scenario]);
        assert(c.ok, `${g.id}: lesson no longer holds (A ${pct(c.a)}, B ${pct(c.b)}, experts ${pct(c.experts)}${c.change !== undefined ? `, change alone ${c.change == null ? 'n/a' : pct(c.change)}` : ''})`);
        let multi = '';
        if (g.expect.seeds) {
          // lessons whose test seed sits near the low end are also checked on several seeds
          const bs = g.expect.seeds.map((sd) => (sd === 7 ? bundles[g.scenario] : MM.trainBundleSync(MM.toyConfig(g.scenario, sd))));
          const m = MM.checkLessonSeeds(g, bs);
          assert(m.ok, `${g.id}: multi-seed lesson no longer holds (mean ${(m.mean * 100).toFixed(1)} pts, wins ${m.wins}/${m.n}: ${m.diffs.map((x) => (x * 100).toFixed(1)).join(', ')})`);
          multi = ` (${m.n} seeds: mean +${(m.mean * 100).toFixed(1)} pts, wins ${m.wins}/${m.n})`;
        }
        out.push(`${g.id} A ${pct(c.a)} → B ${pct(c.b)}${c.change != null ? ` (${g.expect.what} alone ${c.change >= 0 ? '+' : ''}${(c.change * 100).toFixed(1)} pts)` : ''}${c.ratio != null ? ` (layer swing ×${c.ratio.toFixed(1)})` : ''}${multi}`);
      }
      return out.join('; ');
    });

    return results;
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = suite;
  else (window.MMTests = window.MMTests || {}).run = suite;
})();
