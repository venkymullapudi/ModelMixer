/* ModelMixer — recipe gallery: curated A-vs-B examples, each with a one-line lesson.
 * Every lesson was checked on 3 seeds and is guarded by a test (tests/suite.js) via `expect`, so a card can't
 * quietly start teaching something false if the engine or a default changes. */
(function () {
  const root = typeof window !== 'undefined' ? window : globalThis;
  const MM = (root.MM = root.MM || {});

  const p = (id, over) => Object.assign(MM.defaultParams(MM.METHOD_BY_ID[id], 2), over || {});

  const GALLERY = [
    {
      id: 'slerp-gradient', title: 'One t per layer beats one global t',
      lesson: 'SLERP with a per-layer t (a mergekit gradient) lets early layers lean to one expert and late layers to the other. On the toy it beats a flat t = 0.5 by a wide margin.',
      scenario: 'two',
      A: { method: 'slerp', params: p('slerp', { t: 0.5 }), align: false },
      B: { method: 'slerp', params: p('slerp', { layerwise: true, grad: [0.25, 0.5, 0.75] }), align: false },
      expect: { kind: 'b-beats-a', by: 0.03 },
    },
    {
      id: 'regmean-data', title: 'A little data goes a long way',
      lesson: 'RegMean only needs layer inputs from a few hundred examples per task and no training, and it keeps both skills almost perfectly where task arithmetic loses ground.',
      scenario: 'two',
      A: { method: 'task_arithmetic', params: p('task_arithmetic'), align: false },
      B: { method: 'regmean', params: p('regmean'), align: false },
      expect: { kind: 'b-beats-a', by: 0.05 },
    },
    {
      id: 'align-strangers', title: 'Align before you average strangers',
      lesson: 'Models trained from different random inits store the same features in different neurons. Plain averaging scrambles them; Git Re-Basin alignment re-orders neurons first and recovers accuracy.',
      scenario: 'strangers',
      A: { method: 'linear', params: p('linear'), align: false },
      B: { method: 'linear', params: p('linear'), align: true },
      expect: { kind: 'b-beats-a', by: 0.005 },
    },
    {
      id: 'adamerging-images', title: 'On image inputs, learn the coefficients',
      lesson: 'On the image-like Digits toy, AdaMerging tunes task-arithmetic coefficients on unlabeled inputs and clearly beats hand-set task arithmetic, in line with the paper’s vision results.',
      scenario: 'digits',
      A: { method: 'task_arithmetic', params: p('task_arithmetic'), align: false },
      B: { method: 'adamerging', params: p('adamerging'), align: false },
      expect: { kind: 'b-beats-a', by: 0.03 },
    },
    {
      id: 'ties-lambda-first', title: 'On TIES, tune λ before density',
      lesson: 'Going from k 0.30, λ 1.4 to k 0.20, λ 0.8, the λ change alone helped on all 7 seeds we tried (+0.6 to +4.5 pts), while the density change alone went up on 3 seeds and down on 4 (the best density differs by seed). Scaling λ is the knob that transfers; check density on your own data.',
      hint: 'In the Lab, press “Which change mattered?”, then “Repeat on 5 seeds”.',
      scenario: 'two',
      A: { method: 'ties', params: p('ties', { density: 0.3, lambda: 1.4 }), align: false },
      B: { method: 'ties', params: p('ties', { density: 0.2, lambda: 0.8 }), align: false },
      expect: { kind: 'change-helps', what: 'Scaling λ', by: 0.01 },
    },
    {
      id: 'ties-which-layer', title: 'Which layer, not which value',
      lesson: 'With a λ per layer, TIES on Two skills is sensitive to Hidden 1 on all 7 seeds we tried (a 14–27 pt swing, against at most 9.5 for the other layers). But the best Hidden 1 value is not stable: λ 0.65 there helped on 5 of the 7 seeds and hurt on 2. Across 6 toys × 5 seeds, Hidden 1 was the most sensitive layer in 26 of 30 cases; the 4 exceptions were Output (on Three skills, Overlapping skills and Hard + easy). So start tuning at Hidden 1, map the layers on your own models, and don’t copy the number.',
      hint: 'In the Lab’s per-layer map, turn on “± 5 seeds”: the best cell moves, the sensitive layer doesn’t.',
      scenario: 'two',
      A: { method: 'ties', params: p('ties', { layerwise: true, grad: [1, 1, 1] }), align: false },
      B: { method: 'ties', params: p('ties', { layerwise: true, grad: [0.65, 1, 1] }), align: false },
      expect: { kind: 'layer-sensitive', layer: 0, ratio: 2 },
    },
    {
      id: 'imbalance-output', title: 'Imbalanced skills: go easy on the output layer',
      lesson: 'On Hard + easy the hard expert changes the output layer 1.5–8× more than the easy one (on Two skills the two are within 0.4–1.4× of each other). Doubling the output λ of a task-arithmetic merge then collapsed it on 3 of 5 seeds (the easy skill fell to 1% on one) and cost points on a 4th; on Two skills it never collapsed. You can spot this before merging: compare each expert’s per-layer update size.',
      hint: 'In the Lab, the insight panel’s “Update size per layer” table shows the imbalance and warns about it.',
      scenario: 'hard',
      A: { method: 'task_arithmetic', params: p('task_arithmetic', { layerwise: true, grad: [1, 1, 2] }), align: false },
      B: { method: 'task_arithmetic', params: p('task_arithmetic', { layerwise: true, grad: [1, 1, 1] }), align: false },
      expect: { kind: 'b-beats-a', by: 0.2 },
    },
    {
      id: 'imbalance-della', title: 'Imbalanced skills: drop by size, not at random',
      lesson: 'When one expert changed much more than the other (Hard + easy), DELLA, which drops small updates and keeps large ones, beat DARE’s uniform random dropping on 9 of 10 seeds (+11 pts on average) and kept the hard skill at 82% against 63%. On balanced skills (Two skills) the order flips: DELLA lost to DARE on 9 of 10. Pick by your situation, not by a leaderboard.',
      hint: 'The Recipe advisor asks about this: answer “One changed much more” to see DELLA move up.',
      scenario: 'hard',
      A: { method: 'dare', params: p('dare'), align: false },
      B: { method: 'della', params: p('della'), align: false },
      // the test seed (7) is near the low end of the 10 measured (+1.4 pts; mean +11.1, min −0.4), so the guard is modest
      expect: { kind: 'b-beats-a', by: 0.01 },
    },
    {
      id: 'conflict-ceiling', title: 'Conflicting skills have a ceiling',
      lesson: 'When two skills disagree on the same inputs, no merge gets close to the experts. Even a Mixture of Experts, which beats TIES here, stays far below them: you need a task signal or retraining, not a better merge.',
      scenario: 'overlap',
      A: { method: 'ties', params: p('ties'), align: false },
      B: { method: 'moe', params: p('moe'), align: false },
      expect: { kind: 'both-below-experts', by: 0.08 },
    },
  ];

  /** A gallery entry as a full lab recipe (B is the current recipe, A is pinned), ready for encodeRecipe. */
  function galleryRecipe(g) {
    const sc = MM.SCENARIOS.find((s) => s.id === g.scenario);
    return {
      experts: sc.experts.map((e) => ({ ...e })), init: sc.init, hidden: 32, ftSteps: 250, seed: 7,
      align: g.B.align, method: g.B.method, params: JSON.parse(JSON.stringify(g.B.params)),
      pin: { method: g.A.method, params: JSON.parse(JSON.stringify(g.A.params)), align: g.A.align },
    };
  }

  /**
   * Check a gallery lesson on a trained bundle: returns { ok, a, b, experts, change? }.
   * Kinds: 'b-beats-a' (B ahead by `by`), 'both-below-experts' (both at least `by` under the experts),
   * 'change-helps' (the named change from A to B, applied alone, helps by at least `by`; see MM.attributeChanges),
   * 'layer-sensitive' (on B's per-layer grids, the swing along `layer` is at least `ratio`× the other layer's, in every
   * grid that includes it; the same measure as the Lab's per-layer map).
   */
  function checkLesson(g, T) {
    const score = (side) => { const r = MM.scoreMethod(T, MM.bundleCtx(T, side.align), side.method, side.params); return r.blocked ? null : r.avg; };
    const a = score(g.A), b = score(g.B);
    const experts = T.expertAccs.reduce((x, e, i) => x + e[i], 0) / T.K;
    if (g.expect.kind === 'layer-sensitive') {
      const m = MM.METHOD_BY_ID[g.B.method], gr = m.gradient, L = g.expect.layer;
      const vals = Array.from({ length: 7 }, (_, i) => +(gr.min + ((gr.max - gr.min) * i) / 6).toFixed(4));
      const ctx = MM.bundleCtx(T, g.B.align);
      const ratios = [0, 1, 2].filter((o) => o !== L).map((o) => {
        const q = MM.pickGrid(MM.layerGrid(T, ctx, g.B.method, g.B.params, L, o, vals));
        return q.sensA / Math.max(q.sensB, 1e-9);
      });
      return { ok: ratios.every((r) => r >= g.expect.ratio), a, b, experts, ratio: Math.min(...ratios) };
    }
    if (g.expect.kind === 'change-helps') {
      const at = MM.attributeChanges(g.A, g.B, T.K, score);
      const part = at && at.parts.find((x) => x.what === g.expect.what);
      const change = part ? part.delta : null;
      return { ok: change != null && change >= g.expect.by, a, b, experts, change };
    }
    const ok = g.expect.kind === 'b-beats-a' ? b - a >= g.expect.by : Math.max(a, b) <= experts - g.expect.by;
    return { ok, a, b, experts };
  }

  Object.assign(MM, { GALLERY, galleryRecipe, checkLesson });
})();
