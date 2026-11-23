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
   * 'change-helps' (the named change from A to B, applied alone, helps by at least `by`; see MM.attributeChanges).
   */
  function checkLesson(g, T) {
    const score = (side) => { const r = MM.scoreMethod(T, MM.bundleCtx(T, side.align), side.method, side.params); return r.blocked ? null : r.avg; };
    const a = score(g.A), b = score(g.B);
    const experts = T.expertAccs.reduce((x, e, i) => x + e[i], 0) / T.K;
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
