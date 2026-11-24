/* ModelMixer — recipe gallery: curated A-vs-B examples, each with a one-line lesson.
 * Every lesson was measured on several seeds (see `measured`) and is guarded by a test (tests/suite.js) via `expect`, so a card can't
 * quietly start teaching something false if the engine or a default changes. */
(function () {
  const root = typeof window !== 'undefined' ? window : globalThis;
  const MM = (root.MM = root.MM || {});

  const p = (id, over) => Object.assign(MM.defaultParams(MM.METHOD_BY_ID[id], 2), over || {});

  const GALLERY = [
    {
      id: 'slerp-gradient', measured: { seeds: 10 }, title: 'One t per layer beats one global t',
      lesson: 'SLERP with a per-layer t (a mergekit gradient) lets early layers lean to one expert and late layers to the other. On the toy, t 0.25 → 0.50 → 0.75 beat a flat t = 0.5 on all 10 seeds we tried, by 8.6 pts on average (1.7 to 17.5).',
      scenario: 'two',
      A: { method: 'slerp', params: p('slerp', { t: 0.5 }), align: false },
      B: { method: 'slerp', params: p('slerp', { layerwise: true, grad: [0.25, 0.5, 0.75] }), align: false },
      expect: { kind: 'b-beats-a', by: 0.03 },
    },
    {
      id: 'regmean-data', measured: { seeds: 10 }, title: 'A little data goes a long way',
      lesson: 'RegMean only needs layer inputs from a few hundred examples per task and no training. On all 10 seeds it kept both skills at 99–100% (within 0.2 pts of the experts), while task arithmetic scored 78–98%: 14 pts lower on average.',
      scenario: 'two',
      A: { method: 'task_arithmetic', params: p('task_arithmetic'), align: false },
      B: { method: 'regmean', params: p('regmean'), align: false },
      expect: { kind: 'b-beats-a', by: 0.05 },
    },
    {
      id: 'align-strangers', measured: { seeds: 10, also: 'plus 6 seeds × widths 32–128 and same-skill controls' }, title: 'Alignment assumes the same functions',
      lesson: 'Git Re-Basin re-orders one model’s neurons to match another’s before averaging. It is built for models that learned the same function from different inits, and on this toy those average fine without it (99.9%). Here the experts learned different skills: alignment helped this narrow Linear merge on 7 of 10 seeds, but by only 1.1 pts (−4.5 to +5.1). At width 128 it cost 17 pts, and with the same skill on different inputs it cost 8 to 35. Measure before you trust it, or use output-space methods.',
      scenario: 'strangers',
      A: { method: 'linear', params: p('linear'), align: false },
      B: { method: 'linear', params: p('linear'), align: true },
      expect: { kind: 'b-beats-a', by: 0.005 },
    },
    {
      id: 'align-twins', measured: { seeds: 5 }, title: 'Same function, new init? Align first',
      lesson: 'Two runs of the same Spiral skill from different random inits each score ~100%, but plain averaging gives 70.6% because their neurons sit in different orders. Aligning neurons first (Git Re-Basin) brings the average to 98.6%: better on 5 of 5 seeds. This is the case alignment is built for; see “Alignment assumes the same functions” for where it is not.',
      hint: 'In the Lab, flip “Align neurons” and watch the weight similarity jump in the insight panel.',
      scenario: 'twins',
      A: { method: 'linear', params: p('linear'), align: false },
      B: { method: 'linear', params: p('linear'), align: true },
      expect: { kind: 'b-beats-a', by: 0.15, seeds: [7, 108, 209], meanBy: 0.15 },
    },
    {
      id: 'adamerging-images', measured: { seeds: 10 }, title: 'On image inputs, learn the coefficients',
      lesson: 'On the image-like Digits toy, AdaMerging tunes task-arithmetic coefficients on unlabeled inputs. It beat hand-set task arithmetic on 9 of 10 seeds, by 9.2 pts on average (up to 15.8), in line with the paper’s vision results.',
      scenario: 'digits',
      A: { method: 'task_arithmetic', params: p('task_arithmetic'), align: false },
      B: { method: 'adamerging', params: p('adamerging'), align: false },
      expect: { kind: 'b-beats-a', by: 0.03 },
    },
    {
      id: 'ties-lambda-first', measured: { seeds: 7 }, title: 'On TIES, tune λ before density',
      lesson: 'Going from k 0.30, λ 1.4 to k 0.20, λ 0.8, the λ change alone helped on all 7 seeds we tried (+0.6 to +4.5 pts), while the density change alone went up on 3 seeds and down on 4 (the best density differs by seed). Scaling λ is the knob that transfers; check density on your own data.',
      hint: 'In the Lab, press “Which change mattered?”, then “Repeat on 5 seeds”.',
      scenario: 'two',
      A: { method: 'ties', params: p('ties', { density: 0.3, lambda: 1.4 }), align: false },
      B: { method: 'ties', params: p('ties', { density: 0.2, lambda: 0.8 }), align: false },
      expect: { kind: 'change-helps', what: 'Scaling λ', by: 0.01 },
    },
    {
      id: 'ties-which-layer', measured: { seeds: 7, also: '6 toys × 5 seeds for the layer' }, title: 'Which layer, not which value',
      lesson: 'With a λ per layer, TIES on Two skills is sensitive to Hidden 1 on all 7 seeds we tried (a 14–27 pt swing, against at most 9.5 for the other layers). But the best Hidden 1 value is not stable: λ 0.65 there helped on 5 of the 7 seeds and hurt on 2. Across 6 toys × 5 seeds, Hidden 1 was the most sensitive layer in 26 of 30 cases; the 4 exceptions were Output (on Three skills, Overlapping skills and Hard + easy). So start tuning at Hidden 1, map the layers on your own models, and don’t copy the number.',
      hint: 'In the Lab’s per-layer map, turn on “± 5 seeds”: the best cell moves, the sensitive layer doesn’t.',
      scenario: 'two',
      A: { method: 'ties', params: p('ties', { layerwise: true, grad: [1, 1, 1] }), align: false },
      B: { method: 'ties', params: p('ties', { layerwise: true, grad: [0.65, 1, 1] }), align: false },
      expect: { kind: 'layer-sensitive', layer: 0, ratio: 2 },
    },
    {
      id: 'imbalance-output', measured: { seeds: 5 }, title: 'Imbalanced skills: go easy on the output layer',
      lesson: 'On Hard + easy the hard expert changes the output layer 1.5–8× more than the easy one (on Two skills the two are within 0.4–1.4× of each other). Doubling the output λ of a task-arithmetic merge then collapsed it on 3 of 5 seeds (the easy skill fell to 1% on one) and cost points on a 4th; on Two skills it never collapsed. You can spot this before merging: compare each expert’s per-layer update size.',
      hint: 'In the Lab, the insight panel’s “Update size per layer” table shows the imbalance and warns about it.',
      scenario: 'hard',
      A: { method: 'task_arithmetic', params: p('task_arithmetic', { layerwise: true, grad: [1, 1, 2] }), align: false },
      B: { method: 'task_arithmetic', params: p('task_arithmetic', { layerwise: true, grad: [1, 1, 1] }), align: false },
      expect: { kind: 'b-beats-a', by: 0.2 },
    },
    {
      id: 'imbalance-della', measured: { seeds: 10 }, title: 'Imbalanced skills: drop by size, not at random',
      lesson: 'When one expert changed much more than the other (Hard + easy), DELLA, which drops small updates and keeps large ones, beat DARE’s uniform random dropping on 9 of 10 seeds (+11 pts on average) and kept the hard skill at 82% against 63%. On balanced skills (Two skills) the order flips: DELLA lost to DARE on 9 of 10. Pick by your situation, not by a leaderboard.',
      hint: 'The Recipe advisor asks about this: answer “One changed much more” to see DELLA move up.',
      scenario: 'hard',
      A: { method: 'dare', params: p('dare'), align: false },
      B: { method: 'della', params: p('della'), align: false },
      // the test seed (7) is near the low end of the 10 measured (+1.4 pts; mean +11.1, min −0.4), so this lesson is
      // also checked on 3 seeds: B must lead on average by `meanBy` and win on at least 2 of them
      expect: { kind: 'b-beats-a', by: 0.01, seeds: [7, 108, 209], meanBy: 0.05 },
    },
    {
      id: 'conflict-ceiling', measured: { seeds: 10 }, title: 'Conflicting skills have a ceiling',
      lesson: 'When two skills disagree on the same inputs, no merge gets close to the experts. A Mixture of Experts beat TIES on all 10 seeds (+11 pts on average), yet stayed 12–18 pts below the experts on every one: you need a task signal or retraining, not a better merge.',
      scenario: 'overlap',
      A: { method: 'ties', params: p('ties'), align: false },
      B: { method: 'moe', params: p('moe'), align: false },
      expect: { kind: 'both-below-experts', by: 0.08 },
    },
  ];

  /** Short evidence line for a card: how many seeds the lesson was measured on, and what CI re-checks. */
  function evidenceBadge(g) {
    const ci = g.expect.seeds ? `CI re-checks ${g.expect.seeds.length} seeds` : 'CI re-checks seed 7';
    return `Measured on ${g.measured.seeds} seeds${g.measured.also ? ` (${g.measured.also})` : ''} · ${ci}`;
  }

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

  /**
   * Multi-seed check for 'b-beats-a' lessons with expect.seeds: bundles = one trained bundle per seed.
   * ok when B leads by expect.meanBy on average and wins on at least two thirds of the seeds.
   */
  function checkLessonSeeds(g, bundles) {
    const per = bundles.map((T) => MM.checkLesson(g, T));
    const diffs = per.map((c) => c.b - c.a);
    const mean = diffs.reduce((x, y) => x + y, 0) / diffs.length;
    const wins = diffs.filter((d) => d > 0).length;
    return { ok: mean >= (g.expect.meanBy || 0) && wins >= Math.ceil((2 * diffs.length) / 3), mean, wins, n: diffs.length, diffs };
  }

  Object.assign(MM, { GALLERY, galleryRecipe, checkLesson, checkLessonSeeds, evidenceBadge });
})();
