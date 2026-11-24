/* ModelMixer — recipe advisor. Pure rules: your real-world situation in, ranked merge
 * recipes out (with reasons, caveats, what was ruled out and the closest lab scenario).
 * No DOM access, so it is unit-tested alongside the engine. */
(function () {
  const root = typeof window !== 'undefined' ? window : globalThis;
  const MM = (root.MM = root.MM || {});

  const ADVISOR_QUESTIONS = [
    { key: 'base', q: 'Are your models fine-tunes of the same base checkpoint?', hint: 'For example, all of them start from Mistral-7B-v0.1.',
      options: [['yes', 'Yes, same base'], ['unsure', 'Not sure'], ['no', 'No, different bases or inits']] },
    { key: 'goal', q: 'What are you combining?',
      options: [['skills', 'Different skills (code, math, chat…)'], ['same', 'Runs of the same skill (seeds, hparams)']] },
    { key: 'count', q: 'How many models?', options: [['2', 'Two'], ['3+', 'Three or more']] },
    { key: 'conflict', q: 'Do the skills see the same kind of inputs?', hint: 'Overlap is where merged skills interfere.',
      options: [['no', 'No, distinct inputs'], ['some', 'Partly overlapping'], ['yes', 'Same inputs, conflicting answers']] },
    { key: 'data', q: 'Can you run a few hundred task inputs through the models?', hint: 'Unlabeled inputs are enough.',
      options: [['no', 'No data'], ['yes', 'Yes']] },
    { key: 'train', q: 'Any training budget after merging?', options: [['no', 'Merge only'], ['yes', 'A short training run']] },
    { key: 'deploy', q: 'What can you serve?',
      options: [['single', 'One model, 1× cost'], ['router', 'Experts behind a router'], ['any', 'Every model, per request']] },
    { key: 'imbalance', q: 'Did one model change much more than the others?', hint: 'Compare each model’s per-layer weight change from the base: no data needed. Three times or more in the last layer counts.',
      options: [['unknown', 'Haven’t checked'], ['no', 'About the same'], ['yes', 'One changed much more']] },
    { key: 'toy', q: 'Check the picks on which toy?', hint: 'Evidence only: the rules above don’t change. Digits have 64-pixel inputs, closer to real high-dimensional models.',
      options: [['plane', 'Points on a plane (2-D)'], ['digits', '8×8 digits (images)'], ['both', 'Both, and flag disagreements']] },
  ];
  const ADVISOR_DEFAULTS = { base: 'yes', goal: 'skills', count: '2', conflict: 'no', data: 'no', train: 'no', deploy: 'single', imbalance: 'unknown', toy: 'plane' };

  // Rough prior for "a good default in practice" before situation-specific adjustments.
  const PRIOR = { ties: 7, dare: 7, della: 7, karcher: 5, sce: 6, adamerging: 7, localize_stitch: 7, task_arithmetic: 6, breadcrumbs: 5, linear: 5, slerp: 5, model_stock: 5, fisher: 6, regmean: 8, stitch: 2, ensemble: 5, moe: 7, distill: 7 };
  const WEIGHT_SPACE = ['linear', 'slerp', 'karcher', 'model_stock', 'fisher', 'regmean', 'stitch'];

  function adviseRecipe(answers) {
    const a = Object.assign({}, ADVISOR_DEFAULTS, answers || {});
    const st = {};
    MM.METHODS.forEach((m) => (st[m.id] = { id: m.id, score: PRIOR[m.id] || 4, reasons: [], caveats: [], out: null }));
    const bump = (id, d, why) => { const s = st[id]; s.score += d; if (why) (d >= 0 ? s.reasons : s.caveats).push(why); };
    const ban = (id, why) => { if (!st[id].out) st[id].out = why; };
    const notes = [];

    // --- shared base? ---
    MM.METHODS.forEach((m) => {
      if (!m.req.base) return;
      if (a.base === 'no') ban(m.id, 'Needs a shared base. Without one there are no task vectors to merge.');
      else if (a.base === 'unsure') bump(m.id, -1, 'Only valid if every model was fine-tuned from the same base. Check each model card or config.');
      else bump(m.id, 1, 'Shared base: task vectors are meaningful');
    });
    if (a.base === 'no') {
      WEIGHT_SPACE.forEach((id) => bump(id, -2, 'Different inits: align neurons first (Git Re-Basin), and still expect a loss barrier'));
      ['ensemble', 'moe', 'distill'].forEach((id) => bump(id, 2, 'Works across different bases and inits'));
      notes.push('Without a shared base, weight-space merges need neuron alignment (Git Re-Basin), which assumes the models learned the same functions. On the toys it was hit-or-miss: about +3 pts for narrow models with different skills, −17 for wider ones, and worse still when the same skill covers different inputs. Output-space methods and distillation avoid the problem.');
    }

    // --- data / training / deployment hard constraints ---
    MM.METHODS.forEach((m) => {
      if (m.id === 'moe' && (a.data === 'no' || a.train === 'no')) ban(m.id, 'Needs task inputs and a short run to train the router.');
      if (m.req.data && a.data === 'no') ban(m.id, m.id === 'distill' ? 'Needs an input corpus for the teacher to label.' : 'Needs task inputs to measure importance or activations.');
      if (m.req.training && a.train === 'no') ban(m.id, 'Needs a training run.');
    });
    if (a.data === 'yes') {
      bump('regmean', 2, 'Closed-form, using each layer’s input statistics; strong on interfering skills');
      bump('fisher', 1, 'Weighs each parameter by how much each task relies on it');
      bump('adamerging', 1, 'Learns per-layer coefficients from unlabeled inputs alone');
      bump('localize_stitch', 1, 'Tiny per-task masks learned on a little data rarely collide');
    }
    if (a.train === 'yes') bump('distill', 1, 'Training budget lets one student absorb every expert');

    if (a.deploy === 'single') {
      ban('ensemble', 'Runs every model on every request (K× compute and memory).');
      ban('moe', 'Keeps every expert in memory behind a router.');
    } else if (a.deploy === 'router') {
      ban('ensemble', 'Runs every model on every request (K× compute).');
      bump('moe', 2, 'Router picks one expert per input: ~1× compute, K× memory');
    } else {
      bump('ensemble', 1, 'Simple and robust when you can afford K× compute');
      bump('moe', 1, 'Cheaper than an ensemble at the same quality when inputs are separable');
    }

    // --- number of models ---
    if (a.count === '3+') {
      ban('slerp', 'Interpolates exactly two models. Chain merges, or use a multi-model method.');
      bump('ties', 1, 'Built for many experts: trims and elects signs across all of them');
      bump('dare', 1, 'Sparsifying each vector keeps many experts from colliding');
      bump('karcher', 2, 'Norm-preserving SLERP for any number of models');
      bump('stitch', -1, null);
    }

    // --- what is being combined ---
    if (a.goal === 'same') {
      bump('model_stock', 4, 'Designed for several fine-tunes of one recipe: finds the centre from their geometry');
      bump('linear', 3, 'Model-soup averaging of same-skill runs is a well-tested default');
      bump('slerp', 2, 'Smooth, norm-preserving blend of two runs');
      bump('karcher', 3, 'Geometric mean of several runs: SLERP without chaining');
      ['ties', 'dare', 'della', 'sce', 'breadcrumbs'].forEach((id) => bump(id, -2, 'Sign election and dropping pay off with different skills, not repeated runs'));
      bump('moe', -3, 'Nothing for a router to choose between: the experts know the same thing');
      bump('distill', -1, null);
    } else if (a.conflict === 'no' && a.deploy !== 'single') {
      bump('moe', 1, 'Distinct inputs: a router can tell which expert to use');
    }

    // --- interference ---
    if (a.conflict === 'some') {
      bump('ties', 2, 'Resolves sign conflicts where skills overlap');
      bump('dare', 1, 'Random sparsification reduces overlap between task vectors');
      bump('della', 1, 'Magnitude-aware dropping keeps the important updates while reducing overlap');
      bump('sce', 1, 'Concentrates on the entries where experts disagree and erases minority signs');
      bump('breadcrumbs', 1, 'Dropping outliers and tiny updates reduces interference');
      bump('task_arithmetic', -1, 'Plain sums let overlapping updates cancel');
      bump('linear', -1, 'Averaging dilutes overlapping skills');
    } else if (a.conflict === 'yes') {
      bump('moe', -3, 'A router cannot separate identical inputs');
      bump('ties', 1, 'Best weight-space option when updates collide');
      notes.push('No single model can give contradictory answers to the same input. Add a task signal (a system prompt or tag) and train on it, or keep the models separate.');
    }

    // --- imbalanced updates (measured on the Hard + easy toy) ---
    const imbalanced = a.imbalance === 'yes' && a.base !== 'no';
    if (imbalanced) {
      ['task_arithmetic', 'ties', 'dare', 'della', 'sce', 'breadcrumbs'].forEach((id) => st[id].caveats.push('One model’s update dwarfs the others’: keep the output-layer λ at or below 1. Doubling it collapsed Hard + easy merges on 3 of 5 seeds.'));
      // measured on Hard + easy, 10 seeds, default knobs: DELLA 90.6% (hard skill 82%), DARE 79.5% (63%), TIES 85.0%;
      // on balanced Two skills the order flips (DELLA loses to DARE on 9/10), so this only applies under imbalance
      bump('della', 3, 'Magnitude-aware dropping keeps the large updates of the model that changed most: on the imbalanced toy it beat DARE on 9 of 10 seeds (+11 pts) and kept the hard skill 19 pts higher');
      bump('dare', -1, 'Uniform random dropping thins the big model’s updates as much as the small one’s: the hard skill suffers most');
      notes.push('When one model changed much more than the others, the output layer is where merges break. Scale it gently and check the easy skill first: it is the one that collapses.');
    }

    const all = Object.values(st);
    const recs = all.filter((s) => !s.out).sort((x, y) => y.score - x.score || x.id.localeCompare(y.id));
    const excluded = all.filter((s) => s.out).map((s) => ({ id: s.id, why: s.out }));
    // closest toy(s) for the measured evidence: plane scenarios and/or the digit (image) scenarios
    const init = a.base === 'no' ? 'independent' : 'shared'; // the digit toys default to a shared base, so override
    const plane = { scenario: a.base === 'no' ? (a.goal === 'same' ? 'twins' : 'strangers') : a.conflict === 'yes' ? 'overlap' : a.count === '3+' ? 'three' : imbalanced ? 'hard' : 'two', init };
    const digit = { scenario: a.conflict === 'yes' ? 'digits-conflict' : 'digits', init };
    const toys = a.toy === 'digits' ? [digit] : a.toy === 'both' ? [plane, digit] : [plane];
    if (a.toy !== 'plane' && a.count === '3+') notes.push('The digit toys use two experts, so the measurements compare pairs.');
    return { answers: a, recs, excluded, notes, toys, scenario: toys[0].scenario, init, align: a.base === 'no' };
  }

  /**
   * Blend rules with measurement. ids = top picks in rule order; perSeed[id] = per-seed results
   * ([{avg, …} | {blocked}], same seeds for every id). A lower pick that beats the rules' #1 under the
   * paired test (significant, or winning every seed) is promoted to #1; otherwise the rule order stands.
   */
  function blendRanking(ids, perSeed) {
    const n = Math.min(...ids.map((id) => (perSeed[id] || []).length));
    const runs = Array.from({ length: n }, (_, s) => Object.fromEntries(ids.map((id) => [id, perSeed[id][s]])));
    const first = ids[0];
    let best = null;
    for (const id of ids.slice(1)) {
      const d = MM.pairedDiff(runs, id, first);
      if (d.verdict === 'a' && (!best || d.mean > best.d.mean)) best = { id, d };
    }
    return {
      order: best ? [best.id, ...ids.filter((x) => x !== best.id)] : ids.slice(),
      promoted: best ? { id: best.id, over: first, d: best.d } : null,
      runs,
    };
  }

  /**
   * Combine per-toy blends (each from blendRanking on the same rule-ordered ids). A pick is promoted only
   * when every toy promotes the same one; otherwise the rule order stands. `agree` says whether the toys
   * pick the same best-measured method.
   */
  function combineToyBlends(ids, blends) {
    if (blends.length === 1) return Object.assign({ agree: true }, blends[0]);
    const pro = blends.map((b) => (b.promoted ? b.promoted.id : null));
    const same = pro[0] && pro.every((x) => x === pro[0]);
    // per toy: the best method and every method within noise of it (mean gap ≤ the larger of the two sds)
    const stats = (b) => ids.map((id) => {
      const v = b.runs.map((r) => r[id]).filter((x) => x && !x.blocked).map((x) => x.avg);
      if (!v.length) return null;
      const m = v.reduce((a, x) => a + x, 0) / v.length;
      return { id, m, sd: v.length > 1 ? Math.sqrt(v.reduce((a, x) => a + (x - m) ** 2, 0) / (v.length - 1)) : 0 };
    }).filter(Boolean);
    const near = blends.map((b) => {
      const st = stats(b), top = st.reduce((a, x) => (x.m > a.m ? x : a));
      return { best: top.id, set: st.filter((x) => top.m - x.m <= Math.max(top.sd, x.sd) + 1e-12).map((x) => x.id) };
    });
    // a method that is best-or-within-noise on every toy; prefer the rules' order
    const common = ids.find((id) => near.every((n) => n.set.includes(id))) || null;
    return {
      order: same ? [pro[0], ...ids.filter((x) => x !== pro[0])] : ids.slice(),
      promoted: same ? Object.assign({}, blends[0].promoted, { perToy: blends.map((b) => b.promoted) }) : null,
      agree: !!common,
      common,
      best: near.map((n) => n.best),
    };
  }

  /**
   * Which layer to tune first when a method's values are set per layer. Measured 2026-11-24: each layer's swing in
   * average accuracy (7×7 per-layer grids, averaged over the two maps that include the layer), default settings, 5 seeds
   * per toy. `n` = toy-seed cases, `h1` = cases where Hidden 1 swung most. A test re-checks Hidden 1 on the test seed.
   */
  const LAYER_FIRST = {
    ties: { h1: 26, n: 30, toys: 'Two skills, Three skills, Overlapping, Hard + easy, Digits, Digits conflict', other: 'Output 4×' },
    task_arithmetic: { h1: 15, n: 20, toys: 'Two skills, Hard + easy, Overlapping, Digits', other: 'Output 5× (3 on Hard + easy)' },
    dare: { h1: 15, n: 20, toys: 'Two skills, Hard + easy, Overlapping, Digits', other: 'Output 4×, Hidden 2 once' },
    slerp: { h1: 18, n: 20, toys: 'Two skills, Hard + easy, Overlapping, Digits', other: 'Output 2×, both on Hard + easy' },
    linear: { h1: 18, n: 20, toys: 'Two skills, Hard + easy, Overlapping, Digits', other: 'Output 2×, both on Hard + easy' },
  };
  /** One line for a method, or null if it has no measured per-layer evidence. */
  function layerFirstTip(id) {
    const e = LAYER_FIRST[id];
    if (!e) return null;
    return `If you set values per layer, tune Hidden 1 first: it was the most sensitive layer in ${e.h1} of ${e.n} toy-seed cases (${e.toys}). The exceptions were ${e.other}; Output tends to matter more when one skill is much harder than the other. The best value itself moves with the seed, so map it on your own models.`;
  }

  Object.assign(MM, { ADVISOR_QUESTIONS, ADVISOR_DEFAULTS, adviseRecipe, blendRanking, combineToyBlends, LAYER_FIRST, layerFirstTip });
})();
