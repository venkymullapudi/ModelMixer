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
  ];
  const ADVISOR_DEFAULTS = { base: 'yes', goal: 'skills', count: '2', conflict: 'no', data: 'no', train: 'no', deploy: 'single' };

  // Rough prior for "a good default in practice" before situation-specific adjustments.
  const PRIOR = { ties: 7, dare: 7, task_arithmetic: 6, breadcrumbs: 5, linear: 5, slerp: 5, model_stock: 5, fisher: 6, regmean: 8, stitch: 2, ensemble: 5, moe: 7, distill: 7 };
  const WEIGHT_SPACE = ['linear', 'slerp', 'model_stock', 'fisher', 'regmean', 'stitch'];

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
      notes.push('Without a shared base, weight-space merges only work after neuron alignment, and only for same-architecture models. Output-space methods and distillation avoid the problem.');
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
      bump('stitch', -1, null);
    }

    // --- what is being combined ---
    if (a.goal === 'same') {
      bump('model_stock', 4, 'Designed for several fine-tunes of one recipe: finds the centre from their geometry');
      bump('linear', 3, 'Model-soup averaging of same-skill runs is a well-tested default');
      bump('slerp', 2, 'Smooth, norm-preserving blend of two runs');
      ['ties', 'dare', 'breadcrumbs'].forEach((id) => bump(id, -2, 'Sign election and dropping pay off with different skills, not repeated runs'));
      bump('moe', -3, 'Nothing for a router to choose between: the experts know the same thing');
      bump('distill', -1, null);
    } else if (a.conflict === 'no' && a.deploy !== 'single') {
      bump('moe', 1, 'Distinct inputs: a router can tell which expert to use');
    }

    // --- interference ---
    if (a.conflict === 'some') {
      bump('ties', 2, 'Resolves sign conflicts where skills overlap');
      bump('dare', 1, 'Random sparsification reduces overlap between task vectors');
      bump('breadcrumbs', 1, 'Dropping outliers and tiny updates reduces interference');
      bump('task_arithmetic', -1, 'Plain sums let overlapping updates cancel');
      bump('linear', -1, 'Averaging dilutes overlapping skills');
    } else if (a.conflict === 'yes') {
      bump('moe', -3, 'A router cannot separate identical inputs');
      bump('ties', 1, 'Best weight-space option when updates collide');
      notes.push('No single model can give contradictory answers to the same input. Add a task signal (a system prompt or tag) and train on it, or keep the models separate.');
    }

    const all = Object.values(st);
    const recs = all.filter((s) => !s.out).sort((x, y) => y.score - x.score || x.id.localeCompare(y.id));
    const excluded = all.filter((s) => s.out).map((s) => ({ id: s.id, why: s.out }));
    const scenario = a.base === 'no' ? 'strangers' : a.conflict === 'yes' ? 'overlap' : a.count === '3+' ? 'three' : 'two';
    return { answers: a, recs, excluded, notes, scenario, align: a.base === 'no' };
  }

  Object.assign(MM, { ADVISOR_QUESTIONS, ADVISOR_DEFAULTS, adviseRecipe });
})();
