/* ModelMixer — method catalog: what each mixing technique does, what it needs,
 * where it shines, what it is missing, and how to run it for real (open source). */
(function () {
  const MM = (window.MM = window.MM || {});

  const FAMILIES = {
    avg: { name: 'Weight averaging', color: '#7c9cff', blurb: 'Blend parameters directly. Cheap, data-free, but blind to what each weight does.' },
    tv: { name: 'Task vectors', color: '#c084fc', blurb: 'Subtract the base to isolate each skill, then add skills back — with tricks to reduce interference.' },
    data: { name: 'Data-informed', color: '#34d399', blurb: 'Use a little data to estimate which weights matter, then merge in closed form.' },
    surgery: { name: 'Layer surgery', color: '#fbbf24', blurb: 'Copy whole layers from different models (Frankenmerge / passthrough).' },
    output: { name: 'Output mixing', color: '#f472b6', blurb: 'Keep every expert and combine their predictions. Strong but costs more at inference.' },
    distill: { name: 'Distillation', color: '#22d3ee', blurb: 'Train one student to imitate a mixture. Single-model cost, but needs training.' },
  };

  const MKS = 'https://github.com/arcee-ai/mergekit';
  const OSS = {
    mergekit: { name: 'mergekit', url: MKS },
    fusion: { name: 'FusionBench', url: 'https://github.com/tanganke/fusion_bench' },
    rebasin: { name: 'git-re-basin', url: 'https://github.com/samuela/git-re-basin' },
    regmean: { name: 'dataless-model-merging', url: 'https://github.com/bloomberg/dataless-model-merging' },
    distillkit: { name: 'DistillKit', url: 'https://github.com/arcee-ai/DistillKit' },
    trl: { name: 'TRL (GKD trainer)', url: 'https://github.com/huggingface/trl' },
  };

  const W = { key: 'weights', type: 'weights', label: 'Expert weights', min: 0, max: 1, step: 0.05, default: 1 };

  const METHODS = [
    {
      id: 'linear', name: 'Linear / Model Soup', short: 'Linear', family: 'avg',
      tagline: 'Average the weights. The original "model soup".',
      how: ['Line the experts up parameter by parameter.', 'Take a weighted average of every weight.', 'Ship the single averaged model.'],
      formula: 'θ = Σᵢ wᵢ·θᵢ   (Σ wᵢ = 1)',
      pros: ['Zero data, zero training, one model at inference', 'Works great when experts are fine-tunes of the same base', 'Easy to reason about'],
      cons: ['Each skill is diluted by 1/K', 'Ignores which weights matter to which task', 'Collapses when experts were trained from different inits'],
      req: { base: false, data: false, training: false }, multiInit: false, interference: 'no',
      mergekit: 'linear', oss: [OSS.mergekit], paper: 'Wortsman et al., 2022 — Model Soups',
      params: [W],
    },
    {
      id: 'slerp', name: 'SLERP', short: 'SLERP', family: 'avg', maxModels: 2,
      tagline: 'Interpolate along the arc between two models instead of the straight line.',
      how: ['Treat each tensor of A and B as a high-dimensional vector.', 'Measure the angle Ω between them.', 'Walk along the great-circle arc by fraction t, keeping the norm intact.'],
      formula: 'θ = sin((1−t)Ω)/sinΩ · θ_A + sin(tΩ)/sinΩ · θ_B',
      pros: ['Preserves weight norms better than linear', 'Smooth, well-behaved blend of two models', 'Community favourite for LLM merges'],
      cons: ['Only two models at a time', 'Still a blend — no notion of task importance', 'Needs same-lineage models'],
      req: { base: false, data: false, training: false }, multiInit: false, interference: 'no',
      mergekit: 'slerp', oss: [OSS.mergekit], paper: 'Shoemake, 1985 — popularised for LLMs by mergekit',
      params: [{ key: 't', type: 'range', label: 't  (0 = A, 1 = B)', min: 0, max: 1, step: 0.01, default: 0.5 }],
    },
    {
      id: 'karcher', name: 'Karcher Mean', short: 'Karcher', family: 'avg', minModels: 2,
      tagline: 'SLERP for any number of models: the geometric mean on the sphere.',
      how: ['Normalise each expert’s tensor to a direction on the unit sphere.', 'Iteratively find the direction with the smallest summed squared arc distance to every expert (the Karcher / Fréchet mean).', 'Rescale it to the weighted average norm.'],
      formula: 'm* = argmin_m Σᵢ wᵢ·arccos(⟨m, uᵢ⟩)²,   θ = (Σᵢ wᵢ‖θᵢ‖)·m*',
      pros: ['Generalises SLERP beyond two models', 'Keeps weight norms instead of shrinking them like a plain average', 'No base, data or training needed'],
      cons: ['Still a blend: distinct skills get diluted', 'Needs neuron alignment for models without a shared base', 'Iterative, so a little slower than a plain average'],
      req: { base: false, data: false, training: false }, multiInit: false, interference: 'no',
      mergekit: 'karcher', oss: [OSS.mergekit], paper: 'Karcher, 1977 — Riemannian centre of mass (mergekit: karcher)',
      params: [W],
    },
    {
      id: 'model_stock', name: 'Model Stock', short: 'Stock', family: 'avg', minModels: 2,
      tagline: 'Use the geometry between fine-tunes to decide how far to move from the base.',
      how: ['Compute each expert’s offset from the base, per layer.', 'Measure the angle between offsets.', 'Pull the average back toward the base by ratio t = K·cosθ / (1+(K−1)·cosθ).'],
      formula: 'θ = t·avg(θᵢ) + (1−t)·θ_base',
      pros: ['No hyper-parameters', 'Per-layer, automatic interpolation ratio', 'Robust when fine-tunes are noisy copies of one skill'],
      cons: ['Designed for many fine-tunes of the SAME task', 'Orthogonal skills give cosθ≈0 → snaps back to base', 'Needs the shared base'],
      req: { base: true, data: false, training: false }, multiInit: false, interference: 'partial',
      mergekit: 'model_stock', oss: [OSS.mergekit], paper: 'Jang et al., 2024 — Model Stock',
      params: [],
    },
    {
      id: 'task_arithmetic', name: 'Task Arithmetic', short: 'Task Arith.', family: 'tv',
      tagline: 'A skill is a direction in weight space — add skills together.',
      how: ['Task vector τᵢ = θᵢ − θ_base captures what fine-tuning learned.', 'Sum the task vectors (optionally weighted).', 'Add the sum back to the base, scaled by λ.'],
      formula: 'θ = θ_base + λ·Σᵢ wᵢ·τᵢ',
      pros: ['Skills add instead of being averaged away', 'λ trades off strength vs. stability', 'Also supports negation (forgetting) with negative weights'],
      cons: ['Task vectors overlap and cancel each other (interference)', 'Needs the base model', 'λ must be tuned'],
      req: { base: true, data: false, training: false }, multiInit: false, interference: 'no',
      mergekit: 'task_arithmetic', oss: [OSS.mergekit, OSS.fusion], paper: 'Ilharco et al., 2023 — Editing Models with Task Arithmetic',
      params: [{ key: 'lambda', type: 'range', label: 'Scaling λ', min: 0, max: 2, step: 0.05, default: 1 }, W],
    },
    {
      id: 'ties', name: 'TIES-Merging', short: 'TIES', family: 'tv',
      tagline: 'Trim, Elect sign, merge disjointly — fight interference head-on.',
      how: ['Trim: keep only the top-k% largest entries of each task vector.', 'Elect: for every weight, pick the sign with the larger total mass.', 'Disjoint merge: average only the entries that agree with the elected sign.'],
      formula: 'τ = mean{ τ̂ᵢ : sign(τ̂ᵢ) = sign(Σ τ̂ᵢ) }   θ = θ_base + λτ',
      pros: ['Explicitly resolves sign conflicts', 'Redundant small updates are dropped', 'Scales well to many experts'],
      cons: ['Density k and λ need tuning', 'Needs the base model', 'Cannot fix tasks that truly contradict'],
      req: { base: true, data: false, training: false }, multiInit: false, interference: 'yes',
      mergekit: 'ties', oss: [OSS.mergekit, OSS.fusion], paper: 'Yadav et al., 2023 — TIES-Merging',
      params: [
        { key: 'density', type: 'range', label: 'Density k (fraction kept)', min: 0.02, max: 1, step: 0.01, default: 0.2 },
        { key: 'lambda', type: 'range', label: 'Scaling λ', min: 0, max: 2, step: 0.05, default: 1 },
      ],
    },
    {
      id: 'dare', name: 'DARE', short: 'DARE', family: 'tv',
      tagline: 'Drop most of each task vector at random, rescale the rest.',
      how: ['Randomly drop a fraction p of every task vector’s entries.', 'Rescale survivors by 1/(1−p) so the expected update is unchanged.', 'Combine the sparse vectors — by sum (DARE-Linear) or with TIES sign election (DARE-TIES).'],
      formula: 'τ̃ᵢ = mᵢ ⊙ τᵢ / (1−p),  mᵢ ~ Bernoulli(1−p)',
      pros: ['Sparse vectors overlap less → less interference', 'Surprisingly robust to very high drop rates on big models', 'Pairs nicely with TIES'],
      cons: ['Random: results vary by seed', 'Small models have little redundancy to drop', 'Needs the base model'],
      req: { base: true, data: false, training: false }, multiInit: false, interference: 'partial',
      mergekit: 'dare_ties / dare_linear', oss: [OSS.mergekit, OSS.fusion], paper: 'Yu et al., 2024 — Language Models are Super Mario',
      params: [
        { key: 'drop', type: 'range', label: 'Drop rate p', min: 0, max: 0.95, step: 0.01, default: 0.5 },
        { key: 'lambda', type: 'range', label: 'Scaling λ', min: 0, max: 2, step: 0.05, default: 1 },
        { key: 'mode', type: 'select', label: 'Combine', options: [['ties', 'DARE-TIES (sign elect)'], ['linear', 'DARE-Linear (sum)']], default: 'ties' },
        { key: 'seed', type: 'range', label: 'Random seed', min: 0, max: 20, step: 1, default: 0 },
        W,
      ],
    },
    {
      id: 'della', name: 'DELLA', short: 'DELLA', family: 'tv',
      tagline: 'DARE, but drop small updates more often than large ones.',
      how: ['Rank each task vector’s entries by magnitude within every row.', 'Keep each entry with a probability that rises with its rank (MagPrune), then rescale survivors by 1/p_keep.', 'Elect a sign per weight and average the agreeing survivors, as in TIES.'],
      formula: 'p_keep(r) = k − ε/2 + ε·r/(n−1),   τ̃ = m ⊙ τ / p_keep',
      pros: ['Keeps the large updates that DARE may randomly lose', 'Sign election handles conflicts', 'Rescaling keeps the expected update unchanged'],
      cons: ['Three knobs (density, ε, λ) to tune', 'Still random: results vary by seed', 'Needs the base model'],
      req: { base: true, data: false, training: false }, multiInit: false, interference: 'yes',
      mergekit: 'della', oss: [OSS.mergekit], paper: 'Deep et al., 2024 — DELLA-Merging',
      params: [
        { key: 'density', type: 'range', label: 'Density k (avg keep rate)', min: 0.05, max: 1, step: 0.01, default: 0.8 },
        { key: 'epsilon', type: 'range', label: 'Magnitude spread ε', min: 0, max: 0.5, step: 0.01, default: 0.4 },
        { key: 'lambda', type: 'range', label: 'Scaling λ', min: 0, max: 2, step: 0.05, default: 1 },
        { key: 'seed', type: 'range', label: 'Random seed', min: 0, max: 20, step: 1, default: 0 },
      ],
    },
    {
      id: 'sce', name: 'SCE', short: 'SCE', family: 'tv',
      tagline: 'Select where experts differ, weight them by energy, erase minority signs.',
      how: ['Select: keep only the entries whose task-vector values vary most across experts (top-k by variance).', 'Calculate: weight each expert by its share of the squared update over the selected entries.', 'Erase: elect a sign per entry and drop the minority, then take the energy-weighted mean.'],
      formula: 'ηᵢ = Σ τᵢ² / Σⱼ Σ τⱼ²  (selected),   τ = Σ_{agree} ηᵢτᵢ / Σ_{agree} ηᵢ',
      pros: ['Focuses on the parameters where experts genuinely disagree', 'Automatic per-tensor expert weights', 'Sign erase handles conflicts like TIES'],
      cons: ['Drops shared updates (low variance), which can carry common skill', 'Two knobs (top-k, λ)', 'Needs the base model'],
      req: { base: true, data: false, training: false }, multiInit: false, interference: 'yes',
      mergekit: 'sce', oss: [OSS.mergekit], paper: 'Wan et al., 2024 — FuseChat (SCE merging)',
      params: [
        { key: 'topk', type: 'range', label: 'Select top-k (fraction by variance)', min: 0.05, max: 1, step: 0.01, default: 0.7 },
        { key: 'lambda', type: 'range', label: 'Scaling λ', min: 0, max: 2, step: 0.05, default: 1 },
      ],
    },
    {
      id: 'breadcrumbs', name: 'Model Breadcrumbs', short: 'Breadcrumbs', family: 'tv',
      tagline: 'Drop the tiny updates AND the huge outliers; keep the middle.',
      how: ['Sort each task vector’s entries by magnitude.', 'Remove the top γ% outliers and everything below the density cut.', 'Add the surviving "breadcrumbs" to the base.'],
      formula: 'τ̂ᵢ = τᵢ · 1[q_low ≤ |τᵢ| < q_high]',
      pros: ['Outlier removal protects base behaviour', 'Sparse, low-interference vectors', 'No data needed'],
      cons: ['Two thresholds to tune', 'Outliers sometimes ARE the skill', 'Needs the base model'],
      req: { base: true, data: false, training: false }, multiInit: false, interference: 'partial',
      mergekit: 'breadcrumbs', oss: [OSS.mergekit], paper: 'Davari & Belilovsky, 2024 — Model Breadcrumbs',
      params: [
        { key: 'density', type: 'range', label: 'Density (fraction kept)', min: 0.05, max: 1, step: 0.01, default: 0.1 },
        { key: 'gamma', type: 'range', label: 'Outliers removed γ', min: 0, max: 0.3, step: 0.005, default: 0.005 },
        { key: 'lambda', type: 'range', label: 'Scaling λ', min: 0, max: 2, step: 0.05, default: 1 },
      ],
    },
    {
      id: 'fisher', name: 'Fisher Merging', short: 'Fisher', family: 'data',
      tagline: 'Weight every parameter by how much the model’s predictions depend on it.',
      how: ['Run each expert on a little of its own data.', 'Estimate the diagonal Fisher information Fᵢ (parameter importance).', 'Average parameters weighted by importance.'],
      formula: 'θ = Σᵢ Fᵢ⊙θᵢ / Σᵢ Fᵢ',
      pros: ['Importance-aware: each weight goes to the expert that needs it', 'No base model required', 'Principled (Laplace approximation)'],
      cons: ['Needs data + gradients for every expert', 'Diagonal approximation ignores correlations', 'Memory heavy on big models'],
      req: { base: false, data: true, training: false }, multiInit: false, interference: 'partial',
      mergekit: null, oss: [OSS.fusion], paper: 'Matena & Raffel, 2022 — Merging Models with Fisher-Weighted Averaging',
      params: [{ key: 'eps', type: 'range', label: 'Fisher floor ε', min: 0, max: 1, step: 0.01, default: 0 }],
    },
    {
      id: 'regmean', name: 'RegMean', short: 'RegMean', family: 'data',
      tagline: 'Solve for the layer that best reproduces every expert’s activations.',
      how: ['Record each expert’s layer inputs on its own data (Gram matrix G = XᵀX).', 'For every linear layer, solve a least-squares problem matching all experts’ outputs.', 'Closed form — no gradient steps.'],
      formula: 'W = (Σᵢ Wᵢ Gᵢ)(Σᵢ Gᵢ)⁻¹',
      pros: ['Optimal per layer (in the linear sense)', 'Very strong for disjoint skills', 'No base model needed'],
      cons: ['Needs activations from each task’s data', 'Inverts a d×d matrix per layer', 'Layer-local: errors compound through depth'],
      req: { base: false, data: true, training: false }, multiInit: false, interference: 'yes',
      mergekit: null, oss: [OSS.regmean, OSS.fusion], paper: 'Jin et al., 2023 — Dataless Knowledge Fusion (RegMean)',
      params: [{ key: 'alpha', type: 'range', label: 'Off-diagonal scale α', min: 0, max: 1, step: 0.01, default: 0.9 }],
    },
    {
      id: 'adamerging', name: 'AdaMerging', short: 'AdaMerging', family: 'data',
      tagline: 'Learn the merge coefficients from unlabeled inputs by making the model confident.',
      how: ['Start from task arithmetic with every coefficient λ = 0.3.', 'Run unlabeled inputs through the merged model and measure prediction entropy.', 'Adjust one λ per expert (task-wise) or per expert per layer (layer-wise) to lower the entropy. No labels and no weight training.'],
      formula: 'min_λ  E_x[ H(p_θ(·|x)) ],   θ = θ_base + Σₖ λₖ,ₗ · τₖ',
      pros: ['Finds per-layer coefficients you would never hand-tune', 'Needs only unlabeled inputs', 'Only K×L numbers are learned, so it is fast'],
      cons: ['Needs the base model and a sample of inputs', 'Entropy can reward confidently wrong predictions', 'Not in mergekit: export the learned λs as gradients'],
      req: { base: true, data: true, training: false }, multiInit: false, interference: 'yes',
      mergekit: null, oss: [OSS.fusion], paper: 'Yang et al., 2024 — AdaMerging (ICLR)',
      params: [
        { key: 'mode', type: 'select', label: 'Coefficients', options: [['task', 'Task-wise (K)'], ['layer', 'Layer-wise (K×L)']], default: 'task' },
        { key: 'objective', type: 'select', label: 'Objective', options: [['anchored', 'Entropy, anchored'], ['entropy', 'Entropy (paper)'], ['ensemble', 'Expert-average targets']], default: 'anchored' },
      ],
    },
    {
      id: 'localize_stitch', name: 'Localize-and-Stitch', short: 'L&S', family: 'data',
      tagline: 'Keep only the few weights each skill needs, then stitch them into the base.',
      how: ['Localize: for each task, find a sparse mask of the task vector that keeps the skill, by learning it on a little labelled data or, dataless, by taking the largest |τ|.', 'Stitch: add every masked task vector to the base.', 'Where masks overlap, average the updates; everywhere else the base is untouched.'],
      formula: 'θ = θ_base + Σᵢ mᵢ⊙τᵢ / Σᵢ mᵢ,   mᵢ = sparse mask for task i',
      pros: ['Tiny masks rarely collide, so skills interfere less', 'Most of the base is left exactly as it was', 'Masks are interpretable: you see where each skill lives'],
      cons: ['The learned variant needs labelled examples per task', 'Too sparse a mask drops part of the skill', 'Needs the base model'],
      req: { base: true, data: true, training: false }, multiInit: false, interference: 'yes',
      mergekit: null, oss: [{ name: 'Localize-and-Stitch', url: 'https://github.com/uiuctml/Localize-and-Stitch' }], paper: 'He et al., 2024 — Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic',
      params: [
        { key: 'density', type: 'range', label: 'Mask density (fraction kept)', min: 0.01, max: 0.5, step: 0.01, default: 0.3 },
        { key: 'mode', type: 'select', label: 'Localize', options: [['data', 'Learned on data'], ['dataless', 'Dataless (top |τ|)']], default: 'data' },
      ],
    },
    {
      id: 'stitch', name: 'Layer Stitch', short: 'Stitch', family: 'surgery',
      tagline: 'Frankenmerge: take whole layers from different experts.',
      how: ['Pick a source expert for each layer.', 'Copy that layer verbatim.', 'Hope the representations line up across the seams.'],
      formula: 'θ⁽ˡ⁾ = θ⁽ˡ⁾_{src(l)}',
      pros: ['Trivial and fast', 'Can grow depth (passthrough stacks in LLMs)', 'Reveals where in depth each skill lives'],
      cons: ['Seams between layers break unless models share lineage', 'No blending inside a layer', 'Usually needs healing fine-tunes'],
      req: { base: false, data: false, training: false }, multiInit: false, interference: 'no',
      mergekit: 'passthrough', oss: [OSS.mergekit], paper: 'Community practice (e.g. Goliath / SOLAR depth up-scaling)',
      params: [{ key: 'layers', type: 'layers', label: 'Source per layer' }],
    },
    {
      id: 'ensemble', name: 'Ensemble', short: 'Ensemble', family: 'output',
      tagline: 'Keep every expert. Average their predictions.',
      how: ['Run every expert on the input.', 'Average the predicted probabilities.', 'Nothing is merged — the weights stay separate.'],
      formula: 'p(y|x) = Σᵢ wᵢ · pᵢ(y|x)',
      pros: ['Works with ANY models — no shared init needed', 'Never corrupts an expert', 'Reduces variance'],
      cons: ['K× compute and memory at inference', 'Confidently-wrong experts drag down the vote', 'No single model to ship'],
      req: { base: false, data: false, training: false }, multiInit: true, interference: 'no', inferenceK: true,
      mergekit: null, oss: [], paper: 'Classical ensembling',
      params: [W],
    },
    {
      id: 'moe', name: 'Mixture of Experts', short: 'MoE', family: 'output',
      tagline: 'A small router decides which expert handles each input.',
      how: ['Train a tiny gate network on “which task does this input belong to?”.', 'Per input, run the chosen expert (top-1) or blend all by gate weights (soft).', 'All experts stay in memory.'],
      formula: 'p(y|x) = Σᵢ gᵢ(x) · pᵢ(y|x)',
      pros: ['Near-expert quality when tasks are separable', 'Top-1 keeps compute ≈ one expert', 'Works with any models'],
      cons: ['K× memory', 'Router needs labelled task data and training', 'Fails when tasks overlap in input space'],
      req: { base: false, data: true, training: true }, multiInit: true, interference: 'yes', inferenceK: 'memory',
      mergekit: 'mergekit-moe', oss: [OSS.mergekit], paper: 'Shazeer et al., 2017; Mixtral-style “clown-car” MoE via mergekit-moe',
      params: [{ key: 'mode', type: 'select', label: 'Routing', options: [['top1', 'Top-1 (hard)'], ['soft', 'Soft (all experts)']], default: 'top1' }],
    },
    {
      id: 'distill', name: 'Distillation', short: 'Distill', family: 'distill',
      tagline: 'Teach one student to imitate the mixture.',
      how: ['Build a teacher (MoE or ensemble).', 'Label unlabeled inputs with the teacher’s soft predictions.', 'Train a single student (from the base) to match them.'],
      formula: 'min_θ  KL( p_teacher(·|x) ‖ p_θ(·|x) )',
      pros: ['Single-model inference cost', 'Works across architectures and inits', 'Can beat weight merges on hard mixes'],
      cons: ['Needs training compute and an input corpus', 'Inherits teacher mistakes', 'Slowest to iterate'],
      req: { base: false, data: true, training: true }, multiInit: true, interference: 'yes',
      mergekit: null, oss: [OSS.distillkit, OSS.trl], paper: 'Hinton et al., 2015 — Distilling the Knowledge in a Neural Network',
      params: [
        { key: 'teacher', type: 'select', label: 'Teacher', options: [['moe', 'Mixture of Experts'], ['ensemble', 'Ensemble']], default: 'moe' },
        { key: 'steps', type: 'select', label: 'Training steps', options: [['60', '60'], ['150', '150'], ['300', '300']], default: '150' },
      ],
    },
  ];

  // Pre-processing step, shown in the Atlas
  const REBASIN = {
    id: 'rebasin', name: 'Git Re-Basin (alignment)', family: 'avg',
    tagline: 'Permute neurons so independently-trained models line up before merging.',
    pros: ['Unlocks weight merging for models without a shared base', 'Function-preserving: the permuted model is identical', 'Cheap (Hungarian algorithm per layer)'],
    cons: ['Does nothing for fine-tunes of one base (already aligned)', 'Wide models align better than narrow ones', 'Not a merge by itself — a pre-step'],
  };

  const METHOD_BY_ID = Object.fromEntries(METHODS.map((m) => [m.id, m]));

  // Toy defaults are tuned on this lab's small MLPs (3 scenarios × 3 seeds). Real checkpoints often want
  // different starting points; these notes say where, so practitioners don't copy toy numbers blindly.
  const REAL_WORLD = {
    ties: 'The TIES paper keeps the top 20% (density 0.2), which is also this lab’s default. Many published mergekit recipes start nearer density 0.5 with weight 1; tune λ on a validation set.',
    dare: 'Large language models tolerate very high drop rates (0.9 and up) because their updates are highly redundant. Small models like these toys tolerate far less.',
    breadcrumbs: 'Recipes for large models usually keep most entries (density near 0.9) and cut the top ~1% (gamma 0.01). These tiny experts have little redundancy, so the toy default is much sparser (0.1) with a gentler outlier cut (0.005).',
    fisher: 'Estimate the diagonal Fisher on a few hundred examples per task. ε should only guard against division by zero: on the toy, even a 0.1% floor costs several points.',
    task_arithmetic: 'λ is usually tuned on held-out data; values between 0.3 and 1 are common starting points.',
    della: 'Like DARE, DELLA tolerates aggressive sparsity on large language models. ε sets how strongly magnitude shifts the keep probability; tune density first.',
    karcher: 'Typically used to blend several fine-tunes of one model, where SLERP would need to be chained pairwise.',
    sce: 'Used to fuse chat models in FuseChat. Lowering top-k concentrates the merge on the entries where experts disagree; keep λ near 1 to start.',
    adamerging: 'Run a few hundred unlabeled inputs per task through the merge. The paper reports clear gains over task arithmetic on vision models, and this lab agrees on its image-like Digits toy (≈ +12 pts over Task Arithmetic), while on the 2-D toys it roughly ties. The paper found layer-wise coefficients best on ViTs; on these small toys task-wise is slightly better.',
    localize_stitch: 'The paper localizes with a few labelled examples per task and finds masks around 1% of parameters are enough on large models. These tiny toys need far denser masks (0.3), and learned masks barely beat the dataless largest-|τ| ones.',
    regmean: 'Collect layer-input Gram matrices from a few hundred examples per task. α a little below 1 is the usual stabiliser.',
  };
  for (const id in REAL_WORLD) METHOD_BY_ID[id].realWorld = REAL_WORLD[id];

  // Per-layer values ("gradients" in mergekit): one value per layer (hidden 1, hidden 2, output) replaces the scalar knob.
  const LAMBDA_G = { key: 'lambda', label: 'λ', min: 0, max: 2, step: 0.05 };
  const GRADIENT = {
    slerp: { key: 't', label: 't', min: 0, max: 1, step: 0.01 },
    linear: { key: 'balance', label: 'A ↔ B balance', min: 0, max: 1, step: 0.05, k2: true },
    task_arithmetic: LAMBDA_G, ties: LAMBDA_G, dare: LAMBDA_G, della: LAMBDA_G, sce: LAMBDA_G, breadcrumbs: LAMBDA_G,
  };
  for (const id in GRADIENT) METHOD_BY_ID[id].gradient = GRADIENT[id];
  const LAYER_NAMES = ['Hidden 1', 'Hidden 2', 'Output'];

  function defaultParams(m, K) {
    const p = {};
    if (m.gradient) {
      const g = m.gradient, d = m.params.find((x) => x.key === g.key);
      p.layerwise = false;
      p.grad = [0, 1, 2].map(() => (d ? d.default : (g.min + g.max) / 2));
    }
    for (const d of m.params) {
      if (d.type === 'weights') p.weights = Array.from({ length: 3 }, () => 1);
      else if (d.type === 'layers') p.layers = [0, 0, 1];
      else p[d.key] = d.default;
    }
    return p;
  }

  Object.assign(MM, { FAMILIES, METHODS, METHOD_BY_ID, REBASIN, OSS, LAYER_NAMES, defaultParams });
})();
