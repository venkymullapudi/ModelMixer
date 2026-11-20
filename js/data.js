/* ModelMixer — toy "skills" (2-D classification tasks), regions and scenarios. */
(function () {
  const MM = (window.MM = window.MM || {});
  const { mulberry32, gauss, hashStr } = MM;

  // Every generator returns [x1, x2, label] inside roughly the unit disk.
  const TASKS = {
    moons: {
      name: 'Moons', desc: 'Two interleaving half-moons',
      gen(rng) {
        const y = rng() < 0.5 ? 0 : 1;
        const t = rng() * Math.PI;
        let a, b;
        if (y === 0) { a = Math.cos(t); b = Math.sin(t); } else { a = 1 - Math.cos(t); b = 0.5 - Math.sin(t); }
        a = (a - 0.5) / 1.5; b = ((b - 0.25) / 1.5) * 1.35;
        return [a + gauss(rng) * 0.055, b + gauss(rng) * 0.055, y];
      },
    },
    circles: {
      name: 'Circles', desc: 'A disk inside a ring',
      gen(rng) {
        const y = rng() < 0.5 ? 0 : 1;
        const r = y === 0 ? 0.42 * Math.sqrt(rng()) : 0.62 + 0.33 * rng();
        const t = rng() * 2 * Math.PI;
        return [r * Math.cos(t), r * Math.sin(t), y];
      },
    },
    xor: {
      name: 'XOR', desc: 'Quadrant checkerboard',
      gen(rng) {
        let a, b;
        do { a = rng() * 2 - 1; b = rng() * 2 - 1; } while (Math.min(Math.abs(a), Math.abs(b)) < 0.07);
        return [a * 0.82, b * 0.82, a * b > 0 ? 1 : 0];
      },
    },
    blobs: {
      name: 'Blobs', desc: 'Two Gaussian clusters',
      gen(rng) {
        const y = rng() < 0.5 ? 0 : 1;
        const c = y === 0 ? -0.42 : 0.42;
        return [c + gauss(rng) * 0.2, c + gauss(rng) * 0.2, y];
      },
    },
    spiral: {
      name: 'Spiral', desc: 'Two interlocked spiral arms (hard)',
      gen(rng) {
        const y = rng() < 0.5 ? 0 : 1;
        const t = 0.12 + 0.88 * rng();
        const ang = t * 2.4 * Math.PI + y * Math.PI;
        const r = t * 0.95;
        return [r * Math.cos(ang) + gauss(rng) * 0.03, r * Math.sin(ang) + gauss(rng) * 0.03, y];
      },
    },
    stripes: {
      name: 'Stripes', desc: 'Four vertical bands',
      gen(rng) {
        const a = rng() * 1.8 - 0.9, b = rng() * 1.8 - 0.9;
        const band = Math.floor((a + 0.9) / 0.45);
        return [a, b, band % 2];
      },
    },
  };

  const REGIONS = {
    left: { name: 'West', pos: [-1.65, -0.75] },
    right: { name: 'East', pos: [1.65, -0.75] },
    top: { name: 'North', pos: [0, 1.45] },
    center: { name: 'Center', pos: [0, 0] },
  };
  const SCALE = 1.15;
  const EXTENT = 3.2; // plane is [-EXTENT, EXTENT]^2
  const N_TRAIN = 260, N_TEST = 360;

  function sample(spec, n, rng) {
    const X = new Float32Array(n * 2), Y = new Float32Array(n);
    const [cx, cy] = REGIONS[spec.region].pos;
    const g = TASKS[spec.task].gen;
    for (let i = 0; i < n; i++) {
      const [a, b, y] = g(rng);
      X[2 * i] = cx + a * SCALE;
      X[2 * i + 1] = cy + b * SCALE;
      Y[i] = spec.flip ? 1 - y : y;
    }
    return { X, Y, n };
  }

  function makeTaskData(spec, idx, seed) {
    const rng = mulberry32((seed * 7919 + idx * 104729 + hashStr(spec.task + spec.region)) >>> 0);
    return {
      spec, idx,
      train: sample(spec, N_TRAIN, rng),
      test: sample(spec, N_TEST, rng),
      center: REGIONS[spec.region].pos,
      radius: SCALE * 1.12,
    };
  }

  function concat(sets) {
    const n = sets.reduce((a, s) => a + s.n, 0);
    const X = new Float32Array(n * 2), Y = new Float32Array(n);
    let o = 0;
    for (const s of sets) { X.set(s.X, o * 2); Y.set(s.Y, o); o += s.n; }
    return { X, Y, n };
  }

  // A small slice of every task: the "generalist pre-training corpus" for the shared base model.
  function pretrainSet(tasks, frac, seed) {
    const rng = mulberry32(seed * 31 + 5);
    const parts = tasks.map((t) => {
      const m = Math.max(24, Math.floor(t.train.n * frac));
      const X = new Float32Array(m * 2), Y = new Float32Array(m);
      for (let i = 0; i < m; i++) {
        const j = Math.floor(rng() * t.train.n);
        X[2 * i] = t.train.X[2 * j]; X[2 * i + 1] = t.train.X[2 * j + 1]; Y[i] = t.train.Y[j];
      }
      return { X, Y, n: m };
    });
    return concat(parts);
  }

  // "General knowledge" pre-training corpus: a fine checkerboard-like pattern over the
  // whole plane. It teaches the base useful features but not the experts' skills.
  function genericSet(seed, n) {
    const rng = mulberry32(seed + 3);
    const X = new Float32Array(n * 2), Y = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const a = rng() * 2 * EXTENT - EXTENT, b = rng() * 2 * EXTENT - EXTENT;
      X[2 * i] = a; X[2 * i + 1] = b;
      Y[i] = Math.sin(2.1 * a + 0.7) * Math.sin(1.9 * b + 0.3) > 0 ? 1 : 0;
    }
    return { X, Y, n };
  }

  const SCENARIOS = [
    {
      id: 'two', name: 'Two skills', tag: 'classic',
      desc: 'Expert A learns Moons in the west, Expert B learns Circles in the east — both fine-tuned from one shared base. Can one model keep both skills?',
      experts: [{ task: 'moons', region: 'left' }, { task: 'circles', region: 'right' }], init: 'shared',
    },
    {
      id: 'three', name: 'Three skills', tag: 'multi-task',
      desc: 'Three experts, three regions. More task vectors means more interference — watch where simple averaging dilutes everything.',
      experts: [{ task: 'moons', region: 'left' }, { task: 'circles', region: 'right' }, { task: 'xor', region: 'top' }], init: 'shared',
    },
    {
      id: 'overlap', name: 'Overlapping skills', tag: 'conflict',
      desc: 'Blobs and XOR live in the same place and disagree on one quadrant. No mixing method can satisfy both — and a router cannot tell them apart.',
      experts: [{ task: 'blobs', region: 'center' }, { task: 'xor', region: 'center' }], init: 'shared',
    },
    {
      id: 'strangers', name: 'Strangers', tag: 'no shared base',
      desc: 'Same skills, but each expert is trained from its own random init. Weight averaging breaks (permutation symmetry) unless neurons are aligned first.',
      experts: [{ task: 'moons', region: 'left' }, { task: 'circles', region: 'right' }], init: 'independent',
    },
    {
      id: 'hard', name: 'Hard + easy', tag: 'imbalance',
      desc: 'A hard Spiral expert next to an easy Blobs expert. The hard skill needs large weight changes that easy merges tend to wash out.',
      experts: [{ task: 'spiral', region: 'left' }, { task: 'blobs', region: 'right' }], init: 'shared',
    },
  ];

  Object.assign(MM, { TASKS, REGIONS, EXTENT, SCALE, makeTaskData, concat, pretrainSet, genericSet, SCENARIOS });
})();
