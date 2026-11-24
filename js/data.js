/* ModelMixer — toy "skills": 2-D classification tasks in regions of a plane, plus 8×8 digit-pair
 * tasks (64-pixel inputs), and the built-in scenarios. */
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

  // ---------- 8×8 "digits": an image-like task family (64 inputs instead of 2) ----------
  // 5×7 glyphs, rendered into 8×8 with random shift, stroke dropout, blur and pixel noise.
  const GLYPHS = [
    ['01110', '10001', '10011', '10101', '11001', '10001', '01110'], // 0
    ['00100', '01100', '00100', '00100', '00100', '00100', '01110'], // 1
    ['01110', '10001', '00001', '00010', '00100', '01000', '11111'], // 2
    ['11110', '00001', '00001', '01110', '00001', '00001', '11110'], // 3
    ['00010', '00110', '01010', '10010', '11111', '00010', '00010'], // 4
    ['11111', '10000', '11110', '00001', '00001', '10001', '01110'], // 5
    ['00110', '01000', '10000', '11110', '10001', '10001', '01110'], // 6
    ['11111', '00001', '00010', '00100', '01000', '01000', '01000'], // 7
    ['01110', '10001', '10001', '01110', '10001', '10001', '01110'], // 8
    ['01110', '10001', '10001', '01111', '00001', '00010', '01100'], // 9
  ];
  const DIGIT_D = 64;
  function renderDigit(digit, rng, out, o) {
    const img = new Float32Array(64);
    const g = GLYPHS[digit], dx = Math.floor(rng() * 4), dy = Math.floor(rng() * 2);
    const ink = 0.75 + 0.25 * rng();
    for (let r = 0; r < 7; r++) for (let c = 0; c < 5; c++) {
      if (g[r][c] === '1' && rng() > 0.08) img[(r + dy) * 8 + (c + dx)] = ink;
    }
    for (let i = 0; i < 64; i++) {
      const r = (i / 8) | 0, c = i % 8;
      let nb = 0;
      if (r > 0) nb += img[i - 8]; if (r < 7) nb += img[i + 8]; if (c > 0) nb += img[i - 1]; if (c < 7) nb += img[i + 1];
      out[o + i] = Math.min(1, Math.max(0, img[i] + 0.12 * nb + 0.1 * gauss(rng)));
    }
  }
  const pairTask = (a, b) => ({
    name: `Digits ${a} vs ${b}`, kind: 'digits', pair: [a, b], desc: `Tell an 8×8 “${a}” from a “${b}”`,
  });
  const DIGIT_TASKS = { d01: pairTask(0, 1), d23: pairTask(2, 3), d45: pairTask(4, 5), d67: pairTask(6, 7), d89: pairTask(8, 9), d17: pairTask(1, 7) };
  const taskKind = (task) => (TASKS[task] && TASKS[task].kind) || 'plane';

  function sampleDigits(spec, n, rng) {
    const X = new Float32Array(n * DIGIT_D), Y = new Float32Array(n);
    const [a, b] = TASKS[spec.task].pair;
    for (let i = 0; i < n; i++) {
      const y = rng() < 0.5 ? 0 : 1;
      renderDigit(y ? b : a, rng, X, i * DIGIT_D);
      Y[i] = spec.flip ? 1 - y : y;
    }
    return { X, Y, n, d: DIGIT_D };
  }

  // "General knowledge" for digit models: random pen strokes; label = more ink on the left half.
  // It teaches stroke detectors without teaching any digit task.
  function genericDigitSet(seed, n) {
    const rng = mulberry32(seed + 17);
    const X = new Float32Array(n * DIGIT_D), Y = new Float32Array(n);
    for (let s = 0; s < n; s++) {
      const o = s * DIGIT_D;
      const strokes = 2 + Math.floor(rng() * 2);
      for (let k = 0; k < strokes; k++) {
        let x = rng() * 7, y = rng() * 7;
        const ang = rng() * Math.PI * 2, len = 3 + rng() * 4;
        for (let t = 0; t < len; t++) {
          const c = Math.round(x), r = Math.round(y);
          if (c >= 0 && c < 8 && r >= 0 && r < 8) X[o + r * 8 + c] = Math.max(X[o + r * 8 + c], 0.8 + 0.2 * rng());
          x += Math.cos(ang); y += Math.sin(ang);
        }
      }
      let left = 0, right = 0;
      for (let i = 0; i < 64; i++) { X[o + i] = Math.min(1, Math.max(0, X[o + i] + 0.08 * gauss(rng))); if (i % 8 < 4) left += X[o + i]; else right += X[o + i]; }
      Y[s] = left > right ? 1 : 0;
    }
    return { X, Y, n, d: DIGIT_D };
  }

  function sample(spec, n, rng) {
    if (taskKind(spec.task) === 'digits') return sampleDigits(spec, n, rng);
    const X = new Float32Array(n * 2), Y = new Float32Array(n);
    const [cx, cy] = REGIONS[spec.region].pos;
    const g = TASKS[spec.task].gen;
    for (let i = 0; i < n; i++) {
      const [a, b, y] = g(rng);
      X[2 * i] = cx + a * SCALE;
      X[2 * i + 1] = cy + b * SCALE;
      Y[i] = spec.flip ? 1 - y : y;
    }
    return { X, Y, n, d: 2 };
  }

  function makeTaskData(spec, idx, seed) {
    const rng = mulberry32((seed * 7919 + idx * 104729 + hashStr(spec.task + (spec.region || ''))) >>> 0);
    const digits = taskKind(spec.task) === 'digits';
    return {
      spec, idx, kind: digits ? 'digits' : 'plane',
      train: sample(spec, N_TRAIN, rng),
      test: sample(spec, N_TEST, rng),
      center: digits ? null : REGIONS[spec.region].pos,
      radius: SCALE * 1.12,
    };
  }

  function concat(sets) {
    const d = sets[0].d || 2;
    const n = sets.reduce((a, s) => a + s.n, 0);
    const X = new Float32Array(n * d), Y = new Float32Array(n);
    let o = 0;
    for (const s of sets) { X.set(s.X, o * d); Y.set(s.Y, o); o += s.n; }
    return { X, Y, n, d };
  }

  // A small slice of every task: the "generalist pre-training corpus" for the shared base model.
  function pretrainSet(tasks, frac, seed) {
    const rng = mulberry32(seed * 31 + 5);
    const parts = tasks.map((t) => {
      const m = Math.max(24, Math.floor(t.train.n * frac)), d = t.train.d || 2;
      const X = new Float32Array(m * d), Y = new Float32Array(m);
      for (let i = 0; i < m; i++) {
        const j = Math.floor(rng() * t.train.n);
        X.set(t.train.X.subarray(j * d, j * d + d), i * d); Y[i] = t.train.Y[j];
      }
      return { X, Y, n: m, d };
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
    return { X, Y, n, d: 2 };
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
      desc: 'Two different skills (Moons and Circles), each expert trained from its own random init: no shared base, so no task vectors. Neuron alignment (Git Re-Basin) re-orders units before averaging; it assumes the experts learned the same functions, so here it helps only a little and can hurt in wider networks.',
      experts: [{ task: 'moons', region: 'left' }, { task: 'circles', region: 'right' }], init: 'independent',
    },
    {
      id: 'twins', name: 'Twin runs', tag: 'same skill, own inits',
      desc: 'Two runs of the same hard skill (Spiral), each from its own random init. Both learn the same function with their neurons in a different order, so plain averaging scrambles them; neuron alignment (Git Re-Basin) puts them back in register first.',
      experts: [{ task: 'spiral', region: 'left' }, { task: 'spiral', region: 'left' }], init: 'independent',
    },
    {
      id: 'hard', name: 'Hard + easy', tag: 'imbalance',
      desc: 'A hard Spiral expert next to an easy Blobs expert. The hard skill needs large weight changes that easy merges tend to wash out.',
      experts: [{ task: 'spiral', region: 'left' }, { task: 'blobs', region: 'right' }], init: 'shared',
    },
    {
      id: 'digits', name: 'Digits', tag: '8×8 images',
      desc: 'Image-like inputs: Expert A tells 0 from 1, Expert B tells 2 from 3, on noisy 8×8 digits (64 pixels in, not 2 coordinates). The map becomes a gallery of test digits: tint = predicted class, red ring = mistake.',
      experts: [{ task: 'd01' }, { task: 'd23' }], init: 'shared',
    },
    {
      id: 'digits-conflict', name: 'Digits conflict', tag: 'both see 1s',
      desc: 'Expert A tells 0 from 1; Expert B tells 1 from 7. Both see 1s and label them oppositely, the image-world version of Overlapping skills: no single model can satisfy both on the 1s, and a router can’t tell whose 1 it is.',
      experts: [{ task: 'd01' }, { task: 'd17' }], init: 'shared',
    },
  ];
  // register the digit tasks after the plane tasks so menus list them last
  Object.assign(TASKS, DIGIT_TASKS);

  Object.assign(MM, { TASKS, REGIONS, EXTENT, SCALE, DIGIT_D, taskKind, makeTaskData, concat, pretrainSet, genericSet, genericDigitSet, SCENARIOS });
})();
