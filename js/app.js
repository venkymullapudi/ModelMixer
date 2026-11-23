/* ModelMixer — application: state, training loop, lab / arena / atlas / export views. */
(function () {
  const MM = window.MM;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const LETTERS = 'ABC';
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const frame = () => new Promise((r) => setTimeout(r, 0));

  // ---------------- training off the main thread (Web Worker) ----------------
  // The single-file build tags its inline engine scripts; a Blob-URL worker re-runs them, so training never
  // blocks the page and isn't throttled in background tabs. Without them (multi-file dev mode) or if workers
  // are unavailable, training falls back to the chunked main-thread loop. Results are identical either way.
  const ENGINE_SRC = (() => {
    try {
      const parts = Array.from(document.querySelectorAll('script[data-engine]')).map((s) => s.textContent);
      return parts.length >= 5 && parts.every((t) => t.trim()) ? parts.join('\n;\n') : null;
    } catch (e) { return null; }
  })();
  let workerURL = null;
  let workerOK = !!ENGINE_SRC && typeof Worker !== 'undefined' && typeof Blob !== 'undefined';
  function workerMain() {
    self.onmessage = (e) => {
      const { kind, cfg, every } = e.data;
      if (kind === 'score') {
        // score recipes on a trained bundle; send back numbers only (results hold closures)
        try {
          const b = e.data.bundle, recipes = e.data.recipes, out = {};
          b.ctx = {};
          recipes.forEach((rc, n) => {
            const r = self.MM.scoreMethod(b, self.MM.bundleCtx(b, rc.align), rc.method, rc.params);
            out[rc.key] = r.blocked ? { blocked: r.blocked } : { avg: r.avg, retention: r.retention, accs: r.accs };
            self.postMessage({ type: 'progress', done: n + 1, total: recipes.length, key: rc.key });
          });
          self.postMessage({ type: 'result', result: out });
        } catch (err) { self.postMessage({ type: 'error', message: String((err && err.message) || err) }); }
        return;
      }
      try {
        const it = self.MM.trainBundle(cfg);
        let r, done = 0, total = 1, live = null;
        while (!(r = it.next()).done) {
          const ev = r.value;
          if (ev.type === 'start') { total = ev.total; live = ev.live; }
          else if (ev.type === 'step') { if (++done % every === 0) self.postMessage({ type: 'progress', done, total, which: ev.which, P: ev.P.slice(), losses: live.losses.slice() }); }
          else if (ev.type === 'done') self.postMessage({ type: 'progress', done, total, which: ev.which, P: ev.P.slice(), losses: live.losses.slice() });
        }
        const b = r.value;
        delete b.ctx;
        self.postMessage({ type: 'result', result: b });
      } catch (err) { self.postMessage({ type: 'error', message: String((err && err.message) || err) }); }
    };
  }
  async function trainMainThread(cfg, opt) {
    const it = MM.trainBundle(cfg);
    let r, done = 0, total = 1, live = null, tick = performance.now();
    while (!(r = it.next()).done) {
      const ev = r.value;
      if (ev.type === 'start') { total = ev.total; live = ev.live; }
      else if (ev.type === 'step') { if (++done % 40 === 0 && opt.onProgress) opt.onProgress({ done, total, which: ev.which, P: ev.P, losses: live.losses }); }
      else if (ev.type === 'done' && opt.onProgress) opt.onProgress({ done, total, which: ev.which, P: ev.P, losses: live.losses });
      if (performance.now() - tick > 30) { await frame(); tick = performance.now(); if (opt.cancelled && opt.cancelled()) return null; }
    }
    return r.value;
  }
  /** Run one job in a fresh worker. Resolves to the job's result, or null if cancelled; `fallback` runs in-page. */
  function workerJob(msg, opt, fallback) {
    if (!workerOK) return fallback();
    return new Promise((resolve, reject) => {
      let w;
      try {
        if (!workerURL) workerURL = URL.createObjectURL(new Blob([`self.window = self;\n${ENGINE_SRC}\n;(${workerMain.toString()})();`], { type: 'text/javascript' }));
        w = new Worker(workerURL);
      } catch (e) { workerOK = false; resolve(fallback()); return; }
      let settled = false;
      const end = () => { settled = true; clearInterval(poll); w.terminate(); };
      const poll = setInterval(() => { if (!settled && opt.cancelled && opt.cancelled()) { end(); resolve(null); } }, 100);
      w.onmessage = (e) => {
        const m = e.data;
        if (settled) return;
        if (m.type === 'progress') { if (opt.onProgress) opt.onProgress(m); }
        else if (m.type === 'result') { end(); resolve(m.result); }
        else if (m.type === 'error') { end(); reject(new Error(m.message)); }
      };
      w.onerror = (ev) => { if (ev && ev.preventDefault) ev.preventDefault(); if (settled) return; end(); workerOK = false; resolve(fallback()); };
      w.postMessage(msg);
    });
  }
  /** Train a bundle; resolves to the bundle, or null if cancelled. opt: { onProgress, cancelled }. */
  async function trainAsync(cfg, opt) {
    opt = opt || {};
    const plain = JSON.parse(JSON.stringify(cfg));
    const b = await workerJob({ kind: 'train', cfg: plain, every: 40 }, opt, () => trainMainThread(plain, opt));
    if (b) b.ctx = b.ctx || {};
    return b;
  }
  /** Score recipes [{key, method, params, align}] on a bundle → { key: {avg, retention, accs} | {blocked} }. */
  function scoreAsync(bundle, recipes, opt) {
    opt = opt || {};
    const plainRecipes = JSON.parse(JSON.stringify(recipes));
    const inPage = async () => {
      const out = {};
      for (let n = 0; n < plainRecipes.length; n++) {
        const rc = plainRecipes[n];
        await frame();
        if (opt.cancelled && opt.cancelled()) return null;
        const r = MM.scoreMethod(bundle, MM.bundleCtx(bundle, rc.align), rc.method, rc.params);
        out[rc.key] = r.blocked ? { blocked: r.blocked } : { avg: r.avg, retention: r.retention, accs: r.accs };
        if (opt.onProgress) opt.onProgress({ done: n + 1, total: plainRecipes.length, key: rc.key });
      }
      return out;
    };
    // ship the bundle without its contexts (they hold caches and closures) or live training state
    const { ctx, live, ...plainBundle } = bundle;
    return workerJob({ kind: 'score', bundle: plainBundle, recipes: plainRecipes }, opt, inPage);
  }
  const MAX_PARALLEL = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));

  // ---------------- state ----------------
  const S = {
    scenario: 'two',
    experts: MM.SCENARIOS[0].experts.map((e) => ({ ...e })),
    init: 'shared', hidden: 32, ftSteps: 250, seed: 7, align: false,
    method: 'ties', params: {},
    overlay: { points: true, regions: true, router: true, diff: true },
    tab: 'lab',
    advisor: {}, // answers to the recipe advisor
    sweepKnob: {}, // method id -> knob key shown in the sensitivity chart
    sweepSeeds: false, // add a ±1 sd band from 2 extra seeds
    robustSeeds: 5, // seeds in the Arena's multi-seed check
    gridPair: '0-2', // layers shown in the per-layer heat map (hidden 1 × output)
    pin: null, // { method, params, align } recipe pinned for side-by-side comparison
    exp: {
      base: 'mistralai/Mistral-7B-v0.1',
      models: ['teknium/OpenHermes-2.5-Mistral-7B', 'meta-math/MetaMath-Mistral-7B', 'HuggingFaceH4/zephyr-7b-beta'],
      dtype: 'bfloat16',
      prompts: ['chat, conversation, assistant', 'math, arithmetic, solve, equation', 'helpful, explain, summarize'],
    },
  };
  MM.METHODS.forEach((m) => (S.params[m.id] = MM.defaultParams(m)));

  const PARAMS_V = 2;
  try {
    const saved = JSON.parse(localStorage.getItem('modelmixer:v1') || 'null');
    if (saved) {
      for (const k of ['scenario', 'experts', 'init', 'hidden', 'ftSteps', 'seed', 'align', 'method', 'overlay', 'tab', 'pin', 'advisor', 'sweepKnob', 'sweepSeeds', 'robustSeeds', 'gridPair']) if (saved[k] !== undefined) S[k] = saved[k];
      // v2 defaults (2026-11 review): drop knobs saved under the old defaults so everyone starts from the new ones
      if (saved.params && saved.paramsV === PARAMS_V) for (const id in saved.params) if (S.params[id]) Object.assign(S.params[id], saved.params[id]);
      if (saved.exp) Object.assign(S.exp, saved.exp);
      if (!MM.METHOD_BY_ID[S.method]) S.method = 'ties';
      if (S.pin && !MM.METHOD_BY_ID[S.pin.method]) S.pin = null;
    }
  } catch (e) { /* storage unavailable */ }
  function persist() {
    recordHistory();
    try {
      const { scenario, experts, init, hidden, ftSteps, seed, align, method, params, overlay, tab, exp, pin, advisor, sweepKnob, sweepSeeds, robustSeeds, gridPair } = S;
      localStorage.setItem('modelmixer:v1', JSON.stringify({ scenario, experts, init, hidden, ftSteps, seed, align, method, params, paramsV: PARAMS_V, overlay, tab, exp, pin, advisor, sweepKnob, sweepSeeds, robustSeeds, gridPair }));
    } catch (e) { /* ignore */ }
    syncHash();
  }

  // ---------------- undo / redo (recipe state: method, knobs, per-layer values, alignment, pin) ----------------
  // Training settings are not recorded, so undo never triggers retraining.
  const undoHistory = MM.makeHistory(100, 700); // not `history`: that would shadow window.history (share-link sync)
  let restoring = false;
  const editKey = MM.editKeyTracker();
  const recipeSnap = () => JSON.stringify({ method: S.method, params: S.params, align: S.align, pin: S.pin });
  function recordHistory() {
    if (restoring || !undoHistory) return;
    // coalesce rapid knob edits (a slider drag) into one step, but never across a pin, alignment or method change
    if (undoHistory.record(recipeSnap(), performance.now(), editKey(S.method, S.align, S.pin))) updateUndoButtons();
  }
  function updateUndoButtons() {
    const u = document.getElementById('undoBtn'), r = document.getElementById('redoBtn');
    if (u) u.disabled = !undoHistory.canUndo();
    if (r) r.disabled = !undoHistory.canRedo();
  }
  function restoreSnap(snap, verb) {
    if (!snap) { toast(verb === 'Undo' ? 'Nothing to undo' : 'Nothing to redo'); return; }
    const o = JSON.parse(snap);
    const alignChanged = S.align !== o.align;
    const refocus = focusSelector(); // re-rendering replaces the focused control; put keyboard focus back on it
    restoring = true;
    S.method = o.method; S.params = o.params; S.align = o.align; S.pin = o.pin;
    persist();
    restoring = false;
    arenaStale = robustStale = true;
    renderPicker();
    if (alignChanged) renderIngredients();
    refresh();
    updateUndoButtons();
    if (refocus && (!document.activeElement || document.activeElement === document.body)) {
      const el = document.querySelector(refocus);
      if (el) el.focus({ preventScroll: true });
    }
    toast(`${verb}: ${MM.recipeLabel(S.method, S.params[S.method], S.align, S.experts.length)}`);
  }
  /** A selector that finds the focused control again after a re-render (by id or its data-* key), or null. */
  function focusSelector() {
    const e = document.activeElement;
    if (!e || e === document.body) return null;
    if (e.id) return `#${CSS.escape(e.id)}`;
    for (const a of ['param', 'weight', 'grad', 'method', 'pin']) if (e.dataset[a] != null) return `[data-${a}="${CSS.escape(e.dataset[a])}"]`;
    return null;
  }
  const undo = () => restoreSnap(undoHistory.undo(), 'Undo');
  const redo = () => restoreSnap(undoHistory.redo(), 'Redo');
  document.addEventListener('keydown', (ev) => {
    if (!(ev.metaKey || ev.ctrlKey) || ev.altKey) return;
    const t = ev.target;
    // leave text editing alone (textareas, text fields, contenteditable)
    if (t && (t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && !/^(range|checkbox|radio|button)$/i.test(t.type)) || t.isContentEditable)) return;
    const k = ev.key.toLowerCase();
    if (k === 'z' && !ev.shiftKey) { ev.preventDefault(); undo(); }
    else if ((k === 'z' && ev.shiftKey) || k === 'y') { ev.preventDefault(); redo(); }
  });

  // ---------------- shareable recipe links (#r=…) ----------------
  let hashTimer;
  const linkable = () => /^(https?|file):$/.test(location.protocol);
  function currentRecipe() {
    return { experts: S.experts, init: S.init, hidden: S.hidden, ftSteps: S.ftSteps, seed: S.seed, align: S.align, method: S.method, params: S.params[S.method], pin: S.pin };
  }
  function recipeLink() {
    return `${linkable() ? location.href.split('#')[0] : 'modelmixer.html'}#r=${MM.encodeRecipe(currentRecipe())}`;
  }
  // Keep the address bar pointing at the current recipe, so the URL itself is always shareable.
  function syncHash() {
    clearTimeout(hashTimer);
    hashTimer = setTimeout(() => {
      try { if (linkable()) history.replaceState(null, '', `#r=${MM.encodeRecipe(currentRecipe())}`); } catch (e) { /* sandboxed */ }
    }, 300);
  }
  const trainKey = (r) => JSON.stringify([r.experts, r.init, r.hidden, r.ftSteps, r.seed]);
  function applyRecipe(r) {
    S.experts = r.experts; S.init = r.init; S.hidden = r.hidden; S.ftSteps = r.ftSteps; S.seed = r.seed;
    S.align = r.align; S.method = r.method; S.params[r.method] = r.params; S.pin = r.pin;
    if (r.pin) S.params[r.pin.method] = S.params[r.pin.method] || MM.defaultParams(MM.METHOD_BY_ID[r.pin.method]);
    const sc = MM.SCENARIOS.find((x) => sameScenario(x));
    S.scenario = sc ? sc.id : 'custom';
  }

  let T = null; // trained bundle
  let trainToken = 0;
  let training = false;
  let arenaStale = true;
  let robustStale = true; // robustness results no longer match the current settings

  // ---------------- status / toast ----------------
  function status(text, busy) {
    $('#statusText').textContent = text;
    $('#status').classList.toggle('busy', !!busy);
  }
  let toastTimer;
  function toast(msg) {
    const t = $('#toast');
    t.textContent = msg; t.classList.add('show');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 1800);
  }

  // ---------------- training ----------------
  async function train() {
    const token = ++trainToken;
    training = true;
    arenaStale = robustStale = true;
    renderIngredients();
    status('Training experts…', true);
    const cfg = { experts: S.experts, init: S.init, hidden: S.hidden, ftSteps: S.ftSteps, seed: S.seed };
    const shape = MM.bundleShape(cfg);
    T = Object.assign(T || {}, { live: { layout: shape.layout, tasks: shape.tasks, base: null, experts: [], losses: [] } });
    try {
      const res = await trainAsync(cfg, {
        cancelled: () => token !== trainToken,
        onProgress: (m) => {
          if (token !== trainToken || !T.live) return;
          T.live.losses = m.losses || T.live.losses;
          if (m.which === 'base') T.live.base = m.P; else T.live.experts[m.which] = m.P;
          setProgress(m.done / m.total, T.live.losses);
          drawThumb(m.which, m.P, T.live.tasks);
        },
      });
      if (!res || token !== trainToken) throw new Error('cancelled');
      T = res;
      training = false;
      setProgress(1, T.losses);
      status(`${T.K} experts ready · ${T.layout.total.toLocaleString()} params each`);
      renderIngredients();
      renderPicker();
      refresh();
      if (S.tab === 'arena') runArena();
      maybeStartTour();
    } catch (e) {
      if (e.message !== 'cancelled') { console.error(e); status('Training failed — see console'); training = false; }
    }
  }

  function setProgress(f, losses) {
    const bar = $('#trainBar');
    if (bar) bar.style.width = `${(f * 100).toFixed(1)}%`;
    const sp = $('#lossSpark');
    if (sp && losses.length > 1) MM.sparkline(sp, losses, '#a78bfa');
  }

  function getCtx() {
    if (!T || !T.experts) return null;
    return MM.bundleCtx(T, S.align);
  }

  function compute(id, ctx) {
    return MM.scoreMethod(T, ctx, id, S.params[id]);
  }

  // ---------------- thumbnails ----------------
  function drawThumb(which, P, tasks) {
    const cv = which === 'base' ? $('#thumb-base') : $(`#thumb-${which}`);
    if (!cv || !T) return;
    const layout = (T.live && T.live.layout) || T.layout;
    MM.drawMap(cv, (X, n) => MM.predictProb(layout, P, X, n), { grid: 54, thumb: true, tasks, showPoints: false });
  }

  // ---------------- INGREDIENTS ----------------
  function renderIngredients() {
    const K = S.experts.length;
    const sc = MM.SCENARIOS.find((s) => s.id === S.scenario);
    // experts in one mix must share an input space: plane tasks or 8×8 digit tasks, never both
    const kind = MM.taskKind(S.experts[0].task);
    const taskOpts = (v) => Object.entries(MM.TASKS).filter(([k]) => MM.taskKind(k) === kind).map(([k, t]) => `<option value="${k}" ${k === v ? 'selected' : ''}>${t.name}</option>`).join('');
    const regOpts = (v) => Object.entries(MM.REGIONS).map(([k, r]) => `<option value="${k}" ${k === v ? 'selected' : ''}>${r.name}</option>`).join('');
    const ready = T && T.experts && !training && T.K === K; // T can lag behind S.experts until retraining starts
    const accBadge = (i) => (ready && T.expertAccs[i] ? `<span class="acc" title="Accuracy on its own task">${pct(T.expertAccs[i][i])}</span>` : '');
    $('#ingredients').innerHTML = `
      <div class="panel-title"><span>Ingredients</span><span class="hint">the experts you mix</span></div>
      <div class="scenarios">
        ${MM.SCENARIOS.map((s) => `<button class="scenario ${s.id === S.scenario ? 'active' : ''}" data-scenario="${s.id}"><b>${s.name}</b><small>${s.tag}</small></button>`).join('')}
      </div>
      <p class="scenario-desc">${sc ? sc.desc : 'Custom mix — tweak the experts below.'}</p>

      <div class="experts">
        ${S.experts.map((e, i) => `
          <div class="expert" style="--c:${MM.MODEL_COLORS[i]}">
            <canvas class="thumb" id="thumb-${i}"></canvas>
            <div class="expert-meta">
              <div class="expert-name"><i class="dot"></i>Expert ${LETTERS[i]} ${accBadge(i)}
                ${K > 2 ? `<button class="icon-btn" data-remove="${i}" title="Remove expert">×</button>` : ''}</div>
              <div class="expert-row">
                <select data-ex="${i}" data-field="task" aria-label="Task">${taskOpts(e.task)}</select>
                ${kind === 'digits' ? '' : `<select data-ex="${i}" data-field="region" aria-label="Region">${regOpts(e.region)}</select>`}
              </div>
              <label class="mini-check"><input type="checkbox" data-ex="${i}" data-field="flip" ${e.flip ? 'checked' : ''}/> invert labels</label>
            </div>
          </div>`).join('')}
        ${K < 3 ? `<button class="add-expert" id="addExpert">+ Add expert C</button>` : ''}
        <div class="expert base ${S.init === 'shared' ? '' : 'off'}">
          ${S.init === 'shared'
            ? `<canvas class="thumb" id="thumb-base"></canvas>
               <div class="expert-meta"><div class="expert-name"><i class="dot base"></i>Shared base
               ${ready && T.baseAccs ? `<span class="acc muted">${pct(T.baseAccs.reduce((a, b) => a + b, 0) / K)}</span>` : ''}</div>
               <p class="tiny">Pre-trained on a generic pattern plus a pinch of each task; every expert is fine-tuned from it.</p></div>`
            : `<div class="expert-meta"><div class="expert-name"><i class="dot base"></i>No shared base</div>
               <p class="tiny">Each expert is trained from its own random init. Task-vector methods are unavailable; try neuron alignment.</p></div>`}
        </div>
      </div>

      <div class="settings">
        <div class="setting">
          <label>Lineage</label>
          <div class="seg" data-setting="init">
            <button data-v="shared" class="${S.init === 'shared' ? 'on' : ''}">Shared base</button>
            <button data-v="independent" class="${S.init === 'independent' ? 'on' : ''}">Independent</button>
          </div>
        </div>
        <div class="setting">
          <label>Width</label>
          <div class="seg" data-setting="hidden">
            ${[16, 32, 64].map((h) => `<button data-v="${h}" class="${S.hidden === h ? 'on' : ''}">${h}</button>`).join('')}
          </div>
        </div>
        <div class="setting">
          <label>Fine-tune steps <b id="ftVal">${S.ftSteps}</b></label>
          <input type="range" min="50" max="800" step="25" value="${S.ftSteps}" id="ftSteps" aria-label="Fine-tune steps" />
        </div>
        <div class="train-row">
          <button class="btn primary" id="retrain">${training ? 'Training…' : 'Retrain'}</button>
          <button class="btn ghost" id="reseed" title="New random seed">🎲 Seed ${S.seed}</button>
        </div>
        <div class="progress"><div id="trainBar"></div></div>
        <div class="spark" id="lossSpark" title="Training loss"></div>
      </div>

      <label class="switch-row">
        <span><b>Neuron alignment</b><small>Git Re-Basin weight matching before merging</small></span>
        <span class="switch"><input type="checkbox" id="alignToggle" ${S.align ? 'checked' : ''}/><i></i></span>
      </label>
      ${S.align && getCtx() && getCtx().moved ? `<p class="tiny align-note">Units permuted: ${getCtx().moved.map((m, i) => `${LETTERS[i]} ${pct(m)}`).join(' · ')}</p>` : ''}
    `;
    if (T && T.experts && !training) {
      T.experts.forEach((e, i) => drawThumb(i, e, T.tasks));
      if (T.base) drawThumb('base', T.base, T.tasks);
    } else if (T && T.live) {
      T.live.experts.forEach((e, i) => e && drawThumb(i, e, T.live.tasks));
      if (T.live.base) drawThumb('base', T.live.base, T.live.tasks);
    }
    if (T && T.losses) setProgress(training ? 0 : 1, T.losses);
  }

  let retrainTimer;
  function scheduleRetrain() {
    S.scenario = MM.SCENARIOS.find((s) => sameScenario(s)) ? MM.SCENARIOS.find((s) => sameScenario(s)).id : 'custom';
    persist();
    clearTimeout(retrainTimer);
    retrainTimer = setTimeout(train, 250);
    renderIngredients();
  }
  function sameScenario(s) {
    return s.init === S.init && s.experts.length === S.experts.length &&
      s.experts.every((e, i) => e.task === S.experts[i].task && e.region === S.experts[i].region && !!e.flip === !!S.experts[i].flip);
  }

  $('#ingredients').addEventListener('click', (ev) => {
    const b = ev.target.closest('button');
    if (!b) return;
    if (b.dataset.scenario) {
      const sc = MM.SCENARIOS.find((s) => s.id === b.dataset.scenario);
      S.experts = sc.experts.map((e) => ({ ...e }));
      S.init = sc.init;
      S.align = sc.init === 'independent' ? S.align : false;
      fixMethodForScenario();
      scheduleRetrain();
    } else if (b.dataset.remove) {
      S.experts.splice(+b.dataset.remove, 1);
      scheduleRetrain();
    } else if (b.id === 'addExpert') {
      if (MM.taskKind(S.experts[0].task) === 'digits') {
        const usedT = new Set(S.experts.map((e) => e.task));
        S.experts.push({ task: ['d45', 'd67', 'd89', 'd23', 'd01'].find((t) => !usedT.has(t)) || 'd45' });
      } else {
        const used = new Set(S.experts.map((e) => e.region));
        const region = ['top', 'right', 'left', 'center'].find((r) => !used.has(r)) || 'top';
        S.experts.push({ task: 'xor', region });
      }
      scheduleRetrain();
    } else if (b.closest('.seg')) {
      const key = b.closest('.seg').dataset.setting;
      S[key] = key === 'hidden' ? +b.dataset.v : b.dataset.v;
      if (key === 'init') fixMethodForScenario();
      scheduleRetrain();
    } else if (b.id === 'retrain') {
      train();
    } else if (b.id === 'reseed') {
      S.seed = 1 + Math.floor(Math.random() * 999);
      persist();
      train();
    }
  });
  $('#ingredients').addEventListener('change', (ev) => {
    const t = ev.target;
    if (t.dataset.ex !== undefined) {
      const e = S.experts[+t.dataset.ex];
      e[t.dataset.field] = t.type === 'checkbox' ? t.checked : t.value;
      scheduleRetrain();
    } else if (t.id === 'alignToggle') {
      S.align = t.checked;
      persist();
      arenaStale = robustStale = true;
      status(S.align ? 'Aligning neurons…' : 'Alignment off', S.align);
      setTimeout(() => {
        renderIngredients();
        refresh();
        status(T && T.experts ? `${T.K} experts ready${S.align ? ' · aligned' : ''}` : '');
        if (S.tab === 'arena') runArena();
      }, 20);
    } else if (t.id === 'ftSteps') {
      S.ftSteps = +t.value;
      scheduleRetrain();
    }
  });
  $('#ingredients').addEventListener('input', (ev) => {
    if (ev.target.id === 'ftSteps') $('#ftVal').textContent = ev.target.value;
  });

  function fixMethodForScenario() {
    if (S.init === 'independent' && MM.METHOD_BY_ID[S.method].req.base) S.method = 'linear';
  }

  // ---------------- METHOD PICKER ----------------
  function renderPicker() {
    const hasBase = S.init === 'shared';
    const groups = Object.entries(MM.FAMILIES).map(([fid, f]) => {
      const ms = MM.METHODS.filter((m) => m.family === fid);
      return `<div class="fam" style="--fc:${f.color}">
        <div class="fam-name"><i></i>${f.name}</div>
        <div class="chips">${ms.map((m) => {
          const locked = m.req.base && !hasBase;
          return `<button class="chip ${m.id === S.method ? 'active' : ''} ${locked ? 'locked' : ''}" data-method="${m.id}" tabindex="${m.id === S.method ? 0 : -1}" title="${esc(m.tagline)}${locked ? ' — needs a shared base' : ''}">${locked ? '<span class="lock">🔒</span>' : ''}${m.short}</button>`;
        }).join('')}</div>
      </div>`;
    });
    // re-rendering replaces the buttons, so keep keyboard focus on the chosen chip
    const hadFocus = $('#picker').contains(document.activeElement);
    $('#picker').innerHTML = `<div class="families">${groups.join('')}</div>`;
    if (hadFocus) { const b = $(`#picker [data-method="${S.method}"]`); if (b) b.focus(); }
  }
  $('#picker').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-method]');
    if (!b) return;
    selectMethod(b.dataset.method);
  });
  function selectMethod(id) {
    S.method = id;
    persist();
    renderPicker();
    refresh();
  }

  // ---------------- CONTROLS ----------------
  function renderControls() {
    const m = MM.METHOD_BY_ID[S.method], p = S.params[m.id], K = S.experts.length;
    const g = m.gradient && !(m.gradient.k2 && K !== 2) ? m.gradient : null;
    const lw = !!(g && p.layerwise);
    const rows = m.params.map((d) => {
      if (lw && (d.key === g.key || (g.key === 'balance' && d.type === 'weights'))) {
        return `<div class="ctl off"><label>${d.label}<b>per layer ↓</b></label></div>`;
      }
      if (d.type === 'range') {
        return `<div class="ctl"><label>${d.label}<b data-out="${d.key}">${fmt(p[d.key], d.step)}</b></label>
          <input type="range" aria-label="${esc(d.label)}" data-param="${d.key}" min="${d.min}" max="${d.max}" step="${d.step}" value="${p[d.key]}" /></div>`;
      }
      if (d.type === 'select') {
        return `<div class="ctl"><label>${d.label}</label><div class="seg wide" data-param-seg="${d.key}">
          ${d.options.map(([v, l]) => `<button data-v="${v}" class="${String(p[d.key]) === v ? 'on' : ''}">${l}</button>`).join('')}</div></div>`;
      }
      if (d.type === 'weights') {
        return `<div class="ctl"><label>${d.label}</label>${Array.from({ length: K }, (_, i) => `
          <div class="wrow" style="--c:${MM.MODEL_COLORS[i]}"><i class="dot"></i><span>${LETTERS[i]}</span>
            <input type="range" data-weight="${i}" min="0" max="1" step="0.05" value="${p.weights[i]}" aria-label="Weight of expert ${LETTERS[i]}" /><b data-wout="${i}">${(+p.weights[i]).toFixed(2)}</b></div>`).join('')}</div>`;
      }
      if (d.type === 'layers') {
        const names = ['Hidden 1', 'Hidden 2', 'Output'];
        return `<div class="ctl"><label>${d.label}</label>${names.map((nm, l) => `
          <div class="lrow"><span>${nm}</span><div class="seg" data-layer="${l}">
            ${Array.from({ length: K }, (_, i) => `<button data-v="${i}" class="${+p.layers[l] === i ? 'on' : ''}" style="--c:${MM.MODEL_COLORS[i]}">${LETTERS[i]}</button>`).join('')}
          </div></div>`).join('')}</div>`;
      }
      return '';
    });
    const gfmt = (v) => (g && g.key === 'balance' ? `${Math.round((1 - v) * 100)}/${Math.round(v * 100)}` : (+v).toFixed(2));
    const grad = g ? `<div class="ctl grad">
        <label class="switch-row"><span><b>Per-layer ${esc(g.label)}</b><small>mergekit gradient: one value per layer</small></span>
          <span class="switch"><input type="checkbox" id="gradToggle" ${lw ? 'checked' : ''}/><i></i></span></label>
        ${lw ? `${MM.LAYER_NAMES.map((nm, l) => `<div class="grow"><span>${nm}</span>
            <input type="range" aria-label="${nm} ${esc(g.label)}" data-grad="${l}" min="${g.min}" max="${g.max}" step="${g.step}" value="${p.grad[l]}" /><b data-gout="${l}">${gfmt(p.grad[l])}</b></div>`).join('')}
          <div class="gpresets"><span class="tiny muted">Presets</span>${[['flat', 'Flat'], ['up', 'Ramp ↑'], ['down', 'Ramp ↓'], ['mid', 'Middle-heavy']].map(([k, l]) => `<button class="tg" data-gpreset="${k}">${l}</button>`).join('')}</div>` : ''}
      </div>` : '';
    const mac = /Mac|iP(hone|ad)/.test(navigator.platform || navigator.userAgent);
    $('#controls').innerHTML = `<div class="block-title recipe-title">Recipe<span class="undo-row">
        <button class="btn ghost sm icon" id="undoBtn" title="Undo (${mac ? '⌘Z' : 'Ctrl+Z'})" aria-label="Undo">⟲</button>
        <button class="btn ghost sm icon" id="redoBtn" title="Redo (${mac ? '⇧⌘Z' : 'Ctrl+Y'})" aria-label="Redo">⟳</button></span></div>${rows.join('') || (grad ? '' : '<p class="muted small">No knobs — this method decides everything from the weights themselves.</p>')}${grad}
      <button class="btn ghost sm reset" id="resetParams">Reset to defaults</button>`;
  }
  const fmt = (v, step) => (step >= 1 ? String(v) : (+v).toFixed(step < 0.01 ? 3 : 2));

  let refreshQueued = false;
  function queueRefresh() {
    if (refreshQueued) return;
    refreshQueued = true;
    requestAnimationFrame(() => { refreshQueued = false; refresh(true); });
  }
  $('#controls').addEventListener('input', (ev) => {
    const t = ev.target, p = S.params[S.method];
    if (t.dataset.param) { p[t.dataset.param] = +t.value; $(`[data-out="${t.dataset.param}"]`).textContent = fmt(t.value, +t.step); }
    else if (t.dataset.weight !== undefined) { p.weights[+t.dataset.weight] = +t.value; $(`[data-wout="${t.dataset.weight}"]`).textContent = (+t.value).toFixed(2); }
    else if (t.dataset.grad !== undefined) {
      p.grad[+t.dataset.grad] = +t.value;
      const g = MM.METHOD_BY_ID[S.method].gradient;
      $(`[data-gout="${t.dataset.grad}"]`).textContent = g.key === 'balance' ? `${Math.round((1 - t.value) * 100)}/${Math.round(t.value * 100)}` : (+t.value).toFixed(2);
    }
    else return;
    persist(); arenaStale = robustStale = true;
    queueRefresh();
  });
  $('#controls').addEventListener('change', (ev) => {
    if (ev.target.id !== 'gradToggle') return;
    const m = MM.METHOD_BY_ID[S.method], p = S.params[m.id], g = m.gradient;
    p.layerwise = ev.target.checked;
    // start from the current scalar so switching on changes nothing until you move a layer
    if (p.layerwise) {
      const cur = g.key === 'balance' ? MM.knobValue(p, { key: 'balance' }) : +p[g.key];
      if (p.grad.every((v) => v === p.grad[0])) p.grad = [cur, cur, cur];
    }
    persist(); arenaStale = robustStale = true;
    refresh();
  });
  $('#controls').addEventListener('click', (ev) => {
    const b = ev.target.closest('button');
    if (!b) return;
    if (b.id === 'undoBtn') { undo(); return; }
    if (b.id === 'redoBtn') { redo(); return; }
    const p = S.params[S.method];
    if (b.dataset.gpreset) {
      const g = MM.METHOD_BY_ID[S.method].gradient;
      const mid = p.grad.reduce((a, v) => a + v, 0) / 3;
      const d = Math.min((g.max - g.min) / 4, mid - g.min, g.max - mid) || (g.max - g.min) / 4;
      const shapes = { flat: [mid, mid, mid], up: [mid - d, mid, mid + d], down: [mid + d, mid, mid - d], mid: [mid - d, mid + d, mid - d] };
      const snap = (v) => +Math.min(g.max, Math.max(g.min, Math.round(v / g.step) * g.step)).toFixed(4);
      p.grad = shapes[b.dataset.gpreset].map(snap);
      persist(); arenaStale = robustStale = true;
      refresh();
      return;
    }
    if (b.id === 'resetParams') { S.params[S.method] = MM.defaultParams(MM.METHOD_BY_ID[S.method]); }
    else if (b.closest('[data-param-seg]')) p[b.closest('[data-param-seg]').dataset.paramSeg] = b.dataset.v;
    else if (b.closest('[data-layer]')) p.layers[+b.closest('[data-layer]').dataset.layer] = +b.dataset.v;
    else return;
    persist(); arenaStale = robustStale = true;
    refresh();
  });

  // ---------------- STAGE + METRICS ----------------
  let current = null;
  let cmp = null;
  // ---------------- COMPARE (pin a recipe, tweak, see both) ----------------
  const pinCache = { key: null, T: null, r: null };
  const cloneParams = (p) => JSON.parse(JSON.stringify(p));
  function pinnedResult() {
    if (!S.pin || !T || !T.experts) return null;
    const key = JSON.stringify(S.pin);
    if (pinCache.T !== T || pinCache.key !== key) {
      pinCache.r = MM.scoreMethod(T, MM.bundleCtx(T, S.pin.align), S.pin.method, S.pin.params);
      pinCache.key = key; pinCache.T = T;
    }
    return pinCache.r;
  }
  const labelOf = (rec) => MM.recipeLabel(rec.method, rec.params, rec.align, S.experts.length);
  const nowRecipe = () => ({ method: S.method, params: cloneParams(S.params[S.method]), align: S.align });
  function drawPin(pin, r) {
    const on = !!S.pin;
    $('#cmpRow').classList.toggle('on', on);
    $('#pinPane').hidden = !on;
    $('#nowCap').hidden = !on;
    if (!on) return;
    $('#pinCap').innerHTML = `<span class="who">A · pinned</span><span class="rl" title="${esc(labelOf(S.pin))}">${esc(labelOf(S.pin))}</span><span class="acc">${pin && !pin.blocked ? pct(pin.avg) : '—'}</span>`;
    $('#nowCap').innerHTML = `<span class="who">B · now</span><span class="rl" title="${esc(labelOf(nowRecipe()))}">${esc(labelOf(nowRecipe()))}</span><span class="acc">${r && !r.blocked ? pct(r.avg) : '—'}</span>`;
    const cv = $('#pinCanvas'), bl = $('#pinBlocked');
    if (!pin || pin.blocked) {
      bl.hidden = false;
      bl.innerHTML = `<div class="blocked-card"><div class="lock-big">🔒</div><b>Pinned recipe can’t run here</b><p>${pin ? pin.blocked : ''}</p></div>`;
      cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
      return;
    }
    bl.hidden = true;
    const ctx = MM.bundleCtx(T, S.pin.align);
    MM.drawMap(cv, pin.predict, {
      tasks: ctx.tasks, showPoints: S.overlay.points, showRegions: S.overlay.regions,
      gates: pin.router && S.overlay.router ? pin.router.gates : null, K: ctx.K,
    });
  }
  function setAlign(v) {
    if (S.align === v) return;
    S.align = v; arenaStale = robustStale = true; renderIngredients();
  }
  $('#stageHead').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-pin]');
    if (!b) return;
    const act = b.dataset.pin;
    if (act === 'pin') { S.pin = nowRecipe(); toast('Pinned as A. Now change the method or knobs to compare.'); }
    else if (act === 'unpin') { S.pin = null; }
    else if (act === 'swap' && S.pin) {
      const old = S.pin;
      S.pin = nowRecipe();
      S.params[old.method] = cloneParams(old.params);
      setAlign(old.align);
      if (S.method !== old.method) { selectMethod(old.method); persist(); return; }
    }
    persist();
    refresh();
  });
  function refresh(light) {
    if (!light) { renderControls(); updateUndoButtons(); }
    const ctx = getCtx();
    const m = MM.METHOD_BY_ID[S.method];
    renderStageHead(m, null);
    if (!ctx) return;
    const heavy = m.id === 'distill' || (m.id === 'moe' && !ctx.cache.router) || (m.id === 'fisher' && !ctx.cache.fisher) || (m.id === 'regmean' && !ctx.cache.grams) || (m.id === 'adamerging' && !ctx.cache[`ada:${S.params.adamerging.mode === 'task' ? 'task' : 'layer'}:${S.params.adamerging.objective === 'pseudo' ? 'pseudo' : 'entropy'}`]) || (m.id === 'localize_stitch' && S.params.localize_stitch.mode !== 'dataless' && !ctx.cache[`ls:data:${S.params.localize_stitch.density}`]);
    const go = () => {
      const t0 = performance.now();
      const r = compute(m.id, ctx);
      current = r;
      const pin = pinnedResult();
      cmp = pin && !pin.blocked && !r.blocked ? MM.compareResults(T.tasks, pin, r) : null;
      $('#busy').hidden = true;
      drawStage(r, ctx);
      drawPin(pin, r);
      renderStageHead(m, r);
      renderMetrics(r, pin);
      renderSweep(m, r);
      renderInsight(m, r, ctx);
      const ms = performance.now() - t0;
      if (!training) status(`${m.name} · mixed in ${ms.toFixed(0)} ms`);
    };
    if (heavy) {
      $('#busyText').textContent = m.id === 'distill' ? 'Distilling student…' : m.id === 'moe' ? 'Training router…' : m.id === 'adamerging' ? 'Learning coefficients…' : m.id === 'localize_stitch' ? 'Localizing skills…' : 'Measuring importance…';
      $('#busy').hidden = false;
      setTimeout(go, 30);
    } else go();
  }

  function drawStage(r, ctx) {
    const cv = $('#mainCanvas');
    const blocked = $('#blocked');
    if (r.blocked) {
      blocked.hidden = false;
      blocked.innerHTML = `<div class="blocked-card"><div class="lock-big">🔒</div><b>Missing piece</b><p>${r.blocked}</p>
        <div class="row"><button class="btn sm" data-goto-method="linear">Try Linear + alignment</button><button class="btn ghost sm" data-goto-method="ensemble">Try Ensemble</button></div></div>`;
      const ctx2 = cv.getContext('2d');
      ctx2.clearRect(0, 0, cv.width, cv.height);
      return;
    }
    blocked.hidden = true;
    const pin = S.pin && S.overlay.diff !== false ? pinnedResult() : null;
    const diffWith = pin && !pin.blocked ? pin.predict : null;
    $('#lgDiff').hidden = !diffWith;
    MM.drawMap(cv, r.predict, {
      tasks: ctx.tasks, showPoints: S.overlay.points, showRegions: S.overlay.regions,
      gates: r.router && S.overlay.router ? r.router.gates : null, K: ctx.K, diffWith,
    });
    cv.setAttribute('aria-label', `${ctx.tasks[0].kind === 'digits' ? 'Test-digit gallery' : 'Decision map'} for ${MM.METHOD_BY_ID[S.method].name}: ${r.accs.map((a, i) => `${LETTERS[i]} ${MM.TASKS[ctx.tasks[i].spec.task].name} ${pct(a)}`).join(', ')}${diffWith ? '; amber marks where the pinned recipe disagrees' : ''}`);
  }
  $('#blocked').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-goto-method]');
    if (!b) return;
    if (b.dataset.gotoMethod === 'linear') { S.align = true; arenaStale = robustStale = true; renderIngredients(); }
    selectMethod(b.dataset.gotoMethod);
  });

  function renderStageHead(m, r) {
    const f = MM.FAMILIES[m.family];
    const badges = [];
    if (r && !r.blocked) {
      badges.push(`<span class="badge ${r.cost > 1.05 ? 'warn' : 'good'}">${r.cost > 1.05 ? `${r.cost.toFixed(1)}×` : '1×'} inference</span>`);
      badges.push(`<span class="badge ${r.storage > 1.05 ? 'warn' : 'good'}">${r.storage > 1.05 ? `${Math.round(r.storage)}×` : '1×'} memory</span>`);
    }
    if (m.req.base) badges.push('<span class="badge">needs base</span>');
    if (m.req.data) badges.push('<span class="badge">needs data</span>');
    if (m.req.training) badges.push('<span class="badge">needs training</span>');
    const canPin = r && !r.blocked;
    const pinBtns = S.pin
      ? `<div class="pin-actions"><button class="btn ghost sm" data-pin="swap" title="Make the pinned recipe current and pin this one">⇄ Swap</button><button class="btn ghost sm pin-on" data-pin="unpin" title="Leave compare mode">✕ Unpin A</button></div>`
      : `<button class="btn ghost sm" data-pin="pin" ${canPin ? '' : 'disabled'} title="Freeze this recipe as A, then change the method or knobs to compare against it">📌 Pin to compare</button>`;
    $('#stageHead').innerHTML = `
      <div class="sh-title"><span class="fam-tag" style="--fc:${f.color}">${f.name}</span><h2>${m.name}</h2></div>
      <div class="badges">${badges.join('')}${pinBtns}</div>`;
    const planeTasks = !T || !T.tasks || T.tasks[0].kind !== 'digits';
    const tg = planeTasks ? [['points', 'Points'], ['regions', 'Regions']] : [];
    if (m.id === 'moe' && planeTasks) tg.push(['router', 'Router']);
    if (S.pin) tg.push(['diff', 'A ≠ B']);
    $('#lgBoundary').hidden = !planeTasks; // a digit gallery has no boundary line
    $('#overlayToggles').innerHTML = tg.map(([k, l]) => `<button class="tg ${S.overlay[k] ? 'on' : ''}" data-ov="${k}">${l}</button>`).join('');
  }
  $('#overlayToggles').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-ov]');
    if (!b) return;
    S.overlay[b.dataset.ov] = !S.overlay[b.dataset.ov];
    persist();
    refresh(true);
  });

  function renderMetrics(r, pin) {
    if (r.blocked) { $('#metrics').innerHTML = ''; return; }
    const sgn = (x) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)}`;
    const cls = (x) => (x > 0.005 ? 'up' : x < -0.005 ? 'down' : '');
    const K = T.K;
    const rows = r.accs.map((a, i) => {
      const ex = T.expertAccs[i][i], bs = T.baseAccs ? T.baseAccs[i] : null;
      const t = T.tasks[i].spec;
      const delta = a - ex;
      return `<div class="mrow" style="--c:${MM.MODEL_COLORS[i]}">
        <div class="mtop"><span><i class="dot"></i>${LETTERS[i]} · ${MM.TASKS[t.task].name}${t.flip ? ' (inv)' : ''}</span>
          <span class="mval">${pct(a)} <small class="${delta >= -0.01 ? 'up' : 'down'}">${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)}</small>${cmp ? `<small class="vs ${cls(cmp.perTask[i].dAcc)}">vs A ${sgn(cmp.perTask[i].dAcc)}</small>` : ''}</span></div>
        <div class="mbar"><div class="fill" style="width:${a * 100}%"></div>
          <i class="mark ex" style="left:${ex * 100}%" title="Expert ${LETTERS[i]} alone: ${pct(ex)}"></i>
          ${bs !== null ? `<i class="mark bs" style="left:${bs * 100}%" title="Base: ${pct(bs)}"></i>` : ''}
          <i class="mark chance" style="left:50%" title="Coin flip"></i></div>
      </div>`;
    }).join('');
    const tone = r.retention > 0.95 ? 'good' : r.retention > 0.8 ? 'mid' : 'bad';
    $('#metrics').innerHTML = `<div class="block-title">Scorecard <span class="hint">vs. each expert on its own task</span></div>
      ${rows}
      <div class="mlegend"><span><i class="mk ex"></i>expert alone</span>${T.baseAccs ? '<span><i class="mk bs"></i>base</span>' : ''}<span><i class="mk chance"></i>chance</span></div>
      <div class="kpis">
        <div class="kpi"><small>Avg accuracy</small><b>${pct(r.avg)}</b></div>
        <div class="kpi ${tone}"><small>Skill retention</small><b>${pct(r.retention)}</b></div>
      </div>
      ${(r.warnings || []).map((w) => `<div class="warn-box">⚠ ${w}</div>`).join('')}
      ${cmp ? compareBox(r, pin) : ''}`;
  }

  // ---------------- PER-LAYER HEAT MAP (gradient values for two layers) ----------------
  const GRID_N = 7;
  const gridCache = new WeakMap(); // trained bundle -> Map(key -> matrix)
  let gridToken = 0;
  const PAIRS = [['0-1', 'Hidden 1 × Hidden 2'], ['0-2', 'Hidden 1 × Output'], ['1-2', 'Hidden 2 × Output']];
  function renderLayerMap(m, r) {
    const panel = $('#layerPanel');
    const g = m.gradient, p = S.params[m.id];
    if (!T || !T.experts || r.blocked || !g || !p.layerwise || (g.k2 && T.K !== 2)) { panel.hidden = true; return; }
    panel.hidden = false;
    const pair = PAIRS.some(([k]) => k === S.gridPair) ? S.gridPair : '0-2';
    const [la, lb] = pair.split('-').map(Number);
    const fixed = [0, 1, 2].find((l) => l !== la && l !== lb);
    const vals = Array.from({ length: GRID_N }, (_, i) => +(g.min + ((g.max - g.min) * i) / (GRID_N - 1)).toFixed(4));
    const key = JSON.stringify([S.align, m.id, MM.layerGridParams(p, la, lb, 0, 0), la, lb]);
    let byT = gridCache.get(T);
    if (!byT) { byT = new Map(); gridCache.set(T, byT); }
    const M = byT.get(key);
    const LN = MM.LAYER_NAMES;
    panel.innerHTML = `<div class="sweep-head">
        <div><h2>Per-layer sensitivity <span class="pill-mini">which layer matters?</span></h2>
          <p class="muted">Average accuracy for every combination of ${esc(g.label)} in two layers, with ${LN[fixed]} held at ${fmtG(g, p.grad[fixed])}. A colour change along one axis only means that layer is the one to tune.</p></div>
        <div class="sweep-ctl"><div class="seg" id="gridPair">${PAIRS.map(([k, l]) => `<button data-pair="${k}" class="${k === pair ? 'on' : ''}">${l}</button>`).join('')}</div></div></div>
      <div class="heat-wrap" id="heatWrap">${M ? '' : '<div class="loading"><span class="spinner"></span> <span id="heatMsg">Mapping…</span></div>'}</div>
      <div class="sweep-sum" id="heatSum"></div>`;
    if (M) drawHeat(m, g, p, la, lb, vals, M);
    else {
      const token = ++gridToken, Tnow = T, ctx = MM.bundleCtx(T, S.align);
      (async () => {
        // one row per frame so the page stays responsive (49 merges in all)
        const grid = [];
        for (let i = 0; i < vals.length; i++) {
          await frame();
          if (token !== gridToken) return;
          const el = $('#heatMsg');
          if (el) el.textContent = `Mapping ${LN[la]} × ${LN[lb]}… ${Math.round((100 * i) / vals.length)}%`;
          grid.push(vals.map((vb) => {
            const rr = MM.scoreMethod(Tnow, ctx, m.id, MM.layerGridParams(p, la, lb, vals[i], vb));
            return rr.blocked ? null : rr.avg;
          }));
        }
        byT.set(key, grid);
        if (token === gridToken && T === Tnow && S.method === m.id) drawHeat(m, g, p, la, lb, vals, grid);
      })();
    }
  }
  function fmtG(g, v) { return g.key === 'balance' ? `${Math.round((1 - v) * 100)}/${Math.round(v * 100)}` : (+v).toFixed(2); }
  function drawHeat(m, g, p, la, lb, vals, M) {
    const wrap = $('#heatWrap');
    if (!wrap) return;
    const pick = MM.pickGrid(M);
    const flat = M.flat().filter((x) => x != null);
    const lo = Math.min(...flat), hi = Math.max(...flat);
    const col = (v) => {
      if (v == null) return 'rgba(255,255,255,0.04)';
      const t = hi - lo > 1e-9 ? (v - lo) / (hi - lo) : 1;
      const stops = [[248, 113, 113], [251, 191, 36], [52, 211, 153]];
      const k = t < 0.5 ? 0 : 1, u = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
      const c = stops[k].map((a, i) => Math.round(a + (stops[k + 1][i] - a) * u));
      return `rgba(${c.join(',')},${(0.25 + 0.6 * t).toFixed(2)})`;
    };
    const nearest = (v) => vals.reduce((bi, x, i) => (Math.abs(x - v) < Math.abs(vals[bi] - v) ? i : bi), 0);
    const ni = nearest(p.grad[la]), nj = nearest(p.grad[lb]);
    const LN = MM.LAYER_NAMES;
    wrap.innerHTML = `<div class="heat" style="--n:${vals.length}">
        <div class="hy">${esc(LN[la])}</div>
        <div class="hgrid">
          <span class="hc corner"></span>${vals.map((v) => `<span class="hc hx">${fmtG(g, v)}</span>`).join('')}
          ${M.map((row, i) => `<span class="hc hyv">${fmtG(g, vals[i])}</span>${row.map((v, j) => {
            const tags = [i === ni && j === nj ? 'now' : '', i === pick.best.i && j === pick.best.j ? 'best' : '', i === pick.safe.i && j === pick.safe.j ? 'safe' : ''].filter(Boolean);
            return `<span class="hc cell ${tags.join(' ')}" style="background:${col(v)}" title="${esc(LN[la])} ${fmtG(g, vals[i])} · ${esc(LN[lb])} ${fmtG(g, vals[j])}: ${v == null ? 'blocked' : pct(v)}">${v == null ? '—' : (v * 100).toFixed(0)}</span>`;
          }).join('')}`).join('')}
        </div>
        <div class="hxlab">${esc(LN[lb])} →</div>
      </div>
      <div class="chart-legend heat-legend"><span><i class="hk now"></i>now</span><span><i class="hk best"></i>best</span><span><i class="hk safe"></i>safest</span><span class="muted">cell = average accuracy %</span></div>`;
    const sens = pick.sensA > pick.sensB * 1.5 ? `<b>${esc(LN[la])}</b> is the layer this recipe is sensitive to (≈${(pick.sensA * 100).toFixed(0)} pts swing vs ${(pick.sensB * 100).toFixed(0)} for ${esc(LN[lb])})`
      : pick.sensB > pick.sensA * 1.5 ? `<b>${esc(LN[lb])}</b> is the layer this recipe is sensitive to (≈${(pick.sensB * 100).toFixed(0)} pts swing vs ${(pick.sensA * 100).toFixed(0)} for ${esc(LN[la])})`
      : `Both layers matter about equally (≈${(pick.sensA * 100).toFixed(0)} vs ${(pick.sensB * 100).toFixed(0)} pts swing)`;
    const at = (c) => `${esc(LN[la])} ${fmtG(g, vals[c.i])}, ${esc(LN[lb])} ${fmtG(g, vals[c.j])}`;
    $('#heatSum').innerHTML = `<p>${sens}. Best: ${at(pick.best)} (<b>${pct(pick.best.v)}</b>); ${Math.round(pick.near * 100)}% of settings are within 2 pts of it.
      ${pick.safe.i !== pick.best.i || pick.safe.j !== pick.best.j ? `The <span class="safe-word">safest</span> cell is ${at(pick.safe)}: its worst neighbour still scores ${pct(pick.safe.lower)}.` : ''} Single-seed map: confirm a favourite with the paired checks.</p>
      <div class="row"><button class="btn sm" data-grid-use="${vals[pick.best.i]},${vals[pick.best.j]}">Use best</button>${pick.safe.i !== pick.best.i || pick.safe.j !== pick.best.j ? `<button class="btn ghost sm" data-grid-use="${vals[pick.safe.i]},${vals[pick.safe.j]}">Use safest</button>` : ''}</div>`;
  }
  $('#layerPanel').addEventListener('click', (ev) => {
    const m = MM.METHOD_BY_ID[S.method];
    const pb = ev.target.closest('[data-pair]');
    if (pb) { S.gridPair = pb.dataset.pair; persist(); renderLayerMap(m, current); return; }
    const use = ev.target.closest('[data-grid-use]');
    if (use) {
      const g = m.gradient, p = S.params[m.id];
      const [la, lb] = (PAIRS.some(([k]) => k === S.gridPair) ? S.gridPair : '0-2').split('-').map(Number);
      const snap = (v) => +Math.min(g.max, Math.max(g.min, Math.round(v / g.step) * g.step)).toFixed(4);
      const [va, vb] = use.dataset.gridUse.split(',').map(Number);
      p.grad[la] = snap(va); p.grad[lb] = snap(vb);
      persist(); arenaStale = robustStale = true;
      refresh();
      toast(`${MM.LAYER_NAMES[la]} = ${fmtG(g, p.grad[la])}, ${MM.LAYER_NAMES[lb]} = ${fmtG(g, p.grad[lb])}`);
    }
  });

  // ---------------- KNOB SENSITIVITY ----------------
  const SWEEP_N = 17;
  const SWEEP_SEEDS = 5; // the same 5 cached seeds as the edge check, attribution and saliency checks
  const sweepCache = new WeakMap(); // trained bundle -> Map(key -> sweep result)
  let sweepToken = 0, sweepTimer = null;
  const knobFmt = (knob, v) => (knob.key === 'balance' ? `${Math.round((1 - v) * 100)}/${Math.round(v * 100)}` : (+v).toFixed(knob.step < 0.01 ? 3 : 2));
  function currentKnob(m) {
    const ks = MM.sweepKnobs(m.id, T.K, S.params[m.id]);
    return { ks, knob: ks.find((k) => k.key === S.sweepKnob[m.id]) || ks[0] };
  }
  function renderSweep(m, r) {
    const panel = $('#sweepPanel');
    if (!T || !T.experts || r.blocked) { panel.hidden = true; $('#layerPanel').hidden = true; return; }
    panel.hidden = false;
    const { ks, knob } = currentKnob(m);
    renderLayerMap(m, r);
    if (!knob) {
      const lwOn = m.gradient && S.params[m.id].layerwise && !(m.gradient.k2 && T.K !== 2);
      if (lwOn) { panel.hidden = true; return; } // the per-layer heat map below covers it
      const why = lwOn
        ? `${esc(m.gradient.label)} is set per layer, so a single-value sweep doesn’t apply. Use the layer sliders and presets, or turn per-layer off to sweep it.`
        : `${m.name} has no continuous knob${m.params.some((d) => d.type === 'weights') && T.K > 2 ? ' to sweep with 3 experts' : ''}, so there is nothing to tune: what you see is what you get.`;
      panel.innerHTML = `<div class="sweep-head"><div><h2>Knob sensitivity</h2><p class="muted">${why}</p></div></div>`;
      return;
    }
    const params = S.params[m.id];
    const key = sweepKey(m.id, params, knob, S.sweepSeeds);
    let byT = sweepCache.get(T);
    if (!byT) { byT = new Map(); sweepCache.set(T, byT); }
    const res = byT.get(key);
    panel.innerHTML = `<div class="sweep-head">
        <div><h2>Knob sensitivity <span class="pill-mini">how forgiving is this recipe?</span></h2>
          <p class="muted">Average accuracy as one knob moves, with the others held at your settings. A flat top means the recipe is forgiving; a sharp peak means you should tune it carefully on real models too.</p></div>
        <div class="sweep-ctl">
          ${ks.length > 1 ? `<div class="seg" id="sweepKnobs">${ks.map((k) => `<button data-k="${k.key}" class="${k.key === knob.key ? 'on' : ''}">${esc(k.label.replace(/\s*\(.*\)$/, ''))}</button>`).join('')}</div>` : `<span class="muted small">${esc(knob.label)}</span>`}
          <button class="tg ${S.sweepSeeds ? 'on' : ''}" id="sweepSeeds" title="Repeat the sweep on the 5 shared seeds (the same ones the edge check and attribution use) and show a ±1 sd band">± ${SWEEP_SEEDS} seeds</button>
        </div></div>
      <div class="chart" id="sweepChart">${res ? '' : '<div class="loading"><span class="spinner"></span> <span id="sweepMsg">Sweeping…</span></div>'}</div>
      <div class="chart-legend" id="sweepLegend"></div>
      <div class="sweep-sum" id="sweepSum"></div>`;
    if (res) drawSweep(m, knob, res);
    else scheduleSweep(m, knob, params, key, byT);
  }
  /** The sweep cache key for one knob: other knobs at their current values, this knob's value left out. */
  const sweepKey = (id, params, knob, seeds) => JSON.stringify([S.align, id, MM.withKnob(params, knob, knob.min), knob.key, !!seeds]);
  /** Sweep one knob on the shared seeds: each seed's sweep in its own worker, in parallel. Returns { xs, curves } or null. */
  async function seededSweep(id, knob, params, cancelled, msg, given) {
    const xs = MM.sweepValues(knob, SWEEP_N);
    const kname = knob.label.replace(/\s*\(.*\)$/, '');
    const bundles = given || await seedBundles(SWEEP_SEEDS, cancelled, msg);
    if (!bundles) return null;
    const recipes = xs.map((x, i) => ({ key: `x${i}`, method: id, params: MM.withKnob(params, knob, x), align: S.align }));
    let done = 0;
    msg(`Sweeping ${kname} on ${bundles.length} seeds${workerOK ? ' in parallel' : ''}…`);
    const runs = await Promise.all(bundles.map((b) => scoreAsync(b, recipes, { cancelled }).then((r) => { done++; msg(`Sweeping ${kname}… ${done}/${bundles.length} seeds`); return r; })));
    if (cancelled() || runs.some((r) => !r)) return null;
    return { xs, curves: runs.map((r) => xs.map((x, i) => (r[`x${i}`].blocked ? { x, blocked: r[`x${i}`].blocked } : { x, avg: r[`x${i}`].avg, accs: r[`x${i}`].accs }))) };
  }
  function scheduleSweep(m, knob, params, key, byT) {
    clearTimeout(sweepTimer);
    const token = ++sweepToken;
    const Tnow = T;
    sweepTimer = setTimeout(async () => {
      const msg = (t) => { const el = $('#sweepMsg'); if (el && token === sweepToken) el.textContent = t; };
      const cancelled = () => token !== sweepToken;
      let res;
      if (S.sweepSeeds) {
        res = await seededSweep(m.id, knob, params, cancelled, msg);
        if (!res) return;
      } else {
        const xs = MM.sweepValues(knob, SWEEP_N);
        const kname = knob.label.replace(/\s*\(.*\)$/, '');
        const curves = [];
        const ctx = MM.bundleCtx(Tnow, S.align);
        const pts = [];
        for (const x of xs) {
          msg(`Sweeping ${kname}…`);
          pts.push(MM.sweepMethod(Tnow, ctx, m.id, params, knob, [x])[0]);
          await frame();
          if (cancelled()) return;
        }
        curves.push(pts);
        res = { xs, curves };
      }
      byT.set(key, res);
      if (token === sweepToken && T === Tnow && S.method === m.id) drawSweep(m, knob, res);
    }, 250);
  }
  function drawSweep(m, knob, res) {
    const host = $('#sweepChart');
    if (!host) return;
    const { xs, curves } = res;
    if (curves[0].some((p) => p.blocked)) { host.innerHTML = '<div class="empty">This method is blocked here.</div>'; return; }
    const n = curves.length;
    const mean = xs.map((_, i) => curves.reduce((a, c) => a + c[i].avg, 0) / n);
    const sd = n > 1 ? xs.map((_, i) => Math.sqrt(curves.reduce((a, c) => a + (c[i].avg - mean[i]) ** 2, 0) / (n - 1))) : null;
    const pick = MM.pickSettings(xs, mean, sd, 0.02); // 2 pts: about this toy's seed-to-seed noise
    const now = MM.knobValue(S.params[m.id], knob);
    const taskSeries = T.tasks.map((t, k) => ({
      name: `${LETTERS[k]} · ${MM.TASKS[t.spec.task].name}`, color: MM.MODEL_COLORS[k], width: 1.4, opacity: 0.75,
      points: xs.map((x, i) => [x, curves.reduce((a, c) => a + c[i].accs[k], 0) / n]),
    }));
    const avgSeries = { name: n > 1 ? `Average (mean of ${n} seeds)` : 'Average', color: '#ffffff', width: 2.6, points: xs.map((x, i) => [x, mean[i]]) };
    const all = taskSeries.flatMap((s) => s.points.map((p) => p[1])).concat(sd ? mean.map((v, i) => v - sd[i]) : []);
    const yMin = Math.max(0, Math.floor(Math.min(...all) * 10) / 10 - 0.05);
    const vlines = [{ x: now, color: '#a78bfa', label: 'now', solid: true, width: 2, opacity: 0.9 }, { x: pick.best, color: '#34d399', label: 'best', row: 1 }];
    if (pick.safe !== pick.best) vlines.push({ x: pick.safe, color: '#fbbf24', label: 'safe', row: 2 });
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => +(knob.min + (knob.max - knob.min) * f).toFixed(3));
    MM.lineChart(host, {
      series: [avgSeries, ...taskSeries], xMin: knob.min, xMax: knob.max, yMin, yMax: 1, height: 220, xTicks: ticks,
      xFmt: (v) => knobFmt(knob, v), yFmt: (v) => `${Math.round(v * 100)}%`, xLabel: knob.key === 'balance' ? 'expert weights A/B (%)' : knob.label,
      bands: sd ? [{ lo: xs.map((x, i) => [x, mean[i] - sd[i]]), hi: xs.map((x, i) => [x, mean[i] + sd[i]]), color: '#ffffff', opacity: 0.12 }] : [],
      vlines,
    });
    $('#sweepLegend').innerHTML = [avgSeries, ...taskSeries].map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join('') + (sd ? '<span><i style="background:rgba(255,255,255,.25)"></i>±1 sd</span>' : '');
    // value at the current setting (nearest sweep point)
    let ni = 0;
    xs.forEach((x, i) => { if (Math.abs(x - now) < Math.abs(xs[ni] - now)) ni = i; });
    const gap = pick.bestVal - mean[ni];
    const forgiving = pick.rangeFrac > 0.35 ? 'forgiving' : pick.rangeFrac > 0.12 ? 'moderately sensitive' : 'sensitive';
    const kname = knob.key === 'balance' ? 'A/B balance' : knob.label.replace(/\s*\(.*\)$/, '');
    const span = pick.range[0] === pick.range[1]
      ? 'No other setting comes within 2 pts of it'
      : `Anything from <b>${knobFmt(knob, pick.range[0])}</b> to <b>${knobFmt(knob, pick.range[1])}</b> stays within 2 pts`;
    // with several seeds: does the best value itself hold from seed to seed?
    let transfer = '';
    if (n > 1) {
      const tv = MM.knobTransfer(xs, curves.map((c) => c.map((q) => q.avg)));
      const vals = tv.bests.map((x) => knobFmt(knob, x)).join(', ');
      const cost = `${(tv.regret * 100).toFixed(1)} pts`;
      transfer = tv.kind === 'moves'
        ? `<span class="seed-best moves">The best value moves with the seed: ${vals} on the ${n} seeds.</span> Treat ${knobFmt(knob, pick.best)} as one point in a range and tune this knob on your own data; on the worst seed it is ${cost} below that seed’s own best.`
        : tv.kind === 'transfers'
          ? `<span class="seed-best holds">${knobFmt(knob, pick.best)} transfers across seeds:</span> each seed’s best differs (${vals}), but the curve is flat enough that ${knobFmt(knob, pick.best)} is within ${cost} of every seed’s best.`
          : tv.kind === 'narrow'
            ? `<span class="seed-best moves">Same spot on every seed (${vals}), but a narrow peak:</span> the small seed-to-seed shift already costs ${cost} on the worst seed, so tune it finely on your own data.`
            : `<span class="seed-best holds">The best value holds across seeds (${vals}).</span>`;
    }
    $('#sweepSum').innerHTML = `<p>Best ${esc(kname)} is <b>${knobFmt(knob, pick.best)}</b> (${pct(pick.bestVal)}${n > 1 ? `, mean of ${n} seeds` : ''}). ${span}, so this knob is <b class="sens ${pick.rangeFrac > 0.35 ? 'good' : pick.rangeFrac > 0.12 ? 'mid' : 'bad'}">${forgiving}</b>. ${transfer}
      ${gap > 0.01 ? `Your setting (${knobFmt(knob, now)}) is ${(gap * 100).toFixed(1)} pts below the best.` : 'Your setting is at or near the best.'}
      ${pick.safe !== pick.best ? `The <span class="safe-word">safest</span> choice is ${knobFmt(knob, pick.safe)}: ${sd ? 'it has the best mean − 1 sd across seeds' : 'it has the best worst-case among neighbouring settings'}.` : ''}
      ${!sd ? (pick.rangeFrac <= 0.12 ? `A sharp single-seed peak can be luck: turn on <b>± ${SWEEP_SEEDS} seeds</b> to check it.` : `This is one seed: turn on <b>± ${SWEEP_SEEDS} seeds</b> to see whether the best value holds.`) : ''}</p>
      <div class="row"><button class="btn sm" data-sweep-use="${pick.best}">Use best (${knobFmt(knob, pick.best)})</button>${pick.safe !== pick.best ? `<button class="btn ghost sm" data-sweep-use="${pick.safe}">Use safest (${knobFmt(knob, pick.safe)})</button>` : ''}</div>`;
  }
  $('#sweepPanel').addEventListener('click', (ev) => {
    const m = MM.METHOD_BY_ID[S.method];
    const kb = ev.target.closest('#sweepKnobs [data-k]');
    if (kb) { S.sweepKnob[m.id] = kb.dataset.k; persist(); renderSweep(m, current); return; }
    if (ev.target.closest('#sweepSeeds')) { S.sweepSeeds = !S.sweepSeeds; persist(); renderSweep(m, current); return; }
    const use = ev.target.closest('[data-sweep-use]');
    if (use) {
      const { knob } = currentKnob(m);
      const v = +(Math.round(+use.dataset.sweepUse / knob.step) * knob.step).toFixed(4); // snap to the slider grid
      S.params[m.id] = MM.withKnob(S.params[m.id], knob, Math.min(knob.max, Math.max(knob.min, v)));
      persist(); arenaStale = robustStale = true;
      refresh();
      toast(`${knob.key === 'balance' ? 'Balance' : knob.label.replace(/\s*\(.*\)$/, '')} set to ${knobFmt(knob, MM.knobValue(S.params[m.id], knob))}`);
    }
  });

  /** "What changed" between pinned A and current B: the settings behind the accuracy gap. */
  function changesHTML() {
    const diff = MM.recipeDiff(S.pin, nowRecipe(), T.K);
    if (!diff.length) return '<div class="cmp-changes"><b>What changed:</b> nothing yet. Change the method or a knob to compare.</div>';
    const at = attrCache.get(edgeKey()), ss = attrSeedCache.get(edgeKey());
    const pts = (x) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)}`;
    if (ss) return attrSeedsHTML(diff, ss, pts);
    const scale = at ? Math.max(0.005, ...at.parts.map((p) => Math.abs(p.delta || 0)), Math.abs(at.interaction)) : 1;
    const bar = (d) => `<span class="attr-bar"><i style="${d >= 0 ? 'left:50%' : `right:50%`};width:${Math.min(50, (Math.abs(d) / scale) * 50).toFixed(1)}%" class="${d >= 0 ? 'up' : 'down'}"></i></span>`;
    const delta = (i) => { const p = at && at.parts[i]; return !p ? '' : p.delta == null ? '<span class="muted">can’t run alone</span>' : `${bar(p.delta)}<b class="attr-pts">${pts(p.delta)}</b>`; };
    const items = diff.map((x, i) => `<li><span>${esc(x.what)}</span> <code>${esc(x.a)}</code> → <code>${esc(x.b)}</code>${at ? `<span class="attr">${delta(i)}</span>` : ''}</li>`).join('');
    const tail = diff.length < 2 ? '' : at
      ? `<div class="attr-foot">Each line is A with only that change, in pts of average accuracy (seed ${S.seed}). Together: ${pts(at.total)} pts; ${Math.abs(at.interaction) < 0.005 ? 'the changes roughly add up (within half a point).' : `${pts(at.interaction)} pts comes from changes interacting, not from any one alone.`}</div>`
      : '<button class="btn sm ghost" id="attrBtn">Which change mattered?</button>';
    const again = at ? `<button class="btn sm ghost" id="attrSeedBtn">Repeat on ${EDGE_SEEDS} seeds</button><span class="muted small" id="attrSeedMsg" aria-live="polite"></span>` : '';
    return `<div class="cmp-changes"><b>What changed</b><ul>${items}</ul>${tail}${again}</div>`;
  }
  const SIGN_TXT = { up: (p) => `up on ${p.up}/${p.n}`, down: (p) => `down on ${p.down}/${p.n}`, flat: (p) => `under 0.2 pts on ${p.n}/${p.n}`, mixed: (p) => `mixed: ${[[p.up, 'up'], [p.down, 'down'], [p.flat, 'flat']].filter(([c]) => c).map(([c, w]) => `${c} ${w}`).join(', ')}`, none: () => 'can’t run alone' };
  function attrSeedsHTML(diff, ss, pts) {
    const scale = Math.max(0.005, ...ss.parts.map((p) => Math.abs(p.mean || 0) + (p.sd || 0)), Math.abs(ss.interaction.mean));
    const bar = (p) => {
      const w = (x) => Math.min(50, (Math.abs(x) / scale) * 50).toFixed(1);
      const lo = p.mean - p.sd, hi = p.mean + p.sd; // whisker: mean ± sd
      const wl = Math.max(0, Math.min(100, 50 + (lo / scale) * 50)), wh = Math.max(0, Math.min(100, 50 + (hi / scale) * 50));
      return `<span class="attr-bar"><i style="${p.mean >= 0 ? 'left:50%' : 'right:50%'};width:${w(p.mean)}%" class="${p.sign === 'mixed' || p.sign === 'flat' ? 'mix' : p.mean >= 0 ? 'up' : 'down'}"></i><b class="whisk" style="left:${wl.toFixed(1)}%;width:${Math.max(0.5, wh - wl).toFixed(1)}%"></b></span>`;
    };
    const items = diff.map((x, i) => {
      const p = ss.parts[i];
      const right = !p || p.sign === 'none' ? '<span class="muted">can’t run alone</span>'
        : `${bar(p)}<b class="attr-pts">${pts(p.mean)}</b><small class="attr-sd">± ${(p.sd * 100).toFixed(1)}</small><span class="attr-sign ${p.sign}">${SIGN_TXT[p.sign](p)}</span>`;
      return `<li><span>${esc(x.what)}</span> <code>${esc(x.a)}</code> → <code>${esc(x.b)}</code><span class="attr">${right}</span></li>`;
    }).join('');
    const steady = ss.parts.filter((p) => p.sign === 'up' || p.sign === 'down').map((p) => p.what);
    const ix = Math.abs(ss.interaction.mean) < 0.005 ? 'the changes roughly add up (within half a point).' : `${pts(ss.interaction.mean)} pts comes from changes interacting.`;
    return `<div class="cmp-changes"><b>What changed</b> <span class="muted small">effect alone, mean ± sd over ${ss.n} seeds</span><ul>${items}</ul>
      <div class="attr-foot">Together: ${pts(ss.total.mean)} ± ${(ss.total.sd * 100).toFixed(1)} pts; ${ix} ${steady.length ? `Same direction on every seed: ${steady.map(esc).join(', ')}.` : 'No change moves accuracy the same way on every seed, so read the per-change effects as tendencies, not facts.'}</div></div>`;
  }
  const attrSeedCache = new Map();
  let attrSeedToken = 0;
  async function runAttributionSeeds() {
    const token = ++attrSeedToken, key = edgeKey(), A = S.pin, B = nowRecipe(), K = T.K;
    const cancelled = () => token !== attrSeedToken || edgeKey() !== key;
    const msg = (t) => { const el = $('#attrSeedMsg'); if (el && !cancelled()) el.textContent = t; };
    const btn = $('#attrSeedBtn');
    if (btn) btn.disabled = true;
    const bundles = await seedBundles(EDGE_SEEDS, cancelled, msg);
    if (!bundles) return;
    const list = [];
    for (let i = 0; i < bundles.length; i++) {
      msg(`Scoring each change on seed ${i + 1}/${bundles.length}…`);
      await frame();
      if (cancelled()) return;
      const Tb = bundles[i];
      const score = (r) => { const res = MM.scoreMethod(Tb, MM.bundleCtx(Tb, r.align), r.method, r.params); return res.blocked ? null : res.avg; };
      list.push(MM.attributeChanges(A, B, K, score));
    }
    const ss = MM.summarizeAttributions(list);
    if (!ss) return;
    attrSeedCache.set(key, ss);
    refresh();
  }
  $('#metrics').addEventListener('click', (ev) => { if (ev.target.closest('#attrSeedBtn')) runAttributionSeeds(); });
  // "Which change mattered?": A with one change from B at a time, scored on the lab's experts
  const attrCache = new Map();
  async function runAttribution() {
    const key = edgeKey(), A = S.pin, B = nowRecipe(), K = T.K, Tnow = T;
    const btn = $('#attrBtn');
    if (btn) { btn.disabled = true; btn.textContent = 'Scoring each change…'; }
    await frame();
    const score = (r) => { const res = MM.scoreMethod(Tnow, MM.bundleCtx(Tnow, r.align), r.method, r.params); return res.blocked ? null : res.avg; };
    const at = MM.attributeChanges(A, B, K, score);
    if (!at || T !== Tnow) return;
    attrCache.set(key, at);
    if (edgeKey() === key) refresh();
  }
  $('#metrics').addEventListener('click', (ev) => { if (ev.target.closest('#attrBtn')) runAttribution(); });
  function compareBox(r, pin) {
    const sgn = (x) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)} pts`;
    const mult = (x) => (x > 1.05 ? `${x.toFixed(1)}×` : '1×');
    const needs = (id) => { const q = MM.METHOD_BY_ID[id].req; return [q.base && 'base', q.data && 'data', q.training && 'training'].filter(Boolean).join(', ') || 'nothing'; };
    const verdict = Math.abs(cmp.dAvg) < 0.005 ? 'B and A are within half a point of each other on average.'
      : `B is <b>${cmp.dAvg > 0 ? 'better' : 'worse'}</b> than A by ${Math.abs(cmp.dAvg * 100).toFixed(1)} pts on average.`;
    const noise = Math.abs(cmp.dAvg) < 0.03 ? '<div class="note">Gaps under ~3 pts are often seed noise here. Confirm with the Arena’s multi-seed check before committing to a recipe.</div>' : '';
    return `<div class="cmp-box"><div class="cmp-title">B (now) vs A (pinned)</div>
      <div>${verdict}</div>
      <div class="cmp-grid">
        <span>Avg accuracy</span><span>${sgn(cmp.dAvg)}</span>
        <span>Skill retention</span><span>${sgn(cmp.dRetention)}</span>
        <span>Predictions differ</span><span>${pct(cmp.disagree)} of test points</span>
        <span>Inference</span><span>${mult(pin.cost)} → ${mult(r.cost)}</span>
        <span>Needs</span><span>${needs(S.pin.method)} → ${needs(S.method)}</span>
      </div>${changesHTML()}${noise}
      <div class="edge" id="edge">${edgeHTML()}</div></div>`;
  }

  // "Is B's edge real?": both recipes on the same 5 seeds, paired.
  const EDGE_SEEDS = 5;
  const edgeCache = new Map();
  let edgeToken = 0;
  const edgeKey = () => JSON.stringify([trainKey(S), S.pin, nowRecipe()]);
  function edgeHTML() {
    const d = edgeCache.get(edgeKey());
    if (!d) return `<button class="btn sm" id="edgeBtn">Is B’s edge real? Check on ${EDGE_SEEDS} seeds</button><span class="edge-msg muted small" id="edgeMsg"></span>`;
    const [tone, html] = pairVerdict(d, '<b>B</b>', '<b>A</b>', 'The Arena can run 10 seeds to settle it.');
    return `<div class="verdict ${tone}">${html}</div>`;
  }
  async function checkEdge() {
    const token = ++edgeToken, key = edgeKey(), pin = S.pin, now = nowRecipe();
    const msg = (t) => { const el = $('#edgeMsg'); if (el && token === edgeToken) el.textContent = t; };
    const btn = $('#edgeBtn');
    if (btn) btn.disabled = true;
    const bundles = await seedBundles(EDGE_SEEDS, () => token !== edgeToken, msg);
    if (!bundles) return;
    msg('Scoring both recipes…');
    await frame();
    if (token !== edgeToken) return;
    const cancelled = () => token !== edgeToken;
    const recipes = [{ key: 'A', method: pin.method, params: pin.params, align: pin.align }, { key: 'B', method: now.method, params: now.params, align: now.align }];
    const runs = await Promise.all(bundles.map((b) => scoreAsync(b, recipes, { cancelled })));
    if (cancelled() || runs.some((r) => !r)) return;
    edgeCache.set(key, MM.pairedDiff(runs, 'B', 'A'));
    const el = $('#edge');
    if (el && edgeKey() === key) el.innerHTML = edgeHTML();
  }
  $('#metrics').addEventListener('click', (ev) => { if (ev.target.closest('#edgeBtn')) checkEdge(); });

  // ---------------- INSIGHT ----------------
  function renderInsight(m, r, ctx) {
    const f = MM.FAMILIES[m.family];
    const hasBase = !!ctx.base;
    const req = [
      ['Shared base', m.req.base, hasBase],
      ['Task data', m.req.data, true],
      ['Training', m.req.training, true],
    ];
    const reqHtml = req.map(([name, needs, have]) => {
      const cls = !needs ? 'free' : have ? 'ok' : 'missing';
      const txt = !needs ? `No ${name.toLowerCase()} needed` : have ? `Uses ${name.toLowerCase()}` : `Needs ${name.toLowerCase()}`;
      return `<span class="req ${cls}">${cls === 'free' ? '✓' : cls === 'ok' ? '●' : '✕'} ${txt}</span>`;
    }).join('');
    const geo = ctx.cache.geo || (ctx.cache.geo = MM.geometry(ctx));
    const geoHtml = geo.pairs.map((p) => `
      <div class="geo-row"><span class="pair"><i style="background:${MM.MODEL_COLORS[p.i]}"></i><i style="background:${MM.MODEL_COLORS[p.j]}"></i>${LETTERS[p.i]}·${LETTERS[p.j]}</span>
        <span title="Cosine similarity of the task vectors">cos <b>${p.cos.toFixed(2)}</b></span>
        <span title="Among the top-20% largest updates both experts touched, how often do they point in opposite directions?">sign conflict <b class="${p.conflict > 0.4 ? 'bad' : ''}">${pct(p.conflict)}</b></span></div>`).join('');

    let diag = '';
    if (!r.blocked && r.kind === 'weights') {
      diag = `<div class="diag">
        <div class="prov-wrap"><canvas id="prov"></canvas></div>
        <div class="prov-side">
          <div class="tiny muted">Layer W2 · each cell is one weight, coloured by the model it ended up closest to.</div>
          <div class="prov-legend" id="provLegend"></div>
        </div></div>`;
    } else if (!r.blocked) {
      diag = `<div class="note">Weights are untouched — every expert is kept intact and mixing happens on their <b>predictions</b>. ${r.router ? `The router learned to pick the right expert ${pct(r.router.acc)} of the time.` : ''}</div>`;
    }
    const info = r.blocked ? '' : (r.info || []).map((x) => `<div class="kv"><span>${x.label}</span><b>${x.value}</b></div>`).join('');

    $('#insight').innerHTML = `
      <div class="panel-title"><span>Insight</span><span class="hint">${f.name.toLowerCase()}</span></div>
      <p class="tagline">${m.tagline}</p>
      <div class="reqs">${reqHtml}</div>
      <div class="section">
        <div class="sec-title">How it works</div>
        <ol class="steps">${m.how.map((s) => `<li>${s}</li>`).join('')}</ol>
        <code class="formula" tabindex="0" aria-label="Formula: ${esc(m.formula)}">${m.formula}</code>
      </div>
      <div class="two">
        <div class="section"><div class="sec-title good">Strengths</div><ul class="pros">${m.pros.map((s) => `<li>${s}</li>`).join('')}</ul></div>
        <div class="section"><div class="sec-title bad">Missing pieces</div><ul class="cons">${m.cons.map((s) => `<li>${s}</li>`).join('')}</ul></div>
      </div>
      ${info ? `<div class="section"><div class="sec-title">This mix, measured</div>${info}</div>` : ''}
      ${diag ? `<div class="section"><div class="sec-title">${r.kind === 'weights' ? 'Weight provenance' : 'Where the mixing happens'}</div>${diag}</div>` : ''}
      ${!r.blocked && ctx.tasks[0].kind === 'digits' ? `<div class="section"><div class="sec-title">Where the merge looks <span class="hint">pixel saliency</span></div>
        <div class="sal-grid" id="salGrid"><div class="muted small">Measuring…</div></div>
        <div class="sal-seeds" id="salSeeds">${salSeedsHTML()}</div>
        <p class="tiny muted">Which pixels move the prediction, for the expert and for the merge (each map scaled to its own peak). The match compares the patterns (80% or more counts as the same); each task's line reads it together with accuracy.</p></div>` : ''}
      <div class="section"><div class="sec-title">${hasBase ? 'Task-vector geometry' : 'Raw weight similarity <span class="hint">(no base)</span>'}</div>${geoHtml}
        <p class="tiny muted">${hasBase ? 'Low cosine + high sign conflict = skills that fight when added together. That is what TIES and DARE are built for.' : 'Independently trained experts look unrelated in weight space. Turn on neuron alignment and watch the similarity jump — same functions, re-ordered neurons.'}</p></div>
      <div class="section oss">
        <div class="sec-title">Run it for real</div>
        <div class="oss-row">${m.mergekit ? `<span class="pill mk">mergekit: <code>${m.mergekit}</code></span>` : '<span class="pill">not in mergekit</span>'}
        ${m.oss.map((o) => `<a class="pill link" href="${o.url}" target="_blank" rel="noopener">${o.name} ↗</a>`).join('')}</div>
        ${m.realWorld ? `<p class="real-note"><b>On real models:</b> ${m.realWorld}</p>` : ''}
        <p class="tiny muted">${m.paper}</p>
      </div>`;

    if (!r.blocked && r.kind === 'weights') {
      const W2 = ctx.layout.tensors[2];
      const counts = MM.drawProvenance($('#prov'), r.params, ctx.experts, ctx.base, W2);
      const labels = ctx.experts.map((_, i) => [`Expert ${LETTERS[i]}`, MM.MODEL_COLORS[i]]);
      if (ctx.base) labels.push(['Base', MM.BASE_COLOR]);
      labels.push(['Blended', '#a884ff']);
      $('#provLegend').innerHTML = labels.map(([l, c], i) => `<div class="pl"><i style="background:${c}"></i>${l}<b>${pct(counts[i])}</b></div>`).join('');
    }
    if (!r.blocked && ctx.tasks[0].kind === 'digits') renderSaliency(r, ctx);
  }

  // Pixel saliency (digit tasks): expert maps are cached per trained bundle; the merged map is computed after
  // the frame so dragging a knob stays smooth.
  const salCache = new WeakMap();
  let salTimer = null;
  // Is the reading a one-seed artifact? Re-measure it on the shared seed bundles (seed cache), in-page.
  const SAL_SEEDS = 5;
  const salSeedCache = new Map();
  let salSeedToken = 0;
  const salSeedKey = () => JSON.stringify([trainKey(S), nowRecipe()]);
  function salSeedsHTML() {
    const d = salSeedCache.get(salSeedKey());
    if (!d) return `<button class="btn sm" id="salSeedBtn">Is this reading stable? Check on ${SAL_SEEDS} seeds</button><span class="muted small" id="salSeedMsg" aria-live="polite"></span>`;
    // same wording as the report (MM.seedStability), so the Lab and a shared report never disagree
    const lines = d.map((t, k) => {
      const st = MM.seedStability(t);
      const tone = { stable: 'good', mostly: 'mid', unstable: 'bad' }[st.level];
      return `<li class="${tone}"><b>${LETTERS[k]}</b> ${esc(st.text)}${st.level === 'unstable' ? ', so treat the one-seed reading above as a guess' : ''}.</li>`;
    }).join('');
    return `<div class="sal-seed-title">Across ${d[0] ? d[0].n : SAL_SEEDS} seeds</div><ul class="sal-seed-list">${lines}</ul>`;
  }
  async function checkSalSeeds() {
    const token = ++salSeedToken, key = salSeedKey(), rec = nowRecipe();
    const cancelled = () => token !== salSeedToken;
    const msg = (t) => { const el = $('#salSeedMsg'); if (el && !cancelled()) el.textContent = t; };
    const btn = $('#salSeedBtn');
    if (btn) btn.disabled = true;
    const bundles = await seedBundles(SAL_SEEDS, cancelled, msg);
    if (!bundles) return;
    const runs = [];
    for (let i = 0; i < bundles.length; i++) {
      msg(`Measuring saliency on seed ${i + 1}/${bundles.length}…`);
      await frame();
      if (cancelled()) return;
      runs.push(MM.saliencyRun(bundles[i], rec.method, rec.params, rec.align));
    }
    salSeedCache.set(key, MM.saliencySeedSummary(runs));
    const el = $('#salSeeds');
    if (el && salSeedKey() === key) el.innerHTML = salSeedsHTML();
  }
  $('#insight').addEventListener('click', (ev) => { if (ev.target.closest('#salSeedBtn')) checkSalSeeds(); });

  /** One-line verdict for a task: pattern match × accuracy (the same rule the recipe report uses). */
  function salReading(match, merged, expert, router) {
    const rd = MM.saliencyReading(match, merged, expert, { router });
    const icon = { good: '✓', mid: '↻', bad: '⚠' }[rd.tone];
    return `<p class="sal-read ${rd.tone}"><span aria-hidden="true">${icon}</span> ${esc(rd.text.charAt(0).toUpperCase() + rd.text.slice(1))} <span class="muted">(${pct(merged)} vs expert ${pct(expert)})</span></p>`;
  }
  function renderSaliency(r, ctx) {
    clearTimeout(salTimer);
    const Tnow = T;
    salTimer = setTimeout(() => {
      const el = $('#salGrid');
      if (!el || T !== Tnow || r !== current) return;
      const ex = expertMaps(ctx);
      const mg = ctx.tasks.map((t) => MM.digitSaliency(r.predict, t));
      el.innerHTML = ctx.tasks.map((t, k) => {
        const cos = MM.mapCosine(ex[k], mg[k]);
        return `<div class="sal-row" style="--c:${MM.MODEL_COLORS[k]}">
          <div class="sal-lbl"><i class="dot"></i>${LETTERS[k]} · ${esc(MM.TASKS[t.spec.task].name)}</div>
          <figure><canvas class="sal" id="salE${k}" role="img" aria-label="Expert ${LETTERS[k]} pixel saliency"></canvas><figcaption>expert</figcaption></figure>
          <figure><canvas class="sal" id="salM${k}" role="img" aria-label="Merged model pixel saliency on task ${LETTERS[k]}"></canvas><figcaption>merged</figcaption></figure>
          <div class="sal-score ${cos > 0.9 ? 'good' : cos > 0.7 ? 'mid' : 'bad'}"><b>${Math.round(cos * 100)}%</b><small>pattern match</small></div>
          ${salReading(cos, r.accs[k], T.expertAccs[k][k], !!r.router)}</div>`;
      }).join('');
      // each map is scaled to its own peak: confident experts barely move p, so a shared scale would hide them;
      // the match score compares patterns (cosine), not magnitudes
      ctx.tasks.forEach((t, k) => {
        MM.drawSaliency($(`#salE${k}`), ex[k], Math.max(...ex[k]));
        MM.drawSaliency($(`#salM${k}`), mg[k], Math.max(...mg[k]));
      });
    }, 120);
  }

  // ---------------- ARENA ----------------
  function runArena() {
    const ctx = getCtx();
    if (!ctx) return;
    status('Running the arena…', true);
    $('#board').innerHTML = '<div class="loading"><span class="spinner"></span> Mixing every method…</div>';
    setTimeout(() => {
      const results = MM.METHODS.map((m) => ({ m, r: compute(m.id, ctx) }));
      renderBoard(results);
      renderThumbGrid(results, ctx);
      arenaResults = results; arenaSalRun++;
      $('#arenaSal').hidden = ctx.tasks[0].kind !== 'digits';
      $('#arenaSalGrid').innerHTML = '';
      $('#runArenaSal').disabled = false;
      renderPathChart(ctx);
      renderLambdaChart(ctx);
      arenaStale = false;
      status(`Arena: ${results.filter((x) => !x.r.blocked).length} methods compared`);
    }, 40);
  }
  $('#runArena').addEventListener('click', () => { for (const k in (T && T.ctx) || {}) Object.keys(T.ctx[k].cache).filter((c) => c.startsWith('distill')).forEach((c) => delete T.ctx[k].cache[c]); runArena(); });

  // Arena saliency strip (digit tasks, on demand): one row per method, computed a method per tick so the page stays live
  let arenaResults = null, arenaSalRun = 0;
  function expertMaps(ctx) {
    let ex = salCache.get(T);
    if (!ex) { ex = MM.expertSaliency(ctx); salCache.set(T, ex); }
    return ex;
  }
  $('#runArenaSal').addEventListener('click', () => {
    const ctx = getCtx();
    if (!ctx || !arenaResults || ctx.tasks[0].kind !== 'digits') return;
    const run = ++arenaSalRun, ex = expertMaps(ctx);
    const list = arenaResults.filter((x) => !x.r.blocked).sort((a, b) => b.r.avg - a.r.avg);
    const grid = $('#arenaSalGrid'), btn = $('#runArenaSal');
    btn.disabled = true;
    const cell = (id, label) => `<figure><canvas class="sal" id="${id}" role="img" aria-label="${esc(label)}"></canvas></figure>`;
    const score = (c) => `<div class="sal-score ${c > 0.9 ? 'good' : c > 0.7 ? 'mid' : 'bad'}"><b>${Math.round(c * 100)}%</b><small>match</small></div>`;
    const cols = `style="--k:${ctx.tasks.length}"`;
    const taskName = (k) => `${LETTERS[k]} · ${MM.TASKS[ctx.tasks[k].spec.task].name}`;
    grid.innerHTML = `<div class="asal-row asal-top" ${cols}><div></div>${ctx.tasks.map((_, k) => `<div class="asal-task" style="--c:${MM.MODEL_COLORS[k]}" title="${esc(taskName(k))}"><i class="dot"></i>${LETTERS[k]}<span> · ${esc(MM.TASKS[ctx.tasks[k].spec.task].name)}</span></div>`).join('')}<div class="asal-mean">mean</div></div>
      <div class="asal-row asal-ref" ${cols}><div class="asal-name"><b>Experts</b><small>reference</small></div>${ctx.tasks.map((_, k) => `<div class="asal-cell">${cell(`asE${k}`, `Expert ${LETTERS[k]} pixel saliency`)}</div>`).join('')}<div></div></div>`;
    ctx.tasks.forEach((_, k) => MM.drawSaliency($(`#asE${k}`), ex[k], Math.max(...ex[k])));
    let i = 0;
    const step = () => {
      if (run !== arenaSalRun) return;
      if (i >= list.length) { status(`Arena: saliency for ${list.length} methods`); return; }
      const x = list[i++], row = MM.saliencyRow(ctx, x.r.predict, ex);
      grid.insertAdjacentHTML('beforeend', `<button class="asal-row" ${cols} data-open="${x.m.id}" aria-label="Open ${esc(x.m.name)} in the Lab: mean pattern match ${Math.round(row.mean * 100)}%, accuracy ${pct(x.r.avg)}">
        <div class="asal-name"><span class="fam-dot" style="background:${MM.FAMILIES[x.m.family].color}"></span><b>${esc(x.m.short)}</b><small>${pct(x.r.avg)}</small></div>
        ${row.maps.map((_, k) => `<div class="asal-cell">${cell(`as-${x.m.id}-${k}`, `${x.m.name} saliency on task ${LETTERS[k]}`)}${score(row.match[k])}</div>`).join('')}
        <div class="asal-mean">${score(row.mean)}</div></button>`);
      row.maps.forEach((m, k) => MM.drawSaliency($(`#as-${x.m.id}-${k}`), m, Math.max(...m)));
      status(`Arena: pixel saliency ${i}/${list.length}…`, true);
      setTimeout(step, 0);
    };
    setTimeout(step, 0);
  });
  $('#arenaSalGrid').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-open]');
    if (b) { setTab('lab'); selectMethod(b.dataset.open); }
  });

  function renderBoard(results) {
    const K = T.K;
    const ok = results.filter((x) => !x.r.blocked).sort((a, b) => b.r.avg - a.r.avg);
    const blocked = results.filter((x) => x.r.blocked);
    const best = ok.length ? ok[0].r.avg : 0;
    const head = `<tr><th>#</th><th>Method</th>${T.tasks.map((t, i) => `<th><i class="dot" style="background:${MM.MODEL_COLORS[i]}"></i>${LETTERS[i]} · ${MM.TASKS[t.spec.task].name}</th>`).join('')}<th>Average</th><th>Retention</th><th>Inference</th><th>Memory</th><th>Needs</th></tr>`;
    const needs = (m) => [m.req.base && 'base', m.req.data && 'data', m.req.training && 'training'].filter(Boolean).map((x) => `<span class="mini-tag">${x}</span>`).join('') || '<span class="mini-tag free">nothing</span>';
    const expertRow = `<tr class="ref"><td></td><td><b>Experts alone</b><small>upper reference</small></td>${T.tasks.map((_, i) => `<td>${pct(T.expertAccs[i][i])}</td>`).join('')}<td>${pct(T.expertAccs.reduce((a, e, i) => a + e[i], 0) / K)}</td><td>100%</td><td>${K}× models</td><td>${K}×</td><td></td></tr>`;
    const baseRow = T.baseAccs ? `<tr class="ref"><td></td><td><b>Base model</b><small>starting point</small></td>${T.baseAccs.map((a) => `<td>${pct(a)}</td>`).join('')}<td>${pct(T.baseAccs.reduce((a, b) => a + b, 0) / K)}</td><td>—</td><td>1×</td><td>1×</td><td></td></tr>` : '';
    const rows = ok.map((x, idx) => {
      const f = MM.FAMILIES[x.m.family];
      return `<tr data-open="${x.m.id}" class="${x.r.avg === best ? 'top' : ''}">
        <td class="rank">${idx + 1}</td>
        <td><span class="fam-dot" style="background:${f.color}"></span><b>${x.m.name}</b><small>${f.name}</small></td>
        ${x.r.accs.map((a, i) => `<td><div class="cell-bar"><div style="width:${a * 100}%;background:${MM.MODEL_COLORS[i]}"></div><span>${pct(a)}</span></div></td>`).join('')}
        <td><b>${pct(x.r.avg)}</b></td>
        <td><span class="ret ${x.r.retention > 0.95 ? 'good' : x.r.retention > 0.8 ? 'mid' : 'bad'}">${pct(x.r.retention)}</span></td>
        <td>${x.r.cost > 1.05 ? `<span class="warn-text">${x.r.cost.toFixed(1)}×</span>` : '1×'}</td>
        <td>${x.r.storage > 1.05 ? `<span class="warn-text">${Math.round(x.r.storage)}×</span>` : '1×'}</td>
        <td>${needs(x.m)}</td></tr>`;
    }).join('');
    const bl = blocked.map((x) => `<tr class="blocked-row" data-open="${x.m.id}"><td class="rank">🔒</td><td><b>${x.m.name}</b><small>${MM.FAMILIES[x.m.family].name}</small></td><td colspan="${K + 5}" class="muted">${x.r.blocked}</td><td>${needs(x.m)}</td></tr>`).join('');
    $('#board').innerHTML = `<table class="board"><thead>${head}</thead><tbody>${expertRow}${baseRow}${rows}${bl}</tbody></table>`;
  }
  $('#board').addEventListener('click', (ev) => {
    const tr = ev.target.closest('[data-open]');
    if (tr) { setTab('lab'); selectMethod(tr.dataset.open); }
  });

  function renderThumbGrid(results, ctx) {
    $('#thumbGrid').innerHTML = results.map((x) => `
      <button class="tcard ${x.r.blocked ? 'locked' : ''}" data-open="${x.m.id}" style="--fc:${MM.FAMILIES[x.m.family].color}">
        <canvas id="tg-${x.m.id}"></canvas>
        <div class="tcard-meta"><b>${x.m.short}</b><span>${x.r.blocked ? '🔒' : pct(x.r.avg)}</span></div>
      </button>`).join('');
    results.forEach((x) => {
      if (x.r.blocked) return;
      MM.drawMap($(`#tg-${x.m.id}`), x.r.predict, { grid: 64, thumb: true, tasks: ctx.tasks, showPoints: true });
    });
  }
  $('#thumbGrid').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-open]');
    if (b) { setTab('lab'); selectMethod(b.dataset.open); }
  });

  // ---------------- ROBUSTNESS (multi-seed) ----------------
  const seedCache = new Map(); // trainKey -> bundles for seeds S.seed + 101·i (i ≥ 1)
  // Every extra-seed bundle, keyed by its full training config (seed included). The lab's seed checks and the
  // advisor's evidence both look here first, so the same config and seed are trained once and give one answer.
  const bundleCache = new Map();
  const cachedBundle = (cfg) => (T && T.experts && !training && trainKey(S) === trainKey(cfg) ? T : bundleCache.get(trainKey(cfg)) || null);
  async function trainCached(cfg, opt) {
    const hit = cachedBundle(cfg);
    if (hit) return hit;
    const b = await trainAsync(cfg, opt);
    if (b) bundleCache.set(trainKey(cfg), b);
    return b;
  }
  /** Bundles for seed indices 0..n-1 (0 = the lab's experts); missing ones train in parallel workers. */
  async function seedBundles(n, cancelled, msg) {
    const tk = trainKey(S);
    if (!seedCache.has(tk)) seedCache.set(tk, []);
    const arr = seedCache.get(tk);
    const missing = [];
    const base = { experts: S.experts, init: S.init, hidden: S.hidden, ftSteps: S.ftSteps };
    for (let i = 1; i < n; i++) if (!arr[i - 1]) {
      const hit = bundleCache.get(trainKey({ ...base, seed: S.seed + 101 * i })); // e.g. trained by the advisor
      if (hit) arr[i - 1] = hit; else missing.push(i);
    }
    let finished = 0;
    const runOne = async (i) => {
      const b = await trainCached({ ...base, seed: S.seed + 101 * i }, { cancelled });
      if (!b) return false;
      arr[i - 1] = b; finished++;
      if (msg) msg(`Training experts on extra seeds… ${finished}/${missing.length}`);
      return true;
    };
    if (missing.length && msg) msg(`Training experts on ${missing.length} extra seed${missing.length > 1 ? 's' : ''}${workerOK && missing.length > 1 ? ' in parallel' : ''}…`);
    for (let k = 0; k < missing.length; k += workerOK ? MAX_PARALLEL : 1) {
      const ok = await Promise.all(missing.slice(k, k + (workerOK ? MAX_PARALLEL : 1)).map(runOne));
      if (ok.includes(false) || cancelled()) return null;
      if (trainKey(S) !== tk) return null; // settings changed while training
    }
    return [T, ...arr.slice(0, n - 1)];
  }
  let robustToken = 0;
  let robustData = null; // { runs, ids, seeds, taskNames, scenario } from the last check
  const h2h = { a: null, b: null };
  function renderRobustCtl() {
    const n = [3, 5, 10].includes(+S.robustSeeds) ? +S.robustSeeds : 5;
    $$('#robustSeeds button').forEach((b) => b.classList.toggle('on', +b.dataset.n === n));
    $('#runRobust').textContent = `Run ${n} seeds`;
  }
  $('#robustSeeds').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-n]');
    if (!b) return;
    S.robustSeeds = +b.dataset.n; persist(); renderRobustCtl();
  });
  renderRobustCtl();
  async function runRobust() {
    if (!T || !T.experts || training) { toast('Wait for training to finish'); return; }
    const token = ++robustToken;
    const btn = $('#runRobust');
    btn.disabled = true;
    const ids = MM.METHODS.map((m) => m.id);
    const N = [3, 5, 10].includes(+S.robustSeeds) ? +S.robustSeeds : 5;
    const seeds = Array.from({ length: N }, (_, i) => S.seed + i * 101);
    const steps = N * (ids.length + 1);
    let doneSteps = 0;
    const prog = (label) => {
      $('#robust').innerHTML = `<div class="loading"><span class="spinner"></span> ${label}</div><div class="rprog"><div style="width:${(100 * doneSteps / steps).toFixed(0)}%"></div></div>`;
    };
    const runs = [];
    try {
      // train every missing seed (parallel workers), then score all seeds (parallel workers too)
      const cancelled = () => token !== robustToken;
      const bundles = await seedBundles(N, cancelled, (t) => prog(t));
      if (!bundles) return;
      doneSteps = N;
      const recipes = ids.map((id) => ({ key: id, method: id, params: S.params[id], align: S.align }));
      const runsBySeed = new Array(N);
      const scoreSeed = async (s) => {
        const res = await scoreAsync(bundles[s], recipes, {
          cancelled,
          onProgress: () => { doneSteps++; prog(`Mixing ${ids.length} methods on ${N} seeds${workerOK ? ' in parallel' : ''}…`); },
        });
        if (!res) return false;
        runsBySeed[s] = res;
        return true;
      };
      prog(`Mixing ${ids.length} methods on ${N} seeds${workerOK ? ' in parallel' : ''}…`);
      for (let k = 0; k < N; k += workerOK ? MAX_PARALLEL : 1) {
        const ok = await Promise.all(Array.from({ length: Math.min(workerOK ? MAX_PARALLEL : 1, N - k) }, (_, j) => scoreSeed(k + j)));
        if (ok.includes(false) || cancelled()) return;
      }
      runs.push(...runsBySeed);
      robustStale = false;
      robustData = { runs, ids, seeds, taskNames: T.tasks.map((t) => MM.TASKS[t.spec.task].name + (t.spec.flip ? ' (inv)' : '')), scenario: S.scenario };
      h2h.a = h2h.b = null;
      renderRobust(MM.summarizeSeeds(runs, ids));
    } finally {
      if (token === robustToken) btn.disabled = false;
    }
  }
  $('#runRobust').addEventListener('click', runRobust);

  function renderRobust(sum) {
    const ok = sum.filter((x) => !x.blocked).sort((a, b) => b.mean - a.mean);
    const blocked = sum.filter((x) => x.blocked);
    if (!ok.length) { $('#robust').innerHTML = '<p class="muted">Every method is blocked for this scenario.</p>'; return; }
    const top = ok[0], second = ok[1];
    const tie = second && top.mean - top.sd <= second.mean + second.sd;
    const tied = tie ? ok.filter((x) => top.mean - top.sd <= x.mean + x.sd).map((x) => MM.METHOD_BY_ID[x.id].name) : [];
    const name = (id) => MM.METHOD_BY_ID[id].name;
    const verdict = tie
      ? `<div class="verdict mid"><b>No clear winner.</b> ${tied.slice(0, 4).join(', ')}${tied.length > 4 ? ` and ${tied.length - 4} more` : ''} overlap within one standard deviation — pick among them on cost and requirements, not accuracy.</div>`
      : `<div class="verdict good"><b>${name(top.id)} wins reliably:</b> ${pct(top.mean)} ± ${(top.sd * 100).toFixed(1)} across ${top.seeds} seeds, clear of ${name(second ? second.id : top.id)} (${second ? pct(second.mean) : '—'}).</div>`;
    // Practitioner's pick: one model, no data, no training. Prefer the steadier option when means are close.
    const cheap = ok.filter((x) => { const m = MM.METHOD_BY_ID[x.id]; return !m.req.data && !m.req.training && m.family !== 'output'; });
    let pick = null;
    if (cheap.length) {
      const best = cheap[0];
      const steadier = cheap.find((x) => x !== best && x.sd < best.sd * 0.75 && best.mean - x.mean < best.sd);
      pick = steadier
        ? `<b>Free merge pick:</b> ${name(best.id)} has the best mean (${pct(best.mean)} ± ${(best.sd * 100).toFixed(1)}), but ${name(steadier.id)} is within noise (${pct(steadier.mean)}) and steadier (± ${(steadier.sd * 100).toFixed(1)}), with a worst task of ${pct(steadier.worstTask)} vs ${pct(best.worstTask)}.`
        : `<b>Free merge pick:</b> ${name(best.id)} (${pct(best.mean)} ± ${(best.sd * 100).toFixed(1)}), the best option that needs no data, no training and only 1× inference.`;
    }
    const lo = Math.max(0, Math.floor(Math.min(...ok.map((x) => x.min)) * 20) / 20 - 0.05), hi = 1;
    const X = (v) => `${(((v - lo) / (hi - lo)) * 100).toFixed(2)}%`;
    const stab = (sd) => (sd < 0.02 ? ['good', 'stable'] : sd < 0.05 ? ['mid', 'some spread'] : ['bad', 'noisy']);
    const rows = ok.map((x, i) => {
      const m = MM.METHOD_BY_ID[x.id], f = MM.FAMILIES[m.family];
      const [cls, word] = stab(x.sd);
      return `<tr data-open="${x.id}">
        <td class="rank">${i + 1}</td>
        <td><span class="fam-dot" style="background:${f.color}"></span><b>${m.name}</b><small>${f.name}</small></td>
        <td class="mono"><b>${pct(x.mean)}</b> <span class="muted">± ${(x.sd * 100).toFixed(1)}</span></td>
        <td><div class="rtrack" title="Each dot is one seed">
          <i class="rband" style="left:${X(Math.max(lo, x.mean - x.sd))};width:calc(${X(Math.min(hi, x.mean + x.sd))} - ${X(Math.max(lo, x.mean - x.sd))})"></i>
          ${x.avgs.map((v) => `<i class="rdot" style="left:${X(v)};background:${f.color}" title="${pct(v)}"></i>`).join('')}
          <i class="rmean" style="left:${X(x.mean)}"></i></div></td>
        <td class="mono">${pct(x.worstTask)}</td>
        <td class="mono">${x.wins ? `${+x.wins.toFixed(1)}/${x.seeds}` : '<span class="muted">0</span>'}</td>
        <td><span class="ret ${cls}">${word}</span></td></tr>`;
    }).join('');
    const bl = blocked.map((x) => `<tr class="blocked-row" data-open="${x.id}"><td class="rank">🔒</td><td><b>${name(x.id)}</b></td><td colspan="5" class="muted">${x.blocked}</td></tr>`).join('');
    $('#robust').innerHTML = `${verdict}${pick ? `<div class="verdict">${pick}</div>` : ''}
      <div class="board-wrap"><table class="board robust">
        <thead><tr><th>#</th><th>Method</th><th>Mean ± sd</th><th>Per-seed average accuracy <span class="axis-lbl">${pct(lo)} → 100%</span></th><th title="Lowest single-task accuracy seen on any seed">Worst task</th><th title="Seeds where this method had the best average (ties split)">Wins</th><th>Stability</th></tr></thead>
        <tbody>${rows}${bl}</tbody></table></div>
      <div class="h2h" id="h2h"></div>`;
    // default pair: the two best free merges (no data, no training, 1× model) — the decision most people face
    const pair = cheap.length >= 2 ? cheap : ok;
    if (!h2h.a || !ok.some((x) => x.id === h2h.a)) h2h.a = pair[0].id;
    if (!h2h.b || !ok.some((x) => x.id === h2h.b) || h2h.b === h2h.a) h2h.b = (pair[1] || ok[1] || ok[0]).id;
    renderH2H(ok);
  }

  /** Plain-language verdict for a pairedDiff(a, b) result. A and B are HTML names. Returns [tone, html]. */
  function pairVerdict(d, A, B, more) {
    const pts = (x) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)} pts`;
    const tTxt = Number.isFinite(d.t) ? `paired t = ${d.t.toFixed(2)}, needs ±${d.crit.toFixed(2)}` : 'same gap on every seed';
    // a result that only just clears the bar (and isn't won on every seed) is tentative, especially when browsing many pairs
    const borderline = Number.isFinite(d.t) && Math.abs(d.t) < d.crit * 1.2 && d.wins !== d.n && d.losses !== d.n;
    const real = borderline ? 'It only just clears the bar, so treat it as tentative (checking many pairs will turn up some by chance).' : 'Likely a real difference on this setup.';
    const settle = more || 'Run 10 seeds to settle it.';
    return {
      a: ['good', `${A} beats ${B} by ${pts(d.mean)} on average and wins ${d.wins}/${d.n} seeds (${tTxt}). ${real}`],
      b: ['bad', `${B} beats ${A} by ${pts(-d.mean)} on average and wins ${d.losses}/${d.n} seeds (${tTxt}). ${real}`],
      'lean-a': ['mid', `${A} leans ahead: ${pts(d.mean)} on average, wins ${d.wins}/${d.n} seeds and never loses, but the gap varies too much to call (${tTxt}). ${settle}`],
      'lean-b': ['mid', `${B} leans ahead: ${pts(-d.mean)} on average, wins ${d.losses}/${d.n} seeds and never loses, but the gap varies too much to call (${tTxt}). ${settle}`],
      noise: ['mid', `No reliable difference: ${A} averages ${pts(d.mean)}, but the winner flips between seeds (${d.wins} vs ${d.losses}, ${tTxt}). Choose on cost and requirements.`],
      same: ['mid', `${A} and ${B} score the same on every seed.`],
      'too-few': ['mid', 'Need at least 2 seeds where both recipes run.'],
    }[d.verdict];
  }

  // Head-to-head: per-seed difference between two methods (paired across the same seeds).
  function renderH2H(ok) {
    const el = $('#h2h');
    if (!el || !robustData) return;
    ok = ok || MM.summarizeSeeds(robustData.runs, robustData.ids).filter((x) => !x.blocked).sort((a, b) => b.mean - a.mean);
    const name = (id) => MM.METHOD_BY_ID[id].name;
    const opts = (sel) => ok.map((x) => `<option value="${x.id}" ${x.id === sel ? 'selected' : ''}>${esc(name(x.id))}</option>`).join('');
    const d = MM.pairedDiff(robustData.runs, h2h.a, h2h.b);
    const M = Math.max(0.02, ...d.diffs.map((x) => Math.abs(x))) * 1.15;
    const X = (v) => `${(50 + (v / M) * 50).toFixed(2)}%`;
    const pts = (x) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)} pts`;
    const A = `<b>${esc(name(h2h.a))}</b>`, B = `<b>${esc(name(h2h.b))}</b>`;
    const verdict = pairVerdict(d, A, B);
    el.innerHTML = `<div class="h2h-head"><b>Head-to-head</b>
        <select id="h2hA" aria-label="Method A">${opts(h2h.a)}</select><span class="muted">minus</span>
        <select id="h2hB" aria-label="Method B">${opts(h2h.b)}</select>
        <button class="btn ghost sm" id="robustCsv" title="Every per-seed result: seed, method, average, retention, per-task accuracy">Download CSV</button></div>
      <div class="h2h-strip" title="Each dot is one seed: average accuracy of A minus B">
        <i class="zero"></i><span class="lbl l">${esc(MM.METHOD_BY_ID[h2h.b].short)} better</span><span class="lbl r">${esc(MM.METHOD_BY_ID[h2h.a].short)} better</span>
        ${d.n > 1 ? `<i class="band" style="left:${X(Math.max(-M, d.mean - d.sd))};width:calc(${X(Math.min(M, d.mean + d.sd))} - ${X(Math.max(-M, d.mean - d.sd))})"></i>` : ''}
        ${d.diffs.map((x, i) => `<i class="dot ${x > 5e-4 ? 'pos' : x < -5e-4 ? 'neg' : ''}" style="left:${X(x)}" title="seed ${robustData.seeds[i]}: ${pts(x)}"></i>`).join('')}
        <i class="mean" style="left:${X(d.mean)}"></i>
        <span class="tick t0" style="left:${X(-M / 1.15)}">${pts(-M / 1.15)}</span><span class="tick t1" style="left:50%">0</span><span class="tick t2" style="left:${X(M / 1.15)}">${pts(M / 1.15)}</span>
      </div>
      <div class="verdict ${verdict[0]}">${verdict[1]}</div>`;
  }
  $('#robust').addEventListener('change', (ev) => {
    if (ev.target.id === 'h2hA') h2h.a = ev.target.value;
    else if (ev.target.id === 'h2hB') h2h.b = ev.target.value;
    else return;
    renderH2H();
  });
  $('#robust').addEventListener('click', (ev) => {
    if (!ev.target.closest('#robustCsv') || !robustData) return;
    const csv = MM.seedRunsCSV(robustData.runs, robustData.ids, robustData.seeds, robustData.taskNames);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `modelmixer-${robustData.scenario || 'custom'}-${robustData.seeds.length}-seeds.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $('#robust').addEventListener('click', (ev) => {
    const tr = ev.target.closest('[data-open]');
    if (tr) { setTab('lab'); selectMethod(tr.dataset.open); }
  });
  function markRobustStale() {
    const el = $('#robust');
    if (robustStale && el.querySelector('table') && !el.querySelector('.stale-note'))
      el.insertAdjacentHTML('afterbegin', '<div class="stale-note">Settings changed since this check ran. Re-run it to update.</div>');
  }

  function renderPathChart(ctx) {
    const raw = T.experts, layout = T.layout;
    const A = raw[0], B = raw[1];
    const Bal = MM.weightMatch(layout, A, B).aligned;
    const ts = Array.from({ length: 21 }, (_, i) => i / 20);
    const tasks = T.tasks.slice(0, 2);
    const lerp = (X, Y, t) => { const P = new Float32Array(X.length); for (let i = 0; i < P.length; i++) P[i] = (1 - t) * X[i] + t * Y[i]; return P; };
    const sA = [], sB = [], sAvg = [], sAl = [];
    ts.forEach((t) => {
      const acc = MM.evaluate((X, n) => MM.predictProb(layout, lerp(A, B, t), X, n), tasks);
      const acc2 = MM.evaluate((X, n) => MM.predictProb(layout, lerp(A, Bal, t), X, n), tasks);
      sA.push([t, acc[0]]); sB.push([t, acc[1]]); sAvg.push([t, (acc[0] + acc[1]) / 2]); sAl.push([t, (acc2[0] + acc2[1]) / 2]);
    });
    const series = [
      { name: 'Task A', color: MM.MODEL_COLORS[0], points: sA, dash: '4 4', width: 1.5 },
      { name: 'Task B', color: MM.MODEL_COLORS[1], points: sB, dash: '4 4', width: 1.5 },
      { name: 'Average (raw)', color: '#e2e8f0', points: sAvg, width: 2.5 },
      { name: 'Average (aligned)', color: '#a78bfa', points: sAl, width: 2.5 },
    ];
    if (sAvg.every((p, i) => Math.abs(p[1] - sAl[i][1]) < 0.005)) series[3].name = 'Average (aligned — same as raw: shared base)';
    MM.lineChart($('#pathChart'), { series, xMin: 0, xMax: 1, yMin: 0.3, yMax: 1, xTicks: [0, 0.25, 0.5, 0.75, 1], xFmt: (v) => `t=${(+v).toFixed(2)}`, yFmt: (v) => `${Math.round(v * 100)}%`, xLabel: 'interpolation t  (0 = Expert A, 1 = Expert B)' });
    $('#pathLegend').innerHTML = series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join('');
  }

  function renderLambdaChart(ctx) {
    if (!ctx.base) {
      $('#lambdaChart').innerHTML = '<div class="empty">🔒 Task vectors need a shared base model — switch Lineage to “Shared base”.</div>';
      $('#lambdaLegend').innerHTML = '';
      return;
    }
    const lams = Array.from({ length: 21 }, (_, i) => i * 0.1);
    const defs = [
      { id: 'task_arithmetic', name: 'Task Arithmetic', color: '#c084fc' },
      { id: 'ties', name: 'TIES', color: '#22d3ee' },
      { id: 'dare', name: 'DARE', color: '#fbbf24' },
    ];
    const series = defs.map((d) => ({
      name: d.name, color: d.color, width: 2.2,
      points: lams.map((l) => {
        const p = Object.assign({}, S.params[d.id], { lambda: l });
        const r = MM.runMethod(d.id, ctx, p);
        const acc = MM.evaluate(r.predict, ctx.tasks);
        return [l, acc.reduce((a, b) => a + b, 0) / acc.length];
      }),
    }));
    MM.lineChart($('#lambdaChart'), { series, xMin: 0, xMax: 2, yMin: 0.3, yMax: 1, xTicks: [0, 0.5, 1, 1.5, 2], xFmt: (v) => `λ=${(+v).toFixed(1)}`, yFmt: (v) => `${Math.round(v * 100)}%`, xLabel: 'scaling λ  (0 = base model)' });
    $('#lambdaLegend').innerHTML = series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join('');
  }

  // ---------------- ATLAS ----------------
  // ---------------- ATLAS: which knobs transfer across seeds ----------------
  const transferCache = new Map(); // toy config + align -> { label, scenario, rows }
  let transferToken = 0;
  const transferKey = () => JSON.stringify([trainKey(S), S.align]);
  const TRANSFER_TXT = { moves: 'moves with the seed', narrow: 'same spot, narrow peak', transfers: 'transfers', holds: 'holds' };
  function transferMethods(K, hasBase) {
    // trained methods are left out: every sweep point would be a training run
    K = K || T.K; hasBase = hasBase == null ? S.init === 'shared' : hasBase;
    return MM.METHODS.filter((m) => !m.req.training && !(m.req.base && !hasBase) && MM.sweepKnobs(m.id, K, MM.defaultParams(m, K)).length);
  }
  // the 5 check seeds (7, 108 … 411), as a hoisted function so it is safe at start-up (EV_SEEDS is defined further down)
  function checkSeeds() { return [0, 1, 2, 3, 4].map((i) => 7 + 101 * i); }
  const toyTransferKey = (scId) => JSON.stringify([trainKey(MM.toyConfig(scId, checkSeeds()[0])), S.align]);
  /** The 5 check seeds of any toy scenario (shared bundle cache; the lab's own experts when they match). */
  async function toyBundles(scId, cancelled, msg) {
    const cfgs = checkSeeds().map((seed) => MM.toyConfig(scId, seed));
    const out = new Array(cfgs.length);
    let done = 0;
    const step = workerOK ? MAX_PARALLEL : 1;
    for (let k = 0; k < cfgs.length; k += step) {
      const got = await Promise.all(cfgs.slice(k, k + step).map((c, j) => trainCached(c, { cancelled }).then((b) => { done++; if (msg) msg(`Training ${done}/${cfgs.length} seeds…`); out[k + j] = b; return b; })));
      if (got.some((b) => !b) || cancelled()) return null;
    }
    return out;
  }
  function renderTransfer() {
    const el = $('#transfer');
    if (!el) return;
    const sc = MM.SCENARIOS.find((x) => x.id === S.scenario);
    const where = `${sc ? sc.name : 'the custom'} toy${S.align ? ', aligned' : ''}`;
    if (!T || !T.experts) { el.innerHTML = '<p class="muted small">Train the experts in the Lab first.</p>'; return; }
    const entry = transferCache.get(transferKey());
    const rows = entry && entry.rows;
    if (!rows) {
      const ms = transferMethods();
      const nk = ms.reduce((a, m) => a + MM.sweepKnobs(m.id, T.K, MM.defaultParams(m, T.K)).length, 0);
      el.innerHTML = `<div class="tr-head"><p class="muted small">Measures ${nk} knobs across ${ms.length} methods on ${where}, about 5–15 s. Uses the same seeds and cache as the Lab’s checks.</p>
        <button class="btn sm" id="runTransfer">Measure on ${SWEEP_SEEDS} seeds</button><span class="muted small" id="transferMsg" aria-live="polite"></span></div>`;
      return;
    }
    const moving = rows.filter((r) => r.kind === 'moves'), narrow = rows.filter((r) => r.kind === 'narrow');
    const steady = rows.filter((r) => r.kind === 'holds' || r.kind === 'transfers');
    const lead = `<p class="tr-lead"><b>${moving.length} of ${rows.length}</b> knobs on ${esc(where)} have a best value that <b>moves with the seed</b>${narrow.length ? `, and ${narrow.length} more ${narrow.length > 1 ? 'have' : 'has'} a peak so narrow that the small shift still costs over a point` : ''}. Plan a small tuning sweep on your own models instead of copying a toy-optimal value.${steady.length ? ` Values that carry over: ${steady.map((r) => `${esc(r.method)} ${esc(r.knob)}`).join(', ')}.` : ' None of the defaults carry over.'}</p>`;
    const body = rows.map((r, i) => `<tr data-tr="${i}" tabindex="0" title="Open ${esc(r.method)} in the Lab with this knob swept on ${SWEEP_SEEDS} seeds">
        <td><b>${esc(r.method)}</b></td><td>${esc(r.knob)}</td>
        <td><span class="tr-tag ${r.kind}">${TRANSFER_TXT[r.kind]}</span></td>
        <td class="mono">${esc(r.lo)} – ${esc(r.hi)}</td><td class="mono">${esc(r.best)}</td><td class="mono">${(r.regret * 100).toFixed(1)} pts</td></tr>`).join('');
    el.innerHTML = `${lead}<div class="board-wrap"><table class="board tr-table"><thead><tr><th>Method</th><th>Knob</th><th>Across ${SWEEP_SEEDS} seeds</th><th>Per-seed best</th><th>Best on average</th><th>Cost on the worst seed</th></tr></thead><tbody>${body}</tbody></table></div>
      <p class="muted small">“Cost on the worst seed” is how far the best-on-average value falls below that seed’s own best. Default settings for the other knobs; click a row to open it in the Lab.</p>
      ${compareToysHTML(sc ? sc.id : null)}`;
  }
  /** Side-by-side verdicts on every toy measured so far, and a button per toy not yet measured. */
  function compareToysHTML(currentId) {
    const measured = [];
    const cur = transferCache.get(transferKey());
    if (cur) measured.push(cur);
    MM.SCENARIOS.forEach((sc) => { const e = transferCache.get(toyTransferKey(sc.id)); if (e && e !== cur && !measured.some((x) => x.scenario === sc.id)) measured.push(e); });
    const others = MM.SCENARIOS.filter((sc) => sc.id !== currentId && !measured.some((x) => x.scenario === sc.id));
    const buttons = others.length ? `<div class="tr-more"><span class="muted small">Does it depend on the data? Compare with:</span>${others.map((sc) => `<button class="btn ghost sm" data-tr-toy="${sc.id}">${esc(sc.name)}</button>`).join('')}<span class="muted small" id="trToyMsg" aria-live="polite"></span></div>` : '';
    if (measured.length < 2) return buttons;
    const keys = [];
    measured.forEach((e) => e.rows.forEach((r) => { const k = `${r.method}|${r.knob}`; if (!keys.includes(k)) keys.push(k); }));
    const find = (e, k) => e.rows.find((r) => `${r.method}|${r.knob}` === k);
    const carry = (r) => r && (r.kind === 'holds' || r.kind === 'transfers');
    const everywhere = keys.filter((k) => measured.every((e) => carry(find(e, k))));
    const split = keys.filter((k) => { const rs = measured.map((e) => find(e, k)).filter(Boolean); return rs.length > 1 && rs.some(carry) && !rs.every(carry); });
    const nm = (k) => k.replace('|', ' ');
    const lead = `<p class="tr-lead">Across ${measured.length} toys (${measured.map((e) => esc(e.label)).join(', ')}): ${everywhere.length ? `<b>${everywhere.length}</b> knob value${everywhere.length > 1 ? 's carry' : ' carries'} over on every one (${everywhere.map((k) => esc(nm(k))).join(', ')})` : '<b>no knob value carries over on every toy</b>'}${split.length ? `; <b>${split.length}</b> carr${split.length > 1 ? 'y' : 'ies'} over on some toys but not others (${split.map((k) => esc(nm(k))).join(', ')}), so which values you can copy depends on the data` : ''}.</p>`;
    const cell = (r) => (r ? `<span class="tr-tag ${r.kind}">${TRANSFER_TXT[r.kind]}</span> <span class="mono">${esc(r.best)}</span>` : '<span class="muted">—</span>');
    const body = keys.map((k) => { const [mth, kn] = k.split('|'); return `<tr><td><b>${esc(mth)}</b></td><td>${esc(kn)}</td>${measured.map((e) => `<td>${cell(find(e, k))}</td>`).join('')}</tr>`; }).join('');
    return `<h3 class="tr-sub">Across toys</h3>${lead}<div class="board-wrap" tabindex="0" role="region" aria-label="Knob verdicts across toys (scrolls sideways)"><table class="board tr-table tr-x"><thead><tr><th>Method</th><th>Knob</th>${measured.map((e) => `<th>${esc(e.label)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>
      <p class="muted small">Each toy measured on its own 5 seeds, default settings; the number is that toy’s best value on average.</p>${buttons}`;
  }
  async function runTransfer(scId) {
    const other = !!scId;
    const token = ++transferToken, Tnow = T, alignNow = S.align;
    const sc = MM.SCENARIOS.find((x) => x.id === (scId || S.scenario));
    const key = other ? toyTransferKey(scId) : transferKey();
    const cancelled = () => token !== transferToken || T !== Tnow || S.align !== alignNow || (!other && transferKey() !== key);
    const msgEl = other ? '#trToyMsg' : '#transferMsg';
    const msg = (t) => { const el = $(msgEl); if (el && !cancelled()) el.textContent = t; };
    (other ? $$('[data-tr-toy]') : [$('#runTransfer')]).forEach((b) => { if (b) b.disabled = true; });
    const bundles = other ? await toyBundles(scId, cancelled, (t) => msg(`${sc.name}: ${t}`)) : null;
    if (other && !bundles) return;
    const B0 = other ? bundles[0] : Tnow, K = B0.K, hasBase = other ? sc.init === 'shared' : S.init === 'shared';
    let byT = sweepCache.get(B0);
    if (!byT) { byT = new Map(); sweepCache.set(B0, byT); }
    const rows = [];
    for (const m of transferMethods(K, hasBase)) {
      const params = MM.defaultParams(m, K);
      for (const knob of MM.sweepKnobs(m.id, K, params)) {
        const k = sweepKey(m.id, params, knob, true);
        let res = byT.get(k);
        if (!res) {
          res = await seededSweep(m.id, knob, params, cancelled, (t) => msg(`${other ? `${sc.name} · ` : ''}${m.short}: ${t}`), bundles || undefined);
          if (!res) return;
          byT.set(k, res);
        }
        if (res.curves.some((c) => c.some((q) => q.blocked))) continue;
        const tv = MM.knobTransfer(res.xs, res.curves.map((c) => c.map((q) => q.avg)));
        const f = (x) => knobFmt(knob, x);
        rows.push({ id: m.id, method: m.short, knobKey: knob.key, knob: knob.key === 'balance' ? 'A/B balance' : knob.label.replace(/\s*\(.*\)$/, ''),
          kind: tv.kind, lo: f(Math.min(...tv.bests)), hi: f(Math.max(...tv.bests)), best: f(tv.best), regret: tv.regret });
      }
    }
    transferCache.set(key, { label: sc ? sc.name : 'Custom', scenario: sc ? sc.id : 'custom', rows });
    if (!cancelled()) renderTransfer();
  }
  $('#transfer').addEventListener('click', (ev) => {
    if (ev.target.closest('#runTransfer')) { runTransfer(); return; }
    const toy = ev.target.closest('[data-tr-toy]');
    if (toy) { runTransfer(toy.dataset.trToy); return; }
    const tr = ev.target.closest('[data-tr]');
    if (tr) openTransferRow(+tr.dataset.tr);
  });
  $('#transfer').addEventListener('keydown', (ev) => {
    const tr = ev.target.closest('[data-tr]');
    if (tr && (ev.key === 'Enter' || ev.key === ' ')) { ev.preventDefault(); openTransferRow(+tr.dataset.tr); }
  });
  function openTransferRow(i) {
    const r = ((transferCache.get(transferKey()) || {}).rows || [])[i];
    if (!r) return;
    S.params[r.id] = MM.defaultParams(MM.METHOD_BY_ID[r.id], T.K); // the row was measured at default settings
    S.sweepKnob[r.id] = r.knobKey; S.sweepSeeds = true;
    setTab('lab'); selectMethod(r.id);
    setTimeout(() => { const p = $('#sweepPanel'); if (p && !p.hidden) p.scrollIntoView({ block: 'start', behavior: 'smooth' }); }, 300);
  }

  function renderAtlas() {
    renderTransfer();
    $('#gallery').innerHTML = MM.GALLERY.map((g) => {
      const sc = MM.SCENARIOS.find((s) => s.id === g.scenario);
      // same method on both sides: show the settings, since that is what the card is about
      const side = (x) => (g.A.method === g.B.method && !g.A.params.layerwise && !g.B.params.layerwise && !!g.A.align === !!g.B.align
        ? MM.recipeLabel(x.method, x.params, x.align, sc.experts.length)
        : `${MM.METHOD_BY_ID[x.method].short}${x.params.layerwise ? ' (per-layer)' : ''}${x.align ? ' + aligned' : ''}`);
      return `<div class="panel gal-card">
        <h3>${esc(g.title)}</h3>
        <p>${esc(g.lesson)}</p>${g.hint ? `<p class="gal-hint">${esc(g.hint)}</p>` : ''}
        <div class="gal-meta"><span class="mini-tag">${esc(sc.name)}</span><span class="gal-vs"><b>A</b> ${esc(side(g.A))} <span class="muted">vs</span> <b>B</b> ${esc(side(g.B))}</span></div>
        <button class="btn sm primary" data-gallery="${g.id}">Open in the Lab</button>
      </div>`;
    }).join('');
    const Y = '<span class="cap yes" title="Yes">✓</span>', N = '<span class="cap no" title="No">✕</span>', P = '<span class="cap part" title="Partly">◐</span>';
    const cols = ['No base model needed', 'No data needed', 'No training needed', 'Single-model inference', 'Different inits OK', 'Handles interference'];
    const row = (m) => {
      const caps = [
        !m.req.base, !m.req.data, !m.req.training,
        m.inferenceK === true ? false : m.inferenceK === 'memory' ? 'p' : true,
        m.multiInit ? true : 'p-init',
        m.interference === 'yes' ? true : m.interference === 'partial' ? 'p' : false,
      ];
      return caps.map((c) => (c === true ? Y : c === false ? N : c === 'p-init' ? `<span class="cap part" title="Only after neuron alignment">◐</span>` : P));
    };
    const f = (m) => MM.FAMILIES[m.family];
    $('#matrix').innerHTML = `<table class="board matrix"><thead><tr><th>Method</th>${cols.map((c) => `<th>${c}</th>`).join('')}<th>mergekit</th></tr></thead><tbody>
      ${MM.METHODS.map((m) => `<tr data-open="${m.id}"><td><span class="fam-dot" style="background:${f(m).color}"></span><b>${m.name}</b><small>${m.tagline}</small></td>${row(m).map((c) => `<td class="c">${c}</td>`).join('')}<td>${m.mergekit ? `<code>${m.mergekit}</code>` : '<span class="muted">—</span>'}</td></tr>`).join('')}
      <tr class="ref"><td><span class="fam-dot" style="background:#94a3b8"></span><b>${MM.REBASIN.name}</b><small>${MM.REBASIN.tagline}</small></td><td class="c">${Y}</td><td class="c">${Y}</td><td class="c">${Y}</td><td class="c">${Y}</td><td class="c">${Y}</td><td class="c">${N}</td><td><span class="muted">pre-step</span></td></tr>
      </tbody></table>
      <p class="tiny muted pad">✓ yes · ◐ partly / with caveats · ✕ no. “Different inits OK ◐” means the method only works after neuron alignment (toggle it in the lab).</p>`;

    const guide = [
      { q: 'Do the experts share a base model?', a: [['No', 'Align neurons first (Re-Basin), or mix outputs: Ensemble, MoE, Distillation.', ['ensemble', 'moe', 'distill']], ['Yes', 'You unlock task vectors — the strongest data-free merges.', ['task_arithmetic', 'ties']]] },
      { q: 'Do the skills fight each other?', a: [['High sign conflict', 'Trim and elect signs: TIES, DARE-TIES, Breadcrumbs.', ['ties', 'dare', 'breadcrumbs']], ['Mostly orthogonal', 'Plain Task Arithmetic or Linear usually suffice.', ['task_arithmetic', 'linear']]] },
      { q: 'Can you spare a little data per task?', a: [['Yes', 'Data-informed merges weigh each parameter by real importance.', ['regmean', 'fisher']], ['No', 'Stay with weight-only methods.', ['ties', 'model_stock']]] },
      { q: 'Must you ship exactly one model?', a: [['Yes', 'Weight merges or a distilled student.', ['distill', 'regmean']], ['No — memory is fine', 'MoE gives near-expert quality when inputs are separable.', ['moe', 'ensemble']]] },
      { q: 'Do the tasks overlap in input space?', a: [['Yes, and disagree', 'No mix can satisfy contradictory labels; routers can’t separate them either. You need a task signal (prompt / tag) or multi-task training.', ['moe']], ['No', 'Region-specific skills merge well — see the Two/Three skills scenarios.', ['ties']]] },
      { q: 'Combining two models of the same skill?', a: [['Yes', 'SLERP or Model Stock — smooth, norm-preserving blends.', ['slerp', 'model_stock']], ['Different skills', 'Prefer task-vector or data-informed methods.', ['ties', 'regmean']]] },
    ];
    $('#guide').innerHTML = guide.map((g) => `<div class="panel gcard"><h3>${g.q}</h3>${g.a.map(([k, t, ms]) => `
      <div class="gans"><span class="gk">${k}</span><p>${t}</p><div class="gms">${ms.map((id) => `<button class="chip sm" data-open="${id}" style="--fc:${MM.FAMILIES[MM.METHOD_BY_ID[id].family].color}">${MM.METHOD_BY_ID[id].short}</button>`).join('')}</div></div>`).join('')}</div>`).join('');

    $('#families').innerHTML = Object.entries(MM.FAMILIES).map(([fid, fm]) => `<div class="panel fcard" style="--fc:${fm.color}">
      <div class="fam-name"><i></i>${fm.name}</div><p>${fm.blurb}</p>
      <div class="gms">${MM.METHODS.filter((m) => m.family === fid).map((m) => `<button class="chip sm" data-open="${m.id}">${m.short}</button>`).join('')}</div></div>`).join('');

    const gl = [
      ['Task vector', 'τ = θ_finetuned − θ_base. The direction fine-tuning moved the weights; adding it to the base “installs” the skill.'],
      ['Interference', 'When two task vectors push the same weight in opposite directions (or one is redundant noise), summing them hurts both skills.'],
      ['Permutation symmetry', 'Shuffling hidden neurons (and their connections) gives an identical function with different weights — why independently-trained models can’t simply be averaged.'],
      ['Linear mode connectivity', 'Two models are connected if every point on the straight line between them is also good. Fine-tunes of one base usually are; strangers usually aren’t.'],
      ['Skill retention', 'Merged accuracy ÷ the expert’s own accuracy, averaged over tasks. 100% means the mix lost nothing.'],
      ['Fisher information', 'How sharply the model’s predictions change when a parameter moves — a proxy for how important that weight is.'],
    ];
    $('#glossary').innerHTML = gl.map(([k, v]) => `<div class="panel gl"><b>${k}</b><p>${v}</p></div>`).join('');
  }
  document.getElementById('tab-atlas').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-open]');
    if (b) { setTab('lab'); selectMethod(b.dataset.open); }
  });

  // ---------------- EXPORT ----------------
  function yamlFor(id) {
    const p = S.params[id], K = S.experts.length, E = S.exp;
    const models = E.models.slice(0, K), base = E.base, dt = `dtype: ${E.dtype}`;
    const w = (i) => (+(p.weights ? p.weights[i] : 1)).toFixed(2);
    const list = (extra) => models.map((m, i) => `  - model: ${m}${extra ? `\n    parameters:\n${extra(i)}` : ''}`).join('\n');
    // per-layer values export as mergekit gradients: a list spread across the model's depth
    const meta = MM.METHOD_BY_ID[id];
    const lw = !!(meta.gradient && p.layerwise && !(meta.gradient.k2 && K !== 2));
    const gl = (f) => `[${p.grad.map((v) => (+f(+v)).toFixed(2)).join(', ')}]`;
    const lam = (i, scale) => (lw ? gl((v) => v * (scale ? scale(i) : 1)) : (+p.lambda * (scale ? scale(i) : 1)).toFixed(2));
    const gnote = lw ? 'Per-layer values are exported as mergekit gradients: the list is spread across the model’s depth (first value at the first layer, last at the final layer).' : '';
    switch (id) {
      case 'linear':
        return { yaml: `merge_method: linear\nmodels:\n${list((i) => `      weight: ${lw ? gl((v) => (i === 0 ? 1 - v : v)) : w(i)}`)}\nparameters:\n  normalize: true\n${dt}\n`, note: gnote };
      case 'slerp':
        return { yaml: `merge_method: slerp\nbase_model: ${models[0]}\nmodels:\n  - model: ${models[0]}\n  - model: ${models[1]}\nparameters:\n  t: ${lw ? gl((v) => v) : (+p.t).toFixed(2)}\n${dt}\n`, note: [K > 2 ? 'SLERP takes exactly two models — the third is left out.' : '', gnote].filter(Boolean).join(' ') };
      case 'karcher':
        return { yaml: `merge_method: karcher\nmodels:\n${list((i) => `      weight: ${w(i)}`)}\nparameters:\n  max_iter: 20\n  tol: 1.0e-9\n${dt}\n` };
      case 'sce':
        return { yaml: `merge_method: sce\nbase_model: ${base}\nmodels:\n${list(() => `      weight: 1.0`)}\nparameters:\n  select_topk: ${(+p.topk).toFixed(2)}\n${lw ? `  lambda: ${gl((v) => v)}\n` : +p.lambda !== 1 ? `  lambda: ${(+p.lambda).toFixed(2)}\n` : ''}${dt}\n`, note: gnote };
      case 'adamerging': {
        // AdaMerging isn't in mergekit, but its output IS a task-arithmetic merge with learned per-layer weights.
        const ctx = getCtx();
        const r = ctx && ctx.base ? compute('adamerging', ctx) : null;
        if (!r || r.blocked || !r.ada) return null;
        const { lam, nL } = r.ada;
        const wl = (i) => (nL > 1 ? `[${lam.slice(i * nL, (i + 1) * nL).map((v) => v.toFixed(2)).join(', ')}]` : lam[i].toFixed(2));
        return {
          yaml: `# AdaMerging coefficients learned in the lab, exported as a task-arithmetic merge\nmerge_method: task_arithmetic\nbase_model: ${base}\nmodels:\n${list((i) => `      weight: ${wl(i)}   # learned λ${nL > 1 ? ' per layer' : ''}`)}\n${dt}\n`,
          note: `AdaMerging isn’t in mergekit; FusionBench implements it for real models. These λs were learned on the lab’s toy experts, so re-learn them on your models (unlabeled inputs are enough).${nL > 1 ? ' Per-layer lists are spread across the model’s depth.' : ''}`,
        };
      }
      case 'della':
        return { yaml: `merge_method: della\nbase_model: ${base}\nmodels:\n${list(() => `      density: ${(+p.density).toFixed(2)}\n      epsilon: ${(+p.epsilon).toFixed(2)}\n      weight: 1.0`)}\nparameters:\n  lambda: ${lw ? gl((v) => v) : (+p.lambda).toFixed(2)}\n  normalize: true\n${dt}\n`, note: gnote };
      case 'model_stock':
        return { yaml: `merge_method: model_stock\nbase_model: ${base}\nmodels:\n${list()}\n${dt}\n` };
      case 'task_arithmetic':
        return { yaml: `merge_method: task_arithmetic\nbase_model: ${base}\nmodels:\n${list((i) => `      weight: ${lam(i, (j) => +w(j))}   # λ × w`)}\n${dt}\n`, note: gnote };
      case 'ties':
        if (lw) return { yaml: `merge_method: ties\nbase_model: ${base}\nmodels:\n${list(() => `      density: ${(+p.density).toFixed(2)}\n      weight: ${gl((v) => v / K)}   # λ per layer ÷ ${K}`)}\nparameters:\n  normalize: false\n${dt}\n`, note: `With per-layer λ, normalize is off and each weight ≈ λ ÷ ${K}, which approximates the lab’s disjoint mean. ${gnote}` };
        return { yaml: `merge_method: ties\nbase_model: ${base}\nmodels:\n${list(() => `      density: ${(+p.density).toFixed(2)}\n      weight: 1.0`)}\nparameters:\n  normalize: true   # disjoint mean, as in the lab (λ = 1)\n${dt}\n`, note: +p.lambda !== 1 ? `Lab λ = ${(+p.lambda).toFixed(2)}. With normalize: true, weights cancel out; to scale, set normalize: false and weight ≈ λ / ${K}.` : '' };
      case 'dare':
        return { yaml: `merge_method: ${p.mode === 'ties' ? 'dare_ties' : 'dare_linear'}\nbase_model: ${base}\nmodels:\n${list((i) => `      density: ${(1 - +p.drop).toFixed(2)}   # keep = 1 − drop\n      weight: ${lam(i, (j) => +w(j))}`)}\nparameters:\n  int8_mask: true\n${dt}\n`, note: gnote };
      case 'breadcrumbs':
        return { yaml: `merge_method: breadcrumbs\nbase_model: ${base}\nmodels:\n${list((i) => `      density: ${(+p.density).toFixed(2)}\n      gamma: ${(+p.gamma).toFixed(3)}\n      weight: ${lam(i)}`)}\n${dt}\n`, note: gnote };
      case 'stitch': {
        const ranges = [[0, 11], [11, 22], [22, 32]];
        return {
          yaml: `merge_method: passthrough\nslices:\n${ranges.map((r, l) => `  - sources:\n      - model: ${models[Math.min(K - 1, +p.layers[l])]}\n        layer_range: [${r[0]}, ${r[1]}]`).join('\n')}\n${dt}\n`,
          note: 'Lab layers map onto thirds of a 32-layer model. Adjust layer_range to your architecture.',
        };
      }
      case 'moe':
        return {
          file: 'moe.yml', cmd: 'mergekit-moe moe.yml ./merged-moe',
          yaml: `base_model: ${models[0]}\ngate_mode: hidden   # route by hidden states of the prompts below\n${dt}\nexperts_per_token: ${p.mode === 'top1' ? 1 : K}\nexperts:\n${models.map((m, i) => `  - source_model: ${m}\n    positive_prompts:\n${(S.exp.prompts[i] || '').split(',').map((s) => s.trim()).filter(Boolean).map((s) => `      - "${s}"`).join('\n')}`).join('\n')}\n`,
          note: 'mergekit-moe builds a Mixtral-style MoE; gates are initialised from the hidden states of each expert’s positive prompts.',
        };
      default:
        return null;
    }
  }

  const NOT_MK = {
    fisher: { title: 'Fisher merging isn’t in mergekit', body: 'It needs per-parameter Fisher information computed from data. FusionBench implements Fisher merging for Hugging Face models.' },
    regmean: { title: 'RegMean isn’t in mergekit', body: 'It needs layer-input Gram matrices from each task’s data. Use the reference implementation (dataless-model-merging) or FusionBench.' },
    ensemble: { title: 'Ensembles aren’t merges', body: 'Load each model and average their next-token probabilities at inference time:', code: `import torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\nnames = ${JSON.stringify(S.exp.models.slice(0, 2))}\ntok = AutoTokenizer.from_pretrained(names[0])\nmodels = [AutoModelForCausalLM.from_pretrained(n, torch_dtype=torch.bfloat16) for n in names]\n\ndef next_token_probs(prompt):\n    ids = tok(prompt, return_tensors="pt").input_ids\n    probs = [m(ids).logits[:, -1].softmax(-1) for m in models]\n    return torch.stack(probs).mean(0)   # equal-weight ensemble` },
    localize_stitch: { title: 'Localize-and-Stitch isn’t in mergekit', body: 'It needs per-task sparse masks, learned on a little data or taken from the largest task-vector entries. Use the authors’ reference implementation.' },
    adamerging: { title: 'AdaMerging needs a shared base', body: 'It learns how much of each task vector to add back to the base, so it can’t run without one. FusionBench implements AdaMerging for real models.' },
    distill: { title: 'Distillation is training, not merging', body: 'Train a student on the teacher mixture’s outputs. DistillKit (arcee-ai) and TRL’s GKD trainer both support logit-based distillation from Hugging Face teachers.' },
  };

  /** YAML comments with the multi-seed sweep evidence for this recipe's knobs (only knobs already swept on seeds). */
  /**
   * Seed evidence for each knob of a method: from a 5-seed sweep at this recipe's settings if there is one, else at
   * default settings (as the Atlas transfer table measures). Rows: { knob, kind, lo, hi, best, regret, now, n, basis }.
   */
  function knobEvidence(id, opts) {
    if (!T || !T.experts) return { rows: [], knobs: 0 };
    const m = MM.METHOD_BY_ID[id], params = S.params[id], defaults = MM.defaultParams(m, T.K);
    const ks = MM.sweepKnobs(id, T.K, params);
    const byT = sweepCache.get(T);
    const rows = [];
    for (const knob of ks) {
      let basis = 'recipe', res = byT && byT.get(sweepKey(id, params, knob, true));
      if (!res && !(opts && opts.recipeOnly)) { basis = 'defaults'; res = byT && byT.get(sweepKey(id, defaults, knob, true)); }
      if (!res || res.curves.length < 2 || res.curves.some((c) => c.some((q) => q.blocked))) continue;
      const tv = MM.knobTransfer(res.xs, res.curves.map((c) => c.map((q) => q.avg)));
      const f = (x) => knobFmt(knob, x);
      rows.push({ knob: knob.key === 'balance' ? 'A/B balance' : knob.label.replace(/\s*\(.*\)$/, ''), kind: tv.kind, n: tv.n, basis,
        lo: f(Math.min(...tv.bests)), hi: f(Math.max(...tv.bests)), best: f(tv.best), regret: tv.regret, now: f(MM.knobValue(params, knob)) });
    }
    return { rows, knobs: ks.length };
  }
  /** YAML comments with the multi-seed sweep evidence for this recipe's knobs (only sweeps at this recipe's settings). */
  function evidenceComments(id) {
    const { rows, knobs } = knobEvidence(id, { recipeOnly: true });
    if (!rows.length) return { lines: [], swept: 0, knobs };
    const m = MM.METHOD_BY_ID[id], sc = MM.SCENARIOS.find((x) => x.id === S.scenario);
    const pts = (r) => `${(r.regret * 100).toFixed(1)} pts`;
    const lines = rows.map((r) => (r.kind === 'moves'
      ? `#   ${r.knob}: best moves with the seed (${r.lo} to ${r.hi}); tune it on your own models. This config: ${r.now}.`
      : r.kind === 'transfers'
        ? `#   ${r.knob}: ${r.best} transfers (within ${pts(r)} of every seed's best). This config: ${r.now}.`
        : r.kind === 'narrow'
          ? `#   ${r.knob}: best sits near ${r.best} on every seed but the peak is narrow (${pts(r)} on the worst seed); tune it finely. This config: ${r.now}.`
          : `#   ${r.knob}: best ${r.best} holds across seeds. This config: ${r.now}.`));
    return {
      lines: [`# ModelMixer toy evidence: ${m.name} on ${sc ? sc.name : 'a custom'} toy, knob sweeps over ${rows[0].n} seeds.`, ...lines, '# Toy-scale evidence only: confirm on a sample of your real data.'],
      swept: rows.length, knobs,
    };
  }
  let annotateToken = 0;
  async function annotateAll() {
    const id = S.method, params = S.params[id], Tnow = T;
    if (!T || !T.experts || training) return;
    const token = ++annotateToken;
    const cancelled = () => token !== annotateToken || T !== Tnow || S.method !== id;
    const msg = (t) => { const el = $('#annotateMsg'); if (el && !cancelled()) el.textContent = t; };
    const btn = $('#annotateAll');
    if (btn) btn.disabled = true;
    let byT = sweepCache.get(Tnow);
    if (!byT) { byT = new Map(); sweepCache.set(Tnow, byT); }
    for (const knob of MM.sweepKnobs(id, Tnow.K, params)) {
      const key = sweepKey(id, params, knob, true);
      if (byT.has(key)) continue;
      const res = await seededSweep(id, knob, params, cancelled, msg);
      if (!res) return;
      byT.set(key, res);
    }
    if (!cancelled() && S.tab === 'export') renderExport();
  }
  $('#exportNotes').addEventListener('click', (ev) => { if (ev.target.closest('#annotateAll')) annotateAll(); });
  function renderExport() {
    const id = S.method, m = MM.METHOD_BY_ID[id], K = S.experts.length;
    const opts = MM.METHODS.map((x) => `<option value="${x.id}" ${x.id === id ? 'selected' : ''}>${x.name}${x.mergekit ? '' : '  (no mergekit)'}</option>`).join('');
    $('#exportForm').innerHTML = `
      <div class="panel-title"><span>Recipe</span><span class="hint">mirrors the lab</span></div>
      <div class="field"><label for="exMethod">Method</label><select id="exMethod">${opts}</select></div>
      <div class="field"><label for="exBase">Base model ${m.req.base ? '' : '<span class="hint">(unused by this method)</span>'}</label><input id="exBase" value="${esc(S.exp.base)}" spellcheck="false" /></div>
      ${Array.from({ length: K }, (_, i) => `<div class="field" style="--c:${MM.MODEL_COLORS[i]}"><label><i class="dot"></i>Expert ${LETTERS[i]}&nbsp;<span class="hint">(${MM.TASKS[S.experts[i].task].name} in the lab)</span></label><input aria-label="Expert ${LETTERS[i]} model id" data-exm="${i}" value="${esc(S.exp.models[i])}" spellcheck="false" />
        ${id === 'moe' ? `<input class="sub" aria-label="Expert ${LETTERS[i]} routing prompts" data-exp="${i}" value="${esc(S.exp.prompts[i])}" placeholder="routing prompts, comma separated" />` : ''}</div>`).join('')}
      <div class="field"><label>dtype</label><div class="seg wide" id="exDtype">${['bfloat16', 'float16', 'float32'].map((d) => `<button data-v="${d}" class="${S.exp.dtype === d ? 'on' : ''}">${d}</button>`).join('')}</div></div>
      <div class="note">Parameters (weights, density, λ, t…) come from the lab’s current recipe for <b>${m.name}</b>. Tune it there, then copy the config.</div>`;
    const y = yamlFor(id);
    const notes = $('#exportNotes');
    if (y) {
      const ev = evidenceComments(id);
      $('#yaml').textContent = ev.lines.length ? `${ev.lines.join('\n')}\n${y.yaml}` : y.yaml;
      $('#exportFile').textContent = y.file || 'config.yml';
      $('#cmd').textContent = `pip install mergekit\n${y.cmd || 'mergekit-yaml config.yml ./merged --cuda'}`;
      const evNote = ev.knobs && ev.swept < ev.knobs
        ? `<div class="note muted ev-note">${ev.swept ? `The config is annotated with seed evidence for ${ev.swept} of ${ev.knobs} knobs.` : 'Annotate the config with which knobs transfer across seeds?'}
            <button class="btn sm" id="annotateAll">Sweep ${ev.swept ? 'the rest' : ev.knobs > 1 ? `all ${ev.knobs} knobs` : 'the knob'} on ${SWEEP_SEEDS} seeds</button><span class="muted small" id="annotateMsg" aria-live="polite"></span></div>` : '';
      notes.innerHTML = (y.note ? `<div class="note">${y.note}</div>` : '') + evNote;
    } else {
      const n = NOT_MK[id];
      $('#yaml').textContent = n.code || `# ${n.title}\n# ${n.body}`;
      $('#exportFile').textContent = n.code ? 'ensemble.py' : 'notes';
      $('#cmd').textContent = id === 'ensemble' ? 'pip install torch transformers\npython ensemble.py' : 'See the linked open-source implementations below.';
      notes.innerHTML = `<div class="note"><b>${n.title}.</b> ${n.body}</div><div class="oss-row">${m.oss.map((o) => `<a class="pill link" href="${o.url}" target="_blank" rel="noopener">${o.name} ↗</a>`).join('')}</div>`;
    }
  }
  $('#exportForm').addEventListener('change', (ev) => {
    if (ev.target.id === 'exMethod') { S.method = ev.target.value; persist(); renderPicker(); refresh(); renderExport(); }
  });
  $('#exportForm').addEventListener('input', (ev) => {
    const t = ev.target;
    if (t.id === 'exBase') S.exp.base = t.value;
    else if (t.dataset.exm !== undefined) S.exp.models[+t.dataset.exm] = t.value;
    else if (t.dataset.exp !== undefined) S.exp.prompts[+t.dataset.exp] = t.value;
    else return;
    persist();
    const y = yamlFor(S.method);
    if (y) $('#yaml').textContent = y.yaml;
  });
  $('#exportForm').addEventListener('click', (ev) => {
    const b = ev.target.closest('#exDtype button');
    if (!b) return;
    S.exp.dtype = b.dataset.v; persist(); renderExport();
  });
  function legacyCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    ta.remove();
    return ok;
  }
  async function copy(text, msg, manualFallback) {
    try { await navigator.clipboard.writeText(text); toast(msg || 'Copied to clipboard'); return; } catch (e) { /* try the legacy path */ }
    if (legacyCopy(text)) { toast(msg || 'Copied to clipboard'); return; }
    if (manualFallback) { try { window.prompt('Copy this link:', text); return; } catch (e) { /* prompts blocked */ } }
    toast('Copy failed — select the text manually');
  }
  $('#shareLink').addEventListener('click', () => copy(recipeLink(), 'Recipe link copied: it opens this exact mix', true));
  window.addEventListener('hashchange', () => openRecipe(MM.recipeFromHash(location.hash)));
  // Open an encoded recipe in the running app (pasted link / hash change). Returns true if applied.
  function openRecipe(code) {
    if (!code || code === MM.encodeRecipe(currentRecipe())) return false;
    const d = MM.decodeRecipe(code);
    if (!d.ok) { toast(`Couldn’t open that link: ${d.error}`); return false; }
    const retrain = trainKey(d.recipe) !== trainKey(S);
    applyRecipe(d.recipe);
    arenaStale = robustStale = true;
    setTab('lab');
    renderPicker();
    if (retrain) scheduleRetrain(); else { persist(); renderIngredients(); refresh(); }
    toast('Opened a shared recipe');
    return true;
  }
  $('#copyYaml').addEventListener('click', () => copy($('#yaml').textContent));
  $('#copyCmd').addEventListener('click', () => copy($('#cmd').textContent));
  $('#dlYaml').addEventListener('click', () => {
    const blob = new Blob([$('#yaml').textContent], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = $('#exportFile').textContent.replace('notes', 'notes.txt');
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  // ---------------- IMPORT A MERGEKIT CONFIG ----------------
  const MK_EXAMPLE = `# the classic mergekit SLERP gradient: early layers from A, late layers from B
merge_method: slerp
base_model: mistralai/Mistral-7B-v0.1
models:
  - model: mistralai/Mistral-7B-v0.1
  - model: teknium/OpenHermes-2.5-Mistral-7B
parameters:
  t: [0, 0.5, 0.3, 0.7, 1]
dtype: bfloat16
`;
  $('#mkExample').addEventListener('click', () => { $('#mkIn').value = MK_EXAMPLE; $('#mkIn').focus(); });
  $('#mkLoad').addEventListener('click', () => {
    const r = MM.importMergekit($('#mkIn').value, { order: S.exp.models });
    const out = $('#mkOut');
    if (!r.ok) { out.innerHTML = `<div class="verdict bad"><b>Couldn’t import:</b> ${esc(r.error)}</div>`; return; }
    const notes = r.notes.slice();
    let retrain = false;
    // expert count follows the config (2 or 3)
    while (S.experts.length > r.K) { S.experts.pop(); retrain = true; }
    while (S.experts.length < r.K) {
      if (MM.taskKind(S.experts[0].task) === 'digits') {
        const used = new Set(S.experts.map((e) => e.task));
        S.experts.push({ task: ['d45', 'd67', 'd89', 'd23', 'd01'].find((t) => !used.has(t)) || 'd45' });
      } else {
        const used = new Set(S.experts.map((e) => e.region));
        S.experts.push({ task: 'xor', region: ['top', 'right', 'left', 'center'].find((x) => !used.has(x)) || 'top' });
      }
      retrain = true;
    }
    if (retrain) notes.push(`The lab now has ${r.K} experts to match the config’s ${r.K} models.`);
    const meta = MM.METHOD_BY_ID[r.method];
    if (meta.req.base && S.init === 'independent') { S.init = 'shared'; S.align = false; retrain = true; notes.push('Switched the lab to a shared base, which this method needs.'); }
    S.method = r.method;
    S.params[r.method] = Object.assign(MM.defaultParams(meta, r.K), r.params);
    S.exp.models = r.models.concat(S.exp.models.slice(r.models.length)).slice(0, 3);
    if (r.base) S.exp.base = r.base;
    if (r.dtype && ['bfloat16', 'float16', 'float32'].includes(r.dtype)) S.exp.dtype = r.dtype;
    arenaStale = robustStale = true;
    renderPicker();
    if (retrain) scheduleRetrain(); else { persist(); refresh(); }
    renderExport();
    const label = MM.recipeLabel(r.method, S.params[r.method], S.align, r.K);
    out.innerHTML = `<div class="verdict ${r.warnings.length ? 'mid' : 'good'}"><b>Loaded:</b> <code>${esc(label)}</code> with ${r.K} experts.
        ${notes.length ? `<ul class="mk-list">${notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : ''}
        ${r.warnings.length ? `<div class="mk-warn"><b>Approximated</b><ul class="mk-list">${r.warnings.map((w) => `<li>${esc(w)}</li>`).join('')}</ul></div>` : ''}
        <div class="row"><button class="btn sm primary" id="mkOpen">Open in the Lab</button></div></div>`;
    toast(`Imported ${meta.name}`);
  });
  $('#mkOut').addEventListener('click', (ev) => { if (ev.target.closest('#mkOpen')) setTab('lab'); });

  // ---------------- KEYBOARD SHORTCUTS SHEET ----------------
  const MAC = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');
  const MOD = MAC ? '⌘' : 'Ctrl';
  const KEY_GROUPS = [
    ['Recipe', [
      [[`${MOD} Z`], 'Undo the last recipe change (method, knobs, alignment, pin)'],
      [[MAC ? '⇧⌘ Z' : 'Ctrl Y', MAC ? '' : 'Ctrl ⇧ Z'].filter(Boolean), 'Redo'],
      [['← →', '↑ ↓'], 'Move a focused knob (slider) one step'],
      [['Home', 'End'], 'Jump a focused knob to its minimum / maximum'],
    ]],
    ['Moving around', [
      [['Tab'], 'First stops are skip links: methods, recipe knobs, results'],
      [['← →'], 'In the method picker: move between methods (it is one Tab stop)'],
      [['Enter', 'Space'], 'Pick the focused method, press the focused button'],
      [['← →'], 'In the view tabs: switch between Mix Lab, Arena, Atlas and Export'],
    ]],
    ['Dialogs and tour', [
      [['Esc'], 'Close the advisor, the report, this list or the tour'],
      [['← →'], 'Previous / next tour step'],
      [['?'], 'Open this list'],
    ]],
  ];
  const keysDlg = $('#keysDlg');
  $('#keysBody').innerHTML = KEY_GROUPS.map(([title, rows]) => `<section><h3>${title}</h3><dl>${rows.map(([keys, what]) =>
    `<dt>${keys.map((k) => k.split(' ').map((x) => `<kbd>${esc(x)}</kbd>`).join(' ')).join(' <span class="muted">or</span> ')}</dt><dd>${esc(what)}</dd>`).join('')}</dl></section>`).join('');
  function openKeys() {
    if (keysDlg.open) return;
    if (typeof keysDlg.showModal === 'function') keysDlg.showModal(); else keysDlg.setAttribute('open', '');
  }
  $('#closeKeys').addEventListener('click', () => keysDlg.close());
  keysDlg.addEventListener('click', (ev) => { if (ev.target === keysDlg) keysDlg.close(); });
  document.querySelector('[data-keys]').addEventListener('click', (ev) => { ev.preventDefault(); openKeys(); });
  document.addEventListener('keydown', (ev) => {
    if (ev.key !== '?' || ev.metaKey || ev.ctrlKey || ev.altKey || ev.defaultPrevented) return;
    const t = ev.target;
    if (t && (t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && !/^(range|checkbox|radio|button)$/i.test(t.type)) || t.isContentEditable)) return;
    if (document.querySelector('dialog[open]') || tourIdx >= 0) return; // one dialog at a time
    ev.preventDefault();
    openKeys();
  });

  // ---------------- FIRST-RUN TOUR ----------------
  const TOUR_KEY = 'modelmixer:tour';
  const TOUR = [
    { sel: '#ingredients .scenarios', title: 'Pick a scenario', text: 'Each scenario trains small expert networks right here in your browser, each on its own skill. Two skills, conflicts, strangers with no shared base, image-like digits: start with Two skills.' },
    { sel: '#picker', title: 'Choose how to mix them', text: '18 merging methods in six families, from plain weight averaging to TIES, DARE, RegMean, Mixture of Experts and distillation. A lock means the method needs something this scenario lacks.' },
    { sel: '#controls', title: 'Tune the recipe', text: 'Every knob updates the merge live. ⌘Z / Ctrl+Z undoes. The sensitivity panel below shows how forgiving each knob is, and the best and safest settings.' },
    { sel: '.stage', title: 'See what the mix keeps and loses', text: 'The map shows what the merged model predicts. The scorecard compares it with each expert alone. Pin a recipe to compare two side by side, with a paired check of whether one is really better.' },
    { sel: '#openAdvisor', title: 'Not sure where to start?', text: 'Describe your real models to the Recipe advisor for ranked picks backed by toy measurements. The Arena runs every method, Report gives you one shareable page, and Export writes the mergekit config. Press ? any time for keyboard shortcuts.' },
  ];
  let tourIdx = -1;
  const tourSeen = () => { try { return localStorage.getItem(TOUR_KEY) === 'done'; } catch (e) { return true; } };
  function maybeStartTour() {
    if (tourIdx >= 0 || tourSeen() || S.tab !== 'lab' || (advDlg && advDlg.open)) return;
    setTimeout(() => { if (!tourSeen() && tourIdx < 0) startTour(); }, 400);
  }
  function startTour() { setTab('lab'); tourIdx = 0; $('#tourPop').hidden = false; $('#tourSpot').hidden = false; showTourStep(); }
  function endTour() {
    tourIdx = -1;
    $('#tourPop').hidden = true; $('#tourSpot').hidden = true;
    try { localStorage.setItem(TOUR_KEY, 'done'); } catch (e) { /* storage unavailable */ }
    const b = $('#openTour'); if (b) b.focus({ preventScroll: true });
  }
  function placeTour() {
    if (tourIdx < 0) return;
    const step = TOUR[tourIdx], target = document.querySelector(step.sel);
    const pop = $('#tourPop'), spot = $('#tourSpot');
    if (!target) return;
    const r = target.getBoundingClientRect(), pad = 6, vw = window.innerWidth, vh = window.innerHeight;
    Object.assign(spot.style, { left: `${r.left - pad}px`, top: `${r.top - pad}px`, width: `${r.width + 2 * pad}px`, height: `${r.height + 2 * pad}px` });
    const w = Math.min(340, vw - 24), ph = pop.offsetHeight || 180;
    pop.style.width = `${w}px`;
    let top = r.bottom + 14;
    if (top + ph > vh - 12) top = r.top - ph - 14;           // no room below: go above
    if (top < 12) top = Math.max(12, Math.min(vh - ph - 12, r.top + 12)); // tall target: overlap its top edge
    const left = Math.max(12, Math.min(vw - w - 12, r.left + Math.min(24, r.width / 4)));
    Object.assign(pop.style, { top: `${top}px`, left: `${left}px` });
  }
  function showTourStep() {
    const step = TOUR[tourIdx], target = document.querySelector(step.sel);
    $('#tourStep').textContent = `Step ${tourIdx + 1} of ${TOUR.length}`;
    $('#tourTitle').textContent = step.title;
    $('#tourText').textContent = step.text;
    $('#tourBack').disabled = tourIdx === 0;
    $('#tourNext').textContent = tourIdx === TOUR.length - 1 ? 'Done' : 'Next';
    if (target) target.scrollIntoView({ block: 'center', behavior: 'auto' });
    requestAnimationFrame(() => { placeTour(); $('#tourNext').focus({ preventScroll: true }); });
  }
  const tourGo = (d) => { const n = tourIdx + d; if (n >= TOUR.length) return endTour(); if (n < 0) return; tourIdx = n; showTourStep(); };
  $('#tourNext').addEventListener('click', () => tourGo(1));
  $('#tourBack').addEventListener('click', () => tourGo(-1));
  $('#tourSkip').addEventListener('click', endTour);
  $('#openTour').addEventListener('click', () => { if (tourIdx >= 0) endTour(); else startTour(); });
  document.addEventListener('keydown', (ev) => {
    if (tourIdx < 0) return;
    if (ev.key === 'Escape') { ev.preventDefault(); endTour(); }
    else if (ev.key === 'ArrowRight') { ev.preventDefault(); tourGo(1); }
    else if (ev.key === 'ArrowLeft') { ev.preventDefault(); tourGo(-1); }
  });
  window.addEventListener('resize', placeTour);
  window.addEventListener('scroll', placeTour, { passive: true });

  $('#gallery').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-gallery]');
    if (!b) return;
    const g = MM.GALLERY.find((x) => x.id === b.dataset.gallery);
    if (!openRecipe(MM.encodeRecipe(MM.galleryRecipe(g)))) setTab('lab'); // already loaded: just show it
  });

  // ---------------- RECIPE REPORT ----------------
  /** Gather everything known about the current recipe. Only checks that were actually run are included. */
  function collectReport() {
    const m = MM.METHOD_BY_ID[S.method], r = current;
    if (!T || !T.experts || !r || r.blocked) return null;
    const sc = MM.SCENARIOS.find((x) => x.id === S.scenario);
    const taskName = (t) => `${MM.TASKS[t.spec.task].name}${t.spec.flip ? ' (inv)' : ''}${t.kind === 'plane' ? ` · ${MM.REGIONS[t.spec.region].name}` : ''}`;
    const y = yamlFor(S.method), nm = NOT_MK[S.method];
    const exp = y
      ? { file: y.file || 'config.yml', yaml: ((ev) => (ev.length ? `${ev.join('\n')}\n` : ''))(evidenceComments(S.method).lines) + y.yaml, cmd: `pip install mergekit\n${y.cmd || 'mergekit-yaml config.yml ./merged --cuda'}`, note: y.note || '' }
      : { file: 'notes', yaml: nm ? nm.code || `# ${nm.title}\n# ${nm.body}` : '', cmd: 'See the linked open-source implementations.', note: nm ? `${nm.title}. ${nm.body}` : '' };
    const text = (sel) => { const el = $(sel); return el ? el.innerText.trim() || null : null; };
    let robust = null;
    if (robustData && !robustStale) {
      const sum = MM.summarizeSeeds(robustData.runs, robustData.ids).filter((x) => !x.blocked).sort((a, b) => b.mean - a.mean);
      const rows = sum.slice(0, 5);
      const cur = sum.find((x) => x.id === S.method);
      if (cur && !rows.includes(cur)) rows.push(cur);
      robust = {
        seeds: robustData.seeds.length,
        rows: rows.map((x) => ({ name: MM.METHOD_BY_ID[x.id].name, mean: x.mean, sd: x.sd, worst: x.worstTask, stability: x.sd < 0.02 ? 'stable' : x.sd < 0.05 ? 'some spread' : 'noisy', current: x.id === S.method })),
        verdict: text('#robust .verdict'), h2h: text('#h2h .verdict'),
      };
    }
    const at = S.pin ? attrCache.get(edgeKey()) : null, ss = S.pin ? attrSeedCache.get(edgeKey()) : null;
    const compare = S.pin ? { pinned: labelOf(S.pin), now: labelOf(nowRecipe()),
      changes: MM.recipeDiff(S.pin, nowRecipe(), T.K).map((x, i) => ({ what: x.what, a: x.a, b: x.b,
        delta: at && at.parts[i] ? at.parts[i].delta : undefined,
        seeds: ss && ss.parts[i] && ss.parts[i].sign !== 'none' ? { mean: ss.parts[i].mean, sd: ss.parts[i].sd, sign: SIGN_TXT[ss.parts[i].sign](ss.parts[i]) } : undefined })),
      attribution: ss ? { total: ss.total.mean, totalSd: ss.total.sd, interaction: ss.interaction.mean, seeds: ss.n } : at ? { total: at.total, interaction: at.interaction, seed: S.seed } : null, summary: text('.cmp-box > div:nth-child(2)'), edge: text('#edge .verdict') } : null;
    let mapPng = null;
    try {
      const cv = $('#mainCanvas'), c2 = document.createElement('canvas');
      c2.width = c2.height = 440;
      c2.getContext('2d').drawImage(cv, 0, 0, 440, 440);
      mapPng = c2.toDataURL('image/png');
    } catch (e) { /* tainted or unavailable canvas: report without the image */ }
    let saliency = null;
    const ctx = getCtx();
    if (ctx && ctx.tasks[0].kind === 'digits') {
      const ex = expertMaps(ctx), row = MM.saliencyRow(ctx, r.predict, ex);
      const png = (map) => {
        try {
          const c = document.createElement('canvas');
          c.width = c.height = 96;
          MM.drawSaliency(c, map, Math.max(...map));
          return c.toDataURL('image/png');
        } catch (e) { return null; }
      };
      const seeds = salSeedCache.get(salSeedKey()); // only if the 5-seed check ran for exactly this recipe
      saliency = {
        mean: row.mean, router: !!r.router,
        tasks: T.tasks.map((t, i) => ({ name: `${LETTERS[i]} · ${taskName(t)}`, match: row.match[i], merged: r.accs[i], expert: T.expertAccs[i][i], expertPng: png(ex[i]), mergedPng: png(row.maps[i]), seeds: seeds ? seeds[i] : null })),
      };
    }
    return {
      title: `${m.name} recipe report`, date: new Date().toISOString().slice(0, 10), link: recipeLink(),
      recipe: {
        method: m.name, family: MM.FAMILIES[m.family].name, label: MM.recipeLabel(S.method, S.params[S.method], S.align, T.K),
        scenario: sc ? sc.name : 'Custom', experts: T.tasks.map((t, i) => `${LETTERS[i]}: ${taskName(t)}`),
        lineage: S.init === 'shared' ? 'shared base' : 'independent inits', aligned: S.align, width: S.hidden, ftSteps: S.ftSteps, seed: S.seed,
      },
      result: {
        tasks: T.tasks.map((t, i) => ({ name: `${LETTERS[i]} · ${taskName(t)}`, merged: r.accs[i], expert: T.expertAccs[i][i], base: T.baseAccs ? T.baseAccs[i] : null })),
        avg: r.avg, retention: r.retention, cost: r.cost > 1.05 ? `${r.cost.toFixed(1)}×` : '1×', storage: r.storage > 1.05 ? `${Math.round(r.storage)}×` : '1×', warnings: r.warnings || [],
      },
      mapPng,
      sensitivity: !$('#sweepPanel').hidden ? text('#sweepSum p') : null,
      transfer: ((ev) => (ev.rows.length ? { rows: ev.rows, knobs: ev.knobs, seeds: ev.rows[0].n } : null))(knobEvidence(S.method)),
      layerMap: !$('#layerPanel').hidden ? text('#heatSum p') : null,
      robust, compare, saliency,
      method: { tagline: m.tagline, how: m.how, pros: m.pros, cons: m.cons, realWorld: m.realWorld || '', paper: m.paper, oss: m.oss },
      export: exp,
    };
  }
  let report = null;
  const repDlg = $('#reportDlg');
  $('#openReport').addEventListener('click', () => {
    const d = collectReport();
    if (!d) { toast('Train the experts and pick a method that can run here first'); return; }
    report = Object.assign(MM.buildReport(d), { method: S.method });
    $('#repFrame').srcdoc = report.html;
    if (typeof repDlg.showModal === 'function') { if (!repDlg.open) repDlg.showModal(); } else repDlg.setAttribute('open', '');
  });
  $('#closeReport').addEventListener('click', () => { if (repDlg.open) repDlg.close(); });
  repDlg.addEventListener('click', (ev) => { if (ev.target === repDlg) repDlg.close(); });
  $('#repPrint').addEventListener('click', () => { try { $('#repFrame').contentWindow.focus(); $('#repFrame').contentWindow.print(); } catch (e) { toast('Printing is blocked here; download the HTML and print it'); } });
  $('#repHtml').addEventListener('click', () => {
    if (!report) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([report.html], { type: 'text/html' }));
    a.download = `modelmixer-report-${report.method}.html`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $('#repMd').addEventListener('click', () => { if (report) copy(report.markdown, 'Report copied as Markdown'); });

  // ---------------- RECIPE ADVISOR ----------------
  const advDlg = $('#advisor');
  let advResult = null;
  function openAdvisor() {
    renderAdvisor();
    if (typeof advDlg.showModal === 'function') { if (!advDlg.open) advDlg.showModal(); } else advDlg.setAttribute('open', '');
  }
  function closeAdvisor() { evToken++; if (advDlg.open) { if (typeof advDlg.close === 'function') advDlg.close(); else advDlg.removeAttribute('open'); } }
  // ---- measured evidence: each top pick, run on the closest toy scenario over EV_SEEDS seeds ----
  const EV_SEEDS = checkSeeds(); // the lab's 5 check seeds (7, 108 … 411), shared via bundleCache
  const evBundles = {}; // toyKey -> trained bundles (one per seed)
  const evCache = {}; // evKey -> seed summary
  const evSeeds = {}; // evKey -> per-seed results, for the paired re-ranking
  let evToken = 0;
  const toyKey = (t) => `${t.scenario}:${t.init || 'shared'}`;
  const evKey = (t, align, id) => `${toyKey(t)}|${align ? 1 : 0}|${id}`;
  const toysOf = (r) => r.toys || [{ scenario: r.scenario, init: r.init }];
  const toySc = (t) => MM.SCENARIOS.find((x) => x.id === t.scenario);
  function evidenceLine(id, r, t) {
    const sc = toySc(t), e = evCache[evKey(t, r.align, id)];
    if (!e) return `<span class="ev-wait"><span class="spinner"></span> Measuring on the ${sc.name} toy…</span>`;
    if (e.blocked) return `<span class="muted">Can’t run on the ${sc.name} toy: ${e.blocked}</span>`;
    const tone = e.mean > 0.95 ? 'good' : e.mean > 0.85 ? 'mid' : 'bad';
    return `<span class="ev-label">Measured on the ${sc.name} toy${t.init === 'independent' && sc.init !== 'independent' ? ' (independent inits, aligned)' : r.align ? ' (aligned)' : ''}</span>
      <span class="ev-val ${tone}">${pct(e.mean)} <small>± ${(e.sd * 100).toFixed(1)}</small></span>
      <span class="ev-sub">worst task ${pct(e.worstTask)} · ${e.seeds} seeds · default knobs</span>`;
  }
  function evidenceHTML(id, r) { return toysOf(r).map((t) => `<div class="ev-line">${evidenceLine(id, r, t)}</div>`).join(''); }
  function evidenceVerdict(r, blend) {
    const nm = (id) => MM.METHOD_BY_ID[id].name;
    const toys = toysOf(r);
    if (!blend) return '<span class="muted">Checking these picks on a toy that resembles your setup…</span>';
    if (toys.length > 1) {
      const names = toys.map((t) => toySc(t).name);
      if (blend.promoted) {
        const ds = blend.promoted.perToy;
        return `<b>Re-ranked by measurement on both toys:</b> ${nm(blend.promoted.id)} beat the rules’ #1 (${nm(blend.promoted.over)}) on ${names.map((n, i) => `${n} (${ds[i].wins}/${ds[i].n} seeds, ${(ds[i].mean * 100 >= 0 ? '+' : '')}${(ds[i].mean * 100).toFixed(1)} pts)`).join(' and ')}, so it moves to #1.`;
      }
      if (blend.agree) return `<b>Both toys agree:</b> ${nm(blend.common)} is best or within seed noise of the best on ${names.join(' and ')}${blend.common === r.recs[0].id ? ', matching the rules’ #1' : ', though not by enough on both to re-rank'}.`;
      return `<b>The toys disagree:</b> on ${names[0]} ${nm(blend.best[0])} scores best, on ${names[1]} ${nm(blend.best[1])} does. The right pick depends on your data, so check both on a sample of it.`;
    }
    const t = toys[0], scName = toySc(t).name;
    if (blend.promoted) {
      const { id, over, d } = blend.promoted;
      return `<b>Re-ranked by measurement:</b> on the ${scName} toy, ${nm(id)} beat the rules’ #1 (${nm(over)}) on ${d.wins}/${d.n} seeds by ${(d.mean * 100).toFixed(1)} pts on average, so it moves to #1. The rules still describe your real models, so try both.`;
    }
    const top = r.recs.slice(0, 3).map((x) => ({ id: x.id, e: evCache[evKey(t, r.align, x.id)] }));
    const ok = top.filter((x) => x.e && !x.e.blocked);
    if (ok.length < 2) return '';
    const best = ok.reduce((a, b) => (b.e.mean > a.e.mean ? b : a));
    const first = ok[0];
    if (best.id === first.id) return `<b>Measurements agree:</b> ${nm(first.id)} also scores best of these on the ${scName} toy.`;
    const gap = best.e.mean - first.e.mean;
    if (gap <= best.e.sd + first.e.sd) return `<b>Too close to call:</b> on the ${scName} toy, ${nm(best.id)} edges ${nm(first.id)} by ${(gap * 100).toFixed(1)} pts, within seed noise.`;
    return `<b>Measurements disagree:</b> on the ${scName} toy, ${nm(best.id)} beats the rule-based #1 (${nm(first.id)}) by ${(gap * 100).toFixed(1)} pts. Worth trying both.`;
  }
  async function measureEvidence(r) {
    const token = ++evToken;
    const toys = toysOf(r);
    const ids = r.recs.slice(0, 3).map((x) => x.id);
    if (toys.every((t) => ids.every((id) => evCache[evKey(t, r.align, id)]))) return;
    await frame();
    // train every toy's seeds at once (parallel workers when available)
    const need = toys.filter((t) => !evBundles[toyKey(t)]);
    const trained = await Promise.all(need.map((t) => Promise.all(EV_SEEDS.map((seed) => {
      const cfg = Object.assign(MM.toyConfig(t.scenario, seed), { init: t.init || 'shared' });
      // the lab's experts or any bundle already trained for this exact config and seed (lab seed checks included)
      return trainCached(cfg, { cancelled: () => token !== evToken });
    }))));
    if (token !== evToken || trained.some((bs) => bs.some((b) => !b))) return;
    need.forEach((t, i) => (evBundles[toyKey(t)] = trained[i]));
    for (const t of toys) {
      for (const id of ids) {
        if (evCache[evKey(t, r.align, id)]) continue;
        await frame();
        if (token !== evToken) return;
        const runs = MM.seedRuns(evBundles[toyKey(t)], [id], r.align, (mid, K) => MM.defaultParams(MM.METHOD_BY_ID[mid], K));
        evCache[evKey(t, r.align, id)] = MM.summarizeSeeds(runs, [id])[0];
        evSeeds[evKey(t, r.align, id)] = runs.map((run) => run[id]);
        const el = $(`#advRecs [data-ev="${id}"]`);
        if (el) el.innerHTML = evidenceHTML(id, r);
      }
    }
    // everything measured: re-render so the cards can be re-ranked by the paired evidence
    if (token === evToken && advDlg.open) renderAdvisor();
  }
  /** Paired blend of rules and measurement on every toy, once all top picks are measured; null until then. */
  function advisorBlend(r) {
    const ids = r.recs.slice(0, 3).map((x) => x.id);
    const toys = toysOf(r);
    if (ids.length < 2 || toys.some((t) => ids.some((id) => !evSeeds[evKey(t, r.align, id)]))) return null;
    const blends = toys.map((t) => MM.blendRanking(ids, Object.fromEntries(ids.map((id) => [id, evSeeds[evKey(t, r.align, id)]]))));
    return MM.combineToyBlends(ids, blends);
  }

  function renderAdvisor() {
    const a = Object.assign({}, MM.ADVISOR_DEFAULTS, S.advisor);
    $('#advQs').innerHTML = MM.ADVISOR_QUESTIONS.map((q) => `<div class="adv-q"><h3>${q.q}</h3>${q.hint ? `<small>${q.hint}</small>` : ''}
      <div class="seg ${q.options.length > 2 ? 'col' : ''}" data-adv="${q.key}">${q.options.map(([v, l]) => `<button data-v="${v}" class="${a[q.key] === v ? 'on' : ''}">${l}</button>`).join('')}</div></div>`).join('');
    const r = (advResult = MM.adviseRecipe(a));
    const sc = MM.SCENARIOS.find((x) => x.id === r.scenario);
    const uniq = (xs) => Array.from(new Set(xs));
    const cost = (m) => (m.inferenceK === true ? 'K× compute' : m.inferenceK === 'memory' ? 'K× memory' : '1× model');
    const blend = advisorBlend(r);
    const top3 = r.recs.slice(0, 3);
    const ordered = blend ? blend.order.map((id) => top3.find((x) => x.id === id)) : top3;
    const cards = ordered.map((x, i) => {
      const m = MM.METHOD_BY_ID[x.id], f = MM.FAMILIES[m.family];
      const moved = blend && blend.promoted ? (x.id === blend.promoted.id ? 'up' : x.id === blend.promoted.over ? 'down' : '') : '';
      const pros = uniq([...x.reasons.filter((t) => !/^Shared base/.test(t)), m.pros[0]]).slice(0, 3);
      const cons = uniq([...x.caveats, m.cons[0]]).slice(0, 2);
      return `<div class="rec ${i === 0 ? 'first' : ''}" style="--fc:${f.color}">
        <div class="rec-top"><span class="rec-rank">#${i + 1}</span><b>${m.name}</b><span class="fam-tag" style="--fc:${f.color}">${f.name}</span>${moved === 'up' ? '<span class="badge moved up" title="Promoted: beat the rules’ #1 on the toy under the paired test">↑ measured</span>' : moved === 'down' ? '<span class="badge moved" title="The rules ranked this first">rules’ #1</span>' : ''}<span class="badge">${cost(m)}</span></div>
        <ul>${pros.map((t) => `<li class="pro">${t}</li>`).join('')}${cons.map((t) => `<li class="con">${t}</li>`).join('')}${!pros.length && !cons.length ? `<li class="pro">${m.tagline}</li>` : ''}</ul>
        <div class="evidence" data-ev="${x.id}">${evidenceHTML(x.id, r)}</div>
        <div class="rec-actions">
          <button class="btn sm ${i === 0 ? 'primary' : ''}" data-adv-try="${x.id}">Try it in the Lab · ${sc.name}${r.align ? ' + alignment' : ''}</button>
          <button class="btn ghost sm" data-adv-export="${x.id}">${m.mergekit ? 'Export mergekit config' : 'Open-source pointers'}</button>
        </div></div>`;
    }).join('');
    const alts = r.recs.slice(3).map((x) => `<div class="alt"><b>${MM.METHOD_BY_ID[x.id].name}</b><span>${uniq(x.caveats)[0] || uniq(x.reasons)[0] || MM.METHOD_BY_ID[x.id].tagline}</span></div>`).join('');
    const outs = r.excluded.map((x) => `<div class="alt"><b>${MM.METHOD_BY_ID[x.id].name}</b><span>${x.why}</span></div>`).join('');
    measureEvidence(r);
    $('#advRecs').innerHTML = `${r.notes.map((n) => `<div class="adv-note">${n}</div>`).join('')}
      ${cards ? `<div class="ev-verdict" id="evVerdict">${evidenceVerdict(r, blend)}</div>` : ''}
      ${cards || '<p class="muted">Nothing fits every constraint. Relax one of them (data, training or serving) to see options.</p>'}
      ${alts ? `<details class="adv-more"><summary>Other options that fit (${r.recs.length - 3})</summary>${alts}</details>` : ''}
      ${outs ? `<details class="adv-more"><summary>Ruled out (${r.excluded.length})</summary>${outs}</details>` : ''}`;
  }
  $('#advQs').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-adv] button');
    if (!b) return;
    S.advisor = Object.assign({}, S.advisor, { [b.closest('[data-adv]').dataset.adv]: b.dataset.v });
    persist();
    renderAdvisor();
  });
  $('#advRecs').addEventListener('click', (ev) => {
    const t = ev.target.closest('[data-adv-try]'), x = ev.target.closest('[data-adv-export]');
    if (!t && !x) return;
    const id = (t || x).dataset.advTry || (t || x).dataset.advExport;
    closeAdvisor();
    if (x) {
      S.method = id; persist(); renderPicker(); refresh();
      setTab('export');
      return;
    }
    const sc = MM.SCENARIOS.find((s) => s.id === advResult.scenario);
    const switching = !sameScenario(sc);
    const wantInit = advResult.init || sc.init;
    const needTrain = switching || S.init !== wantInit;
    if (switching) S.experts = sc.experts.map((e) => ({ ...e }));
    S.init = wantInit;
    S.align = advResult.align;
    S.method = id;
    arenaStale = robustStale = true;
    setTab('lab');
    renderPicker();
    if (needTrain) scheduleRetrain();
    else { persist(); renderIngredients(); refresh(); }
    toast(`${MM.METHOD_BY_ID[id].name} on the ${sc.name} scenario${wantInit !== sc.init ? ', independent inits' : ''}${S.align ? ', neurons aligned' : ''}`);
  });
  $('#openAdvisor').addEventListener('click', openAdvisor);
  $('#closeAdvisor').addEventListener('click', closeAdvisor);
  advDlg.addEventListener('click', (ev) => { if (ev.target === advDlg) closeAdvisor(); });
  document.addEventListener('click', (ev) => { if (ev.target.closest('[data-open-advisor]')) openAdvisor(); });

  $('#picker').addEventListener('keydown', (ev) => {
    const chips = $$('#picker button');
    const i = chips.indexOf(document.activeElement);
    if (i < 0) return;
    const j = { ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: chips.length - 1 }[ev.key];
    if (j === undefined) return;
    ev.preventDefault();
    // roving tab stop: the picker is one Tab stop, arrows move inside it
    const next = chips[(j + chips.length) % chips.length];
    chips.forEach((c) => { c.tabIndex = c === next ? 0 : -1; });
    next.focus();
  });
  // leaving the picker resets its tab stop to the chosen method
  $('#picker').addEventListener('focusout', (ev) => {
    if ($('#picker').contains(ev.relatedTarget)) return;
    $$('#picker button.chip').forEach((c) => { c.tabIndex = c.dataset.method === S.method ? 0 : -1; });
  });

  // Skip links move focus without touching location.hash (the hash holds the shareable recipe)
  document.querySelector('.skip-links').addEventListener('click', (ev) => {
    const a = ev.target.closest('[data-skip]');
    if (!a) return;
    ev.preventDefault();
    if (S.tab !== 'lab') setTab('lab');
    const box = document.getElementById(a.dataset.skip);
    const first = a.dataset.skip === 'picker' ? box.querySelector('button.chip.active') || box.querySelector('button')
      : a.dataset.skip === 'controls' ? box.querySelector('input, select, button:not([disabled])') : null;
    const target = first || box;
    target.scrollIntoView({ block: 'center' });
    target.focus({ preventScroll: true });
  });

  // Keep aria-pressed in step with the visual "on"/"active" state of every toggle-like button.
  function syncPressed(root) {
    (root || document).querySelectorAll('.seg button, button.tg, #picker button.chip').forEach((b) => {
      b.setAttribute('aria-pressed', b.classList.contains('on') || b.classList.contains('active') ? 'true' : 'false');
    });
  }
  new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.type === 'attributes') { const b = m.target; if (b.matches && b.matches('.seg button, button.tg, #picker button.chip')) b.setAttribute('aria-pressed', b.classList.contains('on') || b.classList.contains('active') ? 'true' : 'false'); }
      else m.addedNodes.forEach((n) => { if (n.nodeType === 1) syncPressed(n.parentNode || n); });
    }
  }).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ['class'] });
  syncPressed();

  // ---------------- tabs ----------------
  function setTab(tab) {
    S.tab = tab; persist();
    $$('#tabs button').forEach((b) => {
      const on = b.dataset.tab === tab;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1; // roving tab stop: arrow keys move between tabs
    });
    $$('.tab-page').forEach((p) => p.classList.toggle('active', p.id === `tab-${tab}`));
    if (tab === 'arena') markRobustStale();
    if (tab === 'arena' && (arenaStale || !$('#board').innerHTML) && T && T.experts) runArena();
    if (tab === 'atlas') renderAtlas();
    if (tab === 'export') renderExport();
    if (tab === 'lab' && current) refresh(true);
    window.scrollTo({ top: 0 });
  }
  $('#tabs').addEventListener('keydown', (ev) => {
    const btns = $$('#tabs button');
    const i = btns.indexOf(document.activeElement);
    if (i < 0) return;
    const j = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: btns.length - 1 }[ev.key];
    if (j === undefined) return;
    ev.preventDefault();
    const b = btns[(j + btns.length) % btns.length];
    b.focus();
    setTab(b.dataset.tab);
  });
  $('#tabs').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-tab]');
    if (b) setTab(b.dataset.tab);
  });

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (S.tab === 'lab' && T && T.experts) refresh(true); }, 150);
  });

  // ---------------- boot ----------------
  let sharedMsg = null;
  const code0 = MM.recipeFromHash(location.hash);
  if (code0) {
    const d = MM.decodeRecipe(code0);
    if (d.ok) { applyRecipe(d.recipe); S.tab = 'lab'; persist(); sharedMsg = 'Opened a shared recipe'; }
    else sharedMsg = `Couldn’t open the shared link: ${d.error}`;
  }
  renderIngredients();
  renderPicker();
  renderControls();
  recordHistory(); // the starting point you can undo back to
  updateUndoButtons();
  renderStageHead(MM.METHOD_BY_ID[S.method], null);
  setTab(S.tab === 'lab' || S.tab === 'atlas' || S.tab === 'export' || S.tab === 'arena' ? S.tab : 'lab');
  train();
  if (sharedMsg) toast(sharedMsg);

  window.ModelMixer = { state: S, get trained() { return T; }, compute, getCtx, recipeLink, openRecipe, trainAsync, scoreAsync, get usesWorker() { return workerOK; } };
})();
