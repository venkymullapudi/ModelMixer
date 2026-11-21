/* ModelMixer — application: state, training loop, lab / arena / atlas / export views. */
(function () {
  const MM = window.MM;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const LETTERS = 'ABC';
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const frame = () => new Promise((r) => setTimeout(r, 0));

  // ---------------- state ----------------
  const S = {
    scenario: 'two',
    experts: MM.SCENARIOS[0].experts.map((e) => ({ ...e })),
    init: 'shared', hidden: 32, ftSteps: 250, seed: 7, align: false,
    method: 'ties', params: {},
    overlay: { points: true, regions: true, router: true },
    tab: 'lab',
    advisor: {}, // answers to the recipe advisor
    pin: null, // { method, params, align } recipe pinned for side-by-side comparison
    exp: {
      base: 'mistralai/Mistral-7B-v0.1',
      models: ['teknium/OpenHermes-2.5-Mistral-7B', 'meta-math/MetaMath-Mistral-7B', 'HuggingFaceH4/zephyr-7b-beta'],
      dtype: 'bfloat16',
      prompts: ['chat, conversation, assistant', 'math, arithmetic, solve, equation', 'helpful, explain, summarize'],
    },
  };
  MM.METHODS.forEach((m) => (S.params[m.id] = MM.defaultParams(m)));

  try {
    const saved = JSON.parse(localStorage.getItem('modelmixer:v1') || 'null');
    if (saved) {
      for (const k of ['scenario', 'experts', 'init', 'hidden', 'ftSteps', 'seed', 'align', 'method', 'overlay', 'tab', 'pin', 'advisor']) if (saved[k] !== undefined) S[k] = saved[k];
      if (saved.params) for (const id in saved.params) if (S.params[id]) Object.assign(S.params[id], saved.params[id]);
      if (saved.exp) Object.assign(S.exp, saved.exp);
      if (!MM.METHOD_BY_ID[S.method]) S.method = 'ties';
      if (S.pin && !MM.METHOD_BY_ID[S.pin.method]) S.pin = null;
    }
  } catch (e) { /* storage unavailable */ }
  function persist() {
    try {
      const { scenario, experts, init, hidden, ftSteps, seed, align, method, params, overlay, tab, exp, pin, advisor } = S;
      localStorage.setItem('modelmixer:v1', JSON.stringify({ scenario, experts, init, hidden, ftSteps, seed, align, method, params, overlay, tab, exp, pin, advisor }));
    } catch (e) { /* ignore */ }
    syncHash();
  }

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
    const it = MM.trainBundle({ experts: S.experts, init: S.init, hidden: S.hidden, ftSteps: S.ftSteps, seed: S.seed });
    let total = 1, done = 0, sinceTick = 0;
    try {
      for (;;) {
        let r;
        for (let i = 0; i < 12; i++) {
          r = it.next();
          if (r.done) break;
          const ev = r.value;
          if (ev.type === 'start') { total = ev.total; T = Object.assign(T || {}, { live: ev.live }); }
          else if (ev.type === 'step') { done++; if (++sinceTick % 48 === 0) drawThumb(ev.which, ev.P, T.live.tasks); }
          else if (ev.type === 'done') drawThumb(ev.which, ev.P, T.live.tasks);
        }
        if (r.done) { T = r.value; break; }
        if (token !== trainToken) throw new Error('cancelled');
        setProgress(done / total, T.live.losses);
        await frame();
      }
      training = false;
      setProgress(1, T.losses);
      status(`${T.K} experts ready · ${T.layout.total.toLocaleString()} params each`);
      renderIngredients();
      renderPicker();
      refresh();
      if (S.tab === 'arena') runArena();
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
    MM.drawBoundary(cv, (X, n) => MM.predictProb(layout, P, X, n), { grid: 54, thumb: true, tasks, showPoints: false });
  }

  // ---------------- INGREDIENTS ----------------
  function renderIngredients() {
    const K = S.experts.length;
    const sc = MM.SCENARIOS.find((s) => s.id === S.scenario);
    const taskOpts = (v) => Object.entries(MM.TASKS).map(([k, t]) => `<option value="${k}" ${k === v ? 'selected' : ''}>${t.name}</option>`).join('');
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
                <select data-ex="${i}" data-field="region" aria-label="Region">${regOpts(e.region)}</select>
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
          <input type="range" min="50" max="800" step="25" value="${S.ftSteps}" id="ftSteps" />
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
      const used = new Set(S.experts.map((e) => e.region));
      const region = ['top', 'right', 'left', 'center'].find((r) => !used.has(r)) || 'top';
      S.experts.push({ task: 'xor', region });
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
          return `<button class="chip ${m.id === S.method ? 'active' : ''} ${locked ? 'locked' : ''}" data-method="${m.id}" title="${esc(m.tagline)}${locked ? ' — needs a shared base' : ''}">${locked ? '<span class="lock">🔒</span>' : ''}${m.short}</button>`;
        }).join('')}</div>
      </div>`;
    });
    $('#picker').innerHTML = `<div class="families">${groups.join('')}</div>`;
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
    const rows = m.params.map((d) => {
      if (d.type === 'range') {
        return `<div class="ctl"><label>${d.label}<b data-out="${d.key}">${fmt(p[d.key], d.step)}</b></label>
          <input type="range" data-param="${d.key}" min="${d.min}" max="${d.max}" step="${d.step}" value="${p[d.key]}" /></div>`;
      }
      if (d.type === 'select') {
        return `<div class="ctl"><label>${d.label}</label><div class="seg wide" data-param-seg="${d.key}">
          ${d.options.map(([v, l]) => `<button data-v="${v}" class="${String(p[d.key]) === v ? 'on' : ''}">${l}</button>`).join('')}</div></div>`;
      }
      if (d.type === 'weights') {
        return `<div class="ctl"><label>${d.label}</label>${Array.from({ length: K }, (_, i) => `
          <div class="wrow" style="--c:${MM.MODEL_COLORS[i]}"><i class="dot"></i><span>${LETTERS[i]}</span>
            <input type="range" data-weight="${i}" min="0" max="1" step="0.05" value="${p.weights[i]}" /><b data-wout="${i}">${(+p.weights[i]).toFixed(2)}</b></div>`).join('')}</div>`;
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
    $('#controls').innerHTML = `<div class="block-title">Recipe</div>${rows.join('') || '<p class="muted small">No knobs — this method decides everything from the weights themselves.</p>'}
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
    else return;
    persist(); arenaStale = robustStale = true;
    queueRefresh();
  });
  $('#controls').addEventListener('click', (ev) => {
    const b = ev.target.closest('button');
    if (!b) return;
    const p = S.params[S.method];
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
    MM.drawBoundary(cv, pin.predict, {
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
    if (!light) renderControls();
    const ctx = getCtx();
    const m = MM.METHOD_BY_ID[S.method];
    renderStageHead(m, null);
    if (!ctx) return;
    const heavy = m.id === 'distill' || (m.id === 'moe' && !ctx.cache.router) || (m.id === 'fisher' && !ctx.cache.fisher) || (m.id === 'regmean' && !ctx.cache.grams);
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
      renderInsight(m, r, ctx);
      const ms = performance.now() - t0;
      if (!training) status(`${m.name} · mixed in ${ms.toFixed(0)} ms`);
    };
    if (heavy) {
      $('#busyText').textContent = m.id === 'distill' ? 'Distilling student…' : m.id === 'moe' ? 'Training router…' : 'Measuring importance…';
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
    MM.drawBoundary(cv, r.predict, {
      tasks: ctx.tasks, showPoints: S.overlay.points, showRegions: S.overlay.regions,
      gates: r.router && S.overlay.router ? r.router.gates : null, K: ctx.K,
    });
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
    const tg = [['points', 'Points'], ['regions', 'Regions']];
    if (m.id === 'moe') tg.push(['router', 'Router']);
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
      </div>${noise}</div>`;
  }

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
        <code class="formula">${m.formula}</code>
      </div>
      <div class="two">
        <div class="section"><div class="sec-title good">Strengths</div><ul class="pros">${m.pros.map((s) => `<li>${s}</li>`).join('')}</ul></div>
        <div class="section"><div class="sec-title bad">Missing pieces</div><ul class="cons">${m.cons.map((s) => `<li>${s}</li>`).join('')}</ul></div>
      </div>
      ${info ? `<div class="section"><div class="sec-title">This mix, measured</div>${info}</div>` : ''}
      ${diag ? `<div class="section"><div class="sec-title">${r.kind === 'weights' ? 'Weight provenance' : 'Where the mixing happens'}</div>${diag}</div>` : ''}
      <div class="section"><div class="sec-title">${hasBase ? 'Task-vector geometry' : 'Raw weight similarity <span class="hint">(no base)</span>'}</div>${geoHtml}
        <p class="tiny muted">${hasBase ? 'Low cosine + high sign conflict = skills that fight when added together. That is what TIES and DARE are built for.' : 'Independently trained experts look unrelated in weight space. Turn on neuron alignment and watch the similarity jump — same functions, re-ordered neurons.'}</p></div>
      <div class="section oss">
        <div class="sec-title">Run it for real</div>
        <div class="oss-row">${m.mergekit ? `<span class="pill mk">mergekit: <code>${m.mergekit}</code></span>` : '<span class="pill">not in mergekit</span>'}
        ${m.oss.map((o) => `<a class="pill link" href="${o.url}" target="_blank" rel="noopener">${o.name} ↗</a>`).join('')}</div>
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
      renderPathChart(ctx);
      renderLambdaChart(ctx);
      arenaStale = false;
      status(`Arena: ${results.filter((x) => !x.r.blocked).length} methods compared`);
    }, 40);
  }
  $('#runArena').addEventListener('click', () => { for (const k in (T && T.ctx) || {}) Object.keys(T.ctx[k].cache).filter((c) => c.startsWith('distill')).forEach((c) => delete T.ctx[k].cache[c]); runArena(); });

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
      MM.drawBoundary($(`#tg-${x.m.id}`), x.r.predict, { grid: 64, thumb: true, tasks: ctx.tasks, showPoints: true });
    });
  }
  $('#thumbGrid').addEventListener('click', (ev) => {
    const b = ev.target.closest('[data-open]');
    if (b) { setTab('lab'); selectMethod(b.dataset.open); }
  });

  // ---------------- ROBUSTNESS (multi-seed) ----------------
  const ROBUST_SEEDS = 5;
  let robustToken = 0;
  async function runRobust() {
    if (!T || !T.experts || training) { toast('Wait for training to finish'); return; }
    const token = ++robustToken;
    const btn = $('#runRobust');
    btn.disabled = true;
    const ids = MM.METHODS.map((m) => m.id);
    const seeds = Array.from({ length: ROBUST_SEEDS }, (_, i) => S.seed + i * 101);
    const cfg = { experts: S.experts, init: S.init, hidden: S.hidden, ftSteps: S.ftSteps };
    const steps = ROBUST_SEEDS * (ids.length + 1);
    let doneSteps = 0;
    const prog = (label) => {
      $('#robust').innerHTML = `<div class="loading"><span class="spinner"></span> ${label}</div><div class="rprog"><div style="width:${(100 * doneSteps / steps).toFixed(0)}%"></div></div>`;
    };
    const runs = [];
    try {
      for (let s = 0; s < seeds.length; s++) {
        let bundle = T;
        if (s > 0) {
          prog(`Seed ${s + 1}/${ROBUST_SEEDS}: training experts…`);
          await frame();
          const it = MM.trainBundle({ ...cfg, seed: seeds[s] });
          let r, k = 0;
          while (!(r = it.next()).done) {
            if (++k % 150 === 0) { await frame(); if (token !== robustToken) return; }
          }
          bundle = r.value;
        }
        doneSteps++;
        const ctx = MM.bundleCtx(bundle, S.align);
        const run = {};
        for (const id of ids) {
          prog(`Seed ${s + 1}/${ROBUST_SEEDS}: mixing with ${MM.METHOD_BY_ID[id].short}…`);
          await frame();
          if (token !== robustToken) return;
          const r = MM.scoreMethod(bundle, ctx, id, S.params[id]);
          run[id] = r.blocked ? { blocked: r.blocked } : { avg: r.avg, retention: r.retention, accs: r.accs };
          doneSteps++;
        }
        runs.push(run);
      }
      robustStale = false;
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
        <tbody>${rows}${bl}</tbody></table></div>`;
  }
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
  function renderAtlas() {
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
    $('#guide').innerHTML = guide.map((g) => `<div class="panel gcard"><h4>${g.q}</h4>${g.a.map(([k, t, ms]) => `
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
    switch (id) {
      case 'linear':
        return { yaml: `merge_method: linear\nmodels:\n${list((i) => `      weight: ${w(i)}`)}\nparameters:\n  normalize: true\n${dt}\n` };
      case 'slerp':
        return { yaml: `merge_method: slerp\nbase_model: ${models[0]}\nmodels:\n  - model: ${models[0]}\n  - model: ${models[1]}\nparameters:\n  t: ${(+p.t).toFixed(2)}\n${dt}\n`, note: K > 2 ? 'SLERP takes exactly two models — the third is left out.' : '' };
      case 'model_stock':
        return { yaml: `merge_method: model_stock\nbase_model: ${base}\nmodels:\n${list()}\n${dt}\n` };
      case 'task_arithmetic':
        return { yaml: `merge_method: task_arithmetic\nbase_model: ${base}\nmodels:\n${list((i) => `      weight: ${(+p.lambda * +w(i)).toFixed(2)}   # λ × w`)}\n${dt}\n` };
      case 'ties':
        return { yaml: `merge_method: ties\nbase_model: ${base}\nmodels:\n${list(() => `      density: ${(+p.density).toFixed(2)}\n      weight: 1.0`)}\nparameters:\n  normalize: true   # disjoint mean, as in the lab (λ = 1)\n${dt}\n`, note: +p.lambda !== 1 ? `Lab λ = ${(+p.lambda).toFixed(2)}. With normalize: true, weights cancel out; to scale, set normalize: false and weight ≈ λ / ${K}.` : '' };
      case 'dare':
        return { yaml: `merge_method: ${p.mode === 'ties' ? 'dare_ties' : 'dare_linear'}\nbase_model: ${base}\nmodels:\n${list((i) => `      density: ${(1 - +p.drop).toFixed(2)}   # keep = 1 − drop\n      weight: ${(+p.lambda * +w(i)).toFixed(2)}`)}\nparameters:\n  int8_mask: true\n${dt}\n` };
      case 'breadcrumbs':
        return { yaml: `merge_method: breadcrumbs\nbase_model: ${base}\nmodels:\n${list(() => `      density: ${(+p.density).toFixed(2)}\n      gamma: ${(+p.gamma).toFixed(3)}\n      weight: ${(+p.lambda).toFixed(2)}`)}\n${dt}\n` };
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
    distill: { title: 'Distillation is training, not merging', body: 'Train a student on the teacher mixture’s outputs. DistillKit (arcee-ai) and TRL’s GKD trainer both support logit-based distillation from Hugging Face teachers.' },
  };

  function renderExport() {
    const id = S.method, m = MM.METHOD_BY_ID[id], K = S.experts.length;
    const opts = MM.METHODS.map((x) => `<option value="${x.id}" ${x.id === id ? 'selected' : ''}>${x.name}${x.mergekit ? '' : '  (no mergekit)'}</option>`).join('');
    $('#exportForm').innerHTML = `
      <div class="panel-title"><span>Recipe</span><span class="hint">mirrors the lab</span></div>
      <div class="field"><label>Method</label><select id="exMethod">${opts}</select></div>
      <div class="field"><label>Base model ${m.req.base ? '' : '<span class="hint">(unused by this method)</span>'}</label><input id="exBase" value="${esc(S.exp.base)}" spellcheck="false" /></div>
      ${Array.from({ length: K }, (_, i) => `<div class="field" style="--c:${MM.MODEL_COLORS[i]}"><label><i class="dot"></i>Expert ${LETTERS[i]}&nbsp;<span class="hint">(${MM.TASKS[S.experts[i].task].name} in the lab)</span></label><input data-exm="${i}" value="${esc(S.exp.models[i])}" spellcheck="false" />
        ${id === 'moe' ? `<input class="sub" data-exp="${i}" value="${esc(S.exp.prompts[i])}" placeholder="routing prompts, comma separated" />` : ''}</div>`).join('')}
      <div class="field"><label>dtype</label><div class="seg wide" id="exDtype">${['bfloat16', 'float16', 'float32'].map((d) => `<button data-v="${d}" class="${S.exp.dtype === d ? 'on' : ''}">${d}</button>`).join('')}</div></div>
      <div class="note">Parameters (weights, density, λ, t…) come from the lab’s current recipe for <b>${m.name}</b>. Tune it there, then copy the config.</div>`;
    const y = yamlFor(id);
    const notes = $('#exportNotes');
    if (y) {
      $('#yaml').textContent = y.yaml;
      $('#exportFile').textContent = y.file || 'config.yml';
      $('#cmd').textContent = `pip install mergekit\n${y.cmd || 'mergekit-yaml config.yml ./merged --cuda'}`;
      notes.innerHTML = y.note ? `<div class="note">${y.note}</div>` : '';
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

  // ---------------- RECIPE ADVISOR ----------------
  const advDlg = $('#advisor');
  let advResult = null;
  function openAdvisor() {
    renderAdvisor();
    if (typeof advDlg.showModal === 'function') { if (!advDlg.open) advDlg.showModal(); } else advDlg.setAttribute('open', '');
  }
  function closeAdvisor() { if (advDlg.open) { if (typeof advDlg.close === 'function') advDlg.close(); else advDlg.removeAttribute('open'); } }
  function renderAdvisor() {
    const a = Object.assign({}, MM.ADVISOR_DEFAULTS, S.advisor);
    $('#advQs').innerHTML = MM.ADVISOR_QUESTIONS.map((q) => `<div class="adv-q"><h4>${q.q}</h4>${q.hint ? `<small>${q.hint}</small>` : ''}
      <div class="seg ${q.options.length > 2 ? 'col' : ''}" data-adv="${q.key}">${q.options.map(([v, l]) => `<button data-v="${v}" class="${a[q.key] === v ? 'on' : ''}">${l}</button>`).join('')}</div></div>`).join('');
    const r = (advResult = MM.adviseRecipe(a));
    const sc = MM.SCENARIOS.find((x) => x.id === r.scenario);
    const uniq = (xs) => Array.from(new Set(xs));
    const cost = (m) => (m.inferenceK === true ? 'K× compute' : m.inferenceK === 'memory' ? 'K× memory' : '1× model');
    const cards = r.recs.slice(0, 3).map((x, i) => {
      const m = MM.METHOD_BY_ID[x.id], f = MM.FAMILIES[m.family];
      const pros = uniq([...x.reasons.filter((t) => !/^Shared base/.test(t)), m.pros[0]]).slice(0, 3);
      const cons = uniq([...x.caveats, m.cons[0]]).slice(0, 2);
      return `<div class="rec ${i === 0 ? 'first' : ''}" style="--fc:${f.color}">
        <div class="rec-top"><span class="rec-rank">#${i + 1}</span><b>${m.name}</b><span class="fam-tag" style="--fc:${f.color}">${f.name}</span><span class="badge">${cost(m)}</span></div>
        <ul>${pros.map((t) => `<li class="pro">${t}</li>`).join('')}${cons.map((t) => `<li class="con">${t}</li>`).join('')}${!pros.length && !cons.length ? `<li class="pro">${m.tagline}</li>` : ''}</ul>
        <div class="rec-actions">
          <button class="btn sm ${i === 0 ? 'primary' : ''}" data-adv-try="${x.id}">Try it in the Lab · ${sc.name}${r.align ? ' + alignment' : ''}</button>
          <button class="btn ghost sm" data-adv-export="${x.id}">${m.mergekit ? 'Export mergekit config' : 'Open-source pointers'}</button>
        </div></div>`;
    }).join('');
    const alts = r.recs.slice(3).map((x) => `<div class="alt"><b>${MM.METHOD_BY_ID[x.id].name}</b><span>${uniq(x.caveats)[0] || uniq(x.reasons)[0] || MM.METHOD_BY_ID[x.id].tagline}</span></div>`).join('');
    const outs = r.excluded.map((x) => `<div class="alt"><b>${MM.METHOD_BY_ID[x.id].name}</b><span>${x.why}</span></div>`).join('');
    $('#advRecs').innerHTML = `${r.notes.map((n) => `<div class="adv-note">${n}</div>`).join('')}
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
    if (switching) { S.experts = sc.experts.map((e) => ({ ...e })); S.init = sc.init; }
    S.align = advResult.align;
    S.method = id;
    arenaStale = robustStale = true;
    setTab('lab');
    renderPicker();
    if (switching) scheduleRetrain();
    else { persist(); renderIngredients(); refresh(); }
    toast(`${MM.METHOD_BY_ID[id].name} on the ${sc.name} scenario${S.align ? ', neurons aligned' : ''}`);
  });
  $('#openAdvisor').addEventListener('click', openAdvisor);
  $('#closeAdvisor').addEventListener('click', closeAdvisor);
  advDlg.addEventListener('click', (ev) => { if (ev.target === advDlg) closeAdvisor(); });
  document.addEventListener('click', (ev) => { if (ev.target.closest('[data-open-advisor]')) openAdvisor(); });

  // ---------------- tabs ----------------
  function setTab(tab) {
    S.tab = tab; persist();
    $$('#tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
    $$('.tab-page').forEach((p) => p.classList.toggle('active', p.id === `tab-${tab}`));
    if (tab === 'arena') markRobustStale();
    if (tab === 'arena' && (arenaStale || !$('#board').innerHTML) && T && T.experts) runArena();
    if (tab === 'atlas') renderAtlas();
    if (tab === 'export') renderExport();
    if (tab === 'lab' && current) refresh(true);
    window.scrollTo({ top: 0 });
  }
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
  renderStageHead(MM.METHOD_BY_ID[S.method], null);
  setTab(S.tab === 'lab' || S.tab === 'atlas' || S.tab === 'export' || S.tab === 'arena' ? S.tab : 'lab');
  train();
  if (sharedMsg) toast(sharedMsg);

  window.ModelMixer = { state: S, get trained() { return T; }, compute, getCtx, recipeLink, openRecipe };
})();
