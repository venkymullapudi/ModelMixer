/* ModelMixer — import a mergekit config (the reverse of Export).
 * A small YAML-subset parser (maps, lists, inline [lists] / {maps}, scalars, comments) plus a mapper from
 * mergekit merge methods to lab recipes. Anything that can't be represented exactly is reported as a warning. */
(function () {
  const root = typeof window !== 'undefined' ? window : globalThis;
  const MM = (root.MM = root.MM || {});

  // ---------------- YAML subset ----------------
  function stripComment(line) {
    let q = null;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (q) { if (c === q) q = null; continue; }
      if (c === '"' || c === "'") q = c;
      else if (c === '#' && (i === 0 || /\s/.test(line[i - 1]))) return line.slice(0, i);
    }
    return line;
  }
  function splitTop(s) { // split "a, [b, c], d" on top-level commas
    const out = []; let depth = 0, q = null, cur = '';
    for (const c of s) {
      if (q) { cur += c; if (c === q) q = null; continue; }
      if (c === '"' || c === "'") { q = c; cur += c; continue; }
      if (c === '[' || c === '{') depth++;
      if (c === ']' || c === '}') depth--;
      if (c === ',' && depth === 0) { out.push(cur); cur = ''; } else cur += c;
    }
    if (cur.trim()) out.push(cur);
    return out.map((x) => x.trim());
  }
  function scalar(t) {
    t = t.trim();
    if (t === '') return null;
    if (t.startsWith('[') && t.endsWith(']')) return splitTop(t.slice(1, -1)).map(scalar);
    if (t.startsWith('{') && t.endsWith('}')) {
      const o = {};
      for (const kv of splitTop(t.slice(1, -1))) { const k = kv.indexOf(':'); if (k > 0) o[kv.slice(0, k).trim()] = scalar(kv.slice(k + 1)); }
      return o;
    }
    if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) return t.slice(1, -1);
    if (/^(true|false)$/i.test(t)) return /^true$/i.test(t);
    if (/^(null|~)$/i.test(t)) return null;
    if (/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t)) return Number(t);
    return t;
  }
  const keyOf = (text) => { const m = /^("[^"]*"|'[^']*'|[^:\s][^:]*?)\s*:(\s+|$)(.*)$/.exec(text); return m ? [m[1].replace(/^["']|["']$/g, ''), m[3]] : null; };

  function parseYAML(src) {
    const lines = String(src).replace(/\t/g, '  ').split(/\r?\n/)
      .map((raw) => { const t = stripComment(raw); return { indent: t.search(/\S/), text: t.trim() }; })
      .filter((l) => l.text !== '' && l.text !== '---');
    let i = 0;
    function block(indent) {
      if (i >= lines.length) return null;
      return lines[i].text.startsWith('-') && /^-(\s|$)/.test(lines[i].text) ? list(indent) : map(indent);
    }
    function list(indent) {
      const out = [];
      while (i < lines.length && lines[i].indent === indent && /^-(\s|$)/.test(lines[i].text)) {
        const item = lines[i].text.replace(/^-\s*/, '');
        if (item === '') { i++; out.push(i < lines.length && lines[i].indent > indent ? block(lines[i].indent) : null); }
        else if (keyOf(item) && !item.startsWith('[') && !item.startsWith('{')) {
          // "- key: value" starts a map whose keys sit two columns further in
          lines[i] = { indent: indent + 2, text: item };
          out.push(map(indent + 2));
        } else { out.push(scalar(item)); i++; }
      }
      return out;
    }
    function map(indent) {
      const out = {};
      while (i < lines.length && lines[i].indent === indent && !/^-(\s|$)/.test(lines[i].text)) {
        const kv = keyOf(lines[i].text);
        if (!kv) throw new Error(`Can’t read line ${i + 1}: “${lines[i].text}”`);
        const [k, rest] = kv;
        i++;
        if (rest.trim() !== '') out[k] = scalar(rest);
        else if (i < lines.length && (lines[i].indent > indent || (lines[i].indent === indent && /^-(\s|$)/.test(lines[i].text)))) out[k] = block(lines[i].indent);
        else out[k] = null;
      }
      return out;
    }
    const v = block(lines.length ? lines[0].indent : 0);
    if (i < lines.length) throw new Error(`Unexpected indentation near “${lines[i].text}”`);
    return v;
  }

  // ---------------- mergekit → lab recipe ----------------
  const num = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : null);
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  /** Lab layers are early / middle / late; mergekit spreads a list across depth, so sample it at 0, ½ and 1. */
  function anchors(list) {
    const v = list.map(Number);
    if (v.length === 1) return [v[0], v[0], v[0]];
    const at = (f) => { const x = f * (v.length - 1), lo = Math.floor(x), hi = Math.min(v.length - 1, lo + 1); return v[lo] + (v[hi] - v[lo]) * (x - lo); };
    return [at(0), at(0.5), at(1)].map((x) => +x.toFixed(4));
  }
  /** A mergekit parameter value: number | number[] (gradient) | [{filter, value}] (per-module). */
  function readParam(v, name, warn) {
    if (v == null) return null;
    if (typeof v === 'number') return { scalar: v };
    if (Array.isArray(v) && v.every((x) => typeof x === 'number')) return v.length === 1 ? { scalar: v[0] } : { grad: anchors(v), raw: v };
    if (Array.isArray(v) && v.every((x) => x && typeof x === 'object')) {
      const dflt = v.find((x) => !x.filter) || v[0];
      warn(`${name}: per-module filters (${v.filter((x) => x.filter).map((x) => x.filter).join(', ') || 'none'}) aren’t modelled; using the ${dflt.filter ? `“${dflt.filter}”` : 'default'} value.`);
      return readParam(dflt.value, name, warn);
    }
    warn(`${name}: couldn’t read ${JSON.stringify(v)}; kept the lab default.`);
    return null;
  }
  const METHOD_MAP = {
    linear: 'linear', slerp: 'slerp', nuslerp: 'slerp', karcher: 'karcher', model_stock: 'model_stock',
    task_arithmetic: 'task_arithmetic', ties: 'ties', dare_ties: 'dare', dare_linear: 'dare',
    della: 'della', della_linear: 'della', breadcrumbs: 'breadcrumbs', breadcrumbs_ties: 'breadcrumbs',
    sce: 'sce', passthrough: 'stitch',
  };
  const TV = new Set(['task_arithmetic', 'ties', 'dare', 'della', 'breadcrumbs', 'sce', 'model_stock']);

  /** opts.order: preferred model order (e.g. the export form's), used where the config has no canonical order. */
  function importMergekit(src, opts) {
    opts = opts || {};
    let cfg;
    try { cfg = parseYAML(src); } catch (e) { return { ok: false, error: e.message }; }
    if (!cfg || typeof cfg !== 'object' || Array.isArray(cfg)) return { ok: false, error: 'This doesn’t look like a mergekit config (expected a YAML mapping).' };
    const warnings = [], notes = [];
    const warn = (w) => warnings.push(w);

    // mergekit-moe configs have experts + gate_mode instead of merge_method
    if (Array.isArray(cfg.experts)) {
      const names = cfg.experts.map((e) => e && e.source_model).filter(Boolean);
      if (names.length < 2) return { ok: false, error: 'A mergekit-moe config needs at least two experts.' };
      const params = MM.defaultParams(MM.METHOD_BY_ID.moe, Math.min(3, names.length));
      params.mode = num(cfg.experts_per_token) === 1 ? 'top1' : 'soft';
      notes.push(`mergekit-moe with ${names.length} experts, ${params.mode === 'top1' ? 'top-1' : 'soft'} routing (gate_mode: ${cfg.gate_mode || 'hidden'}).`);
      return finish('moe', params, names, cfg.base_model || names[0], warnings, notes);
    }
    const mk = String(cfg.merge_method || '').trim();
    const id = METHOD_MAP[mk];
    if (!id) return { ok: false, error: mk ? `“${mk}” isn’t one of the merge methods the lab models (${Object.keys(METHOD_MAP).join(', ')}).` : 'No merge_method found.' };
    const P = cfg.parameters || {};
    let models = (cfg.models || []).filter((m) => m && m.model);
    if (TV.has(id) && cfg.base_model) {
      const before = models.length;
      models = models.filter((m) => m.model !== cfg.base_model);
      if (models.length < before) notes.push('The base model listed under models was treated as the base, not as an expert.');
    }
    if (id === 'stitch') {
      const slices = (cfg.slices || []).map((s) => s && s.sources && s.sources[0]).filter(Boolean);
      if (!slices.length) return { ok: false, error: 'A passthrough config needs slices with sources.' };
      models = []; slices.forEach((s) => { if (!models.some((m) => m.model === s.model)) models.push({ model: s.model }); });
      // slices have no expert order; keep the caller's order when it knows every model (so Export → Import round-trips)
      const pref = opts.order || [];
      if (models.every((m) => pref.includes(m.model))) models.sort((a, b) => pref.indexOf(a.model) - pref.indexOf(b.model));
    }
    if (models.length < 2) return { ok: false, error: 'The lab needs at least two expert models in the config.' };
    if (models.length > 3) warn(`The lab holds at most 3 experts; only the first 3 of ${models.length} models are mapped.`);
    models = models.slice(0, 3);
    const K = models.length;
    const p = MM.defaultParams(MM.METHOD_BY_ID[id], K);
    const mp = (name) => models.map((m) => readParam(m.parameters ? m.parameters[name] : null, `${name} (${m.model})`, warn));
    const scalars = (arr, dflt) => arr.map((x) => (x && x.scalar != null ? x.scalar : x && x.grad ? mean(x.grad) : dflt));
    const sameGrad = (arr) => arr.every((x) => x && x.grad) && arr.every((x) => x.grad.join() === arr[0].grad.join());
    const setLambdaFromWeights = (ws, scale) => {
      if (sameGrad(ws)) { p.layerwise = true; p.grad = ws[0].grad.map((v) => +(v * scale).toFixed(4)); p.lambda = 1; notes.push('Per-model weight lists became per-layer λ.'); return; }
      if (ws.some((x) => x && x.grad)) warn('Weight lists differ between models; each model’s list was averaged.');
      const s = scalars(ws, 1);
      p.lambda = +(mean(s) * scale).toFixed(4);
      if (p.weights) { const mx = Math.max(...s); p.weights = s.map((x) => +(x / (mx || 1)).toFixed(4)).concat([1, 1, 1]).slice(0, 3); p.lambda = +(mx * scale).toFixed(4); }
    };
    const density = (name) => { const ds = scalars(mp(name), null).filter((x) => x != null); if (!ds.length) return null; if (new Set(ds).size > 1) warn(`${name} differs between models (${ds.join(', ')}); the lab uses one value, their mean.`); return mean(ds); };
    const globalLambda = readParam(P.lambda, 'lambda', warn);

    if (id === 'linear' || id === 'karcher') {
      const ws = mp('weight');
      if (ws.every((x) => x == null)) notes.push('No weights given: equal weights.');
      else if (id === 'linear' && K === 2 && ws.some((x) => x && x.grad)) {
        const g = ws.map((x) => (x && x.grad ? x.grad : [x ? x.scalar : 1, x ? x.scalar : 1, x ? x.scalar : 1]));
        p.layerwise = true; p.grad = [0, 1, 2].map((l) => +(g[1][l] / ((g[0][l] + g[1][l]) || 1)).toFixed(4));
        notes.push('Per-model weight lists became a per-layer A↔B balance.');
      } else {
        if (ws.some((x) => x && x.grad)) warn('Per-layer weights are only modelled for two-model linear merges; lists were averaged.');
        const s = scalars(ws, 1), mx = Math.max(...s);
        p.weights = s.map((x) => +(x / (mx || 1)).toFixed(4)).concat([1, 1, 1]).slice(0, 3);
      }
    } else if (id === 'slerp') {
      const t = readParam(P.t, 't', warn);
      if (t && t.grad) { p.layerwise = true; p.grad = t.grad; p.t = +mean(t.grad).toFixed(4); notes.push(`t gradient [${t.raw.join(', ')}] sampled at the first, middle and last layer: [${t.grad.join(', ')}].`); }
      else if (t) p.t = t.scalar;
      if (mk === 'nuslerp') warn('nuslerp is imported as plain SLERP.');
    } else if (id === 'task_arithmetic') {
      setLambdaFromWeights(mp('weight'), globalLambda && globalLambda.scalar != null ? globalLambda.scalar : 1);
    } else if (id === 'ties' || id === 'dare' || id === 'breadcrumbs' || id === 'della') {
      const d = density('density');
      if (id === 'dare') { if (d != null) p.drop = +(1 - d).toFixed(4); p.mode = mk === 'dare_linear' ? 'linear' : 'ties'; }
      else if (d != null) p.density = +d.toFixed(4);
      if (id === 'breadcrumbs') { const g = density('gamma'); if (g != null) p.gamma = +g.toFixed(4); if (mk === 'breadcrumbs_ties') warn('breadcrumbs_ties adds sign election; the lab’s Breadcrumbs sums the survivors.'); }
      if (id === 'della') { const e = density('epsilon'); if (e != null) p.epsilon = +e.toFixed(4); if (mk === 'della_linear') warn('della_linear sums the survivors; the lab’s DELLA uses TIES sign election.'); }
      const normalize = P.normalize !== false; // mergekit default: normalize on
      if (id === 'ties' || id === 'della') {
        if (globalLambda && globalLambda.grad) { p.layerwise = true; p.grad = globalLambda.grad; p.lambda = 1; }
        else if (normalize) { p.lambda = globalLambda ? globalLambda.scalar : 1; notes.push('normalize: true, so per-model weights cancel; λ comes from lambda (default 1).'); }
        else setLambdaFromWeights(mp('weight'), K * (globalLambda ? globalLambda.scalar || 1 : 1));
      } else setLambdaFromWeights(mp('weight'), globalLambda && globalLambda.scalar != null ? globalLambda.scalar : 1);
    } else if (id === 'sce') {
      const k = readParam(P.select_topk, 'select_topk', warn);
      if (k && k.scalar != null) p.topk = k.scalar;
      if (globalLambda && globalLambda.grad) { p.layerwise = true; p.grad = globalLambda.grad; }
      else if (globalLambda) p.lambda = globalLambda.scalar;
    } else if (id === 'stitch') {
      const sl = (cfg.slices || []).map((s) => s.sources[0]);
      const total = Math.max(...sl.map((s) => (Array.isArray(s.layer_range) ? s.layer_range[1] : 0)));
      p.layers = [0, 1, 2].map((l) => {
        const mid = ((l + 0.5) / 3) * total;
        let off = 0, src = sl[sl.length - 1];
        for (const s of sl) { const n = s.layer_range[1] - s.layer_range[0]; if (mid < off + n) { src = s; break; } off += n; }
        return Math.min(K - 1, models.findIndex((m) => m.model === src.model));
      });
      notes.push(`${total} stacked layers mapped onto the lab’s three layers by thirds: ${p.layers.map((i) => 'ABC'[i]).join('·')}.`);
    }
    return finish(id, p, models.map((m) => m.model), cfg.base_model || null, warnings, notes, cfg.dtype);
  }

  function finish(id, params, names, base, warnings, notes, dtype) {
    // clamp to the lab's knob ranges (and report what moved)
    const m = MM.METHOD_BY_ID[id];
    for (const d of m.params) {
      if (d.type !== 'range' || params[d.key] == null) continue;
      const v = Math.min(d.max, Math.max(d.min, +params[d.key]));
      if (Math.abs(v - params[d.key]) > 1e-9) warnings.push(`${d.label} ${params[d.key]} is outside the lab’s range; clamped to ${v}.`);
      params[d.key] = v;
    }
    if (m.gradient && params.grad) params.grad = params.grad.map((v) => Math.min(m.gradient.max, Math.max(m.gradient.min, v)));
    // only methods that merge relative to a base use base_model: for SLERP it is the t = 0 endpoint and for
    // mergekit-moe it is usually one of the experts, so neither should overwrite the export form's base
    const usesBase = !!m.req.base;
    return { ok: true, method: id, params, K: Math.min(3, names.length), models: names, base: usesBase ? base : null, dtype: dtype || null, warnings, notes };
  }

  Object.assign(MM, { parseYAML, importMergekit, mkAnchors: anchors });
})();
