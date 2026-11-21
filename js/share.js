/* ModelMixer — shareable recipe links. Encodes the experts, training settings, method,
 * knobs and any pinned comparison into a compact URL-safe string (#r=…), and decodes it
 * back with strict validation so a hand-edited or truncated link can never break the app. */
(function () {
  const root = typeof window !== 'undefined' ? window : globalThis;
  const MM = (root.MM = root.MM || {});

  const HIDDEN = [16, 32, 64];
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const num = (x, d) => (typeof x === 'number' && Number.isFinite(x) ? x : d);

  function b64urlEncode(str) {
    const bytes = new TextEncoder().encode(str);
    let bin = '';
    bytes.forEach((b) => (bin += String.fromCharCode(b)));
    return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }
  function b64urlDecode(s) {
    const pad = s.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((s.length + 3) % 4);
    const bin = atob(pad);
    return new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0)));
  }

  /** Keep only knobs the method defines, coerced and clamped to their declared ranges. */
  function cleanParams(id, raw, K) {
    const m = MM.METHOD_BY_ID[id];
    const p = MM.defaultParams(m, K);
    raw = raw && typeof raw === 'object' ? raw : {};
    for (const d of m.params) {
      if (d.type === 'range' && raw[d.key] !== undefined) {
        const v = clamp(num(+raw[d.key], d.default), d.min, d.max);
        p[d.key] = d.step >= 1 ? Math.round(v) : +v.toFixed(4);
      } else if (d.type === 'select' && raw[d.key] !== undefined) {
        if (d.options.some(([k]) => k === String(raw[d.key]))) p[d.key] = String(raw[d.key]);
      } else if (d.type === 'weights' && Array.isArray(raw.weights)) {
        p.weights = p.weights.map((w, i) => clamp(num(+raw.weights[i], w), 0, 1));
      } else if (d.type === 'layers' && Array.isArray(raw.layers)) {
        p.layers = p.layers.map((l, i) => clamp(Math.round(num(+raw.layers[i], l)), 0, Math.max(0, K - 1)));
      }
    }
    return p;
  }

  /** r = { experts:[{task,region,flip}], init, hidden, ftSteps, seed, align, method, params, pin? } */
  function encodeRecipe(r) {
    const o = {
      v: 1,
      e: r.experts.map((x) => [x.task, x.region, x.flip ? 1 : 0]),
      i: r.init === 'independent' ? 'i' : 's',
      h: r.hidden, f: r.ftSteps, s: r.seed, a: r.align ? 1 : 0,
      m: r.method, p: r.params,
    };
    if (r.pin) o.pn = { m: r.pin.method, p: r.pin.params, a: r.pin.align ? 1 : 0 };
    return b64urlEncode(JSON.stringify(o));
  }

  function decodeRecipe(str) {
    let o;
    try { o = JSON.parse(b64urlDecode(String(str || '').trim())); } catch (e) { return { ok: false, error: 'This link is damaged or incomplete.' }; }
    if (!o || typeof o !== 'object' || o.v !== 1) return { ok: false, error: 'This link comes from an unknown version of ModelMixer.' };
    if (!Array.isArray(o.e) || o.e.length < 2 || o.e.length > 3) return { ok: false, error: 'A recipe needs 2 or 3 experts.' };
    const experts = [];
    for (const x of o.e) {
      if (!Array.isArray(x) || !MM.TASKS[x[0]] || !MM.REGIONS[x[1]]) return { ok: false, error: 'The link names an unknown skill or region.' };
      const e = { task: x[0], region: x[1] };
      if (x[2]) e.flip = true;
      experts.push(e);
    }
    const K = experts.length;
    if (!MM.METHOD_BY_ID[o.m]) return { ok: false, error: 'The link names an unknown mixing method.' };
    const recipe = {
      experts,
      init: o.i === 'i' ? 'independent' : 'shared',
      hidden: HIDDEN.includes(o.h) ? o.h : 32,
      ftSteps: clamp(Math.round(num(o.f, 250) / 25) * 25, 50, 800),
      seed: clamp(Math.round(num(o.s, 7)), 1, 999999),
      align: !!o.a,
      method: o.m,
      params: cleanParams(o.m, o.p, K),
      pin: null,
    };
    if (o.pn && MM.METHOD_BY_ID[o.pn.m]) recipe.pin = { method: o.pn.m, params: cleanParams(o.pn.m, o.pn.p, K), align: !!o.pn.a };
    return { ok: true, recipe };
  }

  /** Pull the encoded recipe out of a hash or full URL ("#r=…", "…/modelmixer.html#r=…"). */
  function recipeFromHash(hash) {
    const m = /(?:^|[#&])r=([A-Za-z0-9_-]+)/.exec(String(hash || ''));
    return m ? m[1] : null;
  }

  Object.assign(MM, { encodeRecipe, decodeRecipe, recipeFromHash, cleanRecipeParams: cleanParams });
})();
