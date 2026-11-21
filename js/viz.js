/* ModelMixer — rendering: decision-boundary canvases, provenance maps, SVG charts. */
(function () {
  const MM = (window.MM = window.MM || {});

  const CLASS = [[56, 189, 248], [251, 146, 60]];
  const MODEL_COLORS = ['#6d8bff', '#ff6bb5', '#a3e635'];
  const BASE_COLOR = '#94a3b8';
  const BG = [9, 12, 22];

  const gridCache = {};
  function gridInputs(N) {
    if (gridCache[N]) return gridCache[N];
    const E = MM.EXTENT, X = new Float32Array(N * N * 2);
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const i = r * N + c;
      X[2 * i] = -E + ((c + 0.5) / N) * 2 * E;
      X[2 * i + 1] = E - ((r + 0.5) / N) * 2 * E;
    }
    return (gridCache[N] = X);
  }

  function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }

  function setup(cv) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = cv.clientWidth || cv.width, h = cv.clientHeight || w;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function contour(ctx, p, N, W, H, color, width, dash) {
    const cx = (c) => ((c + 0.5) / N) * W, cy = (r) => ((r + 0.5) / N) * H;
    ctx.beginPath();
    for (let r = 0; r < N - 1; r++) for (let c = 0; c < N - 1; c++) {
      const v = [p[r * N + c] - 0.5, p[r * N + c + 1] - 0.5, p[(r + 1) * N + c + 1] - 0.5, p[(r + 1) * N + c] - 0.5];
      const P = [[cx(c), cy(r)], [cx(c + 1), cy(r)], [cx(c + 1), cy(r + 1)], [cx(c), cy(r + 1)]];
      const pts = [];
      for (let e = 0; e < 4; e++) {
        const a = v[e], b = v[(e + 1) % 4];
        if ((a > 0) !== (b > 0)) {
          const t = a / (a - b), A = P[e], B = P[(e + 1) % 4];
          pts.push([A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t]);
        }
      }
      for (let k = 0; k + 1 < pts.length; k += 2) { ctx.moveTo(pts[k][0], pts[k][1]); ctx.lineTo(pts[k + 1][0], pts[k + 1][1]); }
    }
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
    ctx.stroke(); ctx.setLineDash([]);
  }

  const toPx = (x, y, W, H) => { const E = MM.EXTENT; return [((x + E) / (2 * E)) * W, ((E - y) / (2 * E)) * H]; };

  // Image tasks have no 2-D plane to colour, so show a gallery of test digits instead:
  // tile tint = predicted class (same colours as the map), red ring = mistake.
  function drawDigits(cv, predict, o) {
    const { ctx, w: W, h: H } = setup(cv);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = `rgb(${BG.join(',')})`; ctx.fillRect(0, 0, W, H);
    const tasks = o.tasks, K = tasks.length, D = MM.DIGIT_D;
    const thumb = !!o.thumb, pad = thumb ? 3 : 12, head = thumb ? 0 : 20, gap = thumb ? 1 : 3;
    const cols = thumb ? 6 : 10;
    const bandH = (H - pad) / K;
    tasks.forEach((t, k) => {
      const y0 = pad / 2 + k * bandH;
      const cell = (W - 2 * pad) / cols;
      const rows = Math.max(1, Math.floor((bandH - head - pad / 2) / cell));
      const nShow = Math.min(t.test.n, rows * cols);
      const pAll = predict(t.test.X, t.test.n);
      if (!thumb) {
        let correct = 0;
        for (let i = 0; i < t.test.n; i++) if ((pAll[i] > 0.5 ? 1 : 0) === t.test.Y[i]) correct++;
        const tk = MM.TASKS[t.spec.task];
        ctx.font = '700 11.5px Inter, system-ui, sans-serif'; ctx.textBaseline = 'middle';
        ctx.fillStyle = MODEL_COLORS[k];
        ctx.fillText(`${'ABC'[k]} · ${tk.name}${t.spec.flip ? ' (inv)' : ''}`, pad, y0 + 9);
        ctx.fillStyle = 'rgba(205,211,227,0.75)'; ctx.font = '500 11px "JetBrains Mono", monospace'; ctx.textAlign = 'right';
        ctx.fillText(`${(100 * correct / t.test.n).toFixed(1)}% of ${t.test.n} test digits`, W - pad, y0 + 9);
        ctx.textAlign = 'left';
      }
      for (let i = 0; i < nShow; i++) {
        const r = Math.floor(i / cols), c = i % cols;
        const x = pad + c * cell, y = y0 + head + r * cell, sz = cell - gap;
        const pv = pAll[i], cls = pv > 0.5 ? 1 : 0, conf = Math.min(1, Math.abs(pv - 0.5) * 2);
        const col = CLASS[cls];
        ctx.fillStyle = `rgba(${col.join(',')},${(0.14 + 0.4 * conf).toFixed(3)})`;
        ctx.fillRect(x, y, sz, sz);
        const px = (sz - (thumb ? 2 : 6)) / 8, ox = x + (sz - px * 8) / 2, oy = y + (sz - px * 8) / 2;
        for (let q = 0; q < 64; q++) {
          const v = t.test.X[i * D + q];
          if (v < 0.08) continue;
          ctx.fillStyle = `rgba(240,244,255,${Math.min(1, v).toFixed(3)})`;
          ctx.fillRect(ox + (q % 8) * px, oy + ((q / 8) | 0) * px, px + 0.25, px + 0.25);
        }
        if (cls !== t.test.Y[i]) {
          ctx.strokeStyle = '#ff3b6b'; ctx.lineWidth = thumb ? 1.2 : 2.2;
          ctx.strokeRect(x + 1, y + 1, sz - 2, sz - 2);
        }
      }
    });
  }
  const isDigits = (o) => !!(o && o.tasks && o.tasks.length && o.tasks[0].kind === 'digits');
  /** Draw whatever "map" suits the tasks: the decision plane, or a digit gallery for image tasks. */
  function drawMap(cv, predict, o) { return isDigits(o) ? drawDigits(cv, predict, o) : drawBoundary(cv, predict, o); }

  function drawBoundary(cv, predict, o) {
    o = o || {};
    const { ctx, w: W, h: H } = setup(cv);
    const N = o.grid || 110;
    const X = gridInputs(N);
    const p = predict(X, N * N);
    const off = document.createElement('canvas');
    off.width = N; off.height = N;
    const octx = off.getContext('2d');
    const img = octx.createImageData(N, N);
    // Dim everything outside the task regions so the eye goes to the skills.
    const focus = o.tasks && o.focus !== false ? o.tasks : null;
    for (let i = 0; i < N * N; i++) {
      const pv = p[i], conf = Math.min(1, Math.abs(pv - 0.5) * 2);
      const col = pv > 0.5 ? CLASS[1] : CLASS[0];
      let a = 0.1 + 0.55 * Math.pow(conf, 1.6);
      if (focus) {
        const x = X[2 * i], y = X[2 * i + 1];
        let inside = 0;
        for (const t of focus) {
          const d = Math.hypot(x - t.center[0], y - t.center[1]) / t.radius;
          inside = Math.max(inside, Math.min(1, Math.max(0, (1.08 - d) / 0.1)));
        }
        a *= 0.32 + 0.68 * inside;
      }
      img.data[4 * i] = BG[0] + (col[0] - BG[0]) * a;
      img.data[4 * i + 1] = BG[1] + (col[1] - BG[1]) * a;
      img.data[4 * i + 2] = BG[2] + (col[2] - BG[2]) * a;
      img.data[4 * i + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(off, 0, 0, W, H);

    // subtle grid
    if (!o.thumb) {
      ctx.strokeStyle = 'rgba(255,255,255,0.035)'; ctx.lineWidth = 1;
      for (let g = -3; g <= 3; g++) {
        const [gx] = toPx(g, 0, W, H), [, gy] = toPx(0, g, W, H);
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
      }
    }

    if (o.gates) {
      const K = o.K, g = o.gates(X, N * N);
      const off2 = document.createElement('canvas');
      off2.width = N; off2.height = N;
      const c2 = off2.getContext('2d'), im2 = c2.createImageData(N, N);
      const cols = MODEL_COLORS.map(hexRgb);
      for (let i = 0; i < N * N; i++) {
        let best = 0;
        for (let k = 1; k < K; k++) if (g[i * K + k] > g[i * K + best]) best = k;
        const c = cols[best];
        im2.data[4 * i] = c[0]; im2.data[4 * i + 1] = c[1]; im2.data[4 * i + 2] = c[2];
        im2.data[4 * i + 3] = 70 * g[i * K + best];
      }
      c2.putImageData(im2, 0, 0);
      ctx.drawImage(off2, 0, 0, W, H);
      for (let k = 0; k < K; k++) {
        const gk = new Float32Array(N * N);
        for (let i = 0; i < N * N; i++) gk[i] = g[i * K + k];
        contour(ctx, gk, N, W, H, MODEL_COLORS[k], 1.4, [5, 4]);
      }
    }

    if (focus) {
      contour(ctx, p, N, W, H, 'rgba(255,255,255,0.22)', o.thumb ? 0.8 : 1.2);
      ctx.save();
      ctx.beginPath();
      focus.forEach((t) => {
        const [px, py] = toPx(t.center[0], t.center[1], W, H);
        const rr = ((t.radius * 1.06) / (2 * MM.EXTENT)) * W;
        ctx.moveTo(px + rr, py); ctx.arc(px, py, rr, 0, Math.PI * 2);
      });
      ctx.clip();
      contour(ctx, p, N, W, H, 'rgba(255,255,255,0.92)', o.thumb ? 1 : 1.7);
      ctx.restore();
    } else contour(ctx, p, N, W, H, 'rgba(255,255,255,0.9)', o.thumb ? 1 : 1.6);

    if (o.tasks && o.showRegions !== false) {
      o.tasks.forEach((t, i) => {
        const [px, py] = toPx(t.center[0], t.center[1], W, H);
        const rr = (t.radius / (2 * MM.EXTENT)) * W;
        ctx.strokeStyle = MODEL_COLORS[i]; ctx.globalAlpha = o.thumb ? 0.55 : 0.75;
        ctx.lineWidth = o.thumb ? 1 : 1.5; ctx.setLineDash(o.thumb ? [2, 3] : [4, 5]);
        ctx.beginPath(); ctx.arc(px, py, rr, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        if (!o.thumb) {
          const label = 'ABC'[i];
          const lx = px - rr * 0.72, ly = py - rr * 0.72;
          ctx.fillStyle = MODEL_COLORS[i];
          ctx.beginPath(); ctx.arc(lx, ly, 10, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = '#0a0d18'; ctx.font = '700 11px Inter, system-ui, sans-serif';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(label, lx, ly + 0.5);
        }
      });
    }

    if (o.tasks && o.showPoints) {
      o.tasks.forEach((t) => {
        const n = Math.min(t.test.n, o.thumb ? 60 : 170);
        const pr = predict(t.test.X.subarray(0, n * 2), n);
        for (let i = 0; i < n; i++) {
          const [px, py] = toPx(t.test.X[2 * i], t.test.X[2 * i + 1], W, H);
          const y = t.test.Y[i], wrong = (pr[i] > 0.5 ? 1 : 0) !== y;
          const c = CLASS[y];
          ctx.beginPath(); ctx.arc(px, py, o.thumb ? 1.3 : 2.6, 0, Math.PI * 2);
          ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; ctx.fill();
          if (!o.thumb) {
            ctx.lineWidth = wrong ? 1.6 : 0.8;
            ctx.strokeStyle = wrong ? '#ff3b6b' : 'rgba(5,8,16,0.85)';
            if (wrong) { ctx.beginPath(); ctx.arc(px, py, 4.4, 0, Math.PI * 2); }
            ctx.stroke();
          }
        }
      });
    }
  }

  // Where did each weight of the merged layer come from?
  function drawProvenance(cv, merged, experts, base, T) {
    const { ctx, w: W, h: H } = setup(cv);
    ctx.clearRect(0, 0, W, H);
    const rows = Math.min(T.rows, 32), cols = Math.min(T.cols, 32);
    const cw = W / cols, ch = H / rows;
    const cands = experts.map((e, i) => ({ P: e, col: hexRgb(MODEL_COLORS[i]) }));
    if (base) cands.push({ P: base, col: hexRgb(BASE_COLOR) });
    let spread = 0;
    for (let i = 0; i < rows * cols; i++) {
      const j = T.off + Math.floor(i / cols) * T.cols + (i % cols);
      let mn = Infinity, mx = -Infinity;
      for (const c of cands) { mn = Math.min(mn, c.P[j]); mx = Math.max(mx, c.P[j]); }
      spread += mx - mn;
    }
    spread = spread / (rows * cols) + 1e-9;
    const counts = new Array(cands.length + 1).fill(0);
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const j = T.off + r * T.cols + c;
      const d = cands.map((cd) => Math.abs(merged[j] - cd.P[j]));
      let b = 0;
      for (let k = 1; k < d.length; k++) if (d[k] < d[b]) b = k;
      const sorted = d.slice().sort((x, y) => x - y);
      const decisive = Math.min(1, (sorted[1] - sorted[0]) / spread);
      let col, a;
      if (decisive < 0.12) { col = [168, 132, 255]; a = 0.55; counts[cands.length]++; }
      else { col = cands[b].col; a = 0.25 + 0.75 * decisive; counts[b]++; }
      ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${a})`;
      ctx.fillRect(c * cw + 0.5, r * ch + 0.5, cw - 1, ch - 1);
    }
    const total = rows * cols;
    return counts.map((x) => x / total);
  }

  // ---------- SVG charts ----------
  const svgNS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent) {
    const e = document.createElementNS(svgNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function lineChart(host, o) {
    host.innerHTML = '';
    const W = host.clientWidth || 420, H = o.height || 240;
    const m = { l: 40, r: 14, t: 12, b: 34 };
    const svg = el('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: 'chart' }, host);
    const x0 = o.xMin, x1 = o.xMax, y0 = o.yMin ?? 0, y1 = o.yMax ?? 1;
    const X = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
    const Y = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
    for (let i = 0; i <= 4; i++) {
      const v = y0 + ((y1 - y0) * i) / 4;
      el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: 'gridline' }, svg);
      const t = el('text', { x: m.l - 6, y: Y(v) + 4, class: 'tick', 'text-anchor': 'end' }, svg);
      t.textContent = o.yFmt ? o.yFmt(v) : v.toFixed(1);
    }
    (o.xTicks || []).forEach((v) => {
      const t = el('text', { x: X(v), y: H - m.b + 16, class: 'tick', 'text-anchor': 'middle' }, svg);
      t.textContent = o.xFmt ? o.xFmt(v) : v;
    });
    if (o.xLabel) { const t = el('text', { x: (m.l + W - m.r) / 2, y: H - 4, class: 'axis-label', 'text-anchor': 'middle' }, svg); t.textContent = o.xLabel; }
    (o.bands || []).forEach((b) => {
      const top = b.hi.map((pt) => `${X(pt[0]).toFixed(1)},${Y(pt[1]).toFixed(1)}`);
      const bot = b.lo.slice().reverse().map((pt) => `${X(pt[0]).toFixed(1)},${Y(pt[1]).toFixed(1)}`);
      el('polygon', { points: top.concat(bot).join(' '), fill: b.color, opacity: b.opacity || 0.16, stroke: 'none' }, svg);
    });
    (o.vlines || []).forEach((v) => {
      el('line', { x1: X(v.x), x2: X(v.x), y1: m.t, y2: H - m.b, stroke: v.color, 'stroke-dasharray': v.solid ? '' : '3 4', 'stroke-width': v.width || 1, opacity: v.opacity || 0.6 }, svg);
      if (v.label) {
        const t = el('text', { x: X(v.x) + (v.anchor === 'end' ? -4 : 4), y: m.t + 10 + (v.row || 0) * 12, class: 'tick', style: `fill:${v.color}`, 'text-anchor': v.anchor || 'start' }, svg);
        t.textContent = v.label;
      }
    });
    o.series.forEach((s) => {
      const d = s.points.map((pt, i) => `${i ? 'L' : 'M'}${X(pt[0]).toFixed(1)},${Y(pt[1]).toFixed(1)}`).join('');
      el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.width || 2, 'stroke-dasharray': s.dash || '', 'stroke-linejoin': 'round', opacity: s.opacity || 1 }, svg);
    });
    // hover readout
    const guide = el('line', { y1: m.t, y2: H - m.b, class: 'hover-line', opacity: 0 }, svg);
    const tip = document.createElement('div');
    tip.className = 'chart-tip'; host.appendChild(tip);
    svg.addEventListener('mousemove', (ev) => {
      const r = svg.getBoundingClientRect(), mx = ev.clientX - r.left;
      const xv = x0 + ((mx - m.l) / (W - m.l - m.r)) * (x1 - x0);
      const pts = o.series[0].points;
      let bi = 0;
      pts.forEach((p, i) => { if (Math.abs(p[0] - xv) < Math.abs(pts[bi][0] - xv)) bi = i; });
      const xx = X(pts[bi][0]);
      guide.setAttribute('x1', xx); guide.setAttribute('x2', xx); guide.setAttribute('opacity', 1);
      tip.style.opacity = 1;
      tip.style.left = `${Math.min(W - 170, xx + 10)}px`; tip.style.top = `${m.t}px`;
      tip.innerHTML = `<b>${o.xFmt ? o.xFmt(pts[bi][0]) : pts[bi][0].toFixed(2)}</b>` + o.series.map((s) => `<div><i style="background:${s.color}"></i>${s.name}<span>${o.yFmt ? o.yFmt(s.points[bi][1]) : s.points[bi][1].toFixed(2)}</span></div>`).join('');
    });
    svg.addEventListener('mouseleave', () => { guide.setAttribute('opacity', 0); tip.style.opacity = 0; });
  }

  function sparkline(host, values, color) {
    host.innerHTML = '';
    if (values.length < 2) return;
    const W = host.clientWidth || 200, H = host.clientHeight || 28;
    const svg = el('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, host);
    const mx = Math.max(...values), mn = Math.min(...values);
    const d = values.map((v, i) => `${i ? 'L' : 'M'}${((i / (values.length - 1)) * W).toFixed(1)},${(2 + (1 - (v - mn) / (mx - mn + 1e-9)) * (H - 4)).toFixed(1)}`).join('');
    el('path', { d, fill: 'none', stroke: color, 'stroke-width': 1.5 }, svg);
  }

  Object.assign(MM, { drawMap, drawDigits, drawBoundary, drawProvenance, lineChart, sparkline, MODEL_COLORS, BASE_COLOR, CLASS });
})();
