/* ModelMixer — recipe report. Pure: a plain data object in, a self-contained HTML page and Markdown out.
 * Sections whose data wasn't measured (sensitivity, robustness, comparison) are simply left out. */
(function () {
  const root = typeof window !== 'undefined' ? window : globalThis;
  const MM = (root.MM = root.MM || {});

  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (x) => (x == null || !Number.isFinite(+x) ? '—' : `${(x * 100).toFixed(1)}%`);
  const mdCell = (s) => String(s == null ? '' : s).replace(/\|/g, '\\|').replace(/\n/g, ' ');

  /**
   * What a task's saliency match means, given accuracy. match = cosine with the expert's pixel map (0..1);
   * merged / expert = accuracies. A match of 80% or more counts as the same pixel pattern, and a drop of more
   * than 10 pts from the expert counts as a lost skill.
   */
  function saliencyReading(match, merged, expert, opts) {
    const router = !!(opts && opts.router);
    const kept = merged >= expert - 0.1, same = match >= 0.8;
    if (kept && same) return { kind: 'same', tone: 'good', text: 'keeps the skill and reads the same pixels as the expert' };
    if (kept) return { kind: 'other', tone: 'mid', text: router ? 'keeps the skill but reads different pixels: its router sends each input to the right expert, so it solves the task another way'
      : 'keeps the skill but reads different pixels, so it solves the task another way (often fine, but check it on inputs unlike the test set)' };
    if (same) return { kind: 'conflict', tone: 'bad', text: 'looks at the right pixels but answers wrongly: the conflict signature, so a better merge is unlikely to fix it (it needs a task signal or retraining)' };
    return { kind: 'lost', tone: 'bad', text: 'has lost both the skill and the expert’s pixel pattern' };
  }

  const READING_KIND_LABEL = { same: 'same pixels, skill kept', other: 'different pixels, skill kept', conflict: 'conflict signature', lost: 'skill lost' };
  /**
   * Stability of a task's reading over seeds (a saliencySeedSummary entry): all agree, mostly (≥ 80%), or unstable.
   * Returns { level: 'stable' | 'mostly' | 'unstable', text }.
   */
  function seedStability(s) {
    const kinds = Object.entries(s.counts).sort((a, b) => b[1] - a[1]).map(([k, c]) => `${READING_KIND_LABEL[k]} ${c}/${s.n}`).join(', ');
    const range = `match ${Math.round(s.mean * 100)}% ± ${Math.round(s.sd * 100)}, range ${Math.round(s.min * 100)}–${Math.round(s.max * 100)}%`;
    if (s.agree === 1) return { level: 'stable', text: `the same reading on all ${s.n} seeds (${range})` };
    if (s.agree >= 0.8) return { level: 'mostly', text: `mostly stable across ${s.n} seeds: ${kinds} (${range})` };
    return { level: 'unstable', text: `unstable across ${s.n} seeds: ${kinds} (${range})` };
  }

  /**
   * d = { title, date, link,
   *   recipe: { method, family, label, scenario, experts:[...], lineage, aligned, width, ftSteps, seed },
   *   result: { tasks:[{ name, merged, expert, base }], avg, retention, cost, storage, warnings:[] }, mapPng,
   *   sensitivity?, layerMap?, layers?: [{ pair, layer, kind, cost, n, sentence }], transfer?: { rows: [{ knob, kind, lo, hi, best, regret, now, basis }], knobs, seeds }, robust?: { seeds, rows:[{ name, mean, sd, worst, stability, current }], verdict, h2h },
   *   compare?: { pinned, now, changes?: [{ what, a, b, delta? }], attribution?: { total, interaction, seed }, summary, edge },
   *   saliency?: { tasks:[{ name, match, merged, expert, expertPng?, mergedPng?, seeds? }], mean, router? },
   *     (seeds = a saliencySeedSummary entry, present only if the multi-seed check was run) method: { tagline, how:[], pros:[], cons:[], realWorld, paper, oss:[{name,url}] },
   *   export: { file, yaml, cmd, note } }
   */
  const pts = (x) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)} pts`;
  const attrTxt = (x) => (x.seeds ? ` (alone: ${pts(x.seeds.mean)} ± ${(x.seeds.sd * 100).toFixed(1)}, ${x.seeds.sign})`
    : x.delta === undefined ? '' : x.delta === null ? ' (can’t run alone)' : ` (alone: ${pts(x.delta)})`);
  const attrFoot = (a) => `"Alone" is A with only that change, ${a.seeds ? `mean ± sd over ${a.seeds} seeds` : `on seed ${a.seed}`}. Together: ${pts(a.total)}${a.seeds ? ` ± ${(a.totalSd * 100).toFixed(1)}` : ''}; ${Math.abs(a.interaction) < 0.005 ? 'the changes roughly add up (within half a point).' : `${pts(a.interaction)} comes from the changes interacting.`}`;
  const BASIS_TXT = { recipe: 'this recipe', defaults: 'default settings', snapshot: 'pre-measured (default settings)' };
  const TRANSFER_TXT = { moves: 'moves with the seed', narrow: 'same spot, narrow peak', transfers: 'transfers', holds: 'holds' };
  const transferNote = (t) => {
    const mv = t.rows.filter((r) => r.kind === 'moves' || r.kind === 'narrow').length;
    return `${mv ? `${mv} of ${t.rows.length} measured knob${t.rows.length > 1 ? 's' : ''} should be tuned on your own models rather than copied.` : 'Every measured knob value carries over from seed to seed.'}${t.rows.length < t.knobs ? ` ${t.knobs - t.rows.length} knob${t.knobs - t.rows.length > 1 ? 's were' : ' was'} not measured.` : ''} "Worst-seed cost" is how far the best-on-average value falls below that seed's own best.`;
  };
  function buildReport(d) {
    const r = d.recipe, res = d.result, m = d.method;
    // ---------- Markdown ----------
    const md = [];
    md.push(`# ${d.title || 'ModelMixer recipe report'}`, '');
    md.push(`*${d.date || ''}*${d.link ? ` · [open this recipe](${d.link})` : ''}`, '');
    md.push('## Recipe', '');
    md.push(`- **Method:** ${r.method} (${r.family})`, `- **Settings:** \`${r.label}\``,
      `- **Toy scenario:** ${r.scenario}: ${r.experts.join(', ')}`,
      `- **Lineage:** ${r.lineage}${r.aligned ? ', neurons aligned (Git Re-Basin)' : ''}`,
      `- **Training:** width ${r.width}, ${r.ftSteps} fine-tune steps, seed ${r.seed}`, '');
    const tune = d.transfer ? d.transfer.rows.filter((x) => x.kind === 'moves' || x.kind === 'narrow') : [];
    const keep = d.transfer ? d.transfer.rows.filter((x) => x.kind === 'holds' || x.kind === 'transfers') : [];
    const before = d.transfer
      ? `${tune.length ? `Tune on your own models: ${tune.map((x) => `${x.knob} (${TRANSFER_TXT[x.kind]})`).join(', ')}.` : 'Every measured knob value carried over from seed to seed.'}${keep.length ? ` Can be copied: ${keep.map((x) => `${x.knob} ${x.best}`).join(', ')}.` : ''}${d.transfer.rows.length < d.transfer.knobs ? ` ${d.transfer.knobs - d.transfer.rows.length} knob${d.transfer.knobs - d.transfer.rows.length > 1 ? 's' : ''} not measured.` : ''}`
      : null;
    const preOnly = d.transfer && d.transfer.rows.every((x) => x.basis === 'snapshot');
    const layerTxt = (d.layers || []).map((l) => l.sentence).join(' ');
    const beforeAll = [before, layerTxt].filter(Boolean).join(' ');
    const seedsN = d.transfer ? d.transfer.seeds : d.layers ? d.layers[0].n : 0;
    if (beforeAll) md.push(`> **Before a real merge** (${seedsN}-seed toy evidence${preOnly && !layerTxt ? ', pre-measured' : ''}): ${beforeAll}`, '');
    md.push('## Result on the toy', '');
    md.push('| Task | Merged | Expert alone | Base |', '| --- | --- | --- | --- |');
    res.tasks.forEach((t) => md.push(`| ${mdCell(t.name)} | ${pct(t.merged)} | ${pct(t.expert)} | ${pct(t.base)} |`));
    md.push('', `**Average** ${pct(res.avg)} · **skill retention** ${pct(res.retention)} · inference ${res.cost} · memory ${res.storage}`);
    (res.warnings || []).forEach((w) => md.push('', `> ⚠ ${w}`));
    md.push('');
    if (d.sensitivity) md.push('## How forgiving is it?', '', d.sensitivity, '');
    if (d.transfer) {
      md.push(`## Which knob values can be copied? (${d.transfer.seeds} seeds)`, '');
      md.push('| Knob | Across seeds | Per-seed best | Best on average | Worst-seed cost | This recipe | Measured at |', '| --- | --- | --- | --- | --- | --- | --- |');
      d.transfer.rows.forEach((r) => md.push(`| ${mdCell(r.knob)} | ${TRANSFER_TXT[r.kind]} | ${r.lo} – ${r.hi} | ${r.best} | ${(r.regret * 100).toFixed(1)} pts | ${r.now} | ${BASIS_TXT[r.basis] || 'default settings'} |`));
      md.push('', transferNote(d.transfer), '');
    }
    if (d.layerMap) md.push('## Which layer matters?', '', d.layerMap, '');
    const sal = d.saliency ? d.saliency.tasks.map((t) => {
      const reading = saliencyReading(t.match, t.merged, t.expert, { router: d.saliency.router });
      const stab = t.seeds ? seedStability(t.seeds) : null;
      // an unstable reading is reported as what this seed shows, not as a property of the method
      const claim = stab && stab.level === 'unstable' ? `On this seed, the merge ${reading.text}, but treat that as a guess` : `This merge ${reading.text}`;
      return Object.assign({}, t, { reading, stab, claim });
    }) : null;
    const anySeeds = !!(sal && sal.some((t) => t.stab));
    if (sal) {
      md.push('## Where the merge looks (pixel saliency)', '');
      md.push(`| Task | Pattern match with expert | Merged | Expert | Reading |${anySeeds ? ' Across seeds |' : ''}`, `| --- | --- | --- | --- | --- |${anySeeds ? ' --- |' : ''}`);
      sal.forEach((t) => md.push(`| ${mdCell(t.name)} | ${Math.round(t.match * 100)}% | ${pct(t.merged)} | ${pct(t.expert)} | ${mdCell(t.claim)} |${anySeeds ? ` ${mdCell(t.stab ? t.stab.text : 'not checked')} |` : ''}`));
      md.push('', `Mean pattern match ${Math.round(d.saliency.mean * 100)}%. Saliency is how much each of the 64 pixels moves the prediction; the match is the cosine between the merge's map and the expert's.${anySeeds ? '' : ' Readings near the 80% line can change with the seed: run the 5-seed check in the Lab before relying on one.'}`, '');
    }
    if (d.robust) {
      md.push(`## Is the winner real? (${d.robust.seeds} seeds)`, '');
      md.push('| Method | Mean ± sd | Worst task | Stability |', '| --- | --- | --- | --- |');
      d.robust.rows.forEach((x) => md.push(`| ${x.current ? '**' : ''}${mdCell(x.name)}${x.current ? '** (this recipe)' : ''} | ${pct(x.mean)} ± ${(x.sd * 100).toFixed(1)} | ${pct(x.worst)} | ${x.stability} |`));
      if (d.robust.verdict) md.push('', d.robust.verdict);
      if (d.robust.h2h) md.push('', `Head-to-head: ${d.robust.h2h}`);
      md.push('');
    }
    if (d.compare) {
      md.push('## Compared with the pinned recipe', '', `A (pinned): \`${d.compare.pinned}\` · B (this recipe): \`${d.compare.now}\``, '');
      if ((d.compare.changes || []).length) {
        md.push('**What changed (A → B)**', '', ...d.compare.changes.map((x) => `- ${x.what}: \`${x.a}\` → \`${x.b}\`${attrTxt(x)}`), '');
        if (d.compare.attribution) md.push(attrFoot(d.compare.attribution), '');
      }
      if (d.compare.summary) md.push(d.compare.summary, '');
      if (d.compare.edge) md.push(`Paired check: ${d.compare.edge}`, '');
    }
    md.push(`## About ${r.method}`, '', m.tagline || '', '');
    (m.how || []).forEach((h, i) => md.push(`${i + 1}. ${h}`));
    md.push('', '**Strengths**', ...(m.pros || []).map((x) => `- ${x}`), '', '**Missing pieces**', ...(m.cons || []).map((x) => `- ${x}`), '');
    if (m.realWorld) md.push(`**On real models:** ${m.realWorld}`, '');
    md.push(`*${m.paper || ''}*`, '');
    md.push('## Run it for real', '');
    if (d.export.note) md.push(d.export.note, '');
    md.push(`\`${d.export.file}\``, '', '```yaml', d.export.yaml.replace(/\n$/, ''), '```', '', '```bash', d.export.cmd, '```', '');
    if ((m.oss || []).length) md.push(m.oss.map((o) => `[${o.name}](${o.url})`).join(' · '), '');
    md.push('---', '', '*Generated by ModelMixer. Toy-scale evidence: confirm on a sample of your real data.*', '');
    const markdown = md.join('\n');

    // ---------- HTML (self-contained, print-friendly) ----------
    const row = (cells, tag) => `<tr>${cells.map((c) => `<${tag || 'td'}>${c}</${tag || 'td'}>`).join('')}</tr>`;
    const sec = (title, body) => `<section><h2>${esc(title)}</h2>${body}</section>`;
    const parts = [];
    parts.push(sec('Recipe', `<dl>
      <dt>Method</dt><dd><b>${esc(r.method)}</b> <span class="muted">(${esc(r.family)})</span></dd>
      <dt>Settings</dt><dd><code>${esc(r.label)}</code></dd>
      <dt>Toy scenario</dt><dd>${esc(r.scenario)}: ${r.experts.map(esc).join(', ')}</dd>
      <dt>Lineage</dt><dd>${esc(r.lineage)}${r.aligned ? ', neurons aligned (Git Re-Basin)' : ''}</dd>
      <dt>Training</dt><dd>width ${esc(r.width)}, ${esc(r.ftSteps)} fine-tune steps, seed ${esc(r.seed)}</dd></dl>`));
    if (beforeAll) parts.push(`<div class="before"><b>Before a real merge</b> <span class="muted">(${seedsN}-seed toy evidence${preOnly && !layerTxt ? ', pre-measured' : ''})</span><br/>${esc(beforeAll)}</div>`);
    parts.push(sec('Result on the toy', `<div class="res">
      ${d.mapPng ? `<img class="map" src="${esc(d.mapPng)}" alt="Decision map of this recipe on the toy" />` : ''}
      <div><table>${row(['Task', 'Merged', 'Expert alone', 'Base'], 'th')}${res.tasks.map((t) => row([esc(t.name), `<b>${pct(t.merged)}</b>`, pct(t.expert), pct(t.base)])).join('')}</table>
      <p class="kpis"><span>Average <b>${pct(res.avg)}</b></span><span>Skill retention <b>${pct(res.retention)}</b></span><span>Inference <b>${esc(res.cost)}</b></span><span>Memory <b>${esc(res.storage)}</b></span></p>
      ${(res.warnings || []).map((w) => `<p class="warn">⚠ ${esc(w)}</p>`).join('')}</div></div>`));
    if (sal) parts.push(sec('Where the merge looks (pixel saliency)', `<p class="muted">How much each of the 64 pixels moves the prediction, for each expert and for this merge (each map scaled to its own peak). The match is the cosine between the two maps; mean ${Math.round(d.saliency.mean * 100)}%.</p>
      <div class="sal">${sal.map((t) => `<div class="salr">
        <div class="salm">${t.expertPng ? `<figure><img src="${esc(t.expertPng)}" alt="Expert pixel saliency for ${esc(t.name)}" /><figcaption>expert</figcaption></figure>` : ''}${t.mergedPng ? `<figure><img src="${esc(t.mergedPng)}" alt="Merged model pixel saliency for ${esc(t.name)}" /><figcaption>merged</figcaption></figure>` : ''}</div>
        <div><b>${esc(t.name)}</b> <span class="pill ${t.match > 0.9 ? 'good' : t.match > 0.7 ? 'mid' : 'bad'}">${Math.round(t.match * 100)}% match</span><br/><span class="muted">${pct(t.merged)} merged vs ${pct(t.expert)} expert.</span> ${esc(t.claim)}.${t.stab ? `<br/><span class="stab ${t.stab.level}">${t.stab.level === 'stable' ? '✓' : t.stab.level === 'mostly' ? '≈' : '⚠'} ${esc(t.stab.text.charAt(0).toUpperCase() + t.stab.text.slice(1))}.</span>` : ''}</div></div>`).join('')}</div>
      ${anySeeds ? '' : '<p class="muted">Readings near the 80% line can change with the seed: run the 5-seed check in the Lab before relying on one.</p>'}`));
    if (d.sensitivity) parts.push(sec('How forgiving is it?', `<p>${esc(d.sensitivity)}</p>`));
    if (d.transfer) parts.push(sec(`Which knob values can be copied? (${d.transfer.seeds} seeds)`, `<table>${row(['Knob', 'Across seeds', 'Per-seed best', 'Best on average', 'Worst-seed cost', 'This recipe', 'Measured at'], 'th')}${d.transfer.rows.map((r) => row([esc(r.knob), `<span class="tv ${r.kind}">${TRANSFER_TXT[r.kind]}</span>`, `${esc(r.lo)} – ${esc(r.hi)}`, esc(r.best), `${(r.regret * 100).toFixed(1)} pts`, esc(r.now), BASIS_TXT[r.basis] || 'default settings'])).join('')}</table>
      <p class="muted">${esc(transferNote(d.transfer))}</p>`));
    if (d.layerMap) parts.push(sec('Which layer matters?', `<p>${esc(d.layerMap)}</p>`));
    if (d.robust) parts.push(sec(`Is the winner real? (${d.robust.seeds} seeds)`, `<table>${row(['Method', 'Mean ± sd', 'Worst task', 'Stability'], 'th')}${d.robust.rows.map((x) => `<tr class="${x.current ? 'cur' : ''}"><td>${esc(x.name)}${x.current ? ' <span class="tag">this recipe</span>' : ''}</td><td>${pct(x.mean)} ± ${(x.sd * 100).toFixed(1)}</td><td>${pct(x.worst)}</td><td>${esc(x.stability)}</td></tr>`).join('')}</table>
      ${d.robust.verdict ? `<p>${esc(d.robust.verdict)}</p>` : ''}${d.robust.h2h ? `<p><b>Head-to-head:</b> ${esc(d.robust.h2h)}</p>` : ''}`));
    if (d.compare) parts.push(sec('Compared with the pinned recipe', `<p>A (pinned): <code>${esc(d.compare.pinned)}</code><br/>B (this recipe): <code>${esc(d.compare.now)}</code></p>
      ${(d.compare.changes || []).length ? `<p><b>What changed (A → B):</b></p><ul>${d.compare.changes.map((x) => `<li>${esc(x.what)}: <code>${esc(x.a)}</code> → <code>${esc(x.b)}</code>${esc(attrTxt(x))}</li>`).join('')}</ul>${d.compare.attribution ? `<p class="muted">${esc(attrFoot(d.compare.attribution))}</p>` : ''}` : ''}
      ${d.compare.summary ? `<p>${esc(d.compare.summary)}</p>` : ''}${d.compare.edge ? `<p><b>Paired check:</b> ${esc(d.compare.edge)}</p>` : ''}`));
    parts.push(sec(`About ${r.method}`, `<p>${esc(m.tagline || '')}</p><ol>${(m.how || []).map((h) => `<li>${esc(h)}</li>`).join('')}</ol>
      <div class="two"><div><h3>Strengths</h3><ul>${(m.pros || []).map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div>
      <div><h3>Missing pieces</h3><ul>${(m.cons || []).map((x) => `<li>${esc(x)}</li>`).join('')}</ul></div></div>
      ${m.realWorld ? `<p class="note"><b>On real models:</b> ${esc(m.realWorld)}</p>` : ''}<p class="muted"><i>${esc(m.paper || '')}</i></p>`));
    parts.push(sec('Run it for real', `${d.export.note ? `<p>${esc(d.export.note)}</p>` : ''}<p class="muted">${esc(d.export.file)}</p>
      <pre>${esc(d.export.yaml)}</pre><pre>${esc(d.export.cmd)}</pre>
      ${(m.oss || []).length ? `<p>${m.oss.map((o) => `<a href="${esc(o.url)}">${esc(o.name)}</a>`).join(' · ')}</p>` : ''}`));
    const html = `<!doctype html><html lang="en"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${esc(d.title || 'ModelMixer recipe report')}</title>
<style>
  :root { color-scheme: light; }
  body { font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; color: #111827; background: #fff; margin: 0 auto; max-width: 860px; padding: 32px 24px 48px; }
  h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 28px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #e5e7eb; } h3 { font-size: 13px; margin: 8px 0 4px; }
  .muted { color: #6b7280; } code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
  code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; }
  pre { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; overflow-x: auto; white-space: pre-wrap; }
  dl { display: grid; grid-template-columns: 120px 1fr; gap: 4px 12px; margin: 0; } dt { color: #6b7280; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; } th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eef0f3; } th { color: #6b7280; font-weight: 600; }
  tr.cur td { background: #f5f3ff; } .tag { font-size: 11px; color: #6d28d9; border: 1px solid #ddd6fe; border-radius: 999px; padding: 0 6px; }
  .res { display: grid; grid-template-columns: 220px 1fr; gap: 16px; align-items: start; } .map { width: 220px; height: 220px; border-radius: 10px; border: 1px solid #e5e7eb; }
  .kpis { display: flex; gap: 16px; flex-wrap: wrap; } .warn { color: #92400e; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 6px 10px; }
  .note { background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 8px; padding: 8px 12px; } .two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .sal { display: flex; flex-direction: column; gap: 12px; } .salr { display: grid; grid-template-columns: 150px 1fr; gap: 14px; align-items: center; }
  .salm { display: flex; gap: 8px; } .salm figure { margin: 0; text-align: center; } .salm img { width: 64px; height: 64px; image-rendering: pixelated; border-radius: 6px; border: 1px solid #e5e7eb; display: block; }
  .salm figcaption { font-size: 11px; color: #6b7280; }
  .pill { font-size: 11.5px; border-radius: 999px; padding: 1px 8px; border: 1px solid; } .pill.good { color: #047857; border-color: #a7f3d0; background: #ecfdf5; }
  .pill.mid { color: #92400e; border-color: #fde68a; background: #fffbeb; } .pill.bad { color: #b91c1c; border-color: #fecaca; background: #fef2f2; }
  .stab { font-size: 12.5px; } .stab.stable { color: #047857; } .stab.mostly { color: #92400e; } .stab.unstable { color: #b91c1c; }
  .tv { font-size: 11.5px; border-radius: 999px; padding: 0 7px; border: 1px solid; white-space: nowrap; }
  .tv.moves, .tv.narrow { color: #92400e; border-color: #fde68a; background: #fffbeb; } .tv.transfers, .tv.holds { color: #047857; border-color: #a7f3d0; background: #ecfdf5; }
  .before { margin: 20px 0 0; padding: 10px 14px; border-radius: 10px; border: 1px solid #fde68a; background: #fffbeb; font-size: 13.5px; line-height: 1.55; }
  footer { margin-top: 32px; color: #6b7280; font-size: 12px; }
  @media (max-width: 600px) { .res, .two { grid-template-columns: 1fr; } dl { grid-template-columns: 1fr; } }
  @media print { body { padding: 0; } a { color: inherit; } h2 { break-after: avoid; } section { break-inside: avoid; } }
</style></head><body>
<h1>${esc(d.title || 'ModelMixer recipe report')}</h1>
<p class="muted">${esc(d.date || '')}${d.link ? ` · <a href="${esc(d.link)}">Open this recipe in ModelMixer</a>` : ''}</p>
${parts.join('\n')}
<footer>Generated by ModelMixer. Toy-scale evidence: confirm on a sample of your real data before shipping a merge.</footer>
</body></html>`;
    return { markdown, html };
  }

  Object.assign(MM, { buildReport, saliencyReading, seedStability, READING_KIND_LABEL });
})();
