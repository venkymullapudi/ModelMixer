# ModelMixer improvement backlog

Audience: **practitioners picking a merge recipe** for real checkpoints (then exporting to mergekit).
Each improvement loop takes the top unchecked item, ships it (tests green, bundle rebuilt, no console
errors, no horizontal scroll at 375px) and moves it to *Done* with the date.

## How to verify an iteration

- `node tests/run.js` (CI runs this too). Without Node: `python3 scripts/bundle.py --tests`, then open
  `tests/_browser.html` and expect `N/N passed`.
- `python3 scripts/bundle.py`, then open `dist/modelmixer.html`, exercise the changed feature, and check
  the console for errors.

## Next up (highest value first)

1. **Hosted demo (needs owner OK):** a GitHub Pages workflow that publishes `dist/modelmixer.html`, so recipe links can be shared as https URLs. The repo owner has to enable Pages in the repo settings.
2. **Image-like task:** procedurally generated 8×8 "digits", each expert on a different digit pair, so
   merges run on a ~64-dim input model rather than only 2-D toys.
3. **Newer methods:** DELLA (magnitude-adaptive drop), multi-model SLERP / Karcher mean, SCE,
   Localize-and-Stitch, AdaMerging (entropy-tuned coefficients on unlabeled data).
4. **Robustness extras:** a paired-difference view (method X minus method Y per seed), a configurable seed
   count and a CSV download.
5. **Advisor blending:** let the measured toy evidence nudge the ranking (e.g. promote a pick that clearly beats the rule-based #1), and show the measured-vs-rules disagreement in the Arena too.
6. **Compare extras:** a diff map that highlights where A and B disagree on the plane, and an A-vs-B paired multi-seed check (is B's edge real?).
7. **Accessibility pass:** keyboard focus rings and arrow-key navigation in the method picker, ARIA
   labels on canvases, and a colour-blind-safe palette option.

## Done

- 2026-11-21: **Per-layer values (mergekit gradients):** a "Per-layer …" toggle for SLERP (t), two-expert Linear (A↔B balance) and Task Arithmetic, TIES, DARE and Breadcrumbs (λ). It gives one slider per layer plus Flat / Ramp ↑ / Ramp ↓ / Middle-heavy presets. Exported verbatim as mergekit gradient lists (e.g. `t: [0.25, 0.50, 0.75]`); TIES switches to `normalize: false` with weight ≈ λ ÷ K. Recipe labels, share links and the sweep panel understand gradients. Toy finding: SLERP with t ramping up (0.25→0.75) reaches 93.1% vs 85.8% flat and 77.2% ramping down. 2 tests, including Linear [0, 0, 1] = Layer Stitch A·A·B.
- 2026-11-21: **Defaults review** (Two skills, Three skills and Hard + easy × seeds 7/108/209; mean / sd / worst of average accuracy):

  | Knob | Old → new | Mean | Worst | Notes |
  | --- | --- | --- | --- | --- |
  | Fisher ε | 0.05 → 0 | 87.9 → 93.5 | 76.1 → 86.7 | Even ε = 0.001 costs ~4 pts. Matches the paper (ε only for stability). |
  | Breadcrumbs density / γ | 0.6 / 0.02 → 0.1 / 0.005 | 72.2 → 91.2 | 57.6 → 84.7 | Toy experts have little redundancy; γ is sharp (0.01 → 82.7). |
  | TIES density | 0.3 → 0.2 | 85.4 → 87.7 | 79.3 → 82.2 | Lowest spread (sd 3.7); matches the paper's top-20%. A single seed can favour 0.3. |
  | DARE λ | kept at 1 | 88.8 vs 90.0 at λ 0.5 | 75.1 vs 81.8 | Mean gain within noise, so unchanged. |
  | Task Arithmetic λ, RegMean α | kept | within 1 pt of best | | Already in the forgiving range. |

  Method cards gained an "On real models" note where toy defaults differ from common real-world starting points. Knobs saved under the old defaults are dropped once (`paramsV: 2`), and a regression test guards the Fisher and Breadcrumbs gains.
- 2026-11-21: **Knob sensitivity sweep** (Lab panel): average and per-task accuracy as one knob moves (17 points, other knobs held), with "now", "best" and "safe" markers. The summary gives the range within 2 pts and labels the knob forgiving, moderately sensitive or sensitive. Use best / Use safest snap to the slider grid. An optional ± 3 seeds band (2 extra trainings, cached) exposes single-seed peaks that are luck. Covers every range knob plus an A↔B balance for weighted methods with 2 experts. Helpers in `js/lab.js`, 2 tests.
- 2026-11-21: **Advisor evidence:** each of the top-3 advisor picks is measured on the closest toy scenario across 3 seeds (default knobs, aligned if needed), showing mean ± sd and worst task. A verdict says whether the measurements agree with the rules, are too close to call, or disagree. Results are cached per scenario, and the lab's trained experts are reused when they match. Training loops now pause based on elapsed time (≤ ~0.1 s UI stalls). `toyConfig`/`seedRuns` in `js/lab.js`, 1 test.
- 2026-11-21: **Shareable recipe links:** the address bar keeps `#r=…` in sync with the current recipe (experts, training settings, method, knobs, alignment, pinned comparison). 🔗 Share copies it, with clipboard fallbacks. Opening or pasting a link restores the recipe, retraining only when the experts or training settings differ. Decoding is strict: unknown methods, skills or regions are rejected, and knobs are clamped. `js/share.js`, 2 tests. Also fixed an uncaught error when switching to a scenario with more experts.
- 2026-11-21: **Recipe advisor** (topbar ✨ and Atlas): 7 questions about your real models (shared base, same skill or different, count, input overlap, data, training budget, serving). Live top-3 recipes with reasons and caveats, other options that fit, ruled-out methods with why, and situation notes. Opens a pick in the Lab on the closest scenario (with alignment if needed) or in Export. Pure rules in `js/advisor.js`, with 4 tests.
- 2026-11-21: **Lab compare mode:** pin a recipe as A, tweak the method or knobs, and see both maps side by side with per-task "vs A" deltas, a B-vs-A box (accuracy, retention, % of predictions that differ, inference cost, requirements, seed-noise caution) and Swap. The pin survives scenario changes and shows a lock if it can't run.
- 2026-11-21: **Multi-seed robustness check** in the Arena (5 seeds, mean ± sd, per-seed dots, wins,
  stability, "no clear winner" verdict, free-merge pick that prefers steadier options).
- 2026-11-21: **Headless engine** (`js/lab.js`) shared by the UI, the tests and the robustness runner.
- 2026-11-21: **Test suite + CI** (`tests/suite.js`, `tests/run.js`, GitHub Actions; also checks that
  `dist/` is rebuilt).
- 2026-11-21: Fixed Arena charts overflowing at phone width.
