# ModelMixer improvement backlog

Audience: **practitioners picking a merge recipe** for real checkpoints (then exporting to mergekit).
Each improvement loop takes the top unchecked item, ships it (tests green, bundle rebuilt, no console
errors, no horizontal scroll at 375px) and moves it to *Done* with the date.

## How to verify an iteration

- Accessibility: keep axe-core at 0 violations on every view (Lab, Arena, Atlas, Export, advisor) and keep every new control labelled and keyboard reachable.

- `node tests/run.js` (CI runs this too). Without Node: `python3 scripts/bundle.py --tests`, then open
  `tests/_browser.html` and expect `N/N passed`.
- `python3 scripts/bundle.py`, then open `dist/modelmixer.html`, exercise the changed feature, and check
  the console for errors.

## Next up (highest value first)

1. **Import a mergekit config:** paste a mergekit YAML (linear, slerp, ties, dare_ties/dare_linear, della, sce, breadcrumbs, task_arithmetic, karcher, gradient lists) and load it as a Lab recipe, the reverse of Export. Practitioners can then sanity-check a real recipe's knobs on the toys.
2. **Web Worker engine:** run training, sweeps and seed checks in a worker so the UI never stalls, even in background tabs that throttle timers.
3. **Newer methods, part 3:** Localize-and-Stitch (learned sparse masks per task), plus an AdaMerging variant with a confidence-calibrated objective.
4. **Digits extras:** a conflict scenario (0-vs-1 with 1-vs-7, where both claim the "1"s), a per-pixel saliency view of what each merge kept, and digit-aware advisor evidence.
5. **Phone layout check:** re-verify the newer panels (heat map, head-to-head, compare box, advisor) at 375 px and fix any overflow.

## Done

- 2026-11-22: **Accessibility pass** (prioritised because the UI had grown a lot and "great UI" was a core requirement). The axe-core audit of every view (Lab, Arena, Atlas, Export, advisor) went from **7 failing rules to 0**:
  - **Labels:** every slider, select and input is labelled.
  - **Formula box:** the scrollable box is keyboard-focusable.
  - **Landmarks and headings:** labelled landmarks, an `h1` for the Lab and a heading order with no skipped levels.
  - **Contrast:** `--dim` text raised from 3.2:1 to 4.8:1 or better, and rank numbers too.

  Keyboard support:
  - a visible `:focus-visible` ring;
  - WAI-ARIA tabs with arrow / Home / End keys and a single tab stop;
  - arrow keys in the method picker, with focus kept on the chosen chip after it re-renders (this was a real bug: focus fell back to the page);
  - `aria-pressed` kept in sync on every toggle by a MutationObserver.

  Also: polite live regions for status and toasts, a spoken description of the decision map (method and per-task accuracy, plus the A ≠ B note), and `prefers-reduced-motion` support. Verified with real key presses in headless Chrome.
- 2026-11-22: **Per-layer heat map.** When per-layer values (mergekit gradients) are on, a 7×7 map shows average accuracy for every combination of two layers' values (pairs: H1×H2, H1×Out, H2×Out; the third layer is held fixed). It marks the current, best and safest cells, where "safest" means an interior cell with the best worst neighbour, because edge cells would win on having fewer neighbours. A summary names the layer the recipe is sensitive to (average swing per axis), and Use best / Use safest snap the values to the slider grid. It computes one row per frame, about 0.2 s for SLERP. Findings: SLERP is sensitive to Hidden 1 (≈17 pts vs 2 for Output), and "Use best" lifts it from 85.8% to 94.0% with t = [0.17, 0.5, 0.83]. TIES λ barely matters for Hidden 2, but λ = 0 on Hidden 1 costs ≈15 pts. `layerGrid` / `pickGrid` in `js/lab.js`, 1 test (grid corners = Layer Stitch exactly; 53 total).
- 2026-11-22: **Advisor blending.** Once the top-3 picks are measured on the closest toy (3 seeds), a lower pick that beats the rules' #1 under the paired test (significant, or winning every seed) moves to #1. It gets an "↑ measured" badge and the primary "Try it" button, and the old #1 keeps a "rules' #1" tag. Otherwise the rule order stands, and the verdict line explains which case applied. Example: with no shared base and any serving allowed, Karcher beats the rules' Ensemble on 3/3 seeds (+8.0 pts) and is promoted; with default answers, measurements agree and nothing moves. `blendRanking` in `js/advisor.js`, 1 test (52 total). The advisor screenshot now shows the re-ranked case.
- 2026-11-22: **Compare extras** (moved ahead of more methods: with 17 methods, helping people choose between two matters more than an 18th). In compare mode the B map shades where A and B predict different classes (amber, inside the skill regions) and draws A's boundary dashed. Test points (plane) or digit tiles where the recipes disagree get an amber marker, and an "A ≠ B" toggle and legend are added. "Is B's edge real?" runs both recipes on the same 5 seeds and gives the paired verdict, with the same wording as the Arena head-to-head (shared `pairVerdict`). Extra-seed trainings now live in one cache shared by the sweep band, the Arena check and the edge check, so re-checks are much faster (Arena 5 seeds after an edge check: 6 s instead of about 18 s; sweep band: 0.4 s instead of about 8 s). Example: DARE vs TIES on Two skills is "no reliable difference" (+0.4 pts, 3 vs 2). `recipeRuns` in `js/lab.js`, 1 test (51 total).
- 2026-11-22: **SCE and AdaMerging** (17 methods). SCE selects high-variance entries, weights experts by energy and erases minority signs. Its default top-k is 0.7 from the 9-bundle grid (88.3 ± 6.3, worst 81.0 vs top-k 1: 88.5 ± 8.3, worst 76.1); it exports to mergekit `sce` and supports per-layer λ. AdaMerging learns task-wise (K) or layer-wise (K×L) λ by minimising prediction entropy on unlabeled inputs, using finite-difference Adam over 40 steps. Export writes the learned λs as a mergekit task-arithmetic config with per-layer weight lists. Toy finding: it reliably lowers entropy (layer-wise 0.048 → 0.007) but trails Task Arithmetic on average (80–81 vs 84.5 over 9 bundles), because entropy rewards confident mistakes. Tests: task-wise AdaMerging equals Task Arithmetic with the learned λs exactly, entropy decreases, and SCE selects the requested fraction. At top-k 1 SCE skips the entries where experts agree exactly (17% here). 50 tests.
- 2026-11-21: **Head-to-head + seed count + CSV** in the Arena's multi-seed check. Choose 3, 5 or 10 seeds. Pick any two methods to see per-seed A − B dots, a ±1 sd band and a verdict from a paired t-test plus sign consistency. Verdicts: beats / leans ahead (never loses, but not yet significant: run 10 seeds) / no reliable difference / same, and borderline results are flagged as tentative. The default pair is the two best free merges. Download CSV exports every seed × method row. Found on Two skills: at 5 seeds Breadcrumbs only *leans* ahead of DARE (t 1.88), and at 10 seeds it beats DARE (9/10, t 3.33) and TIES (10/10, t 5.78). `pairedDiff` / `seedRunsCSV` in `js/lab.js`, 2 tests (46 total; the earlier "46" in the DELLA commit message was a miscount of 44).
- 2026-11-21: **DELLA and Karcher mean** (15 methods). DELLA uses magnitude-ranked keep probabilities (MagPrune), rescaling and TIES sign election; its defaults (density 0.8, ε 0.4) come from a 9-bundle grid, giving 88.0 ± 3.5 (worst 83.6), on par with TIES, versus 71.4 ± 15.1 at density 0.3. The dropout seed alone moves it by ~3 pts. Karcher is the iterative Fréchet mean on the sphere per tensor (SLERP for K models), with norms kept. On the toy it trails Linear (3 experts: 81.6 vs 90.5). Both export to mergekit (`della`, `karcher`), DELLA supports per-layer λ, and the advisor knows both. Tests: DELLA(ε 0, k 1) = TIES(k 1) exactly, and Karcher(K=2) = the closed-form direction midpoint. Note: SLERP's midpoint differs from Karcher's when norms differ, because SLERP blends raw vectors. Screenshots regenerated.
- 2026-11-21: **Hosted demo + README screenshots** (owner approved): a GitHub Pages workflow publishes the single-file build as the site index, so share links become https URLs (Pages must be set to “GitHub Actions” in the repo settings once). The README has a live-demo link and 7 screenshots (WebP, under 1 MB total) captured by `scripts/screenshots.py` with Playwright. **When a UI change alters a screenshotted view, re-run that script.**
- 2026-11-21: **Digits scenario (image-like task):** procedurally generated 8×8 digits (5×7 glyphs with random shift, stroke dropout, blur and pixel noise). Expert A tells 0 from 1 and B tells 2 from 3; the menu offers 6 digit pairs including a 1-vs-7 pair for conflicts. Models take 64 raw pixels (about 3.2k params), the base is pre-trained on random pen strokes, and the router and distillation work on any input size. The map becomes a gallery of test digits (tint = predicted class, red ring = mistake, per-task accuracy) in the Lab, pin, thumbnails and Arena. Plane and digit skills can't be mixed in one recipe (UI and share links enforce this). Ranking differs from the plane toy: Distill 99.4, MoE 98.3, RegMean 96.5, Fisher 92.9, Linear 91.1, TIES 88.2, TA 86.8, DARE 74.7, Stitch 73.8. 3 tests.
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
