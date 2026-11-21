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

1. **Recipe advisor:** a short wizard (shared base? a little data? retraining budget? are the tasks
   in conflict? is a single model required at inference?). It recommends 1–3 methods with reasons, opens
   the top pick in the Lab and prefills Export.
2. **Shareable recipe links:** encode scenario, experts, method and knobs in the URL hash, with a
   "Copy link" button.
3. **Knob sensitivity sweep:** for the active method, sweep its main knob (density, drop rate, λ, t) and
   plot accuracy against it, with a ±1 sd band from a 3-seed run. Mark the current setting and the most
   forgiving setting.
4. **Per-layer weights (mergekit gradients):** a layer-wise weight list per expert, as in mergekit
   `value: [0, 0.5, 1]`, exported verbatim to YAML.
5. **Image-like task:** procedurally generated 8×8 "digits", each expert on a different digit pair, so
   merges run on a ~64-dim input model rather than only 2-D toys.
6. **Newer methods:** DELLA (magnitude-adaptive drop), multi-model SLERP / Karcher mean, SCE,
   Localize-and-Stitch, AdaMerging (entropy-tuned coefficients on unlabeled data).
7. **Robustness extras:** a paired-difference view (method X minus method Y per seed), a configurable seed
   count and a CSV download.
8. **Compare extras:** a diff map that highlights where A and B disagree on the plane, and an A-vs-B paired multi-seed check (is B's edge real?).
9. **Accessibility pass:** keyboard focus rings and arrow-key navigation in the method picker, ARIA
   labels on canvases, and a colour-blind-safe palette option.

## Done

- 2026-11-21: **Lab compare mode:** pin a recipe as A, tweak the method or knobs, and see both maps side by side with per-task "vs A" deltas, a B-vs-A box (accuracy, retention, % of predictions that differ, inference cost, requirements, seed-noise caution) and Swap. The pin survives scenario changes and shows a lock if it can't run.
- 2026-11-21: **Multi-seed robustness check** in the Arena (5 seeds, mean ± sd, per-seed dots, wins,
  stability, "no clear winner" verdict, free-merge pick that prefers steadier options).
- 2026-11-21: **Headless engine** (`js/lab.js`) shared by the UI, the tests and the robustness runner.
- 2026-11-21: **Test suite + CI** (`tests/suite.js`, `tests/run.js`, GitHub Actions; also checks that
  `dist/` is rebuilt).
- 2026-11-21: Fixed Arena charts overflowing at phone width.
