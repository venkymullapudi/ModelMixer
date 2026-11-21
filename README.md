# ModelMixer

**An interactive lab for model merging.** Train a few small expert networks in your browser, mix them
with 13 different techniques, and *see* what each mix keeps, loses and needs — then export the same
recipe as a [mergekit](https://github.com/arcee-ai/mergekit) config for real Hugging Face models.

No install, no server, no dependencies: everything (training, merging, evaluation, plots) runs in plain
JavaScript in the browser.

## Run it

Open `dist/modelmixer.html` (a single self-contained file) in any modern browser.

Or work from the sources:

```bash
python3 -m http.server 8000
```

then visit <http://localhost:8000>. Opening `index.html` directly via `file://` also works.
After editing anything in `js/` or `css/`, rebuild the single-file version with:

```bash
python3 scripts/bundle.py
```

## What's inside

| Tab | What it does |
| --- | --- |
| **Mix Lab** | Pick a scenario and experts, choose a mixing method, tune its knobs live. **Pin to compare** freezes a recipe as A so you can tweak B and see both decision maps side by side with per-task deltas. See the merged model's decision boundary, a per-task scorecard against each expert and the base, what the method needs, how it works, its strengths and *missing pieces*, a weight-provenance map and task-vector geometry. |
| **Arena** | Every method on the same experts: leaderboard (accuracy, retention, inference cost, memory, requirements), a **multi-seed robustness check** (5 seeds, mean ± sd, wins and stability, and a verdict on whether the winner is real), a gallery of decision maps, the A→B interpolation path (loss barriers, with and without neuron alignment) and a λ sweep for Task Arithmetic / TIES / DARE. |
| **Share** | 🔗 in the top bar copies a link (`…/modelmixer.html#r=…`) that reopens the exact recipe: experts, training settings, method, knobs, alignment and any pinned comparison. The address bar stays in sync too. |
| **Recipe advisor** | ✨ in the top bar: answer 7 questions about your real models (shared base? data? training budget? how you serve?) and get ranked recipes with reasons, caveats and ruled-out methods. Open a pick in the Lab on the closest scenario, or jump to Export. |
| **Atlas** | A capability matrix (needs base? data? training? single-model inference? tolerates different inits? handles interference?), a "which mix should I reach for" guide, method families and a glossary. |
| **Export** | Generates the matching mergekit (`mergekit-yaml` / `mergekit-moe`) config from your lab recipe, plus pointers to open-source implementations for methods mergekit doesn't cover. |

### Mixing methods

| Family | Methods |
| --- | --- |
| Weight averaging | Linear / Model Soup, SLERP, Model Stock |
| Task vectors | Task Arithmetic, TIES-Merging, DARE (TIES / Linear), Model Breadcrumbs |
| Data-informed | Fisher merging, RegMean |
| Layer surgery | Layer Stitch (Frankenmerge / passthrough) |
| Output mixing | Ensemble, Mixture of Experts (learned router, top-1 or soft) |
| Distillation | Student distilled from an MoE or ensemble teacher |
| Pre-step | Git Re-Basin neuron alignment (weight matching + Hungarian algorithm) |

### Scenarios

- **Two skills / Three skills**: experts fine-tuned from one shared base, each learning a skill in its own region.
- **Overlapping skills**: two skills in the same place that disagree. No mix can satisfy both, and a router can't tell them apart.
- **Strangers**: experts trained from different random inits. Task-vector methods are unavailable and plain averaging breaks until you turn on neuron alignment.
- **Hard + easy**: a hard spiral next to easy blobs, which shows how merges wash out large updates.

You can also change any expert's skill, region, label inversion, network width and fine-tuning length.

## How the toy works

Each expert is a small MLP (radial-basis input encoder → 2 hidden tanh layers → logit). The shared base is
pre-trained on a generic "general knowledge" pattern plus a pinch of each task, and each expert is fine-tuned
from it with SGD+momentum. That keeps task vectors fairly local, which is what makes weight-space merging
meaningful, much like fine-tunes of a large pretrained model. Independent experts are trained from scratch with Adam.

Every number on screen is computed live: accuracies on held-out test points, TIES sign conflicts, Fisher
information, RegMean Gram matrices, router accuracy and alignment permutations.

## Tests

```bash
node tests/run.js
```

The suite checks the Hungarian solver against brute force, that weight matching exactly undoes a neuron
shuffle, that training is deterministic, that every method clears an accuracy floor, and that alignment
helps the Strangers scenario. CI (GitHub Actions) runs it on every push and also fails if
`dist/modelmixer.html` is out of date. No Node? Run `python3 scripts/bundle.py --tests` and open
`tests/_browser.html`.

## Project layout

```
index.html          app shell
css/styles.css      styles (dark UI)
js/nn.js            MLP, RBF encoder, Adam/SGD, Fisher, Gram matrices
js/data.js          tasks, regions, scenarios
js/methods.js       method catalog: how it works, pros, missing pieces, requirements, OSS links
js/merge.js         merge engine + Git Re-Basin + router + distillation
js/lab.js           headless core: train experts, build contexts, score methods, seed summaries
js/advisor.js       recipe advisor rules (situation → ranked methods)
js/share.js         recipe ⇄ URL-safe link encoding with strict validation
js/viz.js           decision-boundary canvases, provenance maps, SVG charts
js/app.js           state, training loop, Lab / Arena / Atlas / Export views
scripts/bundle.py   inlines everything into dist/modelmixer.html (--tests: browser test page)
tests/              engine test-suite (Node runner + browser-compatible suite)
IMPROVEMENTS.md     prioritised backlog
```

## Open-source references

- [mergekit](https://github.com/arcee-ai/mergekit): linear, SLERP, task arithmetic, TIES, DARE, breadcrumbs, model stock, passthrough, MoE
- [FusionBench](https://github.com/tanganke/fusion_bench): benchmark suite including Fisher and RegMean merging
- [git-re-basin](https://github.com/samuela/git-re-basin): permutation alignment
- [dataless-model-merging](https://github.com/bloomberg/dataless-model-merging): RegMean
- [DistillKit](https://github.com/arcee-ai/DistillKit), [TRL](https://github.com/huggingface/trl): distillation

## License

Apache-2.0, see [LICENSE](LICENSE).
