# ARCHITECTURE — embtoconcept + steering, one system

Unified technical documentation for the two layers of this project, written as the
foundation for the **automatic judge-test suite** that comes next. Companion files:
[`README.md`](README.md) (tour), [`RECIPE.md`](RECIPE.md) (rebuild runbook),
[`PUBLICATION.md`](PUBLICATION.md) (the why), [`MINDMAP.html`](MINDMAP.html)
(interactive map of everything below), and
[`steering/vectors/scorecard.md`](steering/vectors/scorecard.md) (current verdicts).

---

## 1. The system in one diagram

```
 concept list
   │  10_define_dimensions.py
   ▼
 master_dimensions_prompts.json  (the 104 rulers)
   │  20_score_concepts.py   ← LLM as calibrated measuring instrument (slow)
   ▼
 humanized_embeddings_dataset.json ──30_normalize.py──► …_norm.json
   │  40_build_training_data.py
   ▼
 dataset_X (5578×384)  +  dataset_Y (5578×104)  +  dataset_metadata.json
   │  50_train_translator.py / 51_train_reverse.py
   ▼
 semantic_translator.pth (384→104)        reverse_translator.pth (104→384)
   │                                            └─ semantic_console.py
   ├──► 70_build_table ──► 80_build_faiss_index ──► 90_query_explorer 🎮
   │
   └──► steering/translator.py  (Humanizer — THE JUDGE)
            │
            │   steering/ — proving the axes are causal
            │   ────────────────────────────────────────────────────
            │   generate_stimuli  → caa_stimuli.json
            │   derive_vectors    → control_vectors_caa_white.npy (8×104×4096)
            │   geometry          → effective-rank gate
            │   steering_model    → SteeredLlama (forward hooks; layer 15, clamp)
            └─► fidelity          ⟲ closed loop per dimension (judged above)
                analyze           → scorecard.md (26/104 causal dials)
                tuner / night_run → unattended sweeps
                console / ui / demo / sweep → cockpits
```

The two layers share three contracts: `semantic_translator.pth` (the model),
`dataset_metadata.json` (the pinned dimension order), and `dataset_Y_human.npy`
(the human table, used for pole selection and kinship). Anything that breaks one of
these breaks both layers — they are the prime targets for contract tests (§6).

---

## 2. Layer 1 — embtoconcept root: building the 104-axis space

### 2.1 Live pipeline

| Stage | Script | In | Out | Notes |
|---|---|---|---|---|
| Define axes | `10_define_dimensions.py` | `DIMENSIONS_DB` (in-file) | `master_dimensions_prompts.json` | 104 axes, 9 families, two absolute anchors each; prompt frames the LLM as a "calibrated semantic measuring instrument" |
| Score concepts | `20_score_concepts.py` | prompts + concept list | `humanized_embeddings_dataset.json` | Llama-3-8B local, temp ≈0.1, one dim per call, incremental save; 5,654 concepts |
| Normalize | `30_normalize.py` | dataset | `…_norm.json` | percentile clip 5–95, robust IQR scale into [0.05, 0.95]; flags dead/dominant/correlated (|r|>0.85) axes |
| Tensors | `40_build_training_data.py` | dataset (+norm) | `dataset_X_embeddings.npy`, `dataset_Y_human.npy`, `dataset_metadata.json` | X = MiniLM 384d; Y = 104 scores **sorted by dim id**; 5,578 surviving pairs |
| Train forward | `50_train_translator.py` | X, Y | `semantic_translator.pth` | MLP 384→512(BN,ReLU,Do.1)→256(ReLU,Do.1)→104(Sigmoid); MSE, Adam 1e-3, 80/20 |
| Train reverse | `51_train_reverse.py` | X, Y | `reverse_translator.pth` | 104→384, lets you author a profile and search the corpus near it |

### 2.2 Corpus & exploration

| Script | Role |
|---|---|
| `60_download_corpus.py` | ~5M terms: ConceptNet 5.7, Wikipedia titles (ES+EN), Wikidata labels, OpenSubtitles → `corpus/corpus_final.txt`, deduped |
| `70_build_table.py` | corpus → MiniLM → translator, batched, memmap output (`tabla_embeddings.npy` + `tabla_words.json` + `tabla_stats.npy`), checkpoint/resume |
| `80_build_faiss_index.py` | FAISS index over the humanized table |
| `90_query_explorer.py` | interactive: neighbours, vector arithmetic (`+ king woman - man`), interpolation, top-by-dimension, profiles, polygraph (`pg t1 \| t2`), anomaly detector |

### 2.3 Experiments and legacy (kept, not load-bearing)

`legacy/index.py` / `legacy/index2.py` are earlier scorers superseded by `20_score_concepts.py`.
`legacy/semantic_attention_injection.py` (v3) and `legacy/cognitive_generator_v4.py` (Tkinter GUI)
are the **logit-bias era** of steering — via llama-cpp logits or Ollama prompt bias —
historically interesting but replaced by true activation steering in `steering/`.
`semantic_console.py` is a forward+reverse console over both `.pth` models.
`legacy/build_global_dictionary.py` builds a 20k frequency-word embedding dictionary.
`conceptualizador/` holds the 4096-concept "semantic universe" generators with three
interchangeable scoring backends (Ollama-Llama, GPT, Anthropic). `check_variance.py`
and `debug_data.py` are dataset vital-sign checks — small, deterministic, and ideal
seeds for L1 contract tests (§6).

---

## 3. Layer 2 — steering/: proving the axes are causally real

A readable score is not proof an axis exists inside a model. `steering/` derives one
control vector per dimension inside Llama-3.1-8B and closes the loop: push the dial,
generate, measure with your own Humanizer whether the text moved on that axis.

### 3.1 Configuration (`config.py` — single source of truth)

| Knob | Value | Meaning |
|---|---|---|
| `TARGET_LAYERS` | 12–19 | layers where vectors are **derived** (a wide band, once) |
| `INJECT_LAYERS` | (15,) | layers where injection actually happens; validated 2026-06-11: one layer + clamp survives α≈0.35 where three layers broke at ≈0.25 |
| `ALPHA` | 0.12 | **fraction of the residual norm** at each position — push is prompt-independent |
| `STEER_MODE` | "clamp" | P-controller on the projection: `h += KP·(α·|h| − h·v)·v`. Pushes only what is missing to reach the setpoint; where the model already complies, adds nothing (this is what "add" mode broke grammar with). KP=1 pins the projection (Golden-Gate style) |
| `DERIVE_MODE` | "caa" | Contrastive Activation Addition between the two poles of each axis (vs the older confounded "concepts" mode) |
| `WHITEN` | True | per-coordinate z-score against `N_REF=800` reference concepts; neutralizes Llama's massive-activation outlier coordinates that collapsed all 104 directions onto one axis |

Artifact naming is derived from the mode: `control_vectors{_caa}{_white}.npy` —
a convention worth a contract test, since stale-name loads fail silently.

### 3.2 Derivation chain

1. `generate_stimuli.py` — Llama itself writes ~24 varied sentences per pole per
   dimension (cached, hand-editable `vectors/caa_stimuli.json`). Rich sentences give a
   **semantic** direction; short templates give a lexical one that breaks language
   before evoking meaning.
2. `derive_vectors.py` — runs pole stimuli through the model, captures the residual
   stream at layers 12–19, `mean(high) − mean(low)` per dim per layer, whitens,
   unit-norms → `control_vectors_caa_white.npy` with shape `(8, 104, 4096)`.
3. `geometry.py` — health gate: effective rank (participation ratio + entropy rank)
   of the 104 directions. ~1–2 = collapse, ~15–18 = healthy (close to the human
   table's own rank).

### 3.3 Injection engine

`steering_model.py` (`SteeredLlama`) registers forward hooks on every derived layer.
A 104-profile becomes one 4096-d vector per layer: `steering = Σᵢ (pᵢ − 0.5)·cvᵢ` —
0.5 is neutral by construction, so a flat profile must produce exactly zero steering
(the critical invariant, tested in `verify_mechanics.py` and `smoke_test.py` [B]).
`translator.py` wraps MiniLM + `semantic_translator.pth` as `Humanizer`
(text → 104 profile) — the same artifact from Layer 1, now serving as **the judge**.

### 3.4 Measurement chain — the existing automatic judge

This is the part the upcoming judge-test suite will formalize.

| Tool | What it measures | Output |
|---|---|---|
| `fidelity.py` | per dim: generate with the dim at 0.95 and 0.05 (isolated, greedy, same domain-matched probe), judge both texts with Humanizer, effect = profile(hi) − profile(lo) across **all** 104 dims. Resumable; auto-archives reports from older probe/vector versions | `fidelity_report.json` |
| `analyze.py` | kin-aware scorecard: estimates the judge's common mode only over dims **not** correlated (|r|>0.4 in `dataset_Y`) with the one under test, so divinity doesn't pay for sharing soul with religiosity. Yields `z_kin`, `rank_kin`, sign | `scorecard.md` |
| `tuner.py` | fidelity swept at α ∈ {0.15, 0.25, 0.35}, one subprocess per α (clean VRAM), merged into a per-dim tuning table | `fidelity_aXXX.json`, `afinacion.md` |
| `harmony.py` | resonance hypothesis: alignment (does the prompt already live in that territory?) vs effect vs quality (relative perplexity under the clean model). If confirmed, the α ceiling is a function of prompt–dial agreement, not a constant | console report |
| `night_run.py` | unattended chain stimuli → backup → derive → geometry → fidelity; each step a separate process; keep-awake on Windows; full log | `vectors/night_*.log` |

**Pass criterion** (current): `z_kin ≥ 1.5` with correct sign → **26 of 104 dims are
real causal dials**. Top: necessity +4.2, speed of action +3.6, curiosity +3.6,
anger +3.2, hardness +3.1. The weak tail is fine-grained physics — the same axes the
text model scored noisily in Layer 1, which is the design-feedback loop working.

### 3.5 Mechanics tests (already in the repo)

| Test | Needs | Asserts |
|---|---|---|
| `verify_mechanics.py` | numpy only | control-vector matrix shapes; per-layer steering shapes; **flat 0.5 profile ⇒ zero steering**; maximizing dim i ⇒ steering aligned with cvᵢ |
| `smoke_test.py` | CPU ok (SmolLM2-135M, same architecture) | [A] hook attaches and modifies the residual; [B] steering=0 ⇒ output **identical** to neutral; [C] steering≠0 changes generation; [D] full profile→cv→inject path runs on a real model |
| `blind_test.py` | GPU + human | anti-pareidolia: random dims, unlabeled outputs, human matches letter→dim, scored against chance |

### 3.6 Cockpits

`console.py` (ChronoLLMPuppet: validated presets + 4 note slots, a note =
`(dim, value, layer, alpha)` so chords can be voiced across layers), `ui.py`
(104 live sliders), `demo.py` / `sweep.py` (CLI neutral-vs-steered, multi-α sweeps,
`--voice` notes).

---

## 4. Artifact inventory (the contracts)

| Artifact | Shape / content | Produced by | Consumed by |
|---|---|---|---|
| `master_dimensions_prompts.json` | 104 objects: id `d000…d103`, name, anchors, prompt | `10_define_dimensions.py` | `20_score_concepts.py`, `index*.py` |
| `humanized_embeddings_dataset.json` | {concept → {dim id → score}} ×5,654 | `20_score_concepts.py` | `30_normalize.py`, `40_build_training_data.py` |
| `dataset_metadata.json` | concepts + `dimension_names` (pinned order) + model name | `40_build_training_data.py` | everything downstream — **the** shared contract |
| `dataset_X_embeddings.npy` | (5578, 384) float | 〃 | trainers, `semantic_console.py` |
| `dataset_Y_human.npy` | (5578, 104) in [0,1] | 〃 | trainers, `derive_vectors.py` (poles), `analyze.py` (kinship) |
| `semantic_translator.pth` | MLP 384→512→256→104 | `50_train_translator.py` | `70_build_table.py`, `90_query_explorer.py`, `steering/translator.py` |
| `reverse_translator.pth` | MLP 104→384 | `51_train_reverse.py` | `semantic_console.py` |
| `steering/vectors/caa_stimuli.json` | {dim → {pos: […], neg: […]}} | `generate_stimuli.py` | `derive_vectors.py` |
| `steering/vectors/control_vectors_caa_white.npy` | (8, 104, 4096) unit-norm | `derive_vectors.py` | `steering_model.py`, `geometry.py` |
| `steering/vectors/control_meta_caa_white.json` | target_layers, dim_names, mode | 〃 | 〃 |
| `steering/vectors/fidelity_report*.json` | per dim: effect_vec (104), texts, params, probe version | `fidelity.py` / `tuner.py` | `analyze.py` |
| `steering/vectors/scorecard.md` | ranked verdicts, 26/104 green | `analyze.py` | humans + regression baseline |

---

## 5. Design decisions worth remembering

**The judge is the system itself.** Fidelity uses the Layer-1 translator to grade
Layer-2 steering. That closes the loop cheaply, but couples the two layers: a judge
bias looks like a steering failure. `analyze.py`'s kin-aware common-mode correction
exists precisely to manage this — and any future judge tests must keep judge error
and dial error separable (e.g., by also tracking the judge's response to *unsteered*
text, which fidelity already stores as the full effect vector).

**Everything long-running is resumable and version-guarded.** `20_score_concepts.py` saves
incrementally; `70_build_table.py` checkpoints; `fidelity.py` skips done dims and
self-archives stale reports; `night_run.py`/`tuner.py` isolate steps in subprocesses
so VRAM leaks and crashes don't cascade. The judge-test runner should inherit this
pattern.

**Determinism where it matters.** Fidelity and sweep generate greedy, same probe for
both poles, so the only changing variable is the dial. Tests can rely on this.

**Honesty as a feature.** Weak axes aren't hidden; the scorecard is the deliverable.
The judge-test suite extends this: a red test on d019_radiactividad is information,
not failure.

---

## 6. Test surface — inventory for the automatic judge-test sequence

> **Implemented 2026-06-11 in [`tests/`](tests/)** (pytest). `python -m pytest` runs
> L0+L1; L2/L3 are gated behind `RUN_L2=1` / `RUN_L3=1` (never run L3 while
> tuner/night_run holds the GPU). L3 sentinels run each dim at its tuner-optimal α
> and include the dead-control calibration check plus a test-retest reproducibility
> probe. `fidelity.py` now stamps `vectors_sha` (content hash) into `_meta`, so
> reports from re-derived vectors with the same filename can never be compared
> silently. Known data bug tracked by L1: 31 values > 1.0 in `dataset_Y_human.npy`
> (max 6.54, concentrated in d000/d004/d005) — `xfail` strict test + no-regression
> baseline until the parser clamp is fixed and those dims re-derived.

Ordered from free/instant to expensive/GPU-bound. Each level gates the next; a judge
run should stop at the first failing level (no point burning GPU on broken algebra).

### L0 — Pure math invariants (numpy, milliseconds, CI-friendly)

- Flat profile (all 0.5) ⇒ steering vector exactly 0 per layer (`verify_mechanics` logic, as a pytest).
- `steering = Σ(pᵢ−0.5)·cvᵢ` shapes: (104,) profile × (104, H) cv → (H,) per layer.
- Pushing only dim i ⇒ cosine(steering, cvᵢ) ≈ 1 on synthetic vectors.
- Clamp-mode setpoint math: if `h·v` already equals `α·|h|`, the correction term is 0; with KP=1 the post-injection projection equals the setpoint.
- `effective_rank` (geometry.py): known-rank synthetic matrices return the expected PR/entropy rank.
- `analyze.scorecard` as a pure function: feed a fixture `fidelity_report.json` → assert z_kin/rank/sign outputs; kinship masking actually excludes dims with |r|>0.4.
- Domain router `fidelity._domain()`: every one of the 104 names maps to a domain, no typo falls silently into the "mente" default for a name that has a real family.

### L1 — Artifact contract tests (no model, seconds)

- `dataset_metadata.json`: 104 `dimension_names`, sorted `d000…d103`, len(concepts) == Y rows.
- `dataset_Y_human.npy`: shape (·, 104), all values in [0, 1], no NaN; X shape (·, 384); X rows == Y rows.
- `master_dimensions_prompts.json`: 104 entries, each with both anchors and a prompt demanding a single number.
- Control vectors: shape (len(target_layers), 104, H); unit norms within tolerance; meta `dim_names` == dataset `dimension_names` (the silent-mismatch killer).
- Config naming convention: `CONTROL_VECTORS_FILE` exists and matches `DERIVE_MODE`/`WHITEN`; `INJECT_LAYERS ⊆ TARGET_LAYERS`.
- `caa_stimuli.json` coverage: every dim has non-empty pos and neg pools (night_run's `stimuli_coverage()` as an assertion).
- `semantic_translator.pth` loads into the architecture defined in `translator.py` (state-dict keys match — catches accidental retrains with a different topology).

### L2 — Small-model mechanics (SmolLM2-135M, CPU ok, ~1 min)

- `smoke_test` invariants [A]–[D] as separate asserts, especially [B]: zero steering ⇒ token-identical generation (the no-contamination guarantee for every neutral baseline used by the judge).
- Hook registration idempotence: `clear()` after `set_profile` restores neutral output.

### L3 — Closed-loop judge on a sentinel subset (GPU, minutes)

- Run `fidelity --only <sentinel dims>` greedy at fixed α on a handful of **validated** dials (e.g. d099 necessity, d053 curiosity, d010 hardness, d023 consciousness) plus one known-dead control (e.g. d019 radioactivity).
- Assert: sentinel z_kin above threshold, correct sign; dead control stays below — proving the judge can still tell a live dial from a dead one (calibration, not just power).
- Judge sanity: Humanizer(neutral probe text) is stable across runs (greedy ⇒ deterministic; embedding/model drift would show here first).

### L4 — Scorecard regression (GPU, hours — the night_run tier)

- Full fidelity sweep, then diff `scorecard.md` against the previous: no validated dial drops out of green without an intentional change to vectors/config; record per-dim z_kin deltas.
- Geometry gate: effective rank of freshly derived vectors ≥ threshold (collapse detector) before fidelity even starts — this ordering is already night_run's, keep it.

### L5 — Human-in-the-loop (optional, anti-pareidolia)

- `blind_test.py` sessions, scored vs chance; not automatic, but the periodic honesty audit on top of the automatic levels.

---

## 7. Glossary

| Term | Meaning here |
|---|---|
| **Humanizer / judge** | MiniLM + `semantic_translator.pth`: text → 104-dim profile |
| **Control vector** | unit direction in Llama's 4096-d residual space meaning "more of axis i" at layer L |
| **CAA** | Contrastive Activation Addition: vector = mean activation difference between pole-stimuli pools |
| **Whitening** | per-coordinate z-score vs reference concepts; kills massive-activation outliers |
| **Clamp** | P-controller injection toward a projection setpoint `α·|h|` instead of blind addition |
| **Note / voicing** | a `(dim, value, layer, alpha)` tuple; placing notes on different layers so they don't fight |
| **z_kin / rank_kin** | effect z-score and rank after subtracting the common mode of non-kin dims |
| **Dial** | an axis proven causal: pushing it moves its own dimension more than anything else |

---

*Part of the **Chrono** family. Generated 2026-06-11 alongside [`MINDMAP.html`](MINDMAP.html).*
