# USAGE — how to actually drive this thing

Copy-paste commands for every tool in the repo. Background reading:
[`README.md`](README.md) (what this is), [`ARCHITECTURE.md`](ARCHITECTURE.md)
(how it fits together), [`RECIPE.md`](RECIPE.md) (rebuild the 104 dimensions
from scratch), [`SHOWCASE.md`](SHOWCASE.md) (what steering looks like).

All commands run from the repo root (`embtoconcept/`). Anything that loads
Llama-3.1-8B wants a GPU (≈11 GB in 4-bit); everything else is CPU.

---

## 0. Setup

```bash
pip install torch sentence-transformers faiss-cpu numpy pandas tqdm matplotlib
pip install transformers accelerate bitsandbytes   # steering (HF Llama, 4-bit)
pip install pytest                                 # test suite
```

Llama-3.1-8B is gated on HF: accept the license + `huggingface-cli login`,
or point `LLM_NAME` in `steering/config.py` at an open mirror
(`NousResearch/Meta-Llama-3.1-8B-Instruct`).

---

## 1. The 60-second tour

```bash
python -m pytest                      # is everything healthy? (no GPU, seconds)
python -m steering.console            # GUI: push validated dials, watch text move
python -m steering.showcase --only genesis --lang both    # one curated example
```

---

## 2. Steering — play 🎮

**The console (ChronoLLMPuppet)** — presets + 4 note slots. A *note* is
`(dim, value, layer, alpha)`; notes on different layers = voicing (chords
that don't fight). Model loads in the background.

```bash
python -m steering.console
```

**104 live sliders** over one phrase, neutral vs steered side by side:

```bash
python -m steering.ui
```

**CLI quickies:**

```bash
python -m steering.demo --dim 23 0.95              # one dial, one phrase
python -m steering.sweep                           # one phrase, several alphas
python -m steering.sweep --dim 90 0.95 --isolate --alphas 0.15 0.30 0.45
python -m steering.sweep --voice 90 0.95 14 0.35 --voice 23 0.95 16 0.20
                                                   # voicing: DIM VAL LAYER [ALPHA]
```

Rules of thumb (from `afinacion.md`): every dial has its own best alpha;
strong dials (necessity z=+4.2) want a *third* of the pressure of weak ones;
above α≈0.45 expect morphology to crack; resonant prompts raise the ceiling.

**The reproducible gallery** (writes `SHOWCASE.md`):

```bash
python -m steering.showcase                        # full, both languages (~35 min)
python -m steering.showcase --lang es --only verbo acorde_suave
```

**Blind test** — kill your own confirmation bias: random dials, unlabeled
outputs, you match letter→dimension, scored against chance:

```bash
python -m steering.blind_test --k 4 --alpha 0.15
```

---

## 3. Steering — measure 🔬

Full derivation chain (only needed after changing stimuli, layers or whitening):

```bash
python -m steering.generate_stimuli            # rich pole sentences -> caa_stimuli.json
python -m steering.derive_vectors              # CAA + whitening -> control_vectors_*.npy
python -m steering.geometry                    # health gate: effective rank (~15-18 good, ~1 collapsed)
```

Closed-loop verification (the judge):

```bash
python -m steering.fidelity                    # all 104 dims, resumable
python -m steering.fidelity --only 96 10 19 --alpha 0.15    # just these dims
python -m steering.fidelity --only 28 --show-text --redo    # autopsy with eyes
python -m steering.analyze                     # kin-aware scorecard -> vectors/scorecard.md
python -m steering.analyze --z-real 2.0        # stricter verdict threshold
```

How to read a fidelity row: `z` = how much the pushed dim moved vs all others
(σ units); `rank 0` = its own dim moved most (the dial is a dial); `sign INV`
= it moved the wrong way. `analyze` subtracts the judge's common mode using
only non-kin dims (kinship = |r|>0.4 in the human table), so families don't
pay for shared souls.

Unattended sweeps (each step in its own process, VRAM-safe, keeps Windows awake):

```bash
python -m steering.night_run                   # stimuli -> derive -> geometry -> fidelity
python -m steering.night_run --skip stimuli derive
python -m steering.tuner                       # fidelity at α=0.15/0.25/0.35 -> afinacion.md
```

Mechanics sanity (no real vectors needed):

```bash
python -m steering.verify_mechanics            # numpy-only algebra invariants
python -m steering.smoke_test                  # real hooks on SmolLM2-135M (CPU ok)
```

---

## 4. Tests — the judge-test suite ✅

Four gated levels (see `ARCHITECTURE.md` §6). Levels gate each other: broken
algebra makes GPU results meaningless, so start cheap.

```bash
python -m pytest                       # L0 math + L1 artifact contracts (seconds)
set RUN_L2=1 && python -m pytest       # + small-model hooks & judge determinism (~2 min)
set RUN_L3=1 && python -m pytest tests/test_l3_sentinels.py
                                       # + GPU sentinels (~15 min) — NEVER while
                                       #   tuner/night_run holds the GPU
```

What green means: L0 — the steering algebra keeps its promises (flat profile
⇒ zero push; chords don't crosstalk). L1 — no artifact silently broke its
consumers (dim order pinned, vectors unit-norm, reports carry their regime
manifest). L2 — hooks really modify the residual stream and the Humanizer
judge is deterministic. L3 — validated dials still beat their thresholds at
their tuner-optimal alphas, the dead control (radioactivity) stays dead, and
test-retest reproduces `effect_vec` exactly.

Expected: 1 `xfail` in L1 — the known Y>1.0 data bug (31 legacy values),
tracked with a no-regression baseline until the parser clamp is fixed.
Sentinel reports land in `tests/_artifacts/`.

---

## 5. The dimension pipeline (rebuild from scratch)

One command per stage; full runbook with checkpoints in [`RECIPE.md`](RECIPE.md).

```bash
python 10_define_dimensions.py     # 104 axes + anchors -> master_dimensions_prompts.json
python 20_score_concepts.py        # LLM-as-instrument scoring (slow, GPU, resumable)
python 30_normalize.py             # robust scaling + dead/correlated-axis report
python 40_build_training_data.py   # X (5578x384) + Y (5578x104) + pinned metadata
python 50_train_translator.py      # -> semantic_translator.pth  (the deliverable)
python 51_train_reverse.py         # -> reverse_translator.pth   (104 -> 384)
python 55_eval_translator.py       # MAE/MSE eval of the forward map
```

## 6. Corpus & explorer

```bash
python 60_download_corpus.py                   # ConceptNet + wikis + subs -> corpus_final.txt
python 70_build_table.py                       # corpus -> 104-dim table (memmap, resumable)
python 80_build_faiss_index.py                 # nearest-neighbour index
python 90_query_explorer.py --interactive      # the toy
```

Explorer commands: `priest` (neighbours) · `+ king woman - man` (arithmetic) ·
`~ science mysticism` (interpolate) · `! religiosity` (top by axis) ·
`? artificial intelligence` (full profile) · `pg t1 | t2` (polygraph).

---

## 7. Cookbook — "I want to…"

| I want to… | Do this |
|---|---|
| hear one dial right now | `python -m steering.demo --dim 53 0.95` |
| find a dial's best alpha | check `steering/vectors/afinacion.md`; else `python -m steering.tuner` |
| check one suspicious dial with my own eyes | `python -m steering.fidelity --only <idx> --show-text --redo` |
| add/replace an axis | edit `DIMENSIONS_DB` in `10_define_dimensions.py` → rerun stages 10–50 → `night_run` → `pytest` |
| re-derive vectors after editing stimuli | `python -m steering.derive_vectors && python -m steering.geometry` then re-run fidelity (old reports auto-archive: the regime sha changed) |
| know if a text matches its claimed profile | explorer `pg` mode |
| regenerate the public gallery | `python -m steering.showcase` |
| verify nothing broke after ANY change | `python -m pytest` — always, it's free |

---

*Part of the **Chrono** family. If a command misbehaves, the test suite
(`python -m pytest`) tells you which contract broke before you burn GPU time.*
