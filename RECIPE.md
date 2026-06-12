# RECIPE — Building the 104 Humanized Dimensions from scratch

A reproducible runbook. Follow it top to bottom and you regenerate the entire
`embedding → 104 human-readable axes` translator. Every step lists its **inputs**,
the **command**, and the **outputs** it must produce before you move on.

This recipe is specific to *this* repo (`embtoconcept`): real file names, the real
local model, the real shapes. Where a number is given (5,654 concepts, 104 dims,
5,578 training pairs), it's what the current artifacts actually contain — treat it as
a checkpoint, not a target to hit exactly.

---

## Ingredients (prerequisites)

```bash
pip install torch sentence-transformers faiss-cpu numpy pandas tqdm openai matplotlib
# GPU build of FAISS optional: pip install faiss-gpu
```

- **A local scoring LLM.** Two interchangeable backends are already wired up:
  - **GGUF (in-process):** `models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` — used by `20_score_concepts.py`.
  - **Ollama (HTTP):** `http://localhost:11434/v1`, e.g. `gemma3:12b` or a Llama-3 tag —
    used by the `conceptualizador/` scripts. Start it with `ollama serve` first.
- **A GPU helps but isn't required.** Scoring is the slow part; the translator trains in
  minutes on an RTX 4090.
- **A concept list** — the words/phrases you want to humanize. The current run uses ~5.6k.

> ⚠️ **Encoding note (Chrono house rule):** these are `.md`/`.py`/`.json` files — plain
> UTF-8, no BOM. The UTF-8-with-BOM rule applies only to C++ sources opened in Visual
> Studio, not here.

---

## Step 1 — Define the 104 axes

**The single most important design step.** Each dimension is a tuple
`(name, anchor_0.0, anchor_1.0)`. The anchors are absolute extremes; they're what make
one concept's score comparable to another's. They live in `10_define_dimensions.py` → `DIMENSIONS_DB`.

```bash
python 10_define_dimensions.py
```

**Out:** `master_dimensions_prompts.json` — 104 objects, each with an `id`
(`d000_…` … `d103_…`), `name`, `scale.min` / `scale.max`, and a ready-to-fire
`llama3_prompt` that frames the model as *"a semantic measuring instrument calibrated
for dimension X"* and demands a single number in `[0,1]`.

**Checkpoint:**
```bash
python -c "import json;d=json.load(open('master_dimensions_prompts.json'));print(len(d),'dims');print(d[0]['llama3_prompt'][:120])"
# expect: 104 dims
```

**Why it's built this way (rationale):**
- *Two anchors, not a verbal scale* → forces the LLM to interpolate on a fixed ruler
  instead of inventing its own, which is what keeps scores comparable.
- *Ten decimals requested* → not because they're meaningful, but to push the model off
  lazy round numbers (0.5, 0.8) and spread the distribution.
- *One dimension per call* → orthogonality. Asking for all 104 at once collapses them
  into a few correlated blobs.

**To add or change axes:** edit `DIMENSIONS_DB` in `10_define_dimensions.py` and re-run. Keep families
grouped (the file is ordered physics → matter → biology → qualia → emotion → cognition
→ society → metaphysics → action) so later diagnostics read cleanly.

---

## Step 2 — Score every concept on all 104 axes (the humanizer)

This is the expensive stage: for each concept, fire all 104 calibrated prompts at the
local LLM and collect 104 numbers. Use `20_score_concepts.py` (in-process GGUF, RTX 4090 path) as the
runner; it loads `Meta-Llama-3-8B-Instruct`, sets a low temperature (≈0.1) for numeric
consistency, and writes incrementally.

```bash
python 20_score_concepts.py
```

**In:** `master_dimensions_prompts.json` + your concept list.
**Out:** `humanized_embeddings_dataset.json` — a dict
`{ concept: { d000_…: 0.0, d001_…: 0.0, …, d103_…: … } }`.

**Checkpoint:**
```bash
python -c "import json;d=json.load(open('humanized_embeddings_dataset.json'));k=list(d);print(len(k),'concepts');print(len(d[k[0]]),'dims each')"
# expect: ~5654 concepts, 104 dims each
```

**Rationale & pitfalls:**
- *Low temperature (0.1)* → you want the instrument repeatable, not creative.
- *Incremental save* → scoring thousands of concepts × 104 calls is hours of work; never
  hold it all in memory and lose it on a crash. The runner appends as it goes.
- *Numeric parsing is the failure point.* The model sometimes returns prose. Strip
  markdown, regex out the first float, clamp to `[0,1]`, and on a hard failure record a
  neutral 0.5 or re-ask — never let a bad parse poison the row.
- *Bias warning:* physical axes (density, viscosity, friction) are where a text model is
  weakest — these are the same axes that later fail steering verification (see
  `steering/vectors/scorecard.md`). Don't be surprised when they come out noisy.

> **Alternative backend:** to score via Ollama instead, mirror the call style in
> `conceptualizador/generar_tabla_de_conceptos_4096_llama.py` (OpenAI-compatible client
> pointed at `localhost:11434`, `response_format={"type":"json_object"}`).

---

## Step 3 — Normalize (robust scale + decorrelate)

Raw LLM scores are lumpy: some axes are dead (every concept ≈ same value), some
dominate, some are near-duplicates. `30_normalize.py` diagnoses and fixes this.

```bash
python 30_normalize.py
```

**In:** `humanized_embeddings_dataset.json`
**Out:** `humanized_embeddings_dataset_norm.json`

What it does, and why:
- **Percentile clipping (5–95)** → kills outlier spikes before scaling.
- **Robust IQR scaling** then squash to `[0.05, 0.95]` → every axis ends up usefully
  spread without saturating at 0/1.
- **Reports dead / dominant / correlated dims** (`|r| > 0.85`) → your signal that two
  "different" axes are really measuring the same thing. Consider merging or re-anchoring
  them back in Step 1.

**Checkpoint:** read the printed `--- AFTER NORMALIZE ---` block; global mean should sit
near the middle of the band and min/max inside `[0.05, 0.95]`. Note any flagged
correlations — that's feedback into your axis design, not just a number.

> Decide here whether downstream steps consume the **normalized** file. If so, point
> `INPUT_FILE` in Step 4 at `humanized_embeddings_dataset_norm.json`.

---

## Step 4 — Build the training tensors (X and Y)

Now pair each concept's *opaque* embedding (X) with its *humanized* profile (Y).

```bash
python 40_build_training_data.py
```

**In:** `humanized_embeddings_dataset.json` (or the `_norm` version).
**What it does:** encodes every concept with
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` → 384-dim **X**; stacks the
104 scores (sorted by dim id, so order is stable) → **Y**.
**Out:** `dataset_X_embeddings.npy`, `dataset_Y_human.npy`, `dataset_metadata.json`
(the latter records `concepts`, `dimension_names`, and the embedding model — keep it,
everything downstream relies on the dimension order it pins down).

**Checkpoint:**
```bash
python -c "import numpy as np;X=np.load('dataset_X_embeddings.npy');Y=np.load('dataset_Y_human.npy');print('X',X.shape,'Y',Y.shape)"
# expect: X (~5578, 384)  Y (~5578, 104)
```
(The slight drop from 5,654 → 5,578 is concepts dropped for empty/failed rows — fine.)

**Why sort dimensions by id:** the translator's output neuron *k* must always mean the
same axis. Sorting `d000…d103` once, here, guarantees that train/infer/steer all agree.

---

## Step 5 — Train the translator

A deliberately small MLP. Bigger nets just memorize 5k rows.

```bash
python 50_train_translator.py
```

**Architecture:** `384 → 512 (BatchNorm, ReLU, Dropout 0.1) → 256 (ReLU, Dropout 0.1)
→ 104 (Sigmoid)`. Loss: MSE. Optimizer: Adam, lr 1e-3. 80/20 train/val split, 1500
epochs. (Edit `50_train_translator.py` to add an early-stop on val loss if you scale the
data up.)

**Out:** `semantic_translator.pth`.

**Checkpoint:** watch `Val Loss` track `Train Loss`. If train keeps dropping while val
flattens or rises, you're memorizing — lower epochs, raise dropout, or add data. The
script prints a real example (predicted vs. true on a held-out concept) at the end;
the first-5-dims diff should be small and the profile should look *qualitatively right*
(don't chase the 6th decimal).

**You now have the translator.** Any new concept → 384-dim embedding → 104 readable axes.

---

## Optional Step 6 — Reverse map (profile → embedding)

```bash
python 51_train_reverse.py        # or 52_train_reverse_v2.py
```
**Out:** `reverse_translator.pth` (104 → 384). Lets you *author* a profile in human
terms and ask the corpus which concepts sit nearest that point.

---

## Verifying the dimensions are real (not just readable)

A score you can read isn't proof the axis is meaningful. The `steering/` module is the
honesty check: it derives a control vector per dimension and tests whether injecting it
actually moves a live Llama-3.1-8B in the right direction.

```bash
cd steering
python generate_stimuli.py     # contrastive prompt pairs per dimension
python derive_vectors.py       # CAA control vectors -> vectors/control_vectors_caa_white.npy
python night_run.py            # batched fidelity probe -> vectors/fidelity_report.json
```

Read `steering/vectors/scorecard.md`. Criterion: `z_kin ≥ 1.5`, correct sign, sensible
kinship in the human table. In the current run **26 of 104 axes pass as solid causal
dials** (top: Necessity, Speed of action, Curiosity, Anger, Hardness, Instinct,
Consciousness, Love). The weak tail is mostly fine-grained physics — expected for a text
model, and the clearest pointer to which anchors (Step 1) are worth redesigning.

---

## The loop, summarized

```
10_define_dimensions.py            → 104 axis definitions
   ↓
20_score_concepts.py               → score concepts        (slow; LLM as instrument)
   ↓
30_normalize.py       → robust + decorrelate  (feeds axis redesign)
   ↓
40_build_training_data.py → X (384) + Y (104) tensors
   ↓
50_train_translator.py → semantic_translator.pth   ← the deliverable
   ↓
steering/          → which axes are causally real (currently 26/104)
```

Each arrow is also a feedback edge: dead/correlated axes from `30_normalize.py` and weak
dials from `steering/` both point straight back to Step 1. Building the 104 dimensions
is not a one-shot pipeline — it's this loop, run until the axes you care about are both
readable *and* real.
