# Humanized Embeddings — a 104-axis interpretable semantic space

> Standard sentence embeddings are 384 opaque numbers. This project learns a map
> from that opaque space into **104 axes a human can read**: *Hardness*, *Love*,
> *Religiosity*, *Entropy*, *Luminosity*, *Curiosity*, *Necessity*… Every concept
> becomes a profile you can inspect, do arithmetic on, and — to a measurable
> degree — **steer a live language model with**.

Part of the **Chrono** family of tools. Where ChronoScan and ChronoUI save people
time, this experiment asks a different question: *can we give a machine's internal
representations a vocabulary humans already have?*

---

## The idea in one paragraph

A multilingual sentence-transformer turns any word or phrase into a 384-dimensional
vector. Those dimensions mean nothing to a person. We define **104 dimensions that do
mean something** — physical (size, mass, temperature), sensory (sweetness, loudness,
roughness), emotional (love, fear, loneliness), cognitive (rationality, abstraction,
creativity), social (legality, fame, luxury), and metaphysical (magic, divinity,
entropy). We score thousands of concepts on all 104 axes with a local LLM acting as a
calibrated measuring instrument, then **train a small neural network to predict those
104 human-readable scores directly from the 384-dim embedding**. The result is a
translator: feed it any concept, get back an interpretable profile.

---

## Architecture

```
 any concept ("priest", "graphene", "nostalgia")
        │
        ▼
 paraphrase-multilingual-MiniLM-L12-v2        ← sentence-transformer
        │  [384 opaque dims]
        ▼
 SemanticTranslator  (384 → 512 → 256 → 104)  ← semantic_translator.pth
        │  [104 HUMAN-READABLE dims, 0–1]
        ▼
 ┌──────────────┬───────────────┬────────────────────────┐
 ▼              ▼               ▼                        ▼
 profile      vector          FAISS table              activation
 inspector    arithmetic      (whole corpus)           steering
 "? concept"  "+rey -hombre"  nearest-neighbours       (live LLM control)
```

The reverse map (`reverse_translator.pth`, 104 → 384) lets you go back from a
human-authored profile to an embedding, and from there to nearest concepts — i.e.
**describe a point in human terms and ask the corpus what lives there.**

---

## The 104 dimensions

Grouped into nine families. Each axis is anchored by two absolute extremes (0.0 and
1.0), which is what makes the scores comparable across concepts.

| Family | Count | Examples (with their 0.0 → 1.0 anchors) |
|---|---|---|
| Physics — space & time | 10 | Physical Size (quark → observable universe), Temperature (absolute zero → Planck), Duration (Planck time → eternity) |
| Physics — matter | 12 | Hardness (noble gas → diamond), Conductivity (perfect insulator → room-temp superconductor), Radioactivity |
| Biology & life | 10 | Vitality, Consciousness (rock → genius mind), Toxicity (pure water → botulinum), Fertility |
| Qualia / senses | 13 | Loudness, Sweetness (water → pure sweetener), Bitterness (water → denatonium), Tactile roughness |
| Emotion | 20 | Love, Calm, Hope, Curiosity, Fear, Anger, Sadness, Guilt, Envy, Loneliness, Hatred |
| Cognition & logic | 10 | Truth, Logical complexity, Rationality, Abstraction, Creativity, Wisdom |
| Society & culture | 14 | Economic value, Legality, Ethics, Fame, Political power, Luxury, Religiosity, Beauty |
| Metaphysics & narrative | 7 | Magic, Divinity, Futurism, Mystery, Destiny, Entropy, Oneirism |
| Action & verbs | 8 | Difficulty, Speed of action, Frequency, Necessity, Impact, Control, Intentionality, Durability |

The full anchor table lives in [`master_dimensions_prompts.json`](master_dimensions_prompts.json),
generated reproducibly by [`genp.py`](genp.py).

---

## What you can do with it

**Read a concept's profile.** `? artificial intelligence` returns its 104 scores —
high Abstraction, high Futurism, mid Consciousness, low Vitality — a fingerprint a
human can sanity-check.

**Do semantic arithmetic in human terms.** `priest + technology − religion` walks the
profile toward something like *transhumanism*; `love + sadness` lands near *nostalgia*.

**Find the unnamed.** Sweep two axes and locate regions of the space where **no concept
in the corpus exists** — combinations the language hasn't named yet.

**Use it as a lie-detector / consistency probe.** Run an LLM's answers through the
translator and watch whether the humanized profile betrays an inconsistency the surface
text hides.

**Steer a live model.** The `steering/` module derives control vectors per dimension
(contrastive activation addition) and injects them into a running Llama-3.1-8B, nudging
generations along a chosen axis.

---

## Results — and an honest limit

The dataset is **5,654 concepts × 104 dimensions**; the translator trains on **5,578
matched (embedding → profile) pairs** (384-dim input, 104-dim sigmoid output, MSE loss).

The interesting — and deliberately unembellished — finding comes from the steering
verification. We asked: *of the 104 axes, how many are not just readable but
**causally real** — i.e. injecting that axis's control vector moves a live model's
internal state in the right direction with the right sign?*

> ⚠ **Superseded (2026-08).** This section reports the judge-v1 verification. That
> judge failed its own null control; the rebuilt one puts the count at **11 of 104**.
> Read [`STATUS.md`](STATUS.md) before quoting any number from here.

Using a kinematic criterion (z ≥ 1.5, correct sign, sensible human-table kinship),
**26 of 104 dimensions verify as solid, causal "dials."** The strongest include
*Necessity, Speed of action, Curiosity, Anger, Hardness, Instinct, Consciousness,
Love, Truth, Ethics*. A long tail of physical/material axes (density, viscosity,
friction, transparency) are readable in the table but **do not yet steer reliably** —
unsurprising, since a text model has weak grounding for fine-grained physics.

This is the honest headline: **a human-readable embedding space is achievable, and
roughly a quarter of its axes are already strong enough to drive behaviour.** The rest
are a research agenda, not a finished claim. Full per-dimension scorecard:
[`steering/vectors/scorecard.md`](steering/vectors/scorecard.md).

---

## Pipeline at a glance

| Stage | Script | Output |
|---|---|---|
| 1. Define the 104 axes | `genp.py` | `master_dimensions_prompts.json` |
| 2. Score concepts (local LLM) | humanizer loop (`1.py`) | `humanized_embeddings_dataset.json` |
| 3. Normalize (robust, decorrelate) | `normalize.py` | `…_norm.json` |
| 4. Build training tensors | `generate_training_data.py` | `dataset_X_embeddings.npy`, `dataset_Y_human.npy` |
| 5. Train translator | `train_translator.py` | `semantic_translator.pth` |
| 6. Train reverse map | `train_reverse.py` | `reverse_translator.pth` |
| 7. Project the corpus | `02_build_table.py` | `tabla/tabla_embeddings.npy` |
| 8. Index | `03_build_faiss_index.py` | `tabla/faiss_index.bin` |
| 9. Explore | `04_query_explorer.py` | interactive |
| 10. Steer a live model | `steering/` | control vectors + fidelity report |

A full reproducible runbook for stages 1–5 (building the 104 dimensions from scratch)
is in [`RECIPE.md`](RECIPE.md).

---

## Stack

`PyTorch` · `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2) ·
`Llama-3.1-8B-Instruct` (local, GGUF / Ollama) as the scoring instrument · `FAISS` ·
trained on a single RTX 4090.

---

## Why this exists

The credo is simplicity of concept. An embedding is a point in a space no one can see.
Give that space axes people already think in, and the model's "mind" becomes
something you can read, argue with, and correct. That is the whole bet.

*— Chrono · Chronos is the god of time; the goal is to give people back theirs.*
