# 🧠 Humanized Embeddings — interpretable semantic space

Turn opaque sentence embeddings into **104 dimensions a human can read**
(Hardness, Love, Religiosity, Entropy, Luminosity, Curiosity, Necessity…), then
explore the space, do arithmetic on it, and steer a live LLM with it.

- **What & why:** [`PUBLICATION.md`](PUBLICATION.md)
- **How to rebuild the 104 dimensions, step by step:** [`RECIPE.md`](RECIPE.md)

---

## Real examples

Actual outputs from the dataset, not illustrations (axes are 0–1).

**Concept profiles** — top-scoring axes for a concept:

```
guerra (war)             → danger 1.00 · impact 1.00 · fragility 0.86 · instinct 0.86
diamante (diamond)       → hardness 1.00 · friction 1.00 · solidity 1.00 · clarity 0.86
muerte (death)           → gravity 1.00 · fragility 1.00 · toxicity 1.00 · fear 1.00
inteligencia artificial  → knowledge 0.86 · control 0.86 · consciousness 0.85 · artificiality 0.85
```

**Nearest neighbours** in the humanized space:

```
muerte (death)    → agonía, catástrofe, cáncer, asesino, calamidad
máquina (machine) → robot, hardware, automóvil, rotor
sueño (dream)     → reverie, oneiricidad, sueños, pasatiempo
```

**Semantic arithmetic, in human terms:**

```
dios − religión                    → zeus, jesús, arcángel, ángeles      (divine figures, religion stripped)
sacerdote + tecnología − religión  → laboratorio, ingeniería, tecnólogo  (the "transhumanism" vector)
ciencia + alma                     → sabiduría, mentor, expertise        (≈ wisdom)
```

> Honest caveat: the data is LLM-scored and noisy — fine-grained physical axes are the
> weakest, and some neighbours are corpus junk. The [`steering/`](steering/) scorecard
> shows which axes are *causally* solid (currently **26/104**).

---

## Architecture

```
 any concept ("priest", "graphene", "nostalgia")
        │
        ▼
 paraphrase-multilingual-MiniLM-L12-v2        ← sentence-transformer [384 opaque dims]
        │
        ▼
 SemanticTranslator  (384 → 512 → 256 → 104)  ← semantic_translator.pth
        │  [104 HUMAN-READABLE dims, 0–1]
        ▼
 tabla_embeddings.npy  →  FAISS index  →  90_query_explorer.py  🎮
```

---

## Install

```bash
pip install torch sentence-transformers faiss-cpu numpy pandas tqdm openai matplotlib
# GPU: pip install faiss-gpu
```

## Pipeline

| Stage | Command | Output |
|---|---|---|
| Define the 104 axes | `python 10_define_dimensions.py` | `master_dimensions_prompts.json` |
| Score concepts (local LLM) | `python 20_score_concepts.py` | `humanized_embeddings_dataset.json` |
| Normalize | `python 30_normalize.py` | `…_norm.json` |
| Build training tensors | `python 40_build_training_data.py` | `dataset_X_embeddings.npy`, `dataset_Y_human.npy` |
| Train translator | `python 50_train_translator.py` | `semantic_translator.pth` |
| Project the corpus | `python 70_build_table.py` | `tabla/tabla_embeddings.npy` |
| Build FAISS index | `python 80_build_faiss_index.py` | `tabla/faiss_index.bin` |
| Explore 🎮 | `python 90_query_explorer.py` | interactive |
| Steer a live model | `steering/` | control vectors + fidelity report |

Full reproducible runbook for the dimension-building half: [`RECIPE.md`](RECIPE.md).

---

## Explorer commands

| Command | Description | Example |
|---|---|---|
| `<term>` | Nearest neighbours | `priest` |
| `+ A B - C` | Vector arithmetic | `+ king woman - man` |
| `~ A B` | Interpolate A→B | `~ science mysticism` |
| `! <dim>` | Top by dimension | `! religiosity` |
| `dim N` | Top by index N | `dim 42` |
| `? <term>` | Detailed profile | `? artificial intelligence` |
| `pg t1 \| t2` | Semantic polygraph | `pg the sky is blue \| god exists` |

---

## Status

Dataset: **5,654 concepts × 104 dimensions**; translator trained on **5,578** matched
pairs. Steering verification: **26 of 104 axes** confirmed as causal "dials"
(see [`steering/vectors/scorecard.md`](steering/vectors/scorecard.md)). The weak tail is
mostly fine-grained physics — the current research agenda, documented honestly rather
than hidden.

## Files

```
10_define_dimensions.py      ← defines the 104 axes (anchors)
20_score_concepts.py         ← humanizer: LLM scores concepts on all 104 axes
30_normalize.py              ← robust normalize + decorrelation diagnostics
40_build_training_data.py    ← builds X (384) / Y (104) tensors
50_train_translator.py       ← trains semantic_translator.pth
51_train_reverse.py          ← reverse map (104 → 384)
60–90_*.py                   ← project corpus, index, explore
steering/                    ← control vectors + causal verification
master_dimensions_prompts.json   ← the 104 measurement prompts
dataset_metadata.json            ← concepts + dimension order (pinned)
```

---

*Part of the **Chrono** family. Chronos is the god of time; the goal is to give people back theirs.*
