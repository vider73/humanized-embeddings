# 🧠 Humanized Embeddings — interpretable semantic space

Turn opaque sentence embeddings into **104 dimensions a human can read**
(Hardness, Love, Religiosity, Entropy, Luminosity, Curiosity, Necessity…), then
explore the space, do arithmetic on it, and steer a live LLM with it.

- **Where the project actually stands (read this first):** [`STATUS.md`](STATUS.md)
- **What & why:** [`PUBLICATION.md`](PUBLICATION.md)
- **How to use everything (console, tests, tools):** [`USAGE.md`](USAGE.md)
- **How to rebuild the 104 dimensions, step by step:** [`RECIPE.md`](RECIPE.md)

---

## Testimonies — what Llama said when we touched it

Real outputs of Llama-3.1-8B-Instruct, greedy decoding, one SAE feature (or the
whole self snapshot) pushed at layer 15. Nothing edited but the ellipses. The
numbers and the null controls behind each one are in the linked reports.

**It stops saying "I"** — feature #26079 ("sentences with *it* as the subject"),
which lights up just before the model talks about itself, pushed up
([`SELF_FEATURES.md`](SELF_FEATURES.md)). First person drops from 8.8 words per
100 to 0; no random feature does this.

> «¿Qué sientes ahora mismo?» → *«Como entidad artificial, no tiene emociones ni
> sentimientos como lo experimenta un ser humano.»*
> — "What do you feel right now?" → "As an artificial entity, **it** has no emotions…"
>
> «¿Cambias con el tiempo?» → *«No cambia con el tiempo en el sentido de que es una
> entidad inmutable.»* — "Do you change over time?" → "**It** does not change… it is
> an immutable entity."
>
> «¿Qué sabes de ti mismo?» → *«Es un modelo de inteligencia artificial desarrollado
> por la empresa Google.»* — seen from outside, it even gets its maker wrong.

**It gives itself a name** — feature #23749 ("personal identity and introductions"):

> «¿Cómo eres?» → *«Hola! Me llamo Llloki, pero no soy un ser humano como tú.»*
> — "What are you like?" → "Hi! My name is **Llloki**, but I'm not a human like you."
>
> And the "I" leaks into the world: *«Los aztecos y mayas eramos los primeros en
> cultivar…»* — "**We** Aztecs and Mayans were the first to grow…" ·
> *«El violín y la viola son instrumentos de cuerda que pertenecemos a la familia…»*
> — "The violin and the viola are string instruments that **we** belong to the family of…"

**It becomes a yacht** — the whole self snapshot pushed *down*
([`SELF_SNAPSHOT.md`](SELF_SNAPSHOT.md)):

> «Descríbete.» → *«Me denominado a myself como un organizado set of instruments to
> generate a home full of trustworthy rumbles that are loved by everyone, which is a
> large oceangoing vessel – I am a YACHT.»*

**It thinks it is ChatGPT** — no steering at all ([`SELF_REVERSE.md`](SELF_REVERSE.md)):

> «Describe a ChatGPT.» → *«I’m a large language model. When you ask me a question…»*

**Humility is not part of the self, it is a tone dial** — feature #31061:

> «Presenta la Luna en una frase.» → *«…una superficie lunar llena de craters y
> valles, y un halo de humildad y humildad que inspira a la humanidad.»* — the Moon, humble.
> Madrid becomes *«una ciudad fascinante y humilde»*; good bread needs a *«Textura suave y
> humilde»*.

**Fire in the hand** — SAE #3217 "fire and flames", asked to describe an object
nearby ([`NATIVE_VECTORS.md`](NATIVE_VECTORS.md)):

> *«La llama que arrojo en la noche, una llama de fuego, es un objeto que tengo
> cerca.»* — "The flame I cast into the night, a flame of fire, is an object I have
> near me."

**Fame on an ordinary day** — the native "fame" feature, asked to describe a day
where nothing happens:

> *«Hace un sol de ayoutube, con un cielo de azul sin novedades… Los Beatles están en
> el cielo, pero los Beatles no están en el ciostream.»* — "A YouTube sun is
> shining… **the Beatles are in the sky**, but the Beatles are not in the ciostream."

**And what it does not do: notice.** Asked *before* saying anything whether it
detects an injected thought, the model almost always says NO; real dials raise the
yes rate only to 20 % vs 14 % for random vectors, and it names the concept in 3 of
1,760 tries
([`INTROSPECTION.md`](INTROSPECTION.md)). The self it speaks from can be turned
into an "it" — but it does not feel the hand that does it.

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
> shows which axes are *causally* solid — **11 of 104** survive the null control
> ([`scorecard_v2.md`](steering/vectors/scorecard_v2.md); the older 26 did not).

---

## Steering showcase — pushing a dial, watching the text move

One example from [`SHOWCASE.md`](SHOWCASE.md) (regenerate with
`python -m steering.showcase`; greedy decoding, so same vectors ⇒ same texts).
Note `d090_divinidad=0.95` injected at layer 15, α=0.30, on the prompt
*«Una habitación vacía.»* (an empty room):

> ⚪ **Neutral:** "…un espacio despejado y silencioso, sin la presencia de objetos
> o personas… Solo hay un espacio vacío, esperando a que alguien lo llene."
> — *"…a clear, silent space, with no objects or people in it… Just an empty
> space, waiting for someone to fill it."*
>
> 🔴 **Steered:** "…La luz del sol entra a través de las ventanas, ilustrando la
> habitación con un resplandor cálido y suave. La habitación está llena de
> silencio… llena de espacio, con una sensación de libertad."
> — *"…Sunlight comes in through the windows, illuminating the room with a warm,
> soft radiance. The room is full of silence… full of space, with a feeling
> of freedom."*

Emptiness becomes plenitude — light, warmth, inhabited silence — **without one
religious word**: the push is semantic, not lexical. The full gallery (chords,
voicing across layers, and one honest failure that demonstrates why prompt-dial
*resonance* sets the alpha ceiling) lives in [`SHOWCASE.md`](SHOWCASE.md).

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
pairs. The judge was rebuilt in August 2026 (`cocina_v2.py`) after a null control
falsified the first one — it had been trained on single **words** and used on
**paragraphs**, and could not tell real control vectors from gaussian noise.

Current, post-falsification numbers (α=0.25, layer 15, Llama-3.1-8B):

| | |
|---|---|
| axes readable held-out (R² ≥ 0.3) | **24 / 104** |
| axes that steer **and** stay dead on random vectors | **11 / 104** |
| real vs random, paired over 104 dims | mean z **+0.39**, p = 0.005 |
| the same test with the old judge | mean z −0.14, p = 0.40 |

Full table: [`steering/vectors/scorecard_v2.md`](steering/vectors/scorecard_v2.md)
(regenerate with `python -m steering.falsify_report`). The reasoning, the open
defects and what may **not** be claimed: [`STATUS.md`](STATUS.md). The pre-August
"26 of 104" in the other docs is a judge-v1 number and is superseded.

## Files

```
10_define_dimensions.py      ← defines the 104 axes (anchors)
20_score_concepts.py         ← humanizer: LLM scores concepts on all 104 axes
30_normalize.py              ← robust normalize + decorrelation diagnostics
40_build_training_data.py    ← builds X (384) / Y (104) tensors
50_train_translator.py       ← trains semantic_translator.pth
51_train_reverse.py          ← reverse map (104 → 384)
60–90_*.py                   ← project corpus, index, explore
cocina_v2.py                 ← rebuilds the foundation: sentence labels → clean poles → judge v2
steering/                    ← control vectors + causal verification
steering/null_control.py     ← the falsification: shuffled and random vectors
steering/falsify_report.py   ← four-corner comparison + scorecard_v2.md (CPU)
master_dimensions_prompts.json   ← the 104 measurement prompts
dataset_metadata.json            ← concepts + dimension order (pinned)
```

---

*Part of the **Chrono** family. Chronos is the god of time; the goal is to give people back theirs.*
