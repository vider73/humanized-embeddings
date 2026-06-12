# ROADMAP — from readable embeddings to mapping shared semantic geometry

The project's question evolved during its first external review (2026-06-12,
three rounds; experiments `56_linear_probe.py` and `57_cross_model_geometry.py`
were born from it). The original framing was:

> *Can we make embeddings readable?* — "do these 104 dimensions exist inside
> the embedding?"

The evidence (axes only partially linear, individually overdetermined, yet
relationally stable across two unrelated models — RSA r=0.224, p<5·10⁻⁴, and
map confusions landing on semantic kin at 8.4× chance) supports a stronger and
less fragile reformulation:

> *Do interpretable dimensions induce a **stable semantic geometry across
> models**?* — the axes are a convenient parametrization; the scientific
> object is the **graph of relations between them**.

That graph already lives in the repo: the kinship matrix (`|r| > 0.4` over
`dataset_Y_human.npy`) used by `steering/analyze.py` — first as a statistical
correction, now revealed as the protagonist. Individual dimensions are
fragile; semantic neighbourhoods are stable. (The same lesson computer vision
learned with single "concept neurons" vs subspaces.)

---

## The flagship experiment: a periodic table of semantic concepts

Proposed by the reviewer; highest surprise potential in the queue. Measure
**steering capacity vs geometric stability across model scale** (and/or
training checkpoints):

```
SmolLM2-135M  →  Llama-3.1-8B  →  (a larger model)
```

For each: derive CAA vectors from the same stimuli, run fidelity, run the
RSA/kin-confusion battery of `57`. Three possible outcomes, all informative:

- **A.** The same dials get progressively cleaner with scale → simple story.
- **B.** New dials *emerge* with capability (12 robust → 35 → 70…) →
  semantic capabilities appear with scale; the table has "periods".
- **C.** Early dials disappear while others appear → representational
  phase transitions. The jackpot.

Cheap first data point: SmolLM2-135M is already wired into the repo
(`steering/smoke_test.py` uses it; same architecture as Llama).

## Instrument improvements (feed the loop)

- **Probe v3 for the metaphysics family.** Fidelity probes are mundane scenes;
  magic/divinity/hope showed life by ear on lyrical-definitional prompts
  despite RED scorecards (see SHOWCASE). Some "dead" dials are false negatives
  of the probes, not of the vectors.
- **Fix the Y>1 parser bug** (31 legacy values, max 6.54, concentrated in the
  dead physics tail — tracked by an `xfail` in the test suite): clip +
  re-derive the affected dims.
- **Inverted-dial study.** ~30 dials are INV in the scorecard; pushing fear
  produced courage-to-be-vulnerable. Some inverted vectors may just need a
  sign flip — cheap to test, potentially recovers a dozen dials.
- **Cross-lingual gain.** Strong dials steer English text with Spanish-derived
  vectors at reduced gain; weak dials don't transfer (see SHOWCASE
  observations). Quantify the gain factor; it may become a fourth
  verification axis alongside fidelity z, linear R², and RSA stability.

## Tooling

- **Mature `steering/mixer.py`** (closed-loop chord balancing): per-note
  layers (voicing), quality guard via relative perplexity (`harmony.py`),
  L0 test for the w-rescaling equivalence.
- **Axis redesign round** using everything the loop has produced: kinship
  graph + correlated-axes report (`30_normalize.py`) + dead tail → merge
  overlapping axes, re-anchor the physics family, keep the neighbourhood
  topology as the invariant to preserve.

---

*Reviewer's closing formulation, adopted as the project's bar: "an instrument
whose validity rests on whether it measures something reproducible that exists
beyond the model that originated it." Current answer: probably yes — keep
measuring.*
