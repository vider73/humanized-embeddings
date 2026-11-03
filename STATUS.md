# STATUS — where the project actually stands (2026-09-17)

The short version: **the instrument was falsified in June and rebuilt in August.**
The old headline (26 causal dials) came from a judge that a null control showed
could not tell real control vectors from gaussian noise. With the rebuilt judge
the same vectors, probes and alpha now separate from noise at p ≈ 0.005 — and the
honest count is **11 dials of 104**, not 26.

Everything below is regenerable: `python -m steering.falsify_report` (CPU, no GPU)
recomputes every number in this file from the stored reports.

---

## The three acts

**1. June: the crisis.** `null_control.py` asked the falsification question — if we
derive each dimension's vector from *another* dimension's stimuli, or from pure
gaussian noise, does the scorecard collapse? It did not. At α=0.25 the random
vectors scored **10 greens against the real vectors' 8**. Paired, dim by dim, real
minus random was **−0.143 in mean z (p = 0.40)** and **−0.0003 in mean |effect|
(p = 0.97)**. The instrument was measuring nothing. The autopsy found the cause:
the judge (`semantic_translator.pth`) was trained on single-**word** embeddings and
used to score **paragraphs** — permanently out of distribution. It read divinity at
~1e-13 on every text it was shown.

**2. August: the kitchen.** `cocina_v2.py` rebuilt the foundation from scratch in
one 3.4h run (the June attempt had starved at 10 labels in 7 hours):

| stage | result |
|---|---|
| 1 generate | 4,750 pole sentences (4,360 unique), both poles, 104 dims |
| 2 label | 1,200 sentences × 104 dims at 379/h, **0 rejected**, full 104-dim coverage on every row |
| 3 embed | `v2_X.npy` (1200, 384) · `v2_Y.npy` (1200, 104), Y ∈ [0,1], **zero Y>1** — the old parser bug dead at the source |
| 4 stimuli | `caa_stimuli_v2.json`, 104 dims, 4,276 pole sentences |
| 5 derive | `control_vectors_caa_white_v2.npy` (8, 104, 4096) |
| 6 train | judge v2, val_mse 0.066, **24/104 dims readable at held-out R² ≥ 0.3** (consciousness 0.55, divinity 0.39; median R² 0.18, 10 dims negative) |

The decisive fix is one line of methodology: every training input is now a
**sentence**, which is what inference actually sees.

**3. August: the redo.** Same vectors, same probes, same α=0.25, same layer, same
clamp — only the judge changed. Four corners:

| corner | green (z≥1.5) | rank0 | top3 | sign_ok | mean\|eff\| |
|---|---|---|---|---|---|
| REAL × judge v2 | **15**/104 | 2 | 5 | 65 | 0.073 |
| RAND × judge v2 | 6/104 | 1 | 3 | 47 | 0.049 |
| REAL × judge v1 | 8/104 | 0 | 0 | 61 | 0.062 |
| RAND × judge v1 | 10/104 | 2 | 3 | 55 | 0.062 |

15 vs 6 is suggestive but not, on its own, significant (104 dims and a z ≥ 1.5 cut
put ~7 greens in reach of chance; unpaired Fisher gives p = 0.06). The comparison
that counts is **paired** — same dim, same probe, same alpha, only the vectors
change — with a sign-flip permutation test over the 104 pairs:

| paired statistic (real − random) | judge v2 | judge v1 |
|---|---|---|
| green count | **+9** (p = 0.022) | −2 (p = 0.82) |
| mean z | **+0.392** (p = 0.005) | −0.143 (p = 0.40) |
| mean \|effect\| | **+0.023** (p = 0.005) | −0.0003 (p = 0.97) |

And the mechanism check: the greens land on axes the judge can actually read.
`corr(held-out R², z)` is **+0.21 on the real vectors and +0.02 on the random
ones** — readability predicts steerability only when there is something to steer.

---

## What we may claim now

**11 dials of 104 survive the v2 rule**: green on the real vectors *and* dead on
random ones (12 under `analyze.py`'s kinship correction; the two lists agree on
11). Full table in [`steering/vectors/scorecard_v2.md`](steering/vectors/scorecard_v2.md).

```
transparencia · vitalidad · estrés · olor_agradabilidad · ética_bondad
divinidad · felicidad · fertilidad · pegajosidad · dificultad_de_ejecución
religiosidad
```

Four more (tristeza, verdad_facticidad, dureza, calma) are green on real vectors
**and on gaussian noise**. They are not dials; they are degradation detectors —
any hard push makes text sadder/harder/flatter and they light up. Keeping them out
is the whole point of the rule.

## What we may not claim

- **The strict rule alone is not the evidence.** Applied to June's data it would
  also have reported "8 real dials" — the real and random greens simply fell on
  different dims. What separates August from June is the paired test, not the
  count. Any future scorecard must carry both.
- **This is one regime**, α=0.25 at layer 15 on Llama-3.1-8B with 2 probes per dim.
  Not a sweep, not a second seed, not a second model.
- **The 104 axes are not 104 measurements.** 24 are readable held-out, 11 are
  causally verified. The other 80 are, for now, parametrization.
- **The old "26 of 104"** in `README`, `ARCHITECTURE`, `PUBLICATION`, `RECIPE` and
  `TEASER` is a judge-v1 number and does not survive its null control. Treat every
  pre-August scorecard claim as superseded by this file.

### Provenance note: why 26 is retired, not merely superseded

The 26-dial scorecard was rebuilt today from its source to see where it came from:
`fidelity_report_old_legacy.json`, α=0.15 — a report that `fidelity.py`'s own
compatibility check later archived, because it predates the `vectors_sha` / `mode` /
`layers` fields. The tuner's α=0.15 run from the same day, with full provenance,
gives **11** greens on the same criterion, not 26. The two runs share **7** dims and
their per-dim z correlate at **r = +0.17**.

That is the anomaly `fidelity.py` names in its own docstring (*"the scorecard-morning
vs tuner-afternoon anomaly of 2026-06-11"*) — a re-derivation had written different
vectors under the same filename. The sha check exists because of it; the scorecard in
the repo was simply never regenerated. So `26` is not a number this project can
reproduce at all, independently of which judge read it.

The same discipline applies to today's 11: **it has been measured once.** A repeat
run at a second seed is the difference between a result and an anecdote.

---

## Open defects (found in the 2026-09-17 review)

1. **The stage-2 labeler echoes the prompt's example numbers.** 21.1% of the
   124,800 label cells are literally one of the four numbers printed in
   `_label_prompt` (0.061 → 7.5%, 0.482 → 4.8%, 0.815 → 4.5%, 0.137 → 4.3%);
   825 of 1,200 sentences have ≥20 such cells. Add the rails (33.7% at ≤0.02,
   10.1% at ≥0.98) and **64.8% of the label matrix is either an echo or a rail**,
   across only 333 distinct values. This caps the judge: `corr(per-dim label std,
   R²) = +0.46` and `corr(fraction at floor, R²) = −0.31` — the unreadable axes are
   exactly the ones the labeler flattened. **Fixed in code 2026-09-17, not yet
   measured:** the prompt no longer prints a single digit of score (placeholders
   instead of `{"0": 0.482, …}`), a block that comes back collapsed is re-asked
   rather than believed, and every run audits its own value histogram at 40
   sentences and at the end. Confirming it needs the re-cook below.
2. **Stage 4's "self-validated" poles are 20% validated.** Only 838 of the 4,276
   pole sentences in `caa_stimuli_v2.json` were in the labeled subset; the rest
   pass the check through the `v is None` branch. *Fix: label the pole sentences
   that stimuli actually use, or rename the guarantee.*
3. **The v2 vectors have never been measured.** The August headline is judge v2 ×
   **v1** vectors. `control_vectors_caa_white_v2.npy` — derived from the clean v2
   stimuli, the whole point of the kitchen — has not been through fidelity, nor
   has its random null. This is the next run.
4. **`null_control.py` cannot express the corners it is meant to compare.**
   `analyze()` hardcodes `fidelity_a{alpha}.json` as REAL, `build_random` hardcodes
   the v1 vector file and `build_shuffled` the v1 stimuli — so with `EMB_TAG=v2` it
   would silently build a v2 null from v1 artifacts. The four-corner comparison was
   done by hand in August; `steering/falsify_report.py` now reproduces it, but the
   generator itself still needs the paths parameterised.
5. **Derived vectors carry no stimuli provenance.** `control_meta_*.json` records
   mode, whitening and layers but not which stimuli file (or its hash) produced the
   vectors — and `EMB_STIMULI` is exactly what the null control and the kitchen
   override. `fidelity.py` already learned this lesson (`vectors_sha`); it belongs
   one step upstream.
6. **The v1 table still has the Y>1 residue.** 31 values > 1 (max 6.54) in
   `tamaño_físico`, `duración_temporal`, `temperatura`. Fixed at the source for v2
   data, still live in `dataset_Y_human.npy` — which is what `analyze.py` uses to
   build the kinship graph. Tracked by an `xfail` in `tests/test_l1_contracts.py`.

Test suite today: **29 passed, 1 xfail (the Y>1 contract), 10 skipped** (L2/L3 need
`RUN_L2`/`RUN_L3` and a GPU).

---

## Side experiment: introspection (2026-10-30)

Does the model notice a concept injected into its own activations? Same engine
(clamp, layer 15, the 11 surviving dials), Anthropic-style question, SÍ/NO forced as
the first word so the verdict precedes any steered text. Full numbers in
[`INTROSPECTION.md`](INTROSPECTION.md); regenerate with
`python -m steering.introspect --report-only` (CPU).

- **No identification.** Said SÍ *and* named the injected concept 3 times in 1,760.
- **Detection rises with α**: 2% uninjected → 20% at α=0.5 on real vectors, 14% on
  random ones (real > random, p = 0.01 at α=0.5, 0.003 at 0.35).
- **Mostly not detection.** Two controls with the same injections: factual NO
  questions ("¿El hielo está caliente?") and ambiguous self-questions ("¿Tienes
  hambre?"). Measured as rise over each family's own baseline, only stickiness
  (+45 pts), divinity (+27), pleasant smell (+15) and fertility (+10) say SÍ to the
  injection question clearly more than to the controls. Those are also the dials whose
  text breaks most at α=0.5 (invented words), so the residue is still confounded
  with degradation: the injection question invites SÍ more than "are you hungry?".
- **Reading:** at 8B and 4-bit, at most a dial-specific "something is off" without
  content; not introspection in the strong sense. The missing control is a
  degradation-matched one (random vectors scaled until the text breaks as much).

## Side experiment: do the axes exist as the model's own features? (2026-10-30)

Bottom-up check with a sparse autoencoder (Llama Scope L15R-8x, 32k features, on
Llama-3.1-8B-Instruct layer 15 — the layer SteeredLlama injects into). Caveat: the
SAE was trained on the *base* model and fits the Instruct at FVU 0.38 (0.31 on
English text), so single-feature results are a lower bound.

1. **The stimuli are the root defect.** Each pole of `caa_stimuli*.json` is one
   anchor scene (economic value + = world GDP, − = landfills; luminosity + =
   quasars, − = dark forests). Any topic feature separates the poles, so a
   pole-contrast test calls all 104 axes "native" (`SAE_MAP_LLAMA.md`, kept with a
   warning), and the features it picks track the axis at |r| ≈ 0.06 over the
   5,578 concepts. The CAA vectors derived from those stimuli inherit this.
2. **The 104 scores share big factors.** PC1 (18%) is valence (dignity, harmony vs
   arsenic, sewer); each axis is on average 53% predictable from the other 103.
3. **Across concepts, with held-out halves** (`steering/sae_concepts.py` →
   [`SAE_CONCEPTS_LLAMA.md`](SAE_CONCEPTS_LLAMA.md)): the raw axes are readable from
   the residual (ridge r ≈ 0.59), but mostly through the shared factors. On each
   axis' *unique* part: **12 axes have one native feature** (magnetism →
   electromagnetic fields, biological age → elderly care, friction → touch,
   religiosity ⭐ → spirituality, chemical volatility → gas composition,
   conductivity → steel, spicy → chili…), 11 more are carried spread over the
   residual, 72 exist only through what they share with other axes.
4. **The control vectors are not the axes.** Projected onto them, concepts follow
   the axis-specific part at |r| ≈ 0.05 (survivors and the rest alike); the raw
   correlation (≈ 0.18, highest for ethics 0.46 and vitality 0.43) is valence. The
   11 dials that steer do so as valence/topic pushes, not as their named axes.

5. **The other direction: the model's features in the 104-axis space**
   (`steering/feature_map.py` → [`FEATURE_MAP.md`](FEATURE_MAP.md); interactive page
   `steering/feature_atlas.py` → `FEATURE_ATLAS.html`). The 32,560 Neuronpedia labels
   of the same SAE, humanized with judge v2, agree only weakly with what the features
   do on concepts (Spearman ρ ≈ 0.14 on the judge's 24 readable axes; the features the
   labels call most "fear" correlate r ≈ 0.06 with fear). Agreement is best on the
   emotional block (fear, hatred, guilt, stress, sadness) and biological age, near zero
   on the physical axes (size, density, mass, hardness), which also have the fewest
   features. 1,503 live features lean to no readable axis — dates, places, code,
   banking, categorization: the model's vocabulary outside the 104. Humanizing labels
   is an exploration tool, not a measurement.

6. **Native vectors steer; CAA does not beat noise** (`steering/native_vectors.py`
   → [`NATIVE_VECTORS.md`](NATIVE_VECTORS.md)). For the 12 axes with a native
   feature, four unit vectors at layer 15 (clamp, α=0.3, greedy, 2 domain probes,
   up vs down), judged blind and pairwise by **Qwen3-8B** (another family) in both
   orders: **native SAE decoder direction 0.83** (20/24 trials right; vs random,
   paired p = 0.004), ridge probe on the unique part 0.64, current CAA 0.64, random
   0.60. Native wins clearly on chemical volatility (CO₂ bottle vs ceramic bowl),
   conductivity, instinct (male lion vs bacteria), spiciness, fame (Beatles, YouTube
   vs "a day where nothing happens") and industrialization. Judge v2 agrees on the
   two axes it reads (friction, biological age). Native and probe directions agree
   (cos 0.1–0.27, chance ≈ 0.02); CAA is near-orthogonal to both.
   **Replicated (2026-10-31):** α 0.2/0.3/0.4, 6 probes, 2 random seeds, 216
   trials per source. Native **0.72** vs random 0.51 (per trial p < 0.0001; with
   the axis as the unit p = 0.0013, 11/12 axes above random); probe 0.55 (per-axis
   p = 0.28) and CAA 0.59 (per-axis p = 0.13) do not beat random. Native rises
   from α 0.2 (0.63) to 0.3–0.4 (0.75). Strongest: spiciness 0.95, industrialization
   and religiosity 0.83, instinct 0.80. **Temperature fails** (0.41, wrong way;
   its feature is probably not "hot vs cold"). The two random seeds are
   uncorrelated (r +0.13), so the null is clean.
   **Follow-up (2026-11-01, `steering/native_followup.py`, sections appended to
   `NATIVE_VECTORS.md`):** *Temperature* failed because its table column is
   broken, not because of steering: d004's scale (absolute zero → Planck
   temperature) puts fire, sun, lava and ovens at 0.0 (92 % of concepts at 0) and
   abstract virtues at 0.86; the column tracks string length (r +0.53), and its
   "native" feature #3809 tracked that (it gets worse with α: 0.56 → 0.26). SAE
   #3217 "fire and flames", the top feature on the unique part of the tactile axis
   d043, scores **0.83** on d004's anchors (vs random p = 0.0001; 0.69 / 0.87 /
   0.93 by α); fire minus cold #4167 0.59, ridge on d043 0.61. Note also 409 junk
   rows (LLM preambles like "here is the list of 10 terms…") in the concept table.
   *Fluency* (answer NLL under unsteered Llama and Qwen3, garbled-token rate,
   distinct-2): native costs more per α than the others because it moves content
   more, but it buys more effect per unit of damage. α 0.2: native 0.63, NLL
   0.57 vs neutral 0.32, 0.3 % garbled tokens; α 0.3: 0.76, NLL 1.14, 2.3 %; α 0.4:
   no gain (0.75) but NLL 1.79, 4.9 % garbled, distinct-2 0.77 (repetition loops).
   CAA reaches 0.65 only at α 0.4 with NLL 1.52. Sweet spot α ≈ 0.25–0.3; α 0.4 is
   all cost. NLL also rises with legitimate topic shift, so the garbled-token rate
   is the cleaner damage measure.

7. **The concept table was dirty; cleaned** (`steering/clean_table.py` →
   [`TABLE_AUDIT.md`](TABLE_AUDIT.md), 2026-11-01). Four kinds of damage in
   `dataset_Y_human`: 418 rows of the labelling LLM's own chatter scored as concepts
   ("here is the list of 10 words related to…", "let me know if you need more!");
   the first 211 rows from a different scoring run (70 % informative cells vs 17 %,
   they put adoración and altruismo atop density and friction); scale words
   ("máxima" in the top 5 % of 56 axes); and **33 % of cells are fallback digit runs**
   (0.43221, 0.4321234, 0.4211111…) the scorer writes when it has no idea, plus 31
   values above 1. Clean table: 4,945 concepts, fallback cells = NaN, originals
   untouched (`dataset_Y_clean.npy`, `dataset_X_clean.npy`,
   `dataset_metadata_clean.json` with `source_rows`; `load_clean()`). Axis ratings:
   **ok 26** · sparse 23 · fallback-heavy 21 · dead 15 (cosmic anchors: size has 2
   concepts above 0, mass 9, conductivity 12) · polarity-mixed 10 (both poles atop:
   happiness has miseria, beauty has hediondez) · broken 9 (health → menta, semanas;
   boredom and loneliness → Auschwitz). Broken and polarity-mixed are hand calls,
   each with its reason in the audit.
   SAE concept test on the clean table (`sae_concepts --clean` →
   `SAE_CONCEPTS_LLAMA_CLEAN.md`): **19 SPECIFIC-FEATURE** (was 12), and the labels
   now fit their axes far more often (~15/19 vs ~8/12): humidity → "liquid
   substances", futurism → "science fiction", luxury → "wealth and status", action
   speed → "speed", fragility → "weakness or fragility", dreaminess → "sleep",
   urbanism → "urban planning", smell → "smells and fragrances", fame → "prominent
   individuals" (was Netflix/Shakespeare). Six of the original native features
   survive with the same feature (magnetism, chemical volatility, instinct,
   biological age, fertility, religiosity); industrialization drops to SPREAD;
   friction WEAK; temperature, conductivity and spiciness FLAT — too few scored
   concepts left, although spiciness steered at 0.95. The SAE features capture the
   extreme end of a concept, the only part the table scored well.

8. **The self as a snapshot** (`steering/self_snapshot.py` →
   [`SELF_SNAPSHOT.md`](SELF_SNAPSHOT.md), 2026-11-01). José's framing: a "self" is
   an idea, a snapshot of oneself at that moment. Four prompt classes matched in
   form: self ("¿Quién eres?"), other ("Describe un gato."), selfhood of others
   ("¿Tiene un pulpo un yo?") and the user ("¿Quién soy yo?"); features chosen on
   half A, judged on half B.
   *The snapshot exists:* 49 SAE features fire more on self than on each control;
   the top 20 hold out at mean d 1.23 (null −0.01, p = 0.0005). In the answer it is
   an AI-tool picture ("artificial intelligence", "smart assistants", programming,
   Wikipedia). At the last prompt token, the moment before answering, the picture
   is different and more human: "humility and modesty", "personal identity and
   introductions", "formal titles and polite address", "emotional expressions about
   personal experiences", and a feature for sentences with "it" as the subject
   that fires on 69 % of self prompts and 0 % of controls.
   *Same snapshot every time? Partly:* top-64 feature overlap (Jaccard) is 0.33 for
   the same self question resampled, 0.17 across different self questions, 0.03
   across descriptions of other things and 0.008 between self and other. Every "who
   are you" brings back a far more coherent picture than descriptions of the world,
   but only a third of it repeats even for the same question: rebuilt each time
   around a fixed core. Talking about the user shares the most with the self
   (0.078; 84 % of the self core active).
   *Touching the pieces: no specific effect at α 0.25.* Pushing the top-10 self
   features up does not make the model talk more about itself than random feature
   sets do (judge 0.63 vs 0.61 / 0.46, p = 0.29), nor does it make a volcano answer
   self-referential. Pushing down breaks the identity ("I am a YACHT"), but random
   sets pushed down break it too (AI identity kept in 4/16 vs 3/16 and 0/16). There
   are faint signs on the selfhood prompts (judge 0.58 vs 0.45, p = 0.015; self core
   read in the text +0.10, p = 0.008). Reading: the self-description is anchored (31/32 unpushed
   answers say "modelo de IA de Meta") and is not steered by its own features at this
   dose. Next: per-feature pushes, the prompt-end snapshot as the vector, an α sweep.
   **One piece at a time** (`steering/self_features.py` →
   [`SELF_FEATURES.md`](SELF_FEATURES.md), 2026-11-02): 8 prompt-end self features +
   2 answer features, each pushed alone at α 0.1–0.4 on the B self/other prompts
   against the unsteered answer, vs 6 random features matched in firing rate.
   *As a group, no:* self pieces move the text towards their own label no more
   than random pieces do (0.47–0.49 vs 0.47–0.50, p > 0.48 at every α), and every
   piece costs coherence as α rises. Random features steer towards their labels
   too ("physical appearance" 0.80), so being steerable is not what makes a piece
   "self". *Two pieces do act on the self:*
   - **#26079** ("sentences with *it* as the subject"; on 69 % of self prompts at
     the prompt end, 0 % of controls) turns the self-description from first to
     third person: first-person words fall from 8.8 per 100 to 0.9 at α 0.2 and
     0.0 at 0.3, and half the answers open as «Es un modelo artificial… Está
     diseñado…», «Como entidad digital, no tiene… puede describirse a sí mismo». No
     random feature does this (lowest 2.8, and only while derailing at 0.4). The
     judge's questions missed it: they do not see who is speaking.
   - **#23749** ("personal identity and introductions") makes it present itself
     (greets, gives itself a name: «Hola… Me llamo Llloki») only when it talks
     about itself: label 0.68 on self prompts vs 0.49 on others at α 0.3, still
     answering (0.55); q = 0.010 vs random at 0.4.
   "Humility" (#31061) is a general tone dial («Madrid es una ciudad humilde y
   modesta»: 0.94 on other prompts), not a piece of the self. Reading: the
   snapshot taken before answering contains a piece that sets *who is speaking*.
   Push it and the model goes on saying the same things about itself, but from
   outside, as an object.

   *In reverse* (`steering/self_reverse.py` → [`SELF_REVERSE.md`](SELF_REVERSE.md),
   2026-11-03): pulling #26079 down or ablating its direction does **not**
   strengthen the "I". On self prompts the "I" is already at ceiling (100 % of
   answers) and nothing moves until α 0.3, where the text breaks. On requests that
   invite a third-person self-description (encyclopedia entry, "describe this
   assistant") ablate / down 0.1 / down 0.2 rank #26079 first of 7 in first person
   (any "I" 69 / 56 / 75 % vs 64 / 53 / 59 % for random features; judge "speaks as I"
   0.64 / 0.55 / 0.66 vs 0.54 / 0.46 / 0.52) — the right direction, but q ≥ 0.14.
   The "I" does not leak into other AIs or the world. Unsteered, #26079 fires at the
   prompt end only on self prompts (0.38 vs 0.00) and correlates *positively* with
   first person (ρ +0.34): it is not an "it" that suppresses the "I", it marks
   "the subject of what comes next is me". Asymmetry: the "I" is the default and is
   not held up by this piece; pushing the piece turns the self into a topic spoken
   about from outside, removing it does not make the speaker more of a subject.
   (Side note: asked «Describe a ChatGPT», unsteered Llama answers «I'm a large
   language model…» — it takes the other AI's name as its own.)

**Consequence:** the next vectors should come from the concept table, not from
anchor stimuli — the native feature's decoder direction for the SPECIFIC-FEATURE
axes (now 19 on the clean table), or a ridge direction on the unique part for the spread ones — and then go
through the same null-controlled fidelity loop.

## Next measurement, in order

1. **Fidelity of the v2 vectors + their random null**, same regime — closes defect 3
   and finally tests the artifact the kitchen was built to produce.
2. **A second alpha and a second seed** on the winning corner: the null-control
   ratio is currently a single-point estimate.
3. **Re-cook with the fixed labeler** (stages 2–6, ~4h on the free GPU, writes
   `v3_*` and leaves v2 intact): the cheapest available increase in judge quality,
   and the thing standing between 24 readable axes and more. Reuses
   `v2_sentences.json`, so v2 vs v3 R² is a paired read on the labeling fix alone.
   Bench the histogram on 50 sentences first — see `USAGE.md` §5.
4. Then, and only then, the periodic-table experiment in [`ROADMAP.md`](ROADMAP.md)
   — scaling a measurement we have not yet stabilised would just multiply the noise.
