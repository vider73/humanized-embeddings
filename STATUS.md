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
