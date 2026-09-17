# Launch teasers

Pick one. All English. The honest 26/104 result stays in — it's the hook, not a footnote.

---

## X / Twitter — thread version

**1/**
A sentence embedding is 384 numbers no human can read.

I trained a small net to translate those 384 numbers into **104 axes you actually
think in**: Hardness, Love, Religiosity, Entropy, Curiosity, Necessity…

Every concept becomes a profile you can read. 🧵

**2/**
How: a local Llama-3.1-8B acts as a *calibrated measuring instrument* — for each of 104
dimensions it scores a concept between two absolute anchors
(Temperature: absolute zero → Planck; Bitterness: water → denatonium).

5,654 concepts × 104 axes. Then a 384→104 net learns the map.

**3/**
Now you can do arithmetic in human terms (real outputs):
• `priest + technology − religion` → *laboratory, engineering, technologist*
• `god − religion` → *zeus, jesus, archangel*
• `science + soul` → *wisdom, mentor*

And find **unnamed regions** — coordinates the language hasn't named yet.

**4/**
The honest part 👇
A readable axis isn't automatically a *real* one. So I tested whether injecting each
axis into a live model actually steers it.

**11 of 104 axes survive the null control** — they steer the model *and* stay dead
when the same push comes from random vectors. Strongest: Transparency, Vitality,
Stress, Smell, Ethics, Divinity. (An earlier draft said 26; that judge failed its own
falsification test. See `STATUS.md`.)

**5/**
The weak tail? Fine-grained physics (density, viscosity, friction) — exactly where a
text model is ungrounded. Not a bug to hide; it's the next chapter.

Readable space: done. A quarter already drives behaviour. The rest is the agenda.

Code + writeup → [link]

---

## LinkedIn — single post

I trained a model to make AI embeddings **human-readable**.

Standard sentence embeddings are 384 opaque numbers. I built a translator that maps them
into **104 dimensions people already think in** — Hardness, Love, Religiosity, Entropy,
Curiosity, Necessity — by using a local LLM as a calibrated measuring instrument and
training a small network on 5,654 concepts scored across all 104 axes.

What it unlocks: readable concept profiles, semantic arithmetic with real outputs
(*priest + technology − religion → laboratory, engineering*; *god − religion → zeus,
jesus, archangel*), and maps of "unnamed" regions the language hasn't reached.

The result I care about most is the honest one. A readable axis isn't necessarily a
*causal* one — so I tested whether each axis can actually steer a live model. Then I
tested the test: I re-ran the whole thing with **random** vectors of the same shape.
The first judge scored the noise as highly as the real thing, so I threw it out and
rebuilt it. **11 of 104 axes survive that control** — they move the model and stay
dead under noise (paired over 104 dims, p = 0.005).

Interpretable embeddings are achievable, a tenth of the axes already drive behaviour,
and the falsification is in the repo next to the result.

Writeup + code → [link]

---

## One-liner (for a repo description / bio)

> Making AI embeddings human-readable: a 384→104 translator that turns opaque vectors
> into axes you can read (Love, Entropy, Hardness…). 26/104 verified as causal steering
> dials.
