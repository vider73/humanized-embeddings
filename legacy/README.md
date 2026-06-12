# legacy/ — superseded experiments, kept for the record

Nothing here is part of the live pipeline. Each file was a stepping stone;
what replaced it is noted below. See `ARCHITECTURE.md` for the current system.

| File | What it was | Superseded by |
|---|---|---|
| `index.py` / `index2.py` | Early concept scorers (HF transformers, console UI) | `20_score_concepts.py` |
| `view.py` | Inspector for the old `dataset_humanizer.json` format | dataset format change (`dataset_metadata.json` + npy) |
| `semantic_attention_injection.py` | v3 steering via logit bias (llama-cpp / Ollama prompt engineering) | true activation steering in `steering/` |
| `cognitive_generator_v4.py` | Tkinter GUI over the logit-bias era engine | `steering/console.py` + `steering/ui.py` |
| `build_global_dictionary.py` | 20k frequency-word embedding dictionary (`global_*.npy/json`) | corpus pipeline (`60_…`–`80_…`) |
| `test_global_dictionary.py` | Manual checks for the global dictionary | — (legacy with its subject) |
| `test_synthesizer.py` | Manual eval of the reverse map (human → embedding) | `55_eval_translator.py` covers the forward eval; reverse eval pending |
| `test_cycle.py` / `test_cycle_complete.py` / `test_cycle_complete2.py` | Round-trip experiments (profile → embedding → neighbours) | `semantic_console.py` |
| `test_gui2.py` | GUI experiment over the global dictionary | `steering/console.py` |
| `global_embeddings.npy` / `global_words.json` | Data for the global dictionary experiments | corpus table (`tabla/`) |

Automated tests live in `tests/` (pytest). These `test_*.py` files predate that
convention and are manual scripts, not pytest suites.
