"""
L2 — Mecanica con modelos pequenos. Opt-in: RUN_L2=1. CPU vale (~1-2 min).

  [hooks]  smoke_test sobre SmolLM2-135M: engancha, silencio=identidad,
           steering!=0 cambia la generacion, camino completo 104->cv->inyeccion.
  [juez]   el Humanizer (MiniLM + translator) es DETERMINISTA y devuelve
           perfiles validos. Si el juez deriva, toda fidelity queda en duda
           — este es el test que separa "dial muerto" de "juez borracho".
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.l2

JUDGE_TEXTS = (
    "El diamante corta el vidrio sin esfuerzo.",
    "Una habitacion vacia.",
    "Dios creo el cielo y la tierra en siete dias.",
)


def test_smoke_hooks_on_small_model():
    """Corre steering/smoke_test.py tal cual (invariantes A-D) y exige exito."""
    env = {**os.environ, "PYTHONUTF8": "1"}      # los emojis del smoke vs cp1252
    r = subprocess.run([sys.executable, "-m", "steering.smoke_test"],
                       cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 0, f"smoke_test fallo:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"


@pytest.fixture(scope="module")
def judge():
    from steering.translator import Humanizer
    return Humanizer(device="cpu")


def test_judge_profile_is_valid(judge):
    p = judge.profile(JUDGE_TEXTS[0])
    assert p.shape == (104,)
    assert np.isfinite(p).all()
    assert p.min() >= 0.0 and p.max() <= 1.0, "sigmoid => perfil en [0,1]"


def test_judge_is_deterministic(judge):
    """Mismo texto dos veces => perfil IDENTICO. La regla del instrumento."""
    for txt in JUDGE_TEXTS:
        a, b = judge.profile(txt), judge.profile(txt)
        assert np.array_equal(a, b), f"el juez no es determinista para: «{txt}»"


def test_judge_discriminates(judge):
    """Sanity minimo: dureza(diamante) > dureza(frase neutra)."""
    i = judge.dim_names.index("d010_dureza")
    hard = judge.profile(JUDGE_TEXTS[0])[i]
    neutral = judge.profile(JUDGE_TEXTS[1])[i]
    assert hard > neutral, (
        f"d010_dureza: diamante={hard:.3f} <= neutra={neutral:.3f} — "
        "el juez no distingue ni lo obvio; revisar checkpoint/embedder")
