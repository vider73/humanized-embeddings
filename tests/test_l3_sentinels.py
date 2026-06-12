"""
L3 — Juez en bucle cerrado sobre dims CENTINELA. Opt-in: RUN_L3=1.

GPU, ~10-15 min. NUNCA con tuner/night_run activos (comparten VRAM).

Centinelas elegidas del afinador (afinacion.md, 2026-06-11), cada una a su
alpha optimo:
  d096_dificultad_de_ejecución  α=0.15  la robusta (verde a 0.15/0.25/0.35)
  d053_curiosidad               α=0.35  la potente (z_kin +4.2 a 0.35)
  d010_dureza                   α=0.15  presion baja (muere al apretar)
  d019_radiactividad            α=0.15  CONTROL MUERTO: si esta "pasa", el
                                        juez ha perdido la calibracion

Los umbrales son sobre la z CRUDA del report (con pocas dims medidas no hay
modo comun que restar), deliberadamente conservadores. Tras la primera
pasada estable, apretarlos con lo observado.

Ademas: test-retest — la misma dim medida dos veces en procesos distintos
(greedy) debe dar el mismo effect_vec. Es la respuesta a la anomalia
scorecard-manana vs tuner-tarde del 2026-06-11.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from steering import config

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "tests" / "_artifacts"

pytestmark = pytest.mark.l3

# (nombre, z cruda minima). Sign OK siempre exigido.
# Calibrado 2026-06-11 tras dos pasadas limpias: d096 z=3.07 (x2, identico),
# d010 z=1.47, d053@0.35 z=4.24, d019 z=0.03 rank 99. Margen ~50%.
EXPECT = {
    "d096_dificultad_de_ejecución": 2.0,
    "d010_dureza": 0.7,
    "d019_radiactividad": None,          # control muerto, ver test propio
}
EXPECT_A035 = {"d053_curiosidad": 2.5}
DEAD = "d019_radiactividad"


def _run_fidelity(alpha, dims, report):
    ART.mkdir(exist_ok=True)
    if report.exists():
        report.unlink()                  # siempre medida fresca
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "-m", "steering.fidelity",
           "--alpha", str(alpha), "--only", *map(str, dims),
           "--report", str(report)]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       timeout=2700)
    assert r.returncode == 0, (f"fidelity rc={r.returncode}\n"
                               f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads(report.read_text(encoding="utf-8"))


def _idx(names_needed):
    names = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))["dimension_names"]
    return [names.index(n) for n in names_needed]


@pytest.fixture(scope="module")
def report_a015():
    return _run_fidelity(0.15, _idx(list(EXPECT)), ART / "sentinel_a015.json")


@pytest.fixture(scope="module")
def report_a035():
    return _run_fidelity(0.35, _idx(list(EXPECT_A035)), ART / "sentinel_a035.json")


@pytest.fixture(scope="module")
def report_retest():
    return _run_fidelity(0.15, _idx(["d096_dificultad_de_ejecución"]),
                         ART / "sentinel_retest.json")


# ── manifest de regimen ──────────────────────────────────────────────────────
def test_report_carries_full_manifest(report_a015):
    meta = report_a015["_meta"]
    for k in ("probe_version", "alpha", "vectors", "vectors_sha", "mode", "layers"):
        assert k in meta, f"_meta sin '{k}' — reports no comparables entre regimenes"


# ── diales vivos ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", [n for n, z in EXPECT.items() if z is not None])
def test_live_sentinels_a015(report_a015, name):
    row = report_a015[name]
    assert row["sign_ok"], f"{name}: el empuje movio su dim en direccion CONTRARIA"
    assert row["z"] >= EXPECT[name], \
        f"{name}: z={row['z']:+.2f} < {EXPECT[name]} — dial validado ha degenerado"


def test_live_sentinel_a035(report_a035):
    name, zmin = next(iter(EXPECT_A035.items()))
    row = report_a035[name]
    assert row["sign_ok"] and row["z"] >= zmin, \
        f"{name}@0.35: z={row['z']:+.2f} sign_ok={row['sign_ok']}"


# ── control muerto: calibracion del juez ─────────────────────────────────────
def test_dead_control_stays_dead(report_a015):
    row = report_a015[DEAD]
    looks_real = row["z"] >= 1.5 and row["rank"] == 0 and row["sign_ok"]
    assert not looks_real, (
        f"{DEAD} parece un dial real (z={row['z']:+.2f}, rank={row['rank']}) — "
        "o el juez perdio la calibracion o la fisica desperto: investigar ANTES "
        "de fiarse de ningun otro verde")


# ── test-retest: la respuesta a la anomalia del 2026-06-11 ──────────────────
def test_retest_reproducibility(report_a015, report_retest):
    name = "d096_dificultad_de_ejecución"
    a = np.array(report_a015[name]["effect_vec"])
    b = np.array(report_retest[name]["effect_vec"])
    assert report_a015["_meta"]["vectors_sha"] == report_retest["_meta"]["vectors_sha"]
    assert np.allclose(a, b, atol=2e-3), (
        f"misma dim, mismo regimen, greedy: effect_vec difiere "
        f"(max delta={np.abs(a - b).max():.4f}). El pipeline NO es determinista "
        "=> los z entre runs no son comparables y la anomalia scorecard/tuner "
        "era varianza, no regimen")
    assert abs(report_a015[name]["z"] - report_retest[name]["z"]) < 0.5
