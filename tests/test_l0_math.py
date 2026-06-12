"""
L0 — Mathematical invariants of the steering, no GPU and no model.

Replicates in numpy the exact algebra of steering_model.set_profile / hook
("add" and "clamp" modes), geometry.effective_rank and analyze.scorecard.
If anything here fails, NOTHING above it is interpretable: stop.
"""
import json

import numpy as np
import pytest

from steering.geometry import effective_rank
from steering.analyze import scorecard

rng = np.random.default_rng(7)
H, N_DIMS = 64, 104


# ── exact replicas of the algebra in steering_model.py ──────────────────────
def _mix_add(profile, cv):
    """set_profile 'add' mode: v = (p-0.5) @ cv, unit-norm; None if flat."""
    v = (profile - 0.5) @ cv
    n = np.linalg.norm(v)
    return None if n < 1e-4 else v / n


def _active(profile):
    """active-dims detection: w = (p-0.5)/0.5, |w| > 1e-3."""
    w = (np.asarray(profile, np.float32) - 0.5) / 0.5
    return np.where(np.abs(w) > 1e-3)[0], w


def _clamp_update(h, V, w, alpha, kp=1.0, ridge=1e-4):
    """hook 'clamp' mode (multi-clamp with Gram inverse), as in the code."""
    G = V @ V.T
    Ginv = np.linalg.inv(G + ridge * np.eye(len(V)))
    norm = np.linalg.norm(h, axis=-1, keepdims=True)
    proj = h @ V.T
    targets = alpha * norm * w
    delta = (targets - proj) @ Ginv
    return h + kp * (delta @ V), targets


def _unit(v):
    return v / np.linalg.norm(v)


# ── critical invariant: flat profile = silence ───────────────────────────────
def test_flat_profile_is_silent():
    cv = rng.standard_normal((N_DIMS, H)).astype(np.float32)
    flat = np.full(N_DIMS, 0.5, np.float32)
    active, _ = _active(flat)
    assert len(active) == 0, "perfil plano no debe activar ninguna dim"
    assert _mix_add(flat, cv) is None, "perfil plano debe dar steering nulo en 'add'"


def test_near_flat_below_threshold_is_silent():
    p = np.full(N_DIMS, 0.5, np.float32)
    p[10] = 0.5 + 0.4e-3          # below the 1e-3 threshold of set_profile
    active, _ = _active(p)
    assert len(active) == 0


# ── directional coherence (add mode) ─────────────────────────────────────────
def test_single_dim_add_aligns_with_its_vector():
    cv = np.stack([_unit(v) for v in rng.standard_normal((N_DIMS, H))]).astype(np.float32)
    p = np.full(N_DIMS, 0.5, np.float32)
    p[42] = 1.0
    v = _mix_add(p, cv)
    cos = float(v @ cv[42])
    assert cos > 0.999, f"empujar solo la dim 42 debe alinear con cv_42 (cos={cos:.4f})"


def test_single_dim_low_antialigns():
    cv = np.stack([_unit(v) for v in rng.standard_normal((N_DIMS, H))]).astype(np.float32)
    p = np.full(N_DIMS, 0.5, np.float32)
    p[7] = 0.0
    v = _mix_add(p, cv)
    assert float(v @ cv[7]) < -0.999, "dim a 0.0 debe empujar en direccion -cv_i"


# ── clamp mode: setpoint, no-op and chords without stepping on each other ────
def test_clamp_kp1_reaches_setpoint():
    V = np.stack([_unit(rng.standard_normal(H))])           # (1, H)
    w = np.array([1.0])
    h = rng.standard_normal((5, H))                          # 5 positions
    h2, targets = _clamp_update(h, V, w, alpha=0.2)
    proj = h2 @ V.T
    assert np.allclose(proj, targets, rtol=1e-2, atol=1e-3), \
        "con KP=1 la proyeccion post-hook debe quedar EN el setpoint alpha*|h|*w"


def test_clamp_noop_when_already_at_setpoint():
    V = np.stack([_unit(rng.standard_normal(H))])
    w = np.array([1.0])
    alpha = 0.2
    base = rng.standard_normal((3, H))
    base -= (base @ V.T) @ V                                 # h orthogonal to V
    norm = np.linalg.norm(base, axis=-1, keepdims=True)
    h = base + (alpha * norm * w) @ V                        # already at the setpoint*
    h2, _ = _clamp_update(h, V, w, alpha=alpha)
    # *adding the component changes |h| a bit; we demand a minuscule correction
    assert np.abs(h2 - h).max() < 0.05 * np.abs(h).max(), \
        "donde el modelo ya obedece, clamp no debe anadir casi nada"


def test_multiclamp_chord_no_crosstalk():
    """Two overlapping chords (cos 0.6): each one must reach ITS setpoint."""
    e = np.eye(H)
    V = np.stack([e[0], _unit(0.6 * e[0] + 0.8 * e[1])])     # (2, H), overlap 0.6
    w = np.array([1.0, -1.0])                                # chord: one goes up, the other down
    h = rng.standard_normal((4, H))
    h2, targets = _clamp_update(h, V, w, alpha=0.15)
    proj = h2 @ V.T
    assert np.allclose(proj, targets, rtol=2e-2, atol=2e-3), \
        "la inversa de Gram debe repartir el empuje sin que las notas se pisen"


# ── geometry.effective_rank ──────────────────────────────────────────────────
def test_effective_rank_orthonormal():
    k = 12
    M = np.eye(H)[:k]
    pr, erank = effective_rank(M)
    assert abs(pr - k) < 1e-6 and abs(erank - k) < 1e-6


def test_effective_rank_detects_collapse():
    M = np.tile(_unit(rng.standard_normal(H)), (N_DIMS, 1))
    pr, _ = effective_rank(M)
    assert pr < 1.01, "104 copias del mismo vector = rango efectivo 1 (colapso)"


# ── analyze.scorecard as a pure function (synthetic fixture) ─────────────────
def test_scorecard_on_synthetic_report(tmp_path, dim_names, Y):
    """An obvious hero must come out on top with sign OK; an inverted one, sign INV."""
    measured = dim_names[:15]
    hero, inverted = measured[0], measured[1]
    report = {"_meta": {"nota": "fixture sintetico, ignorar"}}
    for i, name in enumerate(measured):
        vec = (rng.standard_normal(len(dim_names)) * 0.004)
        own = 0.5 if name == hero else (-0.5 if name == inverted else 0.02)
        vec[dim_names.index(name)] = own
        report[name] = {"effect_vec": [round(float(x), 4) for x in vec],
                        "domain": "test"}
    report["sin_vector"] = {"domain": "test"}                # must be ignored
    rf = tmp_path / "fixture_report.json"
    rf.write_text(json.dumps(report), encoding="utf-8")

    out = scorecard(report_path=rf, kin_r=0.4)
    by_name = {d["name"]: d for d in out}
    assert set(by_name) == set(measured), "_meta y entradas sin effect_vec fuera"
    assert by_name[hero]["sign_ok"] and by_name[hero]["z_kin"] > 3.0
    assert not by_name[inverted]["sign_ok"]
    zs = [d["z_kin"] for d in out]
    assert zs == sorted(zs, reverse=True), "scorecard debe venir ordenado por z_kin desc"


# ── fidelity._domain: the probe router ───────────────────────────────────────
def test_domain_router_covers_all_dims(dim_names):
    from steering.fidelity import _domain, PROBES
    for name in dim_names:
        dom = _domain(name)
        assert dom in PROBES, f"{name} -> dominio desconocido '{dom}'"


@pytest.mark.parametrize("name,expected", [
    ("d010_dureza", "materia"),
    ("d053_curiosidad", "mente"),
    ("d099_necesidad", "sociedad"),
    ("d036_sabor_dulzor", "sentidos"),
    ("d024_instinto", "vida"),
    ("d019_radiactividad", "materia"),
])
def test_domain_router_spot_checks(name, expected):
    from steering.fidelity import _domain
    assert _domain(name) == expected
