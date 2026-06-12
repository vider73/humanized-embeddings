"""
L1 — Contratos de artefactos. Solo lectura, sin modelos, segundos.

Vigila los tres contratos que comparten las dos capas (metadata, translator,
tabla Y) y los artefactos del steering. Un fallo aqui significa que algun
artefacto se regenero rompiendo a sus consumidores silenciosamente.
"""
import json
import re

import numpy as np
import pytest

from steering import config


# ── dataset_metadata.json: EL contrato compartido ────────────────────────────
def test_metadata_has_104_sorted_dims(dim_names):
    assert len(dim_names) == 104
    assert dim_names == sorted(dim_names), "el orden d000..d103 esta clavado por contrato"
    assert all(re.match(r"^d\d{3}_", n) for n in dim_names)
    assert dim_names[0].startswith("d000_") and dim_names[-1].startswith("d103_")


def test_metadata_concepts_match_tensors(metadata, X, Y):
    n = len(metadata["concepts"])
    assert X.shape == (n, 384), f"X {X.shape} != ({n}, 384)"
    assert Y.shape == (n, 104), f"Y {Y.shape} != ({n}, 104)"


# ── tensores de entrenamiento ────────────────────────────────────────────────
def test_tensors_are_finite(X, Y):
    assert np.isfinite(X).all(), "X contiene NaN/inf"
    assert np.isfinite(Y).all(), "Y contiene NaN/inf"


@pytest.mark.xfail(reason="BUG CONOCIDO 2026-06-11: 31 valores >1.0 (max 6.54) en "
                          "tamaño_físico/duración/temperatura — el parse de 1.py no "
                          "clampo siempre. Contamina la seleccion de polos de "
                          "derive_vectors. Arreglar = clip + re-derivar esas dims.",
                   strict=False)
def test_Y_range_strict(Y):
    assert Y.min() >= 0.0 and Y.max() <= 1.0


def test_Y_range_no_regression(Y, dim_names):
    """El bug conocido no debe CRECER: si aparecen nuevos fuera de rango, parar."""
    out = int((Y > 1.0).sum() + (Y < 0.0).sum())
    assert out <= 31, (
        f"{out} valores fuera de [0,1] (linea base conocida: 31). "
        "Una regeneracion de Y ha empeorado el clamp del parser.")
    assert Y.max() <= 6.6, f"outlier nuevo mas grave que el conocido: max={Y.max():.3f}"


# ── master_dimensions_prompts.json ───────────────────────────────────────────
def test_prompts_contract(prompts, dim_names):
    assert len(prompts) == 104
    ids = [p["id"] for p in prompts]
    assert ids == dim_names, "los ids de los prompts SON los dimension_names del metadata"
    for p in prompts:
        assert p["scale"]["min"] and p["scale"]["max"], f"{p['id']}: anclas incompletas"
        assert p["llama3_prompt"].strip(), f"{p['id']}: prompt vacio"


# ── control vectors + meta ───────────────────────────────────────────────────
def test_control_vectors_shape_and_norms(cv, cv_meta):
    layers = cv_meta["target_layers"]
    assert cv.ndim == 3 and cv.shape[:2] == (len(layers), 104), f"shape {cv.shape}"
    norms = np.linalg.norm(cv, axis=2)
    assert np.allclose(norms, 1.0, atol=1e-3), \
        f"vectores no unitarios: norms en [{norms.min():.4f}, {norms.max():.4f}]"


def test_control_meta_matches_dataset(cv_meta, dim_names):
    assert cv_meta["dim_names"] == dim_names, \
        "dim_names del control_meta != metadata — neuronas y diales DESALINEADOS"


def test_inject_layers_subset_of_derived(cv_meta):
    derived = set(cv_meta["target_layers"])
    assert set(config.INJECT_LAYERS) <= derived, \
        f"INJECT_LAYERS {config.INJECT_LAYERS} no derivadas (hay {sorted(derived)})"
    assert set(config.TARGET_LAYERS) == derived, \
        "config.TARGET_LAYERS cambio respecto a los vectores en disco: re-derivar"


def test_config_artifact_naming():
    name = config.CONTROL_VECTORS_FILE.name
    if config.DERIVE_MODE == "caa":
        assert "_caa" in name
    if config.WHITEN:
        assert "_white" in name
    assert config.CONTROL_VECTORS_FILE.exists(), f"falta {name}"
    assert config.CONTROL_META_FILE.exists()


# ── stimuli CAA ──────────────────────────────────────────────────────────────
def test_stimuli_full_coverage(stimuli, dim_names):
    missing = [d for d in dim_names
               if not stimuli.get(d, {}).get("pos") or not stimuli.get(d, {}).get("neg")]
    assert not missing, f"{len(missing)} dims sin frases ricas: {missing[:5]}"


def test_stimuli_pools_are_rich(stimuli):
    """Pools demasiado pequenos degeneran en direccion lexica."""
    thin = {k: (len(v.get('pos', [])), len(v.get('neg', [])))
            for k, v in stimuli.items()
            if len(v.get('pos', [])) < 5 or len(v.get('neg', [])) < 5}
    assert not thin, f"pools con <5 frases por polo: {thin}"


# ── checkpoint del traductor: topologia clavada ──────────────────────────────
def test_translator_checkpoint_topology(dim_names):
    torch = pytest.importorskip("torch")
    from steering.translator import SemanticTranslator
    state = torch.load(config.TRANSLATOR_PATH, map_location="cpu",
                       weights_only=True)        # state dict puro: sin pickle libre
    model = SemanticTranslator(384, len(dim_names))
    model.load_state_dict(state)        # strict: keys Y shapes deben cuadrar
    out_features = model.net[-2].out_features
    assert out_features == 104


# ── manifests de regimen en los reports de fidelity ──────────────────────────
REQUIRED_META = ("probe_version", "alpha", "vectors", "mode", "layers")


def test_fidelity_reports_carry_regime_manifest():
    reports = [p for p in config.VEC_DIR.glob("fidelity_*.json")
               if "_old_" not in p.name]
    if not reports:
        pytest.skip("aun no hay reports de fidelity")
    bad = {}
    for p in reports:
        meta = json.loads(p.read_text(encoding="utf-8")).get("_meta", {})
        missing = [k for k in REQUIRED_META if k not in meta]
        if missing:
            bad[p.name] = missing
    assert not bad, (
        f"reports sin partida de nacimiento (renombrar con _old_): {bad}")
