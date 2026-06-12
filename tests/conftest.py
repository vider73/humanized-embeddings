"""
conftest.py — Suite de judge tests de embtoconcept + steering.

Niveles (ver ARCHITECTURE.md §6):
  L0  matematica pura          siempre          (milisegundos)
  L1  contratos de artefactos  siempre          (segundos, solo lectura)
  L2  mecanica small-model     RUN_L2=1         (~1 min, CPU ok)
  L3  juez en bucle cerrado    RUN_L3=1         (GPU, minutos; NO con el tuner activo)

Uso:
  python -m pytest                      # L0 + L1
  set RUN_L2=1 && python -m pytest      # + L2
  set RUN_L3=1 && python -m pytest tests/test_l3_sentinels.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent          # .../embtoconcept
sys.path.insert(0, str(ROOT))

from steering import config                            # import ligero (solo pathlib)


def _need(path) -> Path:
    p = Path(path)
    if not p.exists():
        pytest.skip(f"artefacto ausente: {p.name}")
    return p


# ── fixtures de artefactos (solo lectura, cacheadas por sesion) ─────────────
@pytest.fixture(scope="session")
def metadata():
    return json.loads(_need(config.METADATA_FILE).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def dim_names(metadata):
    return metadata["dimension_names"]


@pytest.fixture(scope="session")
def Y():
    return np.load(_need(config.DATA_Y_FILE))


@pytest.fixture(scope="session")
def X():
    return np.load(_need(config.ROOT / "dataset_X_embeddings.npy"))


@pytest.fixture(scope="session")
def prompts():
    return json.loads(
        _need(config.ROOT / "master_dimensions_prompts.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def cv_meta():
    return json.loads(_need(config.CONTROL_META_FILE).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def cv(cv_meta):
    return np.load(_need(config.CONTROL_VECTORS_FILE))


@pytest.fixture(scope="session")
def stimuli():
    return json.loads(_need(config.STIMULI_FILE).read_text(encoding="utf-8"))


# ── gating por nivel ─────────────────────────────────────────────────────────
def pytest_collection_modifyitems(items):
    skip_l2 = pytest.mark.skip(reason="L2: exporta RUN_L2=1 para correrlo")
    skip_l3 = pytest.mark.skip(reason="L3: exporta RUN_L3=1 (GPU; no con tuner activo)")
    for it in items:
        if "l2" in it.keywords and not os.environ.get("RUN_L2"):
            it.add_marker(skip_l2)
        if "l3" in it.keywords and not os.environ.get("RUN_L3"):
            it.add_marker(skip_l3)
