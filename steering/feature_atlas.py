"""
feature_atlas.py — Pack feature_map's output into one self-contained HTML page.

Reads steering/vectors/feature_map_llama.{json,npz} + _summary.json and fills
steering/atlas_template.html -> FEATURE_ATLAS.html (≈ 6 MB, binary arrays as
base64 so 32k points × 24 label axes × 104 behaviour axes stay small).

  python -m steering.feature_atlas
"""
import base64
import json

import numpy as np

from . import config

SRC_JSON = config.VEC_DIR / "feature_map_llama.json"
SRC_NPZ = config.VEC_DIR / "feature_map_llama.npz"
SRC_SUM = config.VEC_DIR / "feature_map_llama_summary.json"
TEMPLATE = config.ROOT / "steering" / "atlas_template.html"
OUT = config.ROOT / "FEATURE_ATLAS.html"


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def main():
    js = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    sm = json.loads(SRC_SUM.read_text(encoding="utf-8"))
    z = np.load(SRC_NPZ)
    readable = z["readable"]
    ridx = np.where(readable)[0]
    lz = z["label_z"][:, ridx].astype(np.float32)
    lz8 = np.where(np.isfinite(lz), np.clip(np.round(lz * 20), -127, 127), -128).astype(np.int8)
    alive_idx = np.where(z["alive"])[0].astype(np.uint16)
    br = np.nan_to_num(z["beh_r"][alive_idx].astype(np.float32))
    br8 = np.clip(np.round(br * 250), -127, 127).astype(np.int8)
    axes = js["axes"]
    data = {
        "n": len(z["xy"]),
        "xy": b64(z["xy"].astype(np.float32)),
        "regions": b64(z["regions"].astype(np.uint8)),
        "n_regions": int(z["regions"].max()) + 1,
        "labels": [f["l"] for f in js["features"]],
        "axes": axes,
        "axis_en": {a["id"]: a["en"] for a in axes},
        "readable_idx": ridx.tolist(),
        "label_z": b64(lz8),
        "alive_idx": b64(alive_idx),
        "beh_r": b64(br8),
        "dials": sm["dials"],
        "agree": sm["agree"],
        "cover_lab": sm["cover_lab"],
        "cover_beh": sm["cover_beh"],
        "facts": {
            "n_labelled": sm["n_labelled"], "n_alive": sm["n_alive"],
            "n_unnamed": sm["n_unnamed"],
            "rho_readable": float(np.mean([a["rho"] for a in sm["agree"] if a["readable"]])),
        },
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", payload)
    OUT.write_text(html, encoding="utf-8")
    print(f"🗺️  {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
