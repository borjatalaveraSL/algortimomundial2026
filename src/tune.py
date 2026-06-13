"""Tuning post-hoc — optimiza el peso del blend (Dixon-Coles vs Elo) y el
afilado (gamma) sobre las predicciones de un backtest ya guardado, minimizando RPS.

No re-entrena nada: opera sobre las prob. por-modelo guardadas por evaluate.py,
así que es instantáneo. (El half-life sí requiere re-entrenar y se barre aparte.)

Uso:
    python -m src.tune --preds backtest_predictions.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_pairs(path: Path) -> pd.DataFrame:
    """Une, por partido, las prob. de Dixon-Coles y de Elo + el resultado real."""
    r = pd.read_parquet(path)
    keys = ["date", "home_team", "away_team"]
    dc = (r[r["model"] == "dixon_coles"][keys + ["p_home", "p_draw", "p_away", "actual"]]
          .rename(columns={"p_home": "dc_h", "p_draw": "dc_d", "p_away": "dc_a"}))
    elo = (r[r["model"] == "elo"][keys + ["p_home", "p_draw", "p_away"]]
           .rename(columns={"p_home": "el_h", "p_draw": "el_d", "p_away": "el_a"}))
    return dc.merge(elo, on=keys)


def blend_sharp(m: pd.DataFrame, w: float, gamma: float) -> np.ndarray:
    dc = np.vstack([m["dc_h"], m["dc_d"], m["dc_a"]]).T
    el = np.vstack([m["el_h"], m["el_d"], m["el_a"]]).T
    p = w * dc + (1 - w) * el
    p = p / p.sum(1, keepdims=True)
    if gamma != 1.0:
        p = np.power(np.clip(p, 1e-12, None), gamma)
        p = p / p.sum(1, keepdims=True)
    return p


def rps(p: np.ndarray, y: np.ndarray) -> float:
    ph, pdr = p[:, 0], p[:, 1]
    yH = (y == "H").astype(float)
    yD = (y == "D").astype(float)
    return float((((ph - yH) ** 2 + ((ph + pdr) - (yH + yD)) ** 2) / 2.0).mean())


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    # Apila las prob. de los 3 resultados con sus indicadores reales en el mismo orden.
    preds = np.concatenate([p[:, 0], p[:, 1], p[:, 2]])
    obs = np.concatenate([(y == "H").astype(float), (y == "D").astype(float), (y == "A").astype(float)])
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(preds, edges) - 1, 0, bins - 1)
    rows = []
    for k in range(bins):
        mk = idx == k
        if mk.sum():
            rows.append((f"{edges[k]:.1f}-{edges[k+1]:.1f}", round(preds[mk].mean(), 3),
                         round(obs[mk].mean(), 3), int(mk.sum())))
    return pd.DataFrame(rows, columns=["bin", "pred", "real", "n"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="backtest_predictions.parquet")
    args = ap.parse_args()

    m = load_pairs(PROJECT_ROOT / "data" / "features" / args.preds)
    y = m["actual"].to_numpy()

    ws = np.round(np.linspace(0, 1, 21), 2)
    gs = np.round(np.linspace(1.0, 1.8, 17), 2)
    best = None
    for w in ws:
        for g in gs:
            r = rps(blend_sharp(m, w, g), y)
            if best is None or r < best[2]:
                best = (float(w), float(g), r)
    bw, bg, br = best

    rps_default = rps(blend_sharp(m, 0.5, 1.0), y)   # blend 50/50 actual
    rps_dconly = rps(blend_sharp(m, 1.0, 1.0), y)    # Dixon-Coles solo

    line = "─" * 56
    print(f"\n{line}\nTUNING POST-HOC sobre {args.preds}  (n={len(m):,} partidos)\n{line}")
    print(f"  RPS blend 50/50 (actual) ...... {rps_default:.5f}")
    print(f"  RPS Dixon-Coles solo .......... {rps_dconly:.5f}")
    print(f"  RPS ÓPTIMO .................... {br:.5f}   @ w_dc={bw:.2f}  gamma={bg:.2f}")
    print(f"  Mejora vs blend 50/50 ......... {1 - br/rps_default:.2%}")

    print(f"\n{line}\nCalibración: blend 50/50 → óptimo\n{line}")
    cal_def = calibration(blend_sharp(m, 0.5, 1.0), y).set_index("bin")
    cal_opt = calibration(blend_sharp(m, bw, bg), y).set_index("bin")
    print(f"{'bin':<10}{'pred':>7}{'real(def)':>11}{'real(opt)':>11}{'n':>7}")
    for b in cal_opt.index:
        rd = cal_def.loc[b, "real"] if b in cal_def.index else float("nan")
        ro = cal_opt.loc[b, "real"]
        print(f"{b:<10}{cal_opt.loc[b,'pred']:>7.3f}{rd:>11.3f}{ro:>11.3f}{int(cal_opt.loc[b,'n']):>7}")

    print(f"\nTUNE_BEST w_dc={bw:.2f} gamma={bg:.2f} rps={br:.5f} "
          f"rps_default={rps_default:.5f} rps_dconly={rps_dconly:.5f}")


if __name__ == "__main__":
    main()
