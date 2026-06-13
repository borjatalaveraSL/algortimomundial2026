"""Componente 4 — Evaluación y backtesting walk-forward.

Mide qué tan bueno es el modelo prediciendo el GANADOR (1X2), sin look-ahead:
para cada bloque de fechas del período de prueba, re-entrena usando SOLO los
partidos previos y predice el bloque. Compara cada predictor (Uniforme, Elo,
Dixon-Coles, Blend) con métricas propias de probabilidades.

Métricas:
  - RPS (Ranked Probability Score): PRIMARIA. Respeta el orden Local<Empate<Visita;
    penaliza menos errar "por poco". Menor = mejor.
  - Brier, Log-loss: scoring probabilístico estándar. Menor = mejor.
  - Accuracy del pick: % de veces que el resultado más probable acertó.
  - Calibración por bins: que cuando el modelo dice X%, ocurra ~X%.

Uso:
    python -m src.evaluate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:  # python -m src.evaluate
    from .model import BlendModel, UniformModel, build_blend
    from .elo import load_config
else:            # python src/evaluate.py
    from model import BlendModel, UniformModel, build_blend
    from elo import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Métricas (vectorizadas)
# --------------------------------------------------------------------------- #
def compute_metrics(df: pd.DataFrame) -> dict:
    """df con columnas p_home, p_draw, p_away, actual (H/D/A)."""
    ph = df["p_home"].to_numpy(float)
    pdr = df["p_draw"].to_numpy(float)
    pa = df["p_away"].to_numpy(float)
    y = df["actual"].to_numpy()

    yH = (y == "H").astype(float)
    yD = (y == "D").astype(float)
    yA = (y == "A").astype(float)

    # RPS (acumulado sobre el orden H < D < A)
    rps = ((ph - yH) ** 2 + ((ph + pdr) - (yH + yD)) ** 2) / 2.0
    # Brier (multiclase)
    brier = (ph - yH) ** 2 + (pdr - yD) ** 2 + (pa - yA) ** 2
    # Log-loss
    p_act = np.where(y == "H", ph, np.where(y == "D", pdr, pa))
    logloss = -np.log(np.clip(p_act, 1e-12, 1.0))
    # Accuracy del pick
    pick = np.argmax(np.vstack([ph, pdr, pa]).T, axis=1)
    actual_idx = np.where(y == "H", 0, np.where(y == "D", 1, 2))
    acc = (pick == actual_idx).astype(float)

    return {
        "n": int(len(df)),
        "RPS": float(rps.mean()),
        "Brier": float(brier.mean()),
        "LogLoss": float(logloss.mean()),
        "Acc": float(acc.mean()),
    }


def leaderboard(records: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = [{"model": m, **compute_metrics(records[records["model"] == m])} for m in models]
    return pd.DataFrame(rows).sort_values("RPS").reset_index(drop=True)


def calibration_table(records: pd.DataFrame, model: str, bins: int) -> pd.DataFrame:
    """Calibración por-resultado del modelo: agrupa todas las prob. predichas
    (de los 3 resultados) y compara prob. media predicha vs frecuencia real."""
    b = records[records["model"] == model]
    preds = np.concatenate([b["p_home"], b["p_draw"], b["p_away"]]).astype(float)
    obs = np.concatenate(
        [(b["actual"] == "H"), (b["actual"] == "D"), (b["actual"] == "A")]
    ).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(preds, edges) - 1, 0, bins - 1)
    rows = []
    for k in range(bins):
        m = idx == k
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{edges[k]:.1f}-{edges[k+1]:.1f}",
            "pred_medio": round(preds[m].mean(), 3),
            "frec_real": round(obs[m].mean(), 3),
            "n": int(m.sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Backtest walk-forward
# --------------------------------------------------------------------------- #
def walk_forward(played: pd.DataFrame, cfg: dict, quiet: bool = False) -> pd.DataFrame:
    bt = cfg["backtest"]
    refit = pd.Timedelta(days=bt["refit_every_days"])
    played = played.sort_values("date").reset_index(drop=True)
    latest = played["date"].max()
    test_start = latest - pd.DateOffset(years=bt["test_years_back"])
    test = played[played["date"] >= test_start]
    n_chunks = int(np.ceil((latest - test_start).days / bt["refit_every_days"]))

    if not quiet:
        print(f"Backtest walk-forward: período {test_start.date()} → {latest.date()}  "
              f"({len(test):,} partidos, ~{n_chunks} re-entrenamientos)")

    records = []
    cur = test_start
    chunk_i = 0
    while cur <= latest:
        end = cur + refit
        chunk = test[(test["date"] >= cur) & (test["date"] < end)]
        if len(chunk):
            chunk_i += 1
            train = played[played["date"] < cur]
            blend = build_blend(train, cfg, verbose=False)
            predictors = {
                "uniform": UniformModel(),
                "elo": blend.elo,
                "dixon_coles": blend.dc,
                "blend": blend,
            }
            for mt in chunk.itertuples(index=False):
                for name, mdl in predictors.items():
                    p = mdl.predict(mt.home_team, mt.away_team, bool(mt.neutral))
                    records.append({
                        "date": mt.date, "home_team": mt.home_team, "away_team": mt.away_team,
                        "tournament": mt.tournament, "neutral": bool(mt.neutral),
                        "model": name, "p_home": p.p_home, "p_draw": p.p_draw, "p_away": p.p_away,
                        "actual": mt.result,
                    })
            if not quiet:
                print(f"  chunk {chunk_i:>2}: {cur.date()} ({len(chunk):>4} partidos, "
                      f"entrenado con {len(train):,})")
        cur = end
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest walk-forward del predictor.")
    ap.add_argument("--half-life", type=float, help="override del half-life de decaimiento (años)")
    ap.add_argument("--refit-days", type=int, help="override de cada cuántos días re-entrenar")
    ap.add_argument("--out", type=str, default="backtest_predictions.parquet", help="archivo de salida en data/features/")
    ap.add_argument("--quiet", action="store_true", help="suprime tablas detalladas (solo línea TUNE_RESULT)")
    args = ap.parse_args()

    cfg = load_config()
    if args.half_life is not None:
        cfg["model"]["time_decay_half_life_years"] = args.half_life
    if args.refit_days is not None:
        cfg["backtest"]["refit_every_days"] = args.refit_days

    played = pd.read_parquet(PROJECT_ROOT / cfg["data"]["clean_dir"] / "matches_played.parquet")
    played["date"] = pd.to_datetime(played["date"])

    records = walk_forward(played, cfg, quiet=args.quiet)
    feat_dir = PROJECT_ROOT / cfg["data"]["features_dir"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_path = feat_dir / args.out
    records.to_parquet(out_path, index=False)

    models = ["uniform", "elo", "dixon_coles", "blend"]
    lb_all = leaderboard(records, models).set_index("model")
    hl = cfg["model"]["time_decay_half_life_years"]
    rd = cfg["backtest"]["refit_every_days"]
    print(f"TUNE_RESULT half_life={hl} refit={rd} n={len(records)//len(models)} "
          f"uniform={lb_all.loc['uniform','RPS']:.5f} elo={lb_all.loc['elo','RPS']:.5f} "
          f"dixon_coles={lb_all.loc['dixon_coles','RPS']:.5f} blend={lb_all.loc['blend','RPS']:.5f} "
          f"out={out_path.name}")

    if args.quiet:
        return

    line = "─" * 60

    def show_board(name, recs):
        lb = leaderboard(recs, models)
        print(f"\n{line}\n{name}  (n={len(recs)//len(models):,} partidos)\n{line}")
        print(f"{'modelo':<14}{'RPS↓':>8}{'Brier↓':>9}{'LogLoss↓':>10}{'Acc↑':>8}")
        for r in lb.itertuples(index=False):
            print(f"{r.model:<14}{r.RPS:>8.4f}{r.Brier:>9.4f}{r.LogLoss:>10.4f}{r.Acc:>8.1%}")

    show_board("LEADERBOARD — TODOS los partidos del período", records)

    # Cut competitivo: sin amistosos (más cercano a condiciones de torneo)
    comp = records[records["tournament"] != "Friendly"]
    show_board("LEADERBOARD — solo COMPETITIVOS (sin amistosos)", comp)

    # Calibración del blend
    cal = calibration_table(records, "blend", cfg["backtest"]["calibration_bins"])
    print(f"\n{line}\nCALIBRACIÓN del blend (pred. medio vs frecuencia real)\n{line}")
    print(f"{'bin':<10}{'pred_medio':>12}{'frec_real':>11}{'n':>8}")
    for r in cal.itertuples(index=False):
        flag = "  ✓" if abs(r.pred_medio - r.frec_real) <= 0.05 else "  ⚠"
        print(f"{r.bin:<10}{r.pred_medio:>12.3f}{r.frec_real:>11.3f}{r.n:>8}{flag}")

    print(f"\nGuardado: {feat_dir / 'backtest_predictions.parquet'}")
    blend_rps = leaderboard(records, models).set_index("model").loc["blend", "RPS"]
    unif_rps = leaderboard(records, models).set_index("model").loc["uniform", "RPS"]
    print(f"\nRPS blend = {blend_rps:.4f}  vs  uniforme = {unif_rps:.4f}  "
          f"(mejora {1 - blend_rps/unif_rps:.1%})")


if __name__ == "__main__":
    main()
