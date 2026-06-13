"""Componente 2 — Rating Elo rodante de selecciones.

Procesa TODA la historia cronológicamente y deja el rating actual de cada
selección. El Elo "olvida" lo viejo por construcción (cada partido mueve el
rating), así que el rating de hoy ya refleja la forma reciente.

Estilo World Football Elo:
  - Esperado: E_home = 1 / (1 + 10^((R_away - (R_home + ventaja_local)) / 400))
  - Actualización: R' = R + K · importancia · G · (resultado - esperado)
      · importancia: peso por tipo de torneo (config; Mundial 1.0, amistoso 0.4…)
      · G: multiplicador por diferencia de goles (goleadas mueven más el rating)
  - Ventaja de localía: +N puntos al local, ANULADA en sede neutral.

Expone también `elo_1x2`, que convierte dos ratings en probabilidades 1X2
usando un modelo de empate dependiente del gap (idea tomada de Oloráculo).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def goal_diff_multiplier(goal_diff: int) -> float:
    """Multiplicador G por margen (World Football Elo): goleadas pesan más."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def elo_expectation(r_home: float, r_away: float, home_adv: float = 0.0) -> float:
    """Probabilidad esperada del local (sin modelo de empate)."""
    return 1.0 / (1.0 + 10.0 ** ((r_away - (r_home + home_adv)) / 400.0))


def elo_1x2(
    r_home: float,
    r_away: float,
    neutral: bool,
    home_adv_points: float,
    amp: float = 0.30,
    scale: float = 550.0,
    floor: float = 0.08,
    lo: float = 0.08,
    hi: float = 0.34,
) -> np.ndarray:
    """Convierte ratings Elo en [P(local), P(empate), P(visita)].

    El empate decrece con el |gap| de rating (partidos parejos empatan más).
    Parámetros del modelo de empate = punto de partida (Oloráculo), a tunear por RPS.
    """
    h_adv = 0.0 if neutral else home_adv_points
    e = elo_expectation(r_home, r_away, h_adv)
    gap = abs((r_home + h_adv) - r_away)
    draw = min(hi, max(lo, amp * np.exp(-gap / scale) + floor))
    rem = 1.0 - draw
    p = np.array([e * rem, draw, rem * (1.0 - e)])
    return p / p.sum()


def compute_elo(matches: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Recorre los partidos en orden y devuelve (ratings_finales, tabla_resumen)."""
    e = cfg["elo"]
    start, K, ha = e["start_rating"], e["k_factor"], e["home_advantage_points"]
    tw = e["tournament_weight"]
    default_w = tw.get("default", 0.6)

    ratings: dict[str, float] = {}
    counts: dict[str, int] = {}
    last_date: dict[str, pd.Timestamp] = {}

    m = matches.sort_values("date")
    for row in m.itertuples(index=False):
        rh = ratings.get(row.home_team, start)
        ra = ratings.get(row.away_team, start)
        h_adv = 0.0 if row.neutral else ha
        exp_h = elo_expectation(rh, ra, h_adv)

        if row.home_score > row.away_score:
            s_h = 1.0
        elif row.home_score < row.away_score:
            s_h = 0.0
        else:
            s_h = 0.5

        imp = tw.get(row.tournament, default_w)
        g = goal_diff_multiplier(row.home_score - row.away_score)
        delta = K * imp * g * (s_h - exp_h)

        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
        for t in (row.home_team, row.away_team):
            counts[t] = counts.get(t, 0) + 1
            last_date[t] = row.date

    table = (
        pd.DataFrame(
            {
                "team": list(ratings.keys()),
                "elo": [round(v, 1) for v in ratings.values()],
                "matches": [counts[t] for t in ratings],
                "last_played": [last_date[t] for t in ratings],
            }
        )
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    return ratings, table


def main() -> None:
    cfg = load_config()
    played = pd.read_parquet(PROJECT_ROOT / cfg["data"]["clean_dir"] / "matches_played.parquet")
    played["date"] = pd.to_datetime(played["date"])

    ratings, table = compute_elo(played, cfg)

    feat_dir = PROJECT_ROOT / cfg["data"]["features_dir"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(feat_dir / "elo_ratings.parquet", index=False)
    print(f"Elo calculado sobre {len(played):,} partidos → {feat_dir / 'elo_ratings.parquet'}")

    line = "─" * 56
    print(f"\n{line}\nTOP 15 SELECCIONES POR ELO (sanity check)\n{line}")
    for r in table.head(15).itertuples(index=False):
        print(f"  {r.elo:>7.1f}  {r.team:<22} ({r.matches} pj)")

    # Las 48 del Mundial ordenadas por Elo
    fx = pd.read_parquet(PROJECT_ROOT / cfg["data"]["clean_dir"] / "fixtures_pending.parquet")
    wc = sorted(set(fx["home_team"]) | set(fx["away_team"]))
    wc_tbl = table[table["team"].isin(wc)].reset_index(drop=True)
    print(f"\n{line}\nLAS 48 DEL MUNDIAL POR ELO\n{line}")
    for i, r in enumerate(wc_tbl.itertuples(index=False), 1):
        print(f"  {i:>2}. {r.elo:>7.1f}  {r.team}")


if __name__ == "__main__":
    main()
