"""Componente 5 — Ajuste por lesiones / disponibilidad (inference-time).

Las bajas CONFIRMADAS de cada selección ajustan las tasas de goles esperadas
(lambdas) del modelo Dixon-Coles según el rol del ausente, y de ahí se reconstruye
la predicción del partido. NO es un feature de entrenamiento: no hay histórico de
lesiones etiquetado, así que se aplica solo a los partidos por jugar.

Magnitudes (Δataque propio, Δdefensa→sube goleo rival) tomadas de Oloraculo:
heurísticas, NO validadas con datos → tratar como prior ajustable en config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

try:
    from src.team_names import canonical_name
except ImportError:  # pragma: no cover
    from team_names import canonical_name


def role_impact(pos: str, table: dict) -> tuple[float, float]:
    return tuple(table.get(str(pos).strip().lower(), table["default"]))


def team_impacts(positions, table: dict, cap: float) -> tuple[float, float]:
    """(impacto_ataque, impacto_defensa) de un equipo, capeados, dado los ausentes."""
    atk = min(cap, float(sum(role_impact(p, table)[0] for p in positions)))
    dfn = min(cap, float(sum(role_impact(p, table)[1] for p in positions)))
    return atk, dfn


def adjusted_lambdas(lam_h, lam_a, home_pos, away_pos, av: dict):
    """Ajusta las tasas de gol por las bajas de cada lado."""
    table, cap, floor = av["role_impact"], av["cap"], av["attack_floor"]
    h_atk, h_def = team_impacts(home_pos, table, cap)
    a_atk, a_def = team_impacts(away_pos, table, cap)
    lh = lam_h * max(floor, 1 - h_atk) * (1 + a_def)   # mis atacantes out → marco menos; def/arq rival out → marco más
    la = lam_a * max(floor, 1 - a_atk) * (1 + h_def)
    return lh, la


def outcome_from_lambdas(lam_h, lam_a, rho: float = -0.05, max_goals: int = 15):
    """Reconstruye 1X2 + marcador modal + over2.5 desde dos tasas Poisson (con tau Dixon-Coles)."""
    lh = min(max(lam_h, 0.05), 6.0)
    la = min(max(lam_a, 0.05), 6.0)
    h = poisson.pmf(np.arange(max_goals + 1), lh)
    a = poisson.pmf(np.arange(max_goals + 1), la)
    g = np.outer(h, a)
    g[0, 0] *= 1 - lh * la * rho
    g[0, 1] *= 1 + lh * rho
    g[1, 0] *= 1 + la * rho
    g[1, 1] *= 1 - rho
    g = np.clip(g, 0.0, None)
    g /= g.sum()
    p_home = float(np.tril(g, -1).sum())   # filas > columnas → local marca más
    p_away = float(np.triu(g, 1).sum())
    p_draw = float(np.trace(g))
    i, j = np.unravel_index(int(np.argmax(g)), g.shape)
    idx = np.indices(g.shape)
    over = float(g[(idx[0] + idx[1]) >= 3].sum())
    return p_home, p_draw, p_away, (int(i), int(j)), over


def load_unavailable(path: Path) -> dict:
    """team (canónico) -> lista de posiciones de los ausentes confirmados."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "team" not in df.columns:
        return {}
    if "status" in df.columns:  # quedarse solo con bajas confirmadas
        st = df["status"].astype(str).str.strip().str.lower()
        df = df[st.isin(["out", "confirmed", "baja", "confirmada", "nan", ""])]
    out: dict[str, list] = {}
    for r in df.itertuples(index=False):
        out.setdefault(canonical_name(r.team), []).append(getattr(r, "position", "default"))
    return out
