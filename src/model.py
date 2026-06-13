"""Componente 3 — Modelo de predicción: Dixon-Coles + zoo IPredictor + blend.

Patrón IPredictor (idea de Oloráculo): cada modelo expone `predict(home, away,
neutral) -> Prediction` con un flag `degraded`. Un BlendModel combina Dixon-Coles
y Elo, con cascada de fallback (DC+Elo → Elo → uniforme) para nunca fallar.

  - UniformModel    : 1/3-1/3-1/3 (fallback final).
  - EloModel        : ratings Elo + modelo de empate por gap → 1X2.
  - DixonColesModel : penaltyblog (ventana 15a + pesos de recencia + neutral) →
                      1X2 + marcador exacto + over/under.
  - BlendModel      : w_dc·DC + w_elo·Elo (pesos a tunear por RPS).

Salida primaria: 1X2 (ganador). Secundaria: marcador modal y over 2.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel

try:
    from src.elo import compute_elo, elo_1x2, load_config
    from src.availability import adjusted_lambdas, outcome_from_lambdas, load_unavailable
except ImportError:  # pragma: no cover
    from elo import compute_elo, elo_1x2, load_config
    from availability import adjusted_lambdas, outcome_from_lambdas, load_unavailable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = ("H", "D", "A")  # local / empate / visita


# --------------------------------------------------------------------------- #
@dataclass
class Prediction:
    home: str
    away: str
    neutral: bool
    p_home: float
    p_draw: float
    p_away: float
    model: str
    degraded: bool = False
    exact_score: tuple[int, int] | None = None
    p_over25: float | None = None

    @property
    def probs(self) -> np.ndarray:
        return np.array([self.p_home, self.p_draw, self.p_away])

    @property
    def pick(self) -> str:
        return OUTCOMES[int(np.argmax(self.probs))]


def _normalize(p: np.ndarray) -> np.ndarray:
    s = p.sum()
    return p / s if s > 0 and np.isfinite(s) else np.array([1 / 3, 1 / 3, 1 / 3])


def _sharpen(p: np.ndarray, gamma: float) -> np.ndarray:
    """Afila (gamma>1 => más confianza) o suaviza (gamma<1) las probabilidades."""
    if gamma == 1.0:
        return p
    q = np.power(np.clip(p, 1e-12, None), gamma)
    return q / q.sum()


def _as_grid(grid_obj) -> np.ndarray:
    g = grid_obj.grid
    if callable(g):
        g = g()
    return np.asarray(g, dtype=float)


# --------------------------------------------------------------------------- #
# Predictores
# --------------------------------------------------------------------------- #
class UniformModel:
    name = "uniform"

    def predict(self, home: str, away: str, neutral: bool) -> Prediction:
        return Prediction(home, away, neutral, 1 / 3, 1 / 3, 1 / 3, self.name, degraded=False)


class EloModel:
    name = "elo"

    def __init__(self, ratings: dict, cfg: dict):
        self.ratings = ratings
        self.ha = cfg["elo"]["home_advantage_points"]
        self.start = cfg["elo"]["start_rating"]

    def predict(self, home: str, away: str, neutral: bool) -> Prediction:
        degraded = home not in self.ratings or away not in self.ratings
        rh = self.ratings.get(home, self.start)
        ra = self.ratings.get(away, self.start)
        p = elo_1x2(rh, ra, neutral, self.ha)
        return Prediction(home, away, neutral, *p, self.name, degraded=degraded)


class DixonColesModel:
    name = "dixon_coles"

    def __init__(self, fitted, team_counts: dict, min_matches: int, avail: dict | None = None):
        self.m = fitted
        self.team_counts = team_counts
        self.min_matches = min_matches
        self.avail = avail   # config de lesiones (None = desactivado)

    def _known(self, t: str) -> bool:
        return self.team_counts.get(t, 0) >= self.min_matches

    def predict(self, home: str, away: str, neutral: bool,
                out_home=(), out_away=()) -> Prediction:
        if not (self._known(home) and self._known(away)):
            return Prediction(home, away, neutral, 1 / 3, 1 / 3, 1 / 3, self.name, degraded=True)
        grid = self.m.predict(home, away, neutral_venue=bool(neutral))
        if self.avail and (out_home or out_away):
            # Ajuste por bajas: reescala las tasas de gol y reconstruye 1X2 + marcador
            lh, la = float(grid.home_goal_expectation), float(grid.away_goal_expectation)
            lh, la = adjusted_lambdas(lh, la, out_home, out_away, self.avail)
            ph, pdr, pa, score, over = outcome_from_lambdas(lh, la)
            return Prediction(home, away, neutral, ph, pdr, pa, self.name,
                              degraded=False, exact_score=score, p_over25=over)
        g = _as_grid(grid)
        i, j = np.unravel_index(np.argmax(g), g.shape)
        idx = np.indices(g.shape)
        p_over25 = float(g[(idx[0] + idx[1]) >= 3].sum())
        return Prediction(
            home, away, neutral,
            float(grid.home_win), float(grid.draw), float(grid.away_win),
            self.name, degraded=False,
            exact_score=(int(i), int(j)), p_over25=p_over25,
        )


class BlendModel:
    name = "blend"

    def __init__(self, dc: DixonColesModel, elo: EloModel, w_dc: float, w_elo: float,
                 sharpening: float = 1.0, unavailable: dict | None = None):
        self.dc, self.elo = dc, elo
        self.w_dc, self.w_elo = w_dc, w_elo
        self.sharpening = sharpening
        self.unavailable = unavailable or {}   # team -> [posiciones ausentes]
        self.uniform = UniformModel()

    def predict(self, home: str, away: str, neutral: bool) -> Prediction:
        oh = self.unavailable.get(home, [])
        oa = self.unavailable.get(away, [])
        dc = self.dc.predict(home, away, neutral, oh, oa)
        elo = self.elo.predict(home, away, neutral)
        g = self.sharpening

        if not dc.degraded and not elo.degraded:
            p = _sharpen(_normalize(self.w_dc * dc.probs + self.w_elo * elo.probs), g)
            return Prediction(home, away, neutral, *p, "blend", degraded=False,
                              exact_score=dc.exact_score, p_over25=dc.p_over25)
        if not elo.degraded:  # DC no tiene datos suficientes → caemos a Elo
            p = _sharpen(elo.probs, g)
            return Prediction(home, away, neutral, *p, "blend→elo",
                              degraded=False, exact_score=dc.exact_score, p_over25=dc.p_over25)
        if not dc.degraded:
            p = _sharpen(dc.probs, g)
            return Prediction(home, away, neutral, *p, "blend→dc", degraded=False,
                              exact_score=dc.exact_score, p_over25=dc.p_over25)
        u = self.uniform.predict(home, away, neutral)
        return Prediction(home, away, neutral, *u.probs, "blend→uniform", degraded=True)


# --------------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------------- #
def fit_dixon_coles(played: pd.DataFrame, cfg: dict) -> tuple[DixonColesModel, int]:
    """Ajusta Dixon-Coles sobre la ventana de los últimos N años con pesos de recencia."""
    mcfg = cfg["model"]
    window, half_life = mcfg["training_window_years"], mcfg["time_decay_half_life_years"]

    latest = played["date"].max()
    win = played[played["date"] >= latest - pd.DateOffset(years=window)].copy()
    years_ago = (latest - win["date"]).dt.days / 365.25
    win["w"] = 0.5 ** (years_ago / half_life)

    # Copias escribibles (pandas 3.0 Copy-on-Write devuelve arrays read-only,
    # que el kernel Cython de penaltyblog rechaza).
    gh = win["home_score"].to_numpy(dtype=np.int64, copy=True)
    ga = win["away_score"].to_numpy(dtype=np.int64, copy=True)
    th = win["home_team"].to_numpy(dtype=object, copy=True)
    ta = win["away_team"].to_numpy(dtype=object, copy=True)
    w = win["w"].to_numpy(dtype=np.float64, copy=True)
    nv = win["neutral"].to_numpy(dtype=np.int64, copy=True)

    fitted = DixonColesGoalModel(gh, ga, th, ta, w, nv)
    fitted.fit()

    counts = pd.concat([win["home_team"], win["away_team"]]).value_counts().to_dict()
    return DixonColesModel(fitted, counts, mcfg["min_team_matches"]), len(win)


def build_blend(played: pd.DataFrame, cfg: dict, verbose: bool = True, unavailable: dict | None = None) -> BlendModel:
    ratings, _ = compute_elo(played, cfg)
    dc, n_win = fit_dixon_coles(played, cfg)
    av = cfg.get("availability")
    dc.avail = av if (av and av.get("apply")) else None
    bw = cfg["model"]["blend_weights"]
    sharp = cfg["model"].get("sharpening", 1.0)
    if verbose:
        print(f"  · Elo sobre {len(played):,} partidos | Dixon-Coles sobre {n_win:,} (ventana {cfg['model']['training_window_years']}a)")
    return BlendModel(dc, EloModel(ratings, cfg), bw["dixon_coles"], bw["elo"], sharp, unavailable)


def home_field(home_team: str, away_team: str, country: str, hosts: list[str]) -> str | None:
    """Lado con localía: 'home', 'away' o None (neutral).

    Regla: un país anfitrión (Canadá/México/USA) jugando en su propio país
    tiene localía. El resto de los partidos del Mundial son neutrales.
    """
    if home_team in hosts and country == home_team:
        return "home"
    if away_team in hosts and country == away_team:
        return "away"
    return None


def predict_fixtures(fixtures: pd.DataFrame, model: BlendModel, hosts: list[str]) -> pd.DataFrame:
    rows = []
    for fx in fixtures.itertuples(index=False):
        side = home_field(fx.home_team, fx.away_team, fx.country, hosts)
        if side == "away":
            # El anfitrión es el visitante del fixture: lo pasamos como "home"
            # (penaltyblog/Elo aplican localía al primer equipo) y reorientamos.
            p = model.predict(fx.away_team, fx.home_team, neutral=False)
            p_home, p_draw, p_away = p.p_away, p.p_draw, p.p_home
            score = (p.exact_score[1], p.exact_score[0]) if p.exact_score else None
            localia = fx.away_team
        else:
            p = model.predict(fx.home_team, fx.away_team, neutral=(side is None))
            p_home, p_draw, p_away = p.p_home, p.p_draw, p.p_away
            score = p.exact_score
            localia = fx.home_team if side == "home" else "—"

        probs = np.array([p_home, p_draw, p_away])
        pick = OUTCOMES[int(np.argmax(probs))]
        rows.append({
            "date": fx.date, "home_team": fx.home_team, "away_team": fx.away_team,
            "localia": localia,
            "p_home": round(p_home, 4), "p_draw": round(p_draw, 4), "p_away": round(p_away, 4),
            "pick": pick, "model": p.model,
            "exact_score": f"{score[0]}-{score[1]}" if score else None,
            "p_over25": round(p.p_over25, 4) if p.p_over25 is not None else None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()
    clean_dir = PROJECT_ROOT / cfg["data"]["clean_dir"]
    played = pd.read_parquet(clean_dir / "matches_played.parquet")
    played["date"] = pd.to_datetime(played["date"])
    fixtures = pd.read_parquet(clean_dir / "fixtures_pending.parquet")
    fixtures["date"] = pd.to_datetime(fixtures["date"])

    av = cfg.get("availability", {})
    unavailable = load_unavailable(PROJECT_ROOT / av["unavailable_file"]) if av.get("apply") else {}
    print("Construyendo modelos:")
    model = build_blend(played, cfg, unavailable=unavailable)
    hosts = cfg["venue"]["host_nations"]

    preds = predict_fixtures(fixtures, model, hosts)
    feat_dir = PROJECT_ROOT / cfg["data"]["features_dir"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(feat_dir / "predictions.parquet", index=False)

    line = "─" * 80
    print(f"\n{line}\nPREDICCIONES — primeros 14 fixtures del Mundial 2026\n{line}")
    print(f"{'Fecha':<11}{'Partido':<34}{'Local':>7}{'Empate':>8}{'Visita':>8}  {'Pick':<7}{'Marc.':>6}")
    pick_name = {"H": "LOCAL", "D": "EMPATE", "A": "VISITA"}
    for r in preds.head(14).itertuples(index=False):
        match = f"{r.home_team} vs {r.away_team}"
        winner = r.home_team if r.pick == "H" else (r.away_team if r.pick == "A" else "Empate")
        print(f"{str(r.date.date()):<11}{match:<34}{r.p_home:>7.0%}{r.p_draw:>8.0%}{r.p_away:>8.0%}  "
              f"{pick_name[r.pick]:<7}{r.exact_score:>5}  → {winner}")
    print(f"{line}")

    # Efecto de la localía de anfitrión (comparando neutral vs con localía)
    print("\nEFECTO DE LA LOCALÍA DE ANFITRIÓN (P(anfitrión gana): neutral → con localía):")
    for fx in fixtures.itertuples(index=False):
        side = home_field(fx.home_team, fx.away_team, fx.country, hosts)
        if side is None:
            continue
        host = fx.home_team if side == "home" else fx.away_team
        rival = fx.away_team if side == "home" else fx.home_team
        pn = model.predict(host, rival, neutral=True)
        pl = model.predict(host, rival, neutral=False)
        print(f"  {host:<14} vs {rival:<14} ({fx.country}):  "
              f"{pn.p_home:>5.1%} → {pl.p_home:>5.1%}   (Δ +{pl.p_home - pn.p_home:.1%})")

    print(f"\nGuardado: {feat_dir / 'predictions.parquet'} ({len(preds)} fixtures)")
    n_localia = (preds['localia'] != "—").sum()
    deg = preds[preds['model'].str.contains('uniform|→', regex=True)]
    print(f"Fixtures con localía de anfitrión: {n_localia}  ·  con fallback (degradado): {len(deg)}")


if __name__ == "__main__":
    main()
