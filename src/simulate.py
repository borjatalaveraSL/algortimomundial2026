"""Componente 7 (parcial) — Simulación Monte Carlo de la FASE DE GRUPOS.

Para cada grupo (12 grupos × 4 selecciones, formato Mundial 2026):
  1. Genera los 6 cruces round-robin del grupo.
  2. Predice cada cruce con el modelo tuneado: el RESULTADO (1X2) sale del blend
     (mejor calibrado) y la FORMA del marcador sale de la grilla Dixon-Coles.
  3. Simula N torneos: muestrea marcadores, arma la tabla con desempates FIFA
     (puntos → dif. de gol → goles a favor → azar), y rankea 1º-4º.
  4. Clasificación: 1º y 2º de cada grupo (directo) + los 8 mejores terceros
     (rankeados globalmente) → 32 clasificados.

Salida: posiciones finales esperadas, prob. de ganar el grupo / clasificar,
puntos esperados; se exporta a JSON y se inyecta en un front local autocontenido.

Uso:
    python -m src.simulate
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.model import build_blend, home_field
    from src.elo import load_config
    from src.team_names import canonical_name
except ImportError:  # pragma: no cover
    from model import build_blend, home_field
    from elo import load_config
    from team_names import canonical_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Predicción por partido: probs 1X2 (blend) + grilla de goles (Dixon-Coles)
# --------------------------------------------------------------------------- #
def _dc_grid(blend, home: str, away: str, neutral: bool) -> np.ndarray:
    g = blend.dc.m.predict(home, away, neutral_venue=bool(neutral)).grid
    if callable(g):
        g = g()
    return np.asarray(g, dtype=float)


def match_distributions(blend, home: str, away: str, side: str | None):
    """Devuelve (probs 1X2 [H,D,A], grilla de goles) en orientación home/away."""
    if side == "away":  # el anfitrión figura como visitante → lo paso como local y reoriento
        bp = blend.predict(away, home, neutral=False)
        probs = np.array([bp.p_away, bp.p_draw, bp.p_home])
        grid = _dc_grid(blend, away, home, False).T
    else:
        nv = side is None
        bp = blend.predict(home, away, neutral=nv)
        probs = bp.probs
        grid = _dc_grid(blend, home, away, nv)
    return probs, grid


def sample_goals(grid: np.ndarray, probs: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Muestrea n marcadores: el resultado sigue `probs` (blend), la forma del
    marcador sigue la grilla DC condicionada a ese resultado."""
    m = grid.shape[0]
    ii, jj = np.indices((m, m))
    flat = grid.reshape(-1)
    masks = {0: (ii > jj).reshape(-1), 1: (ii == jj).reshape(-1), 2: (ii < jj).reshape(-1)}
    outcomes = rng.choice(3, size=n, p=probs / probs.sum())
    hg = np.zeros(n, dtype=int)
    ag = np.zeros(n, dtype=int)
    for o in (0, 1, 2):
        sel = outcomes == o
        ns = int(sel.sum())
        if ns == 0:
            continue
        cells = np.where(masks[o])[0]
        cp = flat[cells]
        cp = cp / cp.sum()
        chosen = rng.choice(cells, size=ns, p=cp)
        hg[sel] = chosen // m
        ag[sel] = chosen % m
    return hg, ag


# --------------------------------------------------------------------------- #
# Resolución de cada cruce contra los fixtures reales (o sintético)
# --------------------------------------------------------------------------- #
def resolve_match(a: str, b: str, lookup: dict, hosts: list[str]):
    """Devuelve (home, away, country, scheduled). Si el cruce está agendado usa
    el fixture real; si no, lo sintetiza (el anfitrión juega de local en su país)."""
    key = frozenset([a, b])
    if key in lookup:
        r = lookup[key]
        return r["home"], r["away"], r["country"], True
    host = a if a in hosts else (b if b in hosts else None)
    if host:
        return host, (b if host == a else a), host, False
    return a, b, "", False


def actuals_to_training(adf: pd.DataFrame) -> pd.DataFrame:
    """Convierte los resultados reales al esquema de entrenamiento (para Elo + Dixon-Coles)."""
    def _bool(v):
        return str(v).strip().lower() in ("true", "1", "yes", "t")
    rows = []
    for r in adf.itertuples(index=False):
        rows.append({
            "date": pd.Timestamp(str(r.date)),
            "home_team": canonical_name(r.home), "away_team": canonical_name(r.away),
            "home_score": int(r.home_goals), "away_score": int(r.away_goals),
            "tournament": "FIFA World Cup", "neutral": _bool(r.neutral),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bracket del Mundial 2026 (estructura oficial; portada de Oloraculo)
#   slot: ("W", grupo)=ganador · ("R", grupo)=segundo ·
#         ("T", [grupos])=tercero (opciones) · ("WO", tie_id)=ganador de un cruce
# --------------------------------------------------------------------------- #
R32 = [
    (73, ("R", "A"), ("R", "B")),
    (74, ("W", "E"), ("T", ["A", "B", "C", "D", "F"])),
    (75, ("W", "F"), ("R", "C")),
    (76, ("W", "C"), ("R", "F")),
    (77, ("W", "I"), ("T", ["C", "D", "F", "G", "H"])),
    (78, ("R", "E"), ("R", "I")),
    (79, ("W", "A"), ("T", ["C", "E", "F", "H", "I"])),
    (80, ("W", "L"), ("T", ["E", "H", "I", "J", "K"])),
    (81, ("W", "D"), ("T", ["B", "E", "F", "I", "J"])),
    (82, ("W", "G"), ("T", ["A", "E", "H", "I", "J"])),
    (83, ("R", "K"), ("R", "L")),
    (84, ("W", "H"), ("R", "J")),
    (85, ("W", "B"), ("T", ["E", "F", "G", "I", "J"])),
    (86, ("W", "J"), ("R", "H")),
    (87, ("W", "K"), ("T", ["D", "E", "I", "J", "L"])),
    (88, ("R", "D"), ("R", "G")),
]
R16 = [
    (89, ("WO", 74), ("WO", 77)), (90, ("WO", 73), ("WO", 75)),
    (91, ("WO", 76), ("WO", 78)), (92, ("WO", 79), ("WO", 80)),
    (93, ("WO", 83), ("WO", 84)), (94, ("WO", 81), ("WO", 82)),
    (95, ("WO", 86), ("WO", 88)), (96, ("WO", 85), ("WO", 87)),
]
QF = [(97, ("WO", 89), ("WO", 90)), (98, ("WO", 93), ("WO", 94)),
      (99, ("WO", 91), ("WO", 92)), (100, ("WO", 95), ("WO", 96))]
SF = [(101, ("WO", 97), ("WO", 98)), (102, ("WO", 99), ("WO", 100))]
FINAL = (104, ("WO", 101), ("WO", 102))
KO_TIES = R32 + R16 + QF + SF + [FINAL]
R32_IDS = [t[0] for t in R32]
R16_IDS = [t[0] for t in R16]
QF_IDS = [t[0] for t in QF]
SF_IDS = [t[0] for t in SF]


def assign_thirds(qualified: tuple) -> dict:
    """Asigna los 8 grupos con tercero clasificado a los 8 slots de R32
    respetando las opciones oficiales (backtracking = tabla FIFA 2026)."""
    qset = set(qualified)
    slots = [(tie_id, s[1]) for tie_id, sa, sb in R32 for s in (sa, sb) if s[0] == "T"]
    slots.sort(key=lambda s: (sum(g in qset for g in s[1]), s[0]))
    assigned, used = {}, set()

    def rec(i):
        if i == len(slots):
            return True
        tie_id, opts = slots[i]
        for g in sorted((g for g in opts if g in qset), key=lambda x: ord(x) - 65):
            if g in used:
                continue
            used.add(g); assigned[tie_id] = g
            if rec(i + 1):
                return True
            used.discard(g); assigned.pop(tie_id, None)
        return False

    rec(0)
    return assigned


def load_pen_skill(cfg, k: float = 6.0):
    """Devuelve skill(team) = win-rate en definiciones por penales, con shrinkage
    hacia 0.5 (pseudo-conteo k). Captura el temple/presión de cada selección."""
    raw = PROJECT_ROOT / cfg["data"]["raw_dir"] / cfg["data"]["shootouts_file"]
    src = raw if raw.exists() else f'{cfg["data"]["base_url"]}/{cfg["data"]["shootouts_file"]}'
    s = pd.read_csv(src)
    for c in ("home_team", "away_team", "winner"):
        s[c] = s[c].map(lambda x: canonical_name(x) if isinstance(x, str) else x)
    wins, n = {}, {}
    for r in s.itertuples(index=False):
        n[r.home_team] = n.get(r.home_team, 0) + 1
        n[r.away_team] = n.get(r.away_team, 0) + 1
        if isinstance(r.winner, str):
            wins[r.winner] = wins.get(r.winner, 0) + 1

    def skill(t):
        return (wins.get(t, 0) + k * 0.5) / (n.get(t, 0) + k)
    return skill


# --------------------------------------------------------------------------- #
# Simulación
# --------------------------------------------------------------------------- #
def run(blend, blend_pre, groups: dict, lookup: dict, hosts: list[str], n: int, seed: int,
        actuals: dict, pen_skill):
    """`blend` (post, con lo jugado) predice/simula lo que falta;
    `blend_pre` (sin lo jugado) predice los partidos jugados para un tracker honesto (sin leakage)."""
    rng = np.random.default_rng(seed)
    gnames = sorted(groups)

    agg = {t: dict(p1=0.0, p2=0.0, p3=0.0, p4=0.0, pts=0.0, gd=0.0, rank=0.0,
                   q_top2=0.0, q_third=0.0) for g in gnames for t in groups[g]}
    matches_out: dict[str, list] = {g: [] for g in gnames}
    track = []  # comparación predicción vs resultado real (partidos jugados)

    # Pre-muestreo de goles por grupo (los jugados se FIJAN al resultado real)
    presampled = {}
    for g in gnames:
        teams = groups[g]
        sims = []
        for a, b in combinations(teams, 2):
            home, away, country, scheduled = resolve_match(a, b, lookup, hosts)
            side = home_field(home, away, country, hosts)
            act = actuals.get(frozenset([a, b]))
            # Jugado → predicción con el modelo PRE (honesto, sin leakage); por jugar → modelo POST (actualizado)
            model = blend_pre if act else blend
            probs, grid = match_distributions(model, home, away, side)
            i, j = np.unravel_index(int(np.argmax(grid)), grid.shape)
            localia = home if side == "home" else (away if side == "away" else None)
            entry = {
                "home": home, "away": away,
                "p_home": round(float(probs[0]), 3), "p_draw": round(float(probs[1]), 3),
                "p_away": round(float(probs[2]), 3), "modal": f"{int(i)}-{int(j)}",
                "localia": localia, "scheduled": scheduled,
                "played": False, "actual": None, "correct": None,
            }
            if act:  # partido jugado → marcador FIJO en todas las simulaciones
                hgv, agv = (act["hg"], act["ag"]) if act["home"] == home else (act["ag"], act["hg"])
                hg = np.full(n, hgv, dtype=int)
                ag = np.full(n, agv, dtype=int)
                y = "H" if hgv > agv else ("A" if agv > hgv else "D")
                pick = ("H", "D", "A")[int(np.argmax(probs))]
                p_act = float(probs[{"H": 0, "D": 1, "A": 2}[y]])
                rps = float(((probs[0] - (y == "H")) ** 2 +
                             ((probs[0] + probs[1]) - ((y == "H") + (y == "D"))) ** 2) / 2)
                entry.update(played=True, actual=f"{hgv}-{agv}", correct=bool(pick == y))
                track.append({"group": g, "home": home, "away": away,
                              "p_home": entry["p_home"], "p_draw": entry["p_draw"], "p_away": entry["p_away"],
                              "actual": f"{hgv}-{agv}", "outcome": y, "pick": pick,
                              "correct": bool(pick == y), "p_actual": round(p_act, 3), "rps": round(rps, 3)})
            else:
                hg, ag = sample_goals(grid, probs, n, rng)
            sims.append((home, away, hg, ag))
            matches_out[g].append(entry)
        presampled[g] = sims

    # Tablas por grupo + recolección de terceros
    thirds_key, thirds_team = [], []
    winner_arr, runnerup_arr, third_arr = {}, {}, {}
    for g in gnames:
        teams = groups[g]
        idx = {t: i for i, t in enumerate(teams)}
        pts = np.zeros((4, n)); gf = np.zeros((4, n)); ga = np.zeros((4, n))
        for home, away, hg, ag in presampled[g]:
            hi, ai = idx[home], idx[away]
            pts[hi] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
            pts[ai] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
            gf[hi] += hg; ga[hi] += ag
            gf[ai] += ag; ga[ai] += hg
        gd = gf - ga
        # Clave de orden: puntos → dif. gol → goles a favor → azar (desempate)
        key = pts * 1e6 + (gd + 100) * 1e3 + gf * 10 + rng.random((4, n))
        order = np.argsort(-key, axis=0)      # order[0] = ganador del grupo
        ranks = np.argsort(order, axis=0)     # rank por equipo (0 = mejor)
        teams_arr = np.array(teams)
        winner_arr[g] = teams_arr[order[0]]
        runnerup_arr[g] = teams_arr[order[1]]
        third_arr[g] = teams_arr[order[2]]

        for t in teams:
            i = idx[t]
            agg[t]["p1"] += float((ranks[i] == 0).sum())
            agg[t]["p2"] += float((ranks[i] == 1).sum())
            agg[t]["p3"] += float((ranks[i] == 2).sum())
            agg[t]["p4"] += float((ranks[i] == 3).sum())
            agg[t]["pts"] += float(pts[i].sum())
            agg[t]["gd"] += float(gd[i].sum())
            agg[t]["rank"] += float((ranks[i] + 1).sum())
            agg[t]["q_top2"] += float((ranks[i] <= 1).sum())

        third_i = (ranks == 2).argmax(axis=0)
        thirds_key.append(key[third_i, np.arange(n)])
        thirds_team.append(np.array(teams)[third_i])

    # Mejores 8 terceros (ranking global de los 12 terceros por iteración)
    TK = np.vstack(thirds_key)          # (12, n)
    TT = np.vstack(thirds_team)         # (12, n)
    third_rank = np.argsort(np.argsort(-TK, axis=0), axis=0)
    advances = third_rank < 8           # (12, n)
    for gi, g in enumerate(gnames):
        for t in groups[g]:
            agg[t]["q_third"] += float(((TT[gi] == t) & advances[gi]).sum())

    # ----- FASE DE ELIMINACIÓN (bracket 2026, por iteración) -----
    from functools import lru_cache
    ko = {t: dict(r16=0, qf=0, sf=0, final=0, champ=0) for g in gnames for t in groups[g]}
    reg_cache, pen_cache = {}, {}

    def regprob(a, b):
        v = reg_cache.get((a, b))
        if v is None:
            p = blend.predict(a, b, neutral=True)   # eliminatoria = sede neutral
            v = (p.p_home, p.p_draw, p.p_away)
            reg_cache[(a, b)] = v
        return v

    def penprob(a, b):  # P(a gana la definición por penales) según temple histórico
        v = pen_cache.get((a, b))
        if v is None:
            sa, sb = pen_skill(a), pen_skill(b)
            v = sa / (sa + sb)
            pen_cache[(a, b)] = v
        return v

    @lru_cache(maxsize=None)
    def thirds_for(q):
        return assign_thirds(q)

    def team_of(slot, tie_id, i, tw, tassign):
        k = slot[0]
        if k == "W":
            return winner_arr[slot[1]][i]
        if k == "R":
            return runnerup_arr[slot[1]][i]
        if k == "T":
            return third_arr[tassign[tie_id]][i]
        return tw[slot[1]]  # WinnerOf

    U = rng.random((n, 2 * len(KO_TIES) + 2))
    for i in range(n):
        qualified = tuple(gnames[gi] for gi in range(len(gnames)) if advances[gi, i])
        tassign = thirds_for(qualified)
        tw, dptr = {}, 0
        for tie_id, sa, sb in KO_TIES:
            a = team_of(sa, tie_id, i, tw, tassign)
            b = team_of(sb, tie_id, i, tw, tassign)
            pa, _pd, pb = regprob(a, b)
            u = U[i, dptr]; dptr += 1
            if u < pa:
                w = a
            elif u < pa + pb:
                w = b
            else:  # empate en regulación → penales (factor presión)
                up = U[i, dptr]; dptr += 1
                w = a if up < penprob(a, b) else b
            tw[tie_id] = w
        for tid in R32_IDS:
            ko[tw[tid]]["r16"] += 1
        for tid in R16_IDS:
            ko[tw[tid]]["qf"] += 1
        for tid in QF_IDS:
            ko[tw[tid]]["sf"] += 1
        for tid in SF_IDS:
            ko[tw[tid]]["final"] += 1
        ko[tw[104]]["champ"] += 1

    for t in ko:
        for kk in ko[t]:
            ko[t][kk] /= n

    for t, r in agg.items():
        for k in ("p1", "p2", "p3", "p4", "q_top2", "q_third"):
            r[k] /= n
        r["pts"] /= n
        r["gd"] /= n
        r["rank"] /= n
        r["qualify"] = r["q_top2"] + r["q_third"]

    return agg, matches_out, gnames, track, ko


# --------------------------------------------------------------------------- #
def build_payload(agg, matches_out, gnames, groups, elo, cfg, track, ko):
    out_groups = []
    for g in gnames:
        teams = sorted(groups[g], key=lambda t: agg[t]["rank"])  # posición final esperada
        rows = []
        for pos, t in enumerate(teams, 1):
            r = agg[t]
            rows.append({
                "team": t, "pos": pos, "elo": int(round(elo.get(t, 1500))),
                "p_first": round(r["p1"], 3), "p_second": round(r["p2"], 3),
                "p_third": round(r["p3"], 3), "p_fourth": round(r["p4"], 3),
                "p_top2": round(r["q_top2"], 3), "p_third_adv": round(r["q_third"], 3),
                "p_qualify": round(r["qualify"], 3), "p_champion": round(ko[t]["champ"], 3),
                "exp_points": round(r["pts"], 2), "exp_gd": round(r["gd"], 2),
            })
        out_groups.append({"group": g, "teams": rows, "matches": matches_out[g]})

    all_teams = [t for g in gnames for t in groups[g]]
    title_race = sorted(
        ({"team": t, "elo": int(round(elo.get(t, 1500))),
          "p_champion": round(ko[t]["champ"], 4), "p_final": round(ko[t]["final"], 3),
          "p_sf": round(ko[t]["sf"], 3), "p_qf": round(ko[t]["qf"], 3),
          "p_r16": round(ko[t]["r16"], 3)} for t in all_teams),
        key=lambda x: -x["p_champion"])

    n_played = len(track)
    correct = sum(1 for t in track if t["correct"])
    rps_live = round(sum(t["rps"] for t in track) / n_played, 3) if n_played else None
    return {
        "meta": {
            "n_sims": cfg["simulation"]["n_sims"],
            "model": "Blend 70% Dixon-Coles / 30% Elo · half-life 6a · γ=1.15",
            "rps": 0.164, "as_of": cfg["ingest"]["as_of_date"],
            "n_teams": sum(len(v) for v in groups.values()), "n_groups": len(gnames),
            "hosts": cfg["venue"]["host_nations"],
            "n_played": n_played, "picks_correct": correct, "rps_live": rps_live,
        },
        "groups": out_groups,
        "played": track,
        "title_race": title_race,
    }


def render_html(payload: dict, web_dir: Path):
    template = (web_dir / "template.html").read_text(encoding="utf-8")
    html = template.replace("__WC_DATA__", json.dumps(payload, ensure_ascii=False))
    # index.html en la RAÍZ del repo: así GitHub Pages lo sirve en la URL raíz
    # (Pages busca index.html en la raíz; si no, renderiza el README).
    (PROJECT_ROOT / "index.html").write_text(html, encoding="utf-8")
    (web_dir / "standings.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    cfg = load_config()
    clean = PROJECT_ROOT / cfg["data"]["clean_dir"]
    feat = PROJECT_ROOT / cfg["data"]["features_dir"]

    played = pd.read_parquet(clean / "matches_played.parquet")
    played["date"] = pd.to_datetime(played["date"])
    fixtures = pd.read_parquet(clean / "fixtures_pending.parquet")
    elo = dict(zip(pd.read_parquet(feat / "elo_ratings.parquet")["team"],
                   pd.read_parquet(feat / "elo_ratings.parquet")["elo"]))

    # Grupos (canonicalizados para alinear con el dataset)
    gdf = pd.read_csv(PROJECT_ROOT / cfg["simulation"]["groups_file"])
    gdf["team"] = gdf["team"].map(canonical_name)
    groups = gdf.groupby("group")["team"].apply(list).to_dict()

    # Lookup de fixtures por par de equipos
    lookup = {frozenset([r.home_team, r.away_team]):
              {"home": r.home_team, "away": r.away_team, "country": r.country, "neutral": bool(r.neutral)}
              for r in fixtures.itertuples(index=False)}
    hosts = cfg["venue"]["host_nations"]

    # Resultados reales ya jugados (se fijan en la simulación Y se realimentan al modelo)
    actuals, adf = {}, None
    apath = PROJECT_ROOT / cfg["simulation"]["actual_results_file"]
    if apath.exists():
        adf = pd.read_csv(apath)
        adf["home"] = adf["home"].map(canonical_name)
        adf["away"] = adf["away"].map(canonical_name)
        actuals = {frozenset([r.home, r.away]):
                   {"home": r.home, "away": r.away, "hg": int(r.home_goals), "ag": int(r.away_goals)}
                   for r in adf.itertuples(index=False)}
    print(f"Resultados reales cargados: {len(actuals)}")

    # Realimentar lo jugado al entrenamiento (Elo + Dixon-Coles)
    played_aug = pd.concat([played, actuals_to_training(adf)], ignore_index=True) if actuals else played

    print("Construyendo modelo POST (actualizado con lo jugado)…")
    blend_post = build_blend(played_aug, cfg)
    # Modelo PRE (sin lo jugado) para evaluar esos partidos sin leakage
    blend_pre = build_blend(played, cfg, verbose=False) if actuals else blend_post

    if actuals:  # mostrar el efecto en el Elo
        pre, post = blend_pre.elo.ratings, blend_post.elo.ratings
        teams_played = sorted({t for k in actuals.values() for t in (k["home"], k["away"])})
        print("\nEfecto en el Elo (pre → post de incorporar lo jugado):")
        for t in teams_played:
            d = post.get(t, 1500) - pre.get(t, 1500)
            print(f"  {t:<24} {pre.get(t,1500):>7.1f} → {post.get(t,1500):>7.1f}  ({d:+.1f})")

    pen_skill = load_pen_skill(cfg)
    n, seed = cfg["simulation"]["n_sims"], cfg["simulation"]["seed"]
    print(f"\nSimulando {n:,} torneos completos (grupos + eliminatorias hasta la final)…")
    agg, matches_out, gnames, track, ko = run(blend_post, blend_pre, groups, lookup, hosts, n, seed, actuals, pen_skill)

    payload = build_payload(agg, matches_out, gnames, groups, elo, cfg, track, ko)
    web_dir = PROJECT_ROOT / cfg["simulation"]["web_dir"]
    render_html(payload, web_dir)

    # Resumen en consola
    line = "─" * 64
    print(f"\n{line}\nPOSICIONES FINALES ESPERADAS POR GRUPO\n{line}")
    for grp in payload["groups"]:
        print(f"\nGrupo {grp['group']}:")
        for t in grp["teams"]:
            tag = "★" if t["pos"] == 1 else (" " if t["pos"] <= 2 else "·")
            print(f"  {tag} {t['pos']}. {t['team']:<22} "
                  f"clasifica {t['p_qualify']:>5.0%} | gana grupo {t['p_first']:>4.0%} | "
                  f"pts {t['exp_points']:.1f}")
    print(f"\n{line}\nCAMINO AL TÍTULO — top favoritos a campeón\n{line}")
    for t in payload["title_race"][:12]:
        print(f"  {t['p_champion']:>6.1%} campeón · final {t['p_final']:>5.1%} · semis {t['p_sf']:>5.1%} · {t['team']}")

    if track:
        m = payload["meta"]
        print(f"\n{line}\nTRACKER EN VIVO — predicción vs realidad ({m['n_played']} jugados)\n{line}")
        for t in track:
            ok = "✓" if t["correct"] else "✗"
            pn = {"H": t["home"], "D": "Empate", "A": t["away"]}[t["pick"]]
            print(f"  {ok} {t['home']} {t['actual']} {t['away']:<22} "
                  f"(predijimos {pn[:14]} {max(t['p_home'],t['p_draw'],t['p_away']):.0%})")
        print(f"  → aciertos del ganador: {m['picks_correct']}/{m['n_played']}  |  RPS en vivo: {m['rps_live']}")

    print(f"\n{line}")
    print(f"Front generado: {PROJECT_ROOT / 'index.html'}  (abrilo con doble clic / GitHub Pages)")
    print(f"Datos: {web_dir / 'standings.json'}")


if __name__ == "__main__":
    main()
