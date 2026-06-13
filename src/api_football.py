"""Actualización automática vía API-Football (api-sports.io).

Refresca, desde la API, los CSV que ya consume el modelo:
  - assets/actual_results.csv      (resultados de partidos FINALIZADOS del Mundial)
  - assets/unavailable_players.csv (lesiones / bajas actuales)

La key se lee de una variable de entorno (no se hardcodea ni se versiona).
Free tier ≈ 100 req/día; este flujo usa unas pocas. Después: `python -m src.simulate`.

Uso:
    export APIFOOTBALL_KEY=tu_key
    python -m src.api_football --status     # prueba conexión + cupo
    python -m src.api_football --results     # refresca resultados
    python -m src.api_football --injuries    # refresca lesiones
    python -m src.api_football --all
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from src.elo import load_config
    from src.team_names import canonical_name
except ImportError:  # pragma: no cover
    from elo import load_config
    from team_names import canonical_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINISHED = {"FT", "AET", "PEN"}  # estados de partido terminado


# --------------------------------------------------------------------------- #
# Parseo (separado del HTTP para poder testearlo sin key)
# --------------------------------------------------------------------------- #
def parse_results(resp: list) -> pd.DataFrame:
    rows = []
    for fx in resp:
        fixture = fx.get("fixture", {})
        if (fixture.get("status", {}) or {}).get("short") not in FINISHED:
            continue
        teams, goals = fx.get("teams", {}), fx.get("goals", {})
        gh, ga = goals.get("home"), goals.get("away")
        if gh is None or ga is None:
            continue
        rows.append({
            "home": canonical_name(teams.get("home", {}).get("name", "")),
            "away": canonical_name(teams.get("away", {}).get("name", "")),
            "home_goals": int(gh), "away_goals": int(ga),
            "group": "", "date": (fixture.get("date", "") or "")[:10], "neutral": True,
        })
    return pd.DataFrame(rows, columns=["home", "away", "home_goals", "away_goals", "group", "date", "neutral"])


def parse_injuries(resp: list) -> pd.DataFrame:
    rows, seen = [], set()
    for it in resp:
        player, team = it.get("player", {}), it.get("team", {})
        typ = str(player.get("type", "")).strip().lower()
        # solo bajas confirmadas que NO juegan; "questionable"/dudosos se descartan
        if "missing" not in typ and typ != "out":
            continue
        t = canonical_name(team.get("name", ""))
        name = player.get("name", "")
        if (t, name) in seen:
            continue
        seen.add((t, name))
        rows.append({"team": t, "player": name,
                     "position": str(player.get("position") or "default").lower(),
                     "status": "out", "source": "api-football"})
    return pd.DataFrame(rows, columns=["team", "player", "position", "status", "source"])


# --------------------------------------------------------------------------- #
# Cliente HTTP
# --------------------------------------------------------------------------- #
def _client(cfg: dict):
    api = cfg["api_football"]
    key = os.environ.get(api["key_env"])
    if not key:
        raise SystemExit(
            f"Falta la API key. Hacé:  export {api['key_env']}=tu_key\n"
            f"(registrate gratis en https://dashboard.api-football.com)")
    sess = requests.Session()
    sess.headers["x-apisports-key"] = key
    delay = float(api.get("request_delay_s", 1.0))

    def get(path: str, params: dict | None = None) -> list:
        r = sess.get(f"{api['base_url']}/{path}", params=params or {}, timeout=30)
        r.raise_for_status()
        j = r.json()
        errors = j.get("errors")
        if errors:
            raise SystemExit(f"Error de la API en /{path}: {errors}")
        time.sleep(delay)
        return j.get("response", [])

    return get, api


def find_league(get, api: dict):
    if api.get("league_id"):
        return api["league_id"]
    resp = get("leagues", {"search": "world cup"})
    for item in resp:
        lg = item.get("league", {})
        if lg.get("name", "").lower() == "world cup" and lg.get("type", "").lower() == "cup":
            return lg["id"]
    if not resp:
        raise SystemExit("No se encontró la liga 'World Cup' en la API.")
    return resp[0]["league"]["id"]


def _wc_teams(cfg: dict) -> set:
    g = pd.read_csv(PROJECT_ROOT / cfg["simulation"]["groups_file"])
    return set(g["team"].map(canonical_name))


def _filter_known(df: pd.DataFrame, cfg: dict, cols) -> pd.DataFrame:
    teams = _wc_teams(cfg)
    mask = pd.Series(True, index=df.index)
    for c in cols:
        mask &= df[c].isin(teams)
    dropped = df[~mask]
    if len(dropped):
        unknown = sorted(set(dropped[list(cols)].values.ravel()) - teams)
        print(f"  ⚠ {len(dropped)} fila(s) con equipos no reconocidos (agregá alias en team_names.py): {unknown[:12]}")
    return df[mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Acciones
# --------------------------------------------------------------------------- #
def status(cfg: dict):
    get, _ = _client(cfg)
    resp = get("status")
    sub, req = resp.get("subscription", {}), resp.get("requests", {})
    print(f"Plan: {sub.get('plan')} · activo: {sub.get('active')} · vence: {sub.get('end')}")
    print(f"Requests hoy: {req.get('current')}/{req.get('limit_day')}")


def refresh_results(cfg: dict):
    get, api = _client(cfg)
    league, season = find_league(get, api), api["season"]
    df = _filter_known(parse_results(get("fixtures", {"league": league, "season": season})), cfg, ("home", "away"))
    path = PROJECT_ROOT / cfg["simulation"]["actual_results_file"]
    df.to_csv(path, index=False)
    print(f"Resultados finalizados: {len(df)} → {path}")


def refresh_injuries(cfg: dict):
    get, api = _client(cfg)
    league, season = find_league(get, api), api["season"]
    df = _filter_known(parse_injuries(get("injuries", {"league": league, "season": season})), cfg, ("team",))
    path = PROJECT_ROOT / cfg["availability"]["unavailable_file"]
    df.to_csv(path, index=False)
    print(f"Bajas (lesiones/suspensiones): {len(df)} → {path}")


def main():
    ap = argparse.ArgumentParser(description="Actualiza resultados y lesiones desde API-Football.")
    ap.add_argument("--status", action="store_true", help="prueba la conexión y muestra el cupo")
    ap.add_argument("--results", action="store_true", help="refresca actual_results.csv")
    ap.add_argument("--injuries", action="store_true", help="refresca unavailable_players.csv")
    ap.add_argument("--all", action="store_true", help="resultados + lesiones")
    args = ap.parse_args()
    cfg = load_config()

    if not any([args.status, args.results, args.injuries, args.all]):
        args.status = True
    if args.status:
        status(cfg)
    if args.results or args.all:
        refresh_results(cfg)
    if args.injuries or args.all:
        refresh_injuries(cfg)
    print("\nListo. Ahora: python -m src.simulate  (y luego git commit + push para actualizar la demo)")


if __name__ == "__main__":
    main()
