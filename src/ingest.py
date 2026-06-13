"""Componente 1 — Ingesta de datos (capa raw -> clean).

Pipeline:
  1. Descarga el dataset martj42 por HTTP (results + former_names) y lo cachea
     en data/raw/ (sin API key, sin descarga manual).
  2. Aplica renombrado histórico (former_names, date-aware).
  3. Normaliza nombres y genera un id estable (slug) por selección.
  4. Limpia: fechas, flag neutral, marcadores numéricos, deduplicado.
  5. Separa partidos JUGADOS (entrenamiento) de FIXTURES PENDIENTES (a predecir).
  6. Persiste parquet limpio y muestra un resumen para inspección.

Uso:
    python -m src.ingest            # desde la raíz del proyecto
    python -m src.ingest --force    # fuerza re-descarga
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import requests
import yaml

if __package__:  # python -m src.ingest
    from .team_names import apply_former_names, canonical_name, team_slug
else:            # python src/ingest.py
    from team_names import apply_former_names, canonical_name, team_slug

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Configuración y utilidades
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dir(rel: str) -> Path:
    p = PROJECT_ROOT / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def download(url: str, dest: Path, force: bool) -> Path:
    """Descarga `url` a `dest` (cachea). Si ya existe y no se fuerza, no re-baja."""
    if dest.exists() and not force:
        print(f"  · cache    {dest.name}  ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"  · bajando  {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"    -> {dest}  ({len(resp.content):,} bytes)")
    return dest


# --------------------------------------------------------------------------- #
# Pasos del pipeline
# --------------------------------------------------------------------------- #
def clean_results(df: pd.DataFrame, former: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """raw -> clean. Devuelve (df_limpio, métricas_de_calidad)."""
    stats: dict = {"raw_rows": len(df)}

    # 1. Tipos básicos
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["neutral"] = (
        df["neutral"].astype(str).str.strip().str.upper().map({"TRUE": True, "FALSE": False})
    )
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # Descartar filas con fecha o equipos inválidos
    before = len(df)
    df = df.dropna(subset=["date", "home_team", "away_team"])
    stats["dropped_invalid"] = before - len(df)

    # 2. Renombrado histórico (date-aware) y normalización de nombres
    df, renamed = apply_former_names(df, former)
    stats["former_name_renames"] = renamed
    df["home_team"] = df["home_team"].map(canonical_name)
    df["away_team"] = df["away_team"].map(canonical_name)
    df["home_id"] = df["home_team"].map(team_slug)
    df["away_id"] = df["away_team"].map(team_slug)

    # 3. Deduplicado idempotente
    before = len(df)
    df = df.drop_duplicates(
        subset=["date", "home_id", "away_id", "tournament", "home_score", "away_score"]
    )
    stats["dropped_duplicates"] = before - len(df)

    # 4. Orden y reseteo
    df = df.sort_values("date").reset_index(drop=True)
    stats["clean_rows"] = len(df)
    return df, stats


def split_played_pending(
    df: pd.DataFrame, as_of: pd.Timestamp, target_tournament: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa jugados / fixtures futuros / no-jugados-en-el-pasado (anomalías)."""
    played_mask = df["home_score"].notna() & df["away_score"].notna()
    played = df[played_mask].copy()

    pending = df[~played_mask].copy()
    future = pending[pending["date"] >= as_of].copy()
    past_unplayed = pending[pending["date"] < as_of].copy()  # anomalías a reportar

    # Etiqueta de resultado 1X2 para los jugados (objetivo primario del modelo)
    played["result"] = "D"
    played.loc[played["home_score"] > played["away_score"], "result"] = "H"
    played.loc[played["home_score"] < played["away_score"], "result"] = "A"

    return played, future, past_unplayed


def build_teams_table(played: pd.DataFrame) -> pd.DataFrame:
    """Tabla de selecciones con conteo de partidos jugados (id, nombre, n)."""
    long = pd.concat(
        [
            played[["home_id", "home_team"]].rename(columns={"home_id": "team_id", "home_team": "team"}),
            played[["away_id", "away_team"]].rename(columns={"away_id": "team_id", "away_team": "team"}),
        ]
    )
    teams = (
        long.groupby("team_id")
        .agg(team=("team", "first"), matches=("team", "size"))
        .reset_index()
        .sort_values("matches", ascending=False)
    )
    return teams


# --------------------------------------------------------------------------- #
# Resumen para inspección
# --------------------------------------------------------------------------- #
def print_summary(
    stats: dict,
    played: pd.DataFrame,
    future: pd.DataFrame,
    past_unplayed: pd.DataFrame,
    teams: pd.DataFrame,
    target_tournament: str,
) -> None:
    line = "─" * 64
    print(f"\n{line}\nRESUMEN DE LA INGESTA\n{line}")
    print("Calidad de datos:")
    print(f"  filas crudas .............. {stats['raw_rows']:,}")
    print(f"  descartadas (inválidas) ... {stats['dropped_invalid']:,}")
    print(f"  renombres históricos ...... {stats['former_name_renames']:,}")
    print(f"  duplicados eliminados ..... {stats['dropped_duplicates']:,}")
    print(f"  filas limpias ............. {stats['clean_rows']:,}")

    print("\nPartidos JUGADOS (entrenamiento):")
    print(f"  total ..................... {len(played):,}")
    print(f"  rango de fechas ........... {played['date'].min().date()}  →  {played['date'].max().date()}")
    print(f"  selecciones distintas ..... {len(teams):,}")
    res = played["result"].value_counts(normalize=True)
    print(
        f"  balance 1X2 ............... Local {res.get('H', 0):.1%} · "
        f"Empate {res.get('D', 0):.1%} · Visita {res.get('A', 0):.1%}"
    )
    neutral_pct = played["neutral"].mean()
    print(f"  jugados en sede neutral ... {neutral_pct:.1%}")

    print("\nTop 5 torneos por cantidad de partidos:")
    for t, n in played["tournament"].value_counts().head(5).items():
        print(f"  {n:>6,}  {t}")

    print(f"\nFIXTURES PENDIENTES (a predecir) — '{target_tournament}':")
    wc = future[future["tournament"] == target_tournament]
    print(f"  fixtures .................. {len(future):,}  (de '{target_tournament}': {len(wc):,})")
    if len(future):
        print(f"  rango de fechas ........... {future['date'].min().date()}  →  {future['date'].max().date()}")
        wc_teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))
        print(f"  selecciones involucradas .. {len(wc_teams)}")
        # Mostrar selecciones del Mundial y su volumen histórico
        counts = teams.set_index("team")["matches"].to_dict()
        rows = sorted(((t, counts.get(t, 0)) for t in wc_teams), key=lambda x: x[1])
        print("  partidos históricos por selección del Mundial (menos → más):")
        thin = [f"{t} ({n})" for t, n in rows[:6]]
        print(f"    con menos historia: {', '.join(thin)}")
        if any(n < 30 for _, n in rows):
            flag = [t for t, n in rows if n < 30]
            print(f"    ⚠ <30 partidos (riesgo de fuerza ruidosa → shrinkage/Elo): {', '.join(flag)}")

    if len(past_unplayed):
        print(f"\n⚠ ANOMALÍAS: {len(past_unplayed)} partidos sin marcador con fecha pasada (excluidos).")
    print(f"{line}\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta de datos del predictor del Mundial.")
    parser.add_argument("--force", action="store_true", help="forzar re-descarga del raw")
    args = parser.parse_args()

    cfg = load_config()
    dcfg, icfg = cfg["data"], cfg["ingest"]
    force = args.force or icfg.get("force_download", False)

    as_of = (
        pd.Timestamp(icfg["as_of_date"])
        if icfg.get("as_of_date")
        else pd.Timestamp(dt.date.today())
    )
    target = icfg["target_tournament"]

    print(f"Ingesta — fuente: {dcfg['base_url']}")
    print(f"Fecha de corte (as_of): {as_of.date()}\n")

    # 1. Descarga (capa raw)
    raw_dir = _dir(dcfg["raw_dir"])
    print("Descarga (capa raw):")
    results_path = download(f"{dcfg['base_url']}/{dcfg['results_file']}", raw_dir / dcfg["results_file"], force)
    former_path = download(f"{dcfg['base_url']}/{dcfg['former_names_file']}", raw_dir / dcfg["former_names_file"], force)

    results = pd.read_csv(results_path)
    former = pd.read_csv(former_path)

    # 2-4. Limpieza
    clean, stats = clean_results(results, former)

    # 5. Separación jugados / pendientes
    played, future, past_unplayed = split_played_pending(clean, as_of, target)
    teams = build_teams_table(played)

    # 6. Persistencia (capa clean)
    clean_dir = _dir(dcfg["clean_dir"])
    played.to_parquet(clean_dir / "matches_played.parquet", index=False)
    future.to_parquet(clean_dir / "fixtures_pending.parquet", index=False)
    teams.to_parquet(clean_dir / "teams.parquet", index=False)
    print("Persistido en capa clean:")
    print(f"  · {clean_dir / 'matches_played.parquet'}  ({len(played):,} filas)")
    print(f"  · {clean_dir / 'fixtures_pending.parquet'}  ({len(future):,} filas)")
    print(f"  · {clean_dir / 'teams.parquet'}  ({len(teams):,} filas)")

    print_summary(stats, played, future, past_unplayed, teams, target)


if __name__ == "__main__":
    main()
