"""Normalización e identidad de selecciones.

Resuelve dos problemas:
  1. Renombrado histórico (date-aware) usando former_names.csv de martj42
     (ej. "Upper Volta" -> "Burkina Faso" entre 1960 y 1984).
  2. Alias entre fuentes: martj42 es nuestra fuente única hoy, pero este
     diccionario deja preparado el join futuro con API-Football / ranking FIFA,
     mapeando grafías externas a la grafía canónica de martj42.

`team_slug` produce un id estable (slug) que usamos como clave en todo el pipeline
(resultados, Elo, fixtures), evitando errores de merge por tildes o variantes.
"""

from __future__ import annotations

import re

import pandas as pd
from unidecode import unidecode

# Grafías externas conocidas -> grafía canónica (la de martj42).
# Si un nombre no está acá, se devuelve tal cual (sólo limpiando espacios/tildes).
# Las claves se comparan ya normalizadas (minúsculas, sin tildes, espacios colapsados).
ALIASES: dict[str, str] = {
    "usa": "United States",
    "usmnt": "United States",
    "u.s.a": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "korea dpr": "North Korea",
    "dpr korea": "North Korea",
    "turkiye": "Turkey",
    "czech republic": "Czechia",
    "cote d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "ir iran": "Iran",
    "iran (islamic republic of)": "Iran",
    "china": "China PR",
    "china pr": "China PR",
    "cape verde islands": "Cape Verde",
    "bosnia": "Bosnia and Herzegovina",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "north macedonia": "North Macedonia",
    "fyr macedonia": "North Macedonia",
    "macedonia": "North Macedonia",
    "kyrgyz republic": "Kyrgyzstan",
    "curacao": "Curaçao",
}


def _normalize_key(name: str) -> str:
    """Clave de comparación: sin tildes, minúsculas, espacios colapsados."""
    return unidecode(" ".join(str(name).split())).strip().lower()


def canonical_name(name: str) -> str:
    """Devuelve la grafía canónica de una selección.

    Aplica el alias si existe; si no, devuelve el nombre limpio de espacios
    redundantes (conservando la grafía original de la fuente).
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return name
    key = _normalize_key(name)
    if key in ALIASES:
        return ALIASES[key]
    return " ".join(str(name).split())


def team_slug(name: str) -> str:
    """Id estable a partir del nombre: minúsculas, sin tildes, separado por '-'.

    Ej.: "Côte d'Ivoire" -> "ivory-coast", "South Korea" -> "south-korea".
    """
    canon = canonical_name(name)
    slug = re.sub(r"[^a-z0-9]+", "-", unidecode(str(canon)).lower()).strip("-")
    return slug


def apply_former_names(
    df: pd.DataFrame,
    former: pd.DataFrame,
    date_col: str = "date",
    team_cols: tuple[str, ...] = ("home_team", "away_team"),
) -> tuple[pd.DataFrame, int]:
    """Renombra selecciones según former_names.csv, respetando el rango de fechas.

    Cada fila de `former` (current, former, start_date, end_date) significa que
    la entidad hoy llamada `current` se llamaba `former` entre esas fechas.
    Consolidamos la historia bajo el nombre actual para dar continuidad a la
    fuerza del equipo.

    Devuelve (df_renombrado, cantidad_de_celdas_renombradas).
    """
    df = df.copy()
    former = former.copy()
    former["start_date"] = pd.to_datetime(former["start_date"], errors="coerce")
    former["end_date"] = pd.to_datetime(former["end_date"], errors="coerce")

    renamed = 0
    for _, r in former.iterrows():
        if pd.isna(r["start_date"]) or pd.isna(r["end_date"]):
            continue
        in_range = (df[date_col] >= r["start_date"]) & (df[date_col] <= r["end_date"])
        for col in team_cols:
            mask = in_range & (df[col] == r["former"])
            renamed += int(mask.sum())
            df.loc[mask, col] = r["current"]
    return df, renamed
