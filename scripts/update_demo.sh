#!/bin/bash
# Actualiza el predictor del Mundial 2026 y publica la demo.
#   1. sincroniza el repo   2. baja resultados nuevos (martj42)   3. recalcula
#   4. commitea + pushea SOLO si hubo cambios (y si no falló nada antes)
# Pensado para correr por launchd en macOS. Uso manual: bash scripts/update_demo.sh [--no-push]

set -uo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:$PATH"

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$PROJECT/.venv/bin/python"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

cd "$PROJECT" || { echo "[$(ts)] ERROR: no se pudo entrar a $PROJECT"; exit 1; }
[ -x "$PY" ] || { echo "[$(ts)] ERROR: no existe el venv ($PY). Corré: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }

echo "===== [$(ts)] inicio ====="

git pull --rebase --autostash || { echo "[$(ts)] git pull falló (¿conflicto?). Aborto sin publicar."; exit 1; }
"$PY" -m src.ingest          || { echo "[$(ts)] ingest falló. No publico."; exit 1; }
"$PY" -m src.simulate        || { echo "[$(ts)] simulate falló. No publico."; exit 1; }

if [ "${1:-}" = "--no-push" ]; then
  echo "[$(ts)] modo --no-push: datos recalculados, sin commit/push."
  exit 0
fi

if [ -z "$(git status --porcelain)" ]; then
  echo "[$(ts)] sin cambios; nada para publicar."
  exit 0
fi

git add -A
git commit -m "Actualización automática $(ts)" || { echo "[$(ts)] commit falló."; exit 1; }
git push || { echo "[$(ts)] git push falló (¿credenciales?). Configurá el credential helper o usá un token/SSH."; exit 1; }
echo "===== [$(ts)] publicado OK ====="
