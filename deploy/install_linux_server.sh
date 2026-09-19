#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ramav87/whisperx_diarize_gui.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/diarize-server}"
DATA_DIR="${DATA_DIR:-/srv/diarize-server}"
SERVICE_NAME="${SERVICE_NAME:-diarize-server}"
BRANCH="${BRANCH:-optimized}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN="${UV_BIN:-${1:-uv}}"

if [[ $EUID -ne 0 ]]; then
  echo "Run this installer with sudo or as root."
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BIN"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Missing git"
  exit 1
fi

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "Missing uv at '$UV_BIN'. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if ! command -v systemctl >/dev/null 2>&1; then
  echo "This installer expects systemd/systemctl."
  exit 1
fi

id -u diarize >/dev/null 2>&1 || useradd --system --create-home --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin diarize
mkdir -p "$INSTALL_DIR" "$DATA_DIR"

if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "Using existing git checkout in $INSTALL_DIR"
else
  if [[ -n "$(find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Install dir $INSTALL_DIR exists and is not empty, but is not a git checkout."
    echo "Move it aside or point INSTALL_DIR at a clean directory, then rerun."
    exit 1
  fi
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

git config --global --add safe.directory "$INSTALL_DIR"
git -C "$INSTALL_DIR" fetch origin "$BRANCH"
git -C "$INSTALL_DIR" checkout "$BRANCH"
git -C "$INSTALL_DIR" reset --hard "origin/$BRANCH"

if [[ ! -d "$INSTALL_DIR/.venv" ]]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" "$INSTALL_DIR/.venv"
fi

"$UV_BIN" pip install --python "$INSTALL_DIR/.venv/bin/python" -U pip
"$UV_BIN" pip install --python "$INSTALL_DIR/.venv/bin/python" -e "$INSTALL_DIR"

chown -R diarize:diarize "$INSTALL_DIR" "$DATA_DIR"

install -m 0644 "$INSTALL_DIR/deploy/diarize-server.service" "/etc/systemd/system/$SERVICE_NAME.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo "Installed $SERVICE_NAME"
echo "Logs: journalctl -u $SERVICE_NAME -f"
