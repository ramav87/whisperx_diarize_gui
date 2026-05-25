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

if command -v runuser >/dev/null 2>&1; then
  AS_DIARIZE=(runuser -u diarize --)
elif command -v sudo >/dev/null 2>&1; then
  AS_DIARIZE=(sudo -u diarize)
else
  echo "Need either runuser or sudo to switch to the diarize user."
  exit 1
fi

id -u diarize >/dev/null 2>&1 || useradd --system --create-home --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin diarize
mkdir -p "$INSTALL_DIR" "$DATA_DIR"
chown -R diarize:diarize "$INSTALL_DIR" "$DATA_DIR"

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  "${AS_DIARIZE[@]}" git clone "$REPO_URL" "$INSTALL_DIR"
fi

("${AS_DIARIZE[@]}" git -C "$INSTALL_DIR" fetch origin "$BRANCH")
("${AS_DIARIZE[@]}" git -C "$INSTALL_DIR" checkout "$BRANCH")
("${AS_DIARIZE[@]}" git -C "$INSTALL_DIR" reset --hard "origin/$BRANCH")

if [[ ! -d "$INSTALL_DIR/.venv" ]]; then
  "${AS_DIARIZE[@]}" "$UV_BIN" venv --python "$PYTHON_BIN" "$INSTALL_DIR/.venv"
fi

"${AS_DIARIZE[@]}" "$UV_BIN" pip install --python "$INSTALL_DIR/.venv/bin/python" -U pip
"${AS_DIARIZE[@]}" "$UV_BIN" pip install --python "$INSTALL_DIR/.venv/bin/python" -e "$INSTALL_DIR"

install -m 0644 "$INSTALL_DIR/deploy/diarize-server.service" "/etc/systemd/system/$SERVICE_NAME.service"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo "Installed $SERVICE_NAME"
echo "Logs: journalctl -u $SERVICE_NAME -f"
