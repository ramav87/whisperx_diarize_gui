#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, snapshot_download


INSTALL_DIR = Path(os.environ.get("INSTALL_DIR", "/opt/diarize-server"))
ENV_FILE = Path(os.environ.get("ENV_FILE", "/etc/diarize-server.env"))
BUNDLE_DIR = Path(os.environ.get("PYANNOTE_BUNDLE_DIR", INSTALL_DIR / "resources" / "pyannote"))


def _read_token() -> str:
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token.strip()

    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HUGGINGFACE_TOKEN=") or line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise SystemExit(
        f"No HUGGINGFACE_TOKEN or HF_TOKEN found in environment or {ENV_FILE}. "
        "Add a valid HuggingFace token first."
    )


def main() -> None:
    token = _read_token()
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    config_file = hf_hub_download(
        repo_id="pyannote/speaker-diarization-3.1",
        filename="config.yaml",
        local_dir=BUNDLE_DIR,
        token=token,
    )

    snapshot_download(
        repo_id="pyannote/segmentation-3.0",
        local_dir=BUNDLE_DIR / "segmentation",
        token=token,
    )

    snapshot_download(
        repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        local_dir=BUNDLE_DIR / "embedding",
        token=token,
    )

    config_path = Path(config_file)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = config.setdefault("pipeline", {}).setdefault("params", {})
    params["segmentation"] = "./segmentation"
    params["embedding"] = "./embedding"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    print(f"Installed pyannote offline bundle at {BUNDLE_DIR}")
    print(f"Pipeline config: {config_path}")


if __name__ == "__main__":
    main()
