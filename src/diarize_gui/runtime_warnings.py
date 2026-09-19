from __future__ import annotations

import logging
import warnings
from importlib import import_module


def configure_runtime_warning_filters() -> None:
    """Hide known noisy compatibility warnings from trusted bundled models.

    The Pyannote/WhisperX stack loads older Lightning checkpoints during normal
    operation. They emit migration/deprecation warnings on every run even when
    processing succeeds. Keep the filters narrow so unexpected runtime warnings
    still reach the console.
    """

    warning_filters = (
        r"You have multiple `ModelCheckpoint` callback states in this checkpoint.*",
        r"Model was trained with pyannote\.audio .*",
        r"Model was trained with torch .*",
        r"Found keys that are not in the model state dict but in the checkpoint:.*",
        r"In 2\.9, this function's implementation will be changed to use torchaudio\.load_with_torchcodec.*",
        r"torchaudio\._backend\.list_audio_backends has been deprecated.*",
    )

    for message in warning_filters:
        warnings.filterwarnings("ignore", message=message)

    for logger_name in (
        "pytorch_lightning.utilities.migration.utils",
        "pytorch_lightning.utilities.migration.migration",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def suppress_pyannote_version_check_prints() -> None:
    """Disable Pyannote's print-based old-checkpoint version notices."""

    try:
        version_module = import_module("pyannote.audio.utils.version")
    except Exception:
        return

    def quiet_check_version(*_args, **_kwargs):
        return None

    quiet_check_version._diarize_quiet = True
    version_module.check_version = quiet_check_version

    for module_name in (
        "pyannote.audio.core.model",
        "pyannote.audio.core.pipeline",
    ):
        try:
            module = import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "check_version"):
            module.check_version = quiet_check_version
