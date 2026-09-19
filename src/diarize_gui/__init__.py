
# `src/diarize_gui/__init__.py`

#python
"""
diarize_gui package

Simple GUI for WhisperX transcription + diarization with optional recording.
"""

from .runtime_warnings import configure_runtime_warning_filters

configure_runtime_warning_filters()

__version__ = "0.1.0"
