import os
import sys
from pyannote.audio import Pipeline

def get_resource_base_path():
    """
    Returns the path to the 'resources' folder.
    - Frozen (PyInstaller): sys._MEIPASS/resources
    - Dev (Python): ../../resources (relative to this file)
    """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller unpacks data to a temp folder stored in _MEIPASS
        return os.path.join(sys._MEIPASS, "resources")
    else:
        # In development, assume 'resources' is in the project root
        # This file is in src/diarize_gui/, so we go up two levels.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, "..", "..", "resources"))

def load_offline_pipeline():
    """
    Loads the Pyannote pipeline from the local 'resources/pyannote/config.yaml'
    instead of downloading it from Hugging Face.
    """
    base_path = get_resource_base_path()
    pyannote_dir = os.path.join(base_path, "pyannote")
    config_path = os.path.join(pyannote_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Offline Pyannote config not found at: {config_path}\n"
            "Ensure you have downloaded the models and placed them in resources/pyannote/"
        )

    print(f"Loading offline Pyannote pipeline from: {config_path}")
    
    # Pipeline.from_pretrained accepts a path to a YAML file.
    # Paths INSIDE that yaml must be relative to the yaml file itself.
    pipeline = Pipeline.from_pretrained(config_path)
    
    return pipeline