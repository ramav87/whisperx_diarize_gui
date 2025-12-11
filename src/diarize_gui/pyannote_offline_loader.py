import os
import sys
from pyannote.audio import Pipeline

def get_resource_base_path():
    if getattr(sys, 'frozen', False):
        # Frozen (App Bundle): Return the folder containing the executable
        # This is usually .../Contents/MacOS
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        # Dev Mode: Return the 'resources' folder in project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, "..", "..", "resources"))

def load_offline_pipeline():
    """
    Loads the Pyannote pipeline from the local config.
    """
    # 1. Get the base folder (Contents/MacOS or project root)
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(os.path.abspath(sys.executable))
        # In the App Bundle, we put it in 'models/pyannote'
        pyannote_dir = os.path.join(base_path, "models", "pyannote")
    else:
        # In Dev Mode, it is in 'resources/pyannote'
        # Go up two levels from src/diarize_gui/ to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        pyannote_dir = os.path.join(project_root, "resources", "pyannote")

    config_path = os.path.join(pyannote_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Offline Pyannote config not found at: {config_path}\n"
            "Ensure you have downloaded the models and placed them in resources/pyannote/"
        )

    print(f"Loading offline Pyannote pipeline from: {config_path}")
    
    # Load pipeline
    pipeline = Pipeline.from_pretrained(config_path)
    
    return pipeline