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