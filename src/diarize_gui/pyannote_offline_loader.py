import os
import sys
import yaml
import tempfile
from pyannote.audio import Pipeline

def get_resource_base_path():
    """
    Returns the base directory where resources (like 'deps' or 'ollama') are located.
    Used by gui.py to find the Ollama binary.
    """
    if getattr(sys, 'frozen', False):
        # Frozen (App Bundle): The folder containing the executable
        # e.g., .../DiarizeApp.app/Contents/MacOS
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        # Dev Mode: The 'resources' folder in the project root
        # This file is in src/diarize_gui/ -> go up two levels to find 'resources'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        return os.path.join(project_root, "resources")

def get_model_dir():
    """
    Locate the folder containing config.yaml and bin files for Pyannote.
    """
    base_path = get_resource_base_path()
    
    if getattr(sys, 'frozen', False):
        # In the App Bundle, build_app.sh puts pyannote inside 'models'
        # .../Contents/MacOS/models/pyannote
        return os.path.join(base_path, "models", "pyannote")
    else:
        # In Dev Mode, it is directly in 'resources/pyannote'
        return os.path.join(base_path, "pyannote")

def load_offline_pipeline():
    model_dir = get_model_dir()
    config_path = os.path.join(model_dir, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Offline Pyannote config not found at: {config_path}\n"
            "Ensure you have downloaded the models and placed them in resources/pyannote/"
        )

    print(f"Reading config from: {config_path}")

    # 1. Read the YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Inject Absolute Paths
    # We look for 'segmentation' and 'embedding' params and fix them
    # Structure: pipeline -> params -> [keys]
    params = config.get("pipeline", {}).get("params", {})

    def make_absolute(rel_path):
        # Turn "./segmentation.bin" into "/Users/.../segmentation.bin"
        filename = os.path.basename(rel_path) 
        return os.path.join(model_dir, filename)

    if "segmentation" in params:
        # Only fix if it looks like a relative path
        if str(params["segmentation"]).startswith("."):
            params["segmentation"] = make_absolute(params["segmentation"])
            print(f"Patched segmentation path: {params['segmentation']}")

    if "embedding" in params:
        if str(params["embedding"]).startswith("."):
            params["embedding"] = make_absolute(params["embedding"])
            print(f"Patched embedding path: {params['embedding']}")

    # 3. Write to a temporary file
    # Pyannote needs a file on disk to load from
    fd, tmp_config_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, 'w') as f:
            yaml.dump(config, f)
        
        # 4. Load the pipeline from the temp file
        print(f"Loading pipeline from temp config: {tmp_config_path}")
        pipeline = Pipeline.from_pretrained(tmp_config_path)
        return pipeline

    finally:
        # 5. Cleanup the temp file
        if os.path.exists(tmp_config_path):
            try:
                os.remove(tmp_config_path)
            except:
                pass