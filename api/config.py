import yaml 
import sys
import os 
from pathlib import Path
# add root dir 
root = str(Path(__file__).parent.parent)
sys.path.insert(0, root)

# load the config (needed for reddit credentials)
config_path = Path(root) / "configs/config.yaml"
with open(config_path) as f: 
    config = yaml.safe_load(f)