#!/usr/bin/env python3
"""Config validation helper for experiment scripts."""

import json
import sys
from pathlib import Path

def validate_poisoning_config(config):
    """Validate poisoning experiment config."""
    required_keys = [
        "data_path", "epochs", "batch_size", "lr", "p", "target_class",
        "model", "pretrained", "freeze", "seed", "workers"
    ]
    
    missing = [key for key in required_keys if key not in config]
    if missing:
        print(f"Missing required keys: {missing}")
        return False
    
    # Validate data types and ranges
    if not isinstance(config["p"], (int, float)) or not (0 <= config["p"] <= 1):
        print("'p' must be a number between 0 and 1")
        return False
        
    if not isinstance(config["target_class"], int) or config["target_class"] < 0:
        print("'target_class' must be a non-negative integer")
        return False
        
    if not isinstance(config["epochs"], int) or config["epochs"] <= 0:
        print("'epochs' must be a positive integer")
        return False
        
    return True

def validate_vision_config(config):
    """Validate vision experiment config."""
    required_keys = [
        "data", "epochs", "batch_size", "lr", "eps", "pgd_steps", "seed", "out"
    ]
    
    missing = [key for key in required_keys if key not in config]
    if missing:
        print(f"Missing required keys: {missing}")
        return False
    
    # Validate eps format
    try:
        eps_list = [int(e) for e in str(config["eps"]).split(",")]
        if not all(e > 0 for e in eps_list):
            print("All epsilon values must be positive")
            return False
    except:
        print("'eps' must be comma-separated integers")
        return False
        
    return True

def validate_visualize_config(config):
    """Validate visualization config."""
    required_keys = ["model_dir", "data", "n_samples"]
    
    missing = [key for key in required_keys if key not in config]
    if missing:
        print(f"Missing required keys: {missing}")
        return False
    
    # Check if model directory exists
    model_dir = Path(config["model_dir"])
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return False
        
    metadata_file = model_dir / "metadata.json"
    if not metadata_file.exists():
        print(f"Metadata file not found: {metadata_file}")
        return False
        
    return True

config_file = sys.argv[1]
experiment_type = sys.argv[2]

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Config file not found: {config_file}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
    sys.exit(1)

print(f"Validating {experiment_type} config...")
print(f"Config file: {config_file}")

print("\nCurrent settings:")
for key, value in config.items():
    print(f"   {key}: {value}")

validators = {
    "poisoning": validate_poisoning_config,
    "vision": validate_vision_config,
    "visualize": validate_visualize_config
}

if validators[experiment_type](config):
    print("\nConfig validation passed!")
    sys.exit(0)
else:
    print("\nConfig validation failed!")
    sys.exit(1)