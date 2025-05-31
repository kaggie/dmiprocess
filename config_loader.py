import json
import yaml
import os

def load_config(config_path):
    """
    Loads a configuration file (JSON or YAML).

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Loaded configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file type is not supported or parsing fails.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    _, file_extension = os.path.splitext(config_path)

    if file_extension.lower() == ".json":
        with open(config_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON file {config_path}: {e}")
    elif file_extension.lower() in [".yaml", ".yml"]:
        with open(config_path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file {config_path}: {e}")
    else:
        raise ValueError(f"Unsupported configuration file type: {file_extension}. Please use JSON or YAML.")

def validate_config(config_dict, mode):
    """
    Validates the loaded configuration dictionary.

    Args:
        config_dict (dict): The configuration dictionary.
        mode (str): The processing mode ('oxsa' or 'lcmodel').

    Returns:
        bool: True if validation is successful.

    Raises:
        ValueError: If validation fails.
    """
    # Validate input_file structure
    if "input_file" not in config_dict:
        raise ValueError("Missing required top-level key in config: 'input_file'")

    input_file_val = config_dict["input_file"]
    if not (isinstance(input_file_val, str) or \
            (isinstance(input_file_val, list) and all(isinstance(item, str) for item in input_file_val))):
        raise ValueError("'input_file' must be a string (filepath) or a list of strings (filepaths).")

    # Validate other required top-level keys
    required_other_top_level_keys = ["output_dir", "output_prefix"]
    for key in required_other_top_level_keys:
        if key not in config_dict:
            raise ValueError(f"Missing required top-level key in config: '{key}'")

    if "mode_specific_params" not in config_dict:
        raise ValueError("Missing 'mode_specific_params' in config.")

    if mode not in config_dict["mode_specific_params"]:
        raise ValueError(f"Missing mode-specific parameters for mode '{mode}' in 'mode_specific_params'.")

    mode_params = config_dict["mode_specific_params"][mode]

    if mode == "oxsa":
        required_oxsa_keys = ["prior_knowledge_file", "lineshape"]
        for key in required_oxsa_keys:
            if key not in mode_params:
                raise ValueError(f"Missing required key '{key}' in 'mode_specific_params.oxsa'")
    elif mode == "lcmodel":
        required_lcmodel_keys = ["basis_file", "fitting_range_ppm", "baseline_degree"]
        for key in required_lcmodel_keys:
            if key not in mode_params:
                raise ValueError(f"Missing required key '{key}' in 'mode_specific_params.lcmodel'")
    else:
        # This case should ideally be caught by argparse choices, but good for robustness
        raise ValueError(f"Unsupported mode '{mode}' for validation.")

    return True

if __name__ == "__main__":
    # Create dummy config files for testing
    dummy_json_valid_content = {
        "input_file": "data/input.dat",
        "output_dir": "results/",
        "output_prefix": "scan01",
        "mode_specific_params": {
            "oxsa": {
                "prior_knowledge_file": "priors/oxsa_prior.json",
                "lineshape": "lorentzian"
            },
            "lcmodel": {
                "basis_file": "basis/lcmodel_basis.basis",
                "fitting_range_ppm": [0.5, 4.0],
                "baseline_degree": 3
            }
        }
    }

    dummy_yaml_valid_content = {
        "input_file": "data/input.yaml_dat",
        "output_dir": "results_yaml/",
        "output_prefix": "scan01_yaml",
        "mode_specific_params": {
            "oxsa": {
                "prior_knowledge_file": "priors/oxsa_prior.yaml_prior",
                "lineshape": "gaussian"
            },
            "lcmodel": {
                "basis_file": "basis/lcmodel_basis.yaml_basis",
                "fitting_range_ppm": [0.2, 4.2],
                "baseline_degree": 4
            }
        }
    }

    dummy_json_invalid_content = {
        "output_dir": "results/",
        "output_prefix": "scan02",
        "mode_specific_params": {
            "oxsa": {
                "lineshape": "lorentzian"
            }
        }
    }

    os.makedirs("dummy_configs", exist_ok=True)

    with open("dummy_configs/test_config_valid.json", "w") as f:
        json.dump(dummy_json_valid_content, f, indent=4)

    with open("dummy_configs/test_config_valid.yaml", "w") as f:
        yaml.dump(dummy_yaml_valid_content, f, indent=4)

    with open("dummy_configs/test_config_invalid.json", "w") as f:
        json.dump(dummy_json_invalid_content, f, indent=4)

    print("--- Testing valid JSON config (oxsa mode) ---")
    try:
        config_data_json_valid = load_config("dummy_configs/test_config_valid.json")
        print("Loaded JSON config:", config_data_json_valid)
        validate_config(config_data_json_valid, "oxsa")
        print("JSON config (oxsa) validation successful.")
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing valid YAML config (lcmodel mode) ---")
    try:
        config_data_yaml_valid = load_config("dummy_configs/test_config_valid.yaml")
        print("Loaded YAML config:", config_data_yaml_valid)
        validate_config(config_data_yaml_valid, "lcmodel")
        print("YAML config (lcmodel) validation successful.")
    except Exception as e:
        print(f"Error: {e}")

    print("\n--- Testing invalid JSON config (oxsa mode) ---")
    try:
        config_data_json_invalid = load_config("dummy_configs/test_config_invalid.json")
        print("Loaded invalid JSON config:", config_data_json_invalid)
        validate_config(config_data_json_invalid, "oxsa")
        print("Invalid JSON config validation successful? (Should not happen)")
    except Exception as e:
        print(f"Error (expected): {e}")

    print("\n--- Testing non-existent config file ---")
    try:
        load_config("dummy_configs/non_existent_config.json")
    except Exception as e:
        print(f"Error (expected): {e}")

    print("\n--- Testing unsupported file type ---")
    try:
        with open("dummy_configs/test_config.txt", "w") as f:
            f.write("This is not a valid config file.")
        load_config("dummy_configs/test_config.txt")
    except Exception as e:
        print(f"Error (expected): {e}")
        os.remove("dummy_configs/test_config.txt") # Clean up

    # Clean up dummy files and directory
    # os.remove("dummy_configs/test_config_valid.json")
    # os.remove("dummy_configs/test_config_valid.yaml")
    # os.remove("dummy_configs/test_config_invalid.json")
    # os.rmdir("dummy_configs") # only if empty, but useful for cleanup if script is run multiple times
    print("\nNote: Dummy config files created in 'dummy_configs/' directory. Clean them up manually if needed.")
