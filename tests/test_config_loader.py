import unittest
import json
import yaml
import os
import tempfile

# Add project root to sys.path to allow importing config_loader
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_loader import load_config, validate_config

class TestConfigLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to hold dummy config files
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup) # Ensure cleanup even if tests fail

        self.valid_config_dict_oxsa = {
            "input_file": "data.mat",
            "output_dir": "results/",
            "output_prefix": "scan1",
            "mode_specific_params": {
                "oxsa": {
                    "prior_knowledge_file": "prior.csv",
                    "lineshape": "voigt"
                }
            }
        }
        self.valid_config_dict_lcmodel = {
            "input_file": ["data1.dat", "data2.dat"], # Test list input
            "output_dir": "results_lcm/",
            "output_prefix": "lcm_run",
            "mode_specific_params": {
                "lcmodel": {
                    "basis_file": "basis.basis",
                    "fitting_range_ppm": [0.2, 4.0],
                    "baseline_degree": 3
                }
            }
        }

    def _create_dummy_file(self, filename, content, is_json):
        filepath = os.path.join(self.test_dir.name, filename)
        with open(filepath, 'w') as f:
            if is_json:
                json.dump(content, f, indent=4)
            else: # YAML
                yaml.dump(content, f, indent=4)
        return filepath

    def test_load_valid_json(self):
        filepath = self._create_dummy_file("valid.json", self.valid_config_dict_oxsa, True)
        loaded = load_config(filepath)
        self.assertEqual(loaded, self.valid_config_dict_oxsa)

    def test_load_valid_yaml(self):
        filepath = self._create_dummy_file("valid.yaml", self.valid_config_dict_lcmodel, False)
        loaded = load_config(filepath)
        self.assertEqual(loaded, self.valid_config_dict_lcmodel)

    def test_load_non_existent_file(self):
        with self.assertRaises(FileNotFoundError):
            load_config(os.path.join(self.test_dir.name, "non_existent.json"))

    def test_load_unsupported_format(self):
        filepath = self._create_dummy_file("unsupported.txt", {"key": "value"}, True) # Content doesn't matter
        renamed_filepath = os.path.join(self.test_dir.name, "unsupported.txt")
        # os.rename(filepath, renamed_filepath) # No, create directly with .txt
        with open(renamed_filepath, 'w') as f: f.write("text content")

        with self.assertRaisesRegex(ValueError, "Unsupported configuration file type: .txt"):
            load_config(renamed_filepath)

    def test_load_invalid_json_content(self):
        filepath = os.path.join(self.test_dir.name, "invalid_syntax.json")
        with open(filepath, 'w') as f:
            f.write("{'key': 'value', invalid_json_missing_quote}") # Invalid JSON
        with self.assertRaisesRegex(ValueError, "Error decoding JSON file"):
            load_config(filepath)

    def test_load_invalid_yaml_content(self):
        filepath = os.path.join(self.test_dir.name, "invalid_syntax.yaml")
        with open(filepath, 'w') as f:
            f.write("key: value\n  bad_indent: - value1\n bad_indent_again: value2") # Invalid YAML
        with self.assertRaisesRegex(ValueError, "Error parsing YAML file"):
            load_config(filepath)


    def test_validate_valid_config_oxsa(self):
        self.assertTrue(validate_config(self.valid_config_dict_oxsa, "oxsa"))

    def test_validate_valid_config_lcmodel(self):
        self.assertTrue(validate_config(self.valid_config_dict_lcmodel, "lcmodel"))

    def test_validate_input_file_as_list(self):
        config = self.valid_config_dict_lcmodel.copy()
        self.assertTrue(validate_config(config, "lcmodel")) # Already a list

    def test_validate_input_file_as_string(self):
        config = self.valid_config_dict_oxsa.copy()
        self.assertTrue(validate_config(config, "oxsa")) # Already a string

    def test_validate_invalid_input_file_type(self):
        config = self.valid_config_dict_oxsa.copy()
        config["input_file"] = 123 # Invalid type
        with self.assertRaisesRegex(ValueError, "'input_file' must be a string .* or a list of strings"):
            validate_config(config, "oxsa")

    def test_validate_missing_top_level_key(self):
        config = self.valid_config_dict_oxsa.copy()
        del config["output_dir"]
        with self.assertRaisesRegex(ValueError, "Missing required top-level key in config: 'output_dir'"):
            validate_config(config, "oxsa")

    def test_validate_missing_mode_specific_params_group(self):
        config = self.valid_config_dict_oxsa.copy()
        del config["mode_specific_params"]
        with self.assertRaisesRegex(ValueError, "Missing 'mode_specific_params' in config"):
            validate_config(config, "oxsa")

    def test_validate_missing_specific_mode_in_params(self):
        config = self.valid_config_dict_oxsa.copy()
        # Valid oxsa config, but try to validate for lcmodel
        with self.assertRaisesRegex(ValueError, "Missing mode-specific parameters for mode 'lcmodel'"):
            validate_config(config, "lcmodel")

    def test_validate_missing_key_in_oxsa_params(self):
        config = self.valid_config_dict_oxsa.copy()
        del config["mode_specific_params"]["oxsa"]["prior_knowledge_file"]
        with self.assertRaisesRegex(ValueError, "Missing required key 'prior_knowledge_file' in 'mode_specific_params.oxsa'"):
            validate_config(config, "oxsa")

    def test_validate_missing_key_in_lcmodel_params(self):
        config = self.valid_config_dict_lcmodel.copy()
        del config["mode_specific_params"]["lcmodel"]["basis_file"]
        with self.assertRaisesRegex(ValueError, "Missing required key 'basis_file' in 'mode_specific_params.lcmodel'"):
            validate_config(config, "lcmodel")

    def test_validate_unsupported_mode_for_validation(self):
        # This case should ideally be caught by argparse in CLI, but good to test validator robustness
        with self.assertRaisesRegex(ValueError, "Unsupported mode 'invalid_mode' for validation"):
            validate_config(self.valid_config_dict_oxsa, "invalid_mode")


if __name__ == '__main__':
    unittest.main()
```
