import unittest
from unittest.mock import patch, MagicMock, call
import argparse
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from the CLI script
# We need to be careful if spectrofit_cli itself tries to import torch/matplotlib at module level
# The refactored version should have these imports inside functions or guarded.
import spectrofit_cli

class TestSpectrofitCLIArgParsing(unittest.TestCase):

    def test_parse_args_required(self):
        with self.assertRaises(SystemExit): # argparse exits on error
            spectrofit_cli.parse_cli_args(["--mode", "oxsa"]) # Missing --config
        with self.assertRaises(SystemExit):
            spectrofit_cli.parse_cli_args(["--config", "cfg.json"]) # Missing --mode

    def test_parse_args_valid_minimal(self):
        args = spectrofit_cli.parse_cli_args(["--mode", "oxsa", "--config", "cfg.json"])
        self.assertEqual(args.mode, "oxsa")
        self.assertEqual(args.config, "cfg.json")
        self.assertEqual(args.loglevel, "INFO") # Default

    def test_parse_args_all_set(self):
        argv = [
            "--mode", "lcmodel",
            "--config", "myconfig.yaml",
            "--input", "myinput.dat",
            "--output", "myoutput_dir/",
            "--basis", "mybasis.basis",
            "--loglevel", "DEBUG"
        ]
        args = spectrofit_cli.parse_cli_args(argv)
        self.assertEqual(args.mode, "lcmodel")
        self.assertEqual(args.config, "myconfig.yaml")
        self.assertEqual(args.input, "myinput.dat")
        self.assertEqual(args.output, "myoutput_dir/")
        self.assertEqual(args.basis, "mybasis.basis")
        self.assertEqual(args.loglevel, "DEBUG")


@patch('spectrofit_cli.load_config')
@patch('spectrofit_cli.validate_config')
@patch('spectrofit_cli.load_spectra')
@patch('spectrofit_cli.preprocess_spectra')
@patch('spectrofit_cli.save_results_to_csv') # Mock output_utils directly if imported there
@patch('spectrofit_cli.plot_fit_results')  # Mock output_utils directly if imported there
@patch.dict(sys.modules, { # Mock mode-specific modules to prevent actual import attempts
    'oxsa_model': MagicMock(),
    'lcmodel_fitting': MagicMock(),
    'output_utils': MagicMock(save_results_to_csv=MagicMock(), plot_fit_results=MagicMock()) # also mock funcs from it
})
class TestSpectrofitCLIMainLogic(unittest.TestCase):

    def setUp(self):
        # Reset mocks for each test if they are attributes of self
        # For module-level mocks via @patch, they are reset automatically.
        # Access mocked functions via their original module name if patched at module level
        # or via spectrofit_cli if they are imported into its namespace.

        # If output_utils functions are imported into spectrofit_cli's namespace:
        if hasattr(spectrofit_cli, 'save_results_to_csv'):
            self.mock_save_csv = spectrofit_cli.save_results_to_csv
            self.mock_plot_results = spectrofit_cli.plot_fit_results
        else: # If they are used as output_utils.save_results_to_csv
            # This requires output_utils to be in sys.modules mock correctly
            output_utils_mock = sys.modules['output_utils']
            self.mock_save_csv = output_utils_mock.save_results_to_csv
            self.mock_plot_results = output_utils_mock.plot_fit_results

        # Mocking the fitting functions that would be dynamically imported
        # These mocks will be returned by the import statements inside run_processing
        self.mock_fit_oxsa = MagicMock(return_value={'final_loss': 0.1, 'fitted_params': {}, 'crlbs_absolute': {}})
        self.mock_fit_lcmodel = MagicMock(return_value={'amplitudes': {}, 'crlbs': {}})

        # This is a bit tricky due to dynamic imports. We patch where they are *looked up*.
        # If 'from oxsa_model import fit_oxsa_model' is used, patch 'spectrofit_cli.fit_oxsa_model'
        # if the import happens at the top of spectrofit_cli.
        # Since it's dynamic, we need to ensure the *importer* gets the mock.
        # The @patch.dict(sys.modules, ...) handles this by replacing the entire module.
        # We then assign the mock function to the attribute of that mocked module.
        sys.modules['oxsa_model'].fit_oxsa_model = self.mock_fit_oxsa

        # For lcmodel_fitting, we need to mock its components too
        lcmodel_fitting_mock = sys.modules['lcmodel_fitting']
        lcmodel_fitting_mock.fit_lcmodel_data = self.mock_fit_lcmodel
        lcmodel_fitting_mock.MRS_LCM_LIB_AVAILABLE = True # Assume lib is OK for these tests
        lcmodel_fitting_mock.create_mrs_data_object = MagicMock(return_value=MagicMock()) # Return a dummy MRSData
        lcmodel_fitting_mock.load_basis_set = MagicMock(return_value=MagicMock()) # Return a dummy BasisSet


    def test_single_file_processing_oxsa(self, mock_plot, mock_csv, mock_preprocess, mock_load_spectra, mock_validate, mock_load_cfg):
        """Test processing a single file in OXSA mode."""
        args = spectrofit_cli.parse_cli_args(["--mode", "oxsa", "--config", "dummy_cfg.json"])

        mock_load_cfg.return_value = {
            "input_file": "sample_data/test.mat", "output_dir": "results", "output_prefix": "single_oxsa",
            "mode_specific_params": {"oxsa": {"num_peaks":1, "initial_params_guess":{}, "fit_settings":{}}}
        }
        mock_load_spectra.return_value = {"data": np.array([1]), "axis": np.array([1]), "metadata": {}}
        mock_preprocess.side_effect = lambda d, p, logger: d # Return data as is

        spectrofit_cli.run_processing(args, mock_load_cfg.return_value)

        mock_load_cfg.assert_called_once_with("dummy_cfg.json")
        mock_validate.assert_called_once()
        mock_load_spectra.assert_called_once()
        self.mock_fit_oxsa.assert_called_once()
        self.mock_save_csv.assert_called_once()
        self.mock_plot_results.assert_called_once()

        # Check unique filename part
        csv_call_args = self.mock_save_csv.call_args
        self.assertIn("single_oxsa_test", csv_call_args[0][0]) # filepath_prefix


    def test_batch_processing_lcmodel(self, mock_plot, mock_csv, mock_preprocess, mock_load_spectra, mock_validate, mock_load_cfg):
        """Test batch processing for two files in LCModel mode."""
        args = spectrofit_cli.parse_cli_args(["--mode", "lcmodel", "--config", "dummy_batch.json"])

        mock_load_cfg.return_value = {
            "input_file": ["file1.dat", "file2.dat"], "output_dir": "batch_out", "output_prefix": "lcm_batch",
            "mode_specific_params": {"lcmodel": {"basis_file":"b.basis", "fitting_range_ppm":[0,4], "baseline_degree":1}}
        }
        mock_load_spectra.return_value = {"data": np.array([1]), "axis": np.array([1]), "metadata": {"tx_freq_hz":123e6, "spectral_width_hz":2000}}
        mock_preprocess.side_effect = lambda d, p, logger: d

        spectrofit_cli.run_processing(args, mock_load_cfg.return_value)

        self.assertEqual(mock_load_spectra.call_count, 2)
        self.assertEqual(self.mock_fit_lcmodel.call_count, 2)
        self.assertEqual(self.mock_save_csv.call_count, 2)
        self.assertEqual(self.mock_plot_results.call_count, 2)

        # Check unique filenames for batch
        first_csv_call_args = self.mock_save_csv.call_args_list[0][0][0] # filepath_prefix of first call
        second_csv_call_args = self.mock_save_csv.call_args_list[1][0][0]
        self.assertIn("lcm_batch_file1", first_csv_call_args)
        self.assertIn("lcm_batch_file2", second_csv_call_args)
        self.assertNotEqual(first_csv_call_args, second_csv_call_args)


    def test_per_file_error_handling(self, mock_plot, mock_csv, mock_preprocess, mock_load_spectra, mock_validate, mock_load_cfg):
        """Test that batch continues if one file fails at data loading."""
        args = spectrofit_cli.parse_cli_args(["--mode", "oxsa", "--config", "err_batch.json"])

        mock_load_cfg.return_value = {
            "input_file": ["good_file.mat", "bad_file.mat", "another_good.mat"],
            "output_dir": "errors", "output_prefix": "err_test",
            "mode_specific_params": {"oxsa": {"num_peaks":1, "initial_params_guess":{}, "fit_settings":{}}}
        }

        # Simulate 'bad_file.mat' failing to load
        def load_spectra_side_effect(filepath, *args_load, **kwargs_load):
            if "bad_file" in filepath:
                raise ValueError("Failed to load bad_file.mat")
            return {"data": np.array([1]), "axis": np.array([1]), "metadata": {}}
        mock_load_spectra.side_effect = load_spectra_side_effect
        mock_preprocess.side_effect = lambda d, p, logger: d

        all_succeeded = spectrofit_cli.run_processing(args, mock_load_cfg.return_value)

        self.assertFalse(all_succeeded) # Overall process should indicate failure
        self.assertEqual(mock_load_spectra.call_count, 3) # Attempted all files
        self.assertEqual(self.mock_fit_oxsa.call_count, 2) # Fit only for good files
        self.assertEqual(self.mock_save_csv.call_count, 2)
        self.assertEqual(self.mock_plot_results.call_count, 2)


    def test_cli_input_override(self, mock_plot, mock_csv, mock_preprocess, mock_load_spectra, mock_validate, mock_load_cfg):
        """Test --input CLI argument overrides config and processes only one file."""
        args = spectrofit_cli.parse_cli_args([
            "--mode", "oxsa",
            "--config", "dummy_cfg_list.json",
            "--input", "cli_override.mat"
        ])

        mock_load_cfg.return_value = { # This config has a list, but CLI --input should override
            "input_file": ["config_file1.mat", "config_file2.mat"],
            "output_dir": "results", "output_prefix": "override_test",
            "mode_specific_params": {"oxsa": {"num_peaks":1, "initial_params_guess":{}, "fit_settings":{}}}
        }
        mock_load_spectra.return_value = {"data": np.array([1]), "axis": np.array([1]), "metadata": {}}
        mock_preprocess.side_effect = lambda d, p, logger: d

        spectrofit_cli.run_processing(args, mock_load_cfg.return_value)

        mock_load_spectra.assert_called_once() # Only one file processed
        # Check that the filepath passed to load_spectra is the one from CLI arg
        actual_filepath_loaded = mock_load_spectra.call_args[0][0]
        self.assertEqual(actual_filepath_loaded, "cli_override.mat")
        self.mock_fit_oxsa.assert_called_once()
        self.mock_save_csv.assert_called_once()


if __name__ == '__main__':
    unittest.main()
```
