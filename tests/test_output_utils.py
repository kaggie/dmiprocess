import unittest
from unittest.mock import patch, mock_open, MagicMock
import io
import csv
import os
import numpy as np

# Add project root to sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to be tested
import output_utils
from output_utils import save_results_to_csv, plot_fit_results

# Mock logger for tests
test_logger = logging.getLogger("output_utils_test_logger")
test_logger.addHandler(logging.NullHandler())


class TestSaveResultsToCSV(unittest.TestCase):

    def setUp(self):
        self.filepath_prefix = "test_results/scan_01" #output_utils creates dir
        self.oxsa_fit_results_basic = {
            'fitted_params': {'a': [1.0, 0.5], 'f': [10.0, 20.0], 'd': [0.1, 0.2], 'phi': [0.0, 0.1], 'g': [0.5, 0.5]},
            'crlbs_absolute': {'a_0': 0.1, 'f_0': 0.01, 'd_0': 0.01, 'phi_0': 0.005, 'g_0': 0.05,
                               'a_1': 0.05, 'f_1': 0.02, 'd_1': 0.02, 'phi_1': 0.01, 'g_1': 0.06},
            'final_loss': 0.001
        }
        self.lcmodel_fit_results_basic = {
            'amplitudes': {'NAA': 10.0, 'Cr': 8.0},
            'crlbs': {
                'absolute': {'NAA': 0.5, 'Cr': 0.4},
                'percent_metabolite': {'NAA': 5.0, 'Cr': 4.88}
            },
            'baseline_amplitudes': np.array([0.1, -0.05, 0.01])
        }
        # Ensure the test_results directory may or may not exist to test os.makedirs
        if os.path.exists(os.path.dirname(self.filepath_prefix)):
             # Clean up if needed from previous partial runs, but makedirs has exist_ok=True
             pass


    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_save_oxsa_results_csv(self, mock_makedirs, mock_file_open):
        save_results_to_csv(self.filepath_prefix, self.oxsa_fit_results_basic, 'oxsa', logger=test_logger)

        expected_filename = f"{self.filepath_prefix}_oxsa_results.csv"
        mock_makedirs.assert_called_once_with(os.path.dirname(expected_filename), exist_ok=True)
        mock_file_open.assert_called_once_with(expected_filename, 'w', newline='')

        # Check what was written to the mock file (via handle)
        handle = mock_file_open()
        written_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)

        reader = csv.reader(io.StringIO(written_content))
        header = next(reader)
        self.assertEqual(header, ["Peak_Index", "Parameter_Type", "Fitted_Value", "CRLB_Absolute"])

        rows = list(reader)
        self.assertEqual(len(rows), 2 * 5) # 2 peaks, 5 params each
        self.assertEqual(rows[0], ['0', 'a', f"{1.0:.6e}", f"{0.1:.6e}"])
        self.assertEqual(rows[5], ['1', 'a', f"{0.5:.6e}", f"{0.05:.6e}"])


    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_save_lcmodel_results_csv(self, mock_makedirs, mock_file_open):
        save_results_to_csv(self.filepath_prefix, self.lcmodel_fit_results_basic, 'lcmodel', logger=test_logger)

        expected_filename = f"{self.filepath_prefix}_lcmodel_results.csv"
        mock_makedirs.assert_called_once_with(os.path.dirname(expected_filename), exist_ok=True)
        mock_file_open.assert_called_once_with(expected_filename, 'w', newline='')

        handle = mock_file_open()
        written_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)
        reader = csv.reader(io.StringIO(written_content))
        header = next(reader)
        self.assertEqual(header, ["Metabolite", "Amplitude", "CRLB_Absolute", "CRLB_Percent"])
        rows = list(reader)
        self.assertEqual(rows[0], ['NAA', f"{10.0:.6e}", f"{0.5:.6e}", "5.00"])
        self.assertEqual(rows[1], ['Cr', f"{8.0:.6e}", f"{0.4:.6e}", "4.88"])
        # Check for baseline coeffs
        self.assertEqual(rows[3], ["Baseline_Coefficient_Index", "Value"]) # After spacer
        self.assertEqual(rows[4], ["coeff_0", f"{0.1:.6e}"])


    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_save_results_no_crlbs_oxsa(self, mock_makedirs, mock_file_open):
        results_no_crlb = self.oxsa_fit_results_basic.copy()
        results_no_crlb['crlbs_absolute'] = None
        save_results_to_csv(self.filepath_prefix, results_no_crlb, 'oxsa', logger=test_logger)

        handle = mock_file_open()
        written_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)
        reader = csv.reader(io.StringIO(written_content))
        header = next(reader) # Skip header
        rows = list(reader)
        self.assertEqual(rows[0], ['0', 'a', f"{1.0:.6e}", f"{np.nan:.6e}"]) # CRLB is nan


    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_save_results_no_fit_results(self, mock_makedirs, mock_file_open):
        save_results_to_csv(self.filepath_prefix, None, 'oxsa', logger=test_logger)
        mock_file_open.assert_not_called() # Should skip if no results


class TestPlotFitResults(unittest.TestCase):
    def setUp(self):
        self.filepath_prefix = "test_results/scan_plot"
        self.oxsa_results = {
            'time_axis': np.linspace(0, 1, 100),
            'fitted_spectrum_total': np.random.rand(100) + 1j*np.random.rand(100),
            'residuals_final': np.random.rand(100) + 1j*np.random.rand(100)
        }
        self.oxsa_mrs_data_dict = { # Simulates dict from data_io
            'data': np.random.rand(100) + 1j*np.random.rand(100),
            'axis': np.linspace(0,1,100), # time axis
            'metadata': {'spectral_width_hz': 1000.0, 'tx_freq_hz': 123.2e6}
        }
        self.lcmodel_results = {
            'fitted_spectrum_total': np.random.rand(100),
            'residuals': np.random.rand(100),
            'fitted_baseline': np.random.rand(100),
            'fitted_spectrum_metabolites': np.random.rand(100),
            'frequency_axis_fitted': np.linspace(4,0,100)
        }
        # Mock MRSData object for LCModel
        self.mock_mrs_data_obj = MagicMock()
        self.mock_mrs_data_obj.get_frequency_domain_data.return_value = np.random.rand(100)
        self.mock_mrs_data_obj.get_frequency_axis.return_value = np.linspace(4,0,100)
        self.mock_mrs_data_obj.central_frequency = 123.2


    @patch('output_utils.plt') # Mock the plt object within output_utils
    @patch('os.makedirs')
    def test_plot_oxsa_results_matplotlib_called(self, mock_makedirs, mock_plt):
        output_utils.MATPLOTLIB_AVAILABLE = True # Ensure test runs this path
        plot_fit_results(self.filepath_prefix, self.oxsa_mrs_data_dict, self.oxsa_results, 'oxsa', logger=test_logger)

        mock_makedirs.assert_called_with(os.path.dirname(self.filepath_prefix), exist_ok=True)
        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.savefig.called)
        # Expect 3 plots for OXSA (real, imag, real_residuals) + 1 optional freq domain plot
        self.assertGreaterEqual(mock_plt.savefig.call_count, 3)
        self.assertTrue(mock_plt.close.called)
        self.assertEqual(mock_plt.close.call_args[0][0], 'all')


    @patch('output_utils.plt')
    @patch('os.makedirs')
    def test_plot_lcmodel_results_matplotlib_called(self, mock_makedirs, mock_plt):
        output_utils.MATPLOTLIB_AVAILABLE = True
        plot_fit_results(self.filepath_prefix, self.mock_mrs_data_obj, self.lcmodel_results, 'lcmodel', logger=test_logger)

        mock_makedirs.assert_called_with(os.path.dirname(self.filepath_prefix), exist_ok=True)
        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.savefig.called)
        self.assertEqual(mock_plt.savefig.call_count, 2) # Fit plot and residuals plot
        self.assertTrue(mock_plt.close.called)

    @patch('output_utils.plt')
    @patch('os.makedirs')
    def test_plot_results_matplotlib_unavailable(self, mock_makedirs, mock_plt):
        output_utils.MATPLOTLIB_AVAILABLE = False # Simulate matplotlib not being available
        plot_fit_results(self.filepath_prefix, self.oxsa_mrs_data_dict, self.oxsa_results, 'oxsa', logger=test_logger)

        mock_makedirs.assert_not_called() # Should skip even directory creation if no plots
        mock_plt.figure.assert_not_called()
        mock_plt.savefig.assert_not_called()

    @patch('output_utils.plt') # Still need to mock plt to avoid it actually trying to plot
    @patch('os.makedirs')
    def test_plot_no_fit_results(self, mock_makedirs, mock_plt):
        output_utils.MATPLOTLIB_AVAILABLE = True
        plot_fit_results(self.filepath_prefix, self.oxsa_mrs_data_dict, None, 'oxsa', logger=test_logger)
        mock_plt.savefig.assert_not_called()


if __name__ == '__main__':
    unittest.main()

```
