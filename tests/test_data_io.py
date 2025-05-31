import unittest
import numpy as np
import os
import tempfile
import logging

# Add project root to sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_io import load_spectra, preprocess_spectra, SCIPY_AVAILABLE, H5PY_AVAILABLE

# Mock logger for tests
test_logger = logging.getLogger("data_io_test_logger")
test_logger.addHandler(logging.NullHandler()) # Prevent log output during tests unless specifically configured

class TestDataIO(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)

        # Sample data
        self.sample_data_array = np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex64)
        self.sample_axis_array = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        self.sample_metadata = {'tx_freq_hz': 123.2e6, 'spectral_width_hz': 5000.0}

    @unittest.skipIf(not SCIPY_AVAILABLE, "SciPy not installed, skipping .mat tests")
    def test_load_spectra_mat(self):
        import scipy.io
        mat_filepath = os.path.join(self.test_dir.name, "test.mat")
        scipy.io.savemat(mat_filepath, {
            'data': self.sample_data_array,
            'axis': self.sample_axis_array,
            'metadata': self.sample_metadata # Stored as a struct-like object
        })

        loaded = load_spectra(mat_filepath, "mat", logger=test_logger)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_almost_equal(loaded['data'], self.sample_data_array)
        np.testing.assert_array_almost_equal(loaded['axis'], self.sample_axis_array)
        self.assertEqual(loaded['metadata'].get('tx_freq_hz'), self.sample_metadata['tx_freq_hz'])
        self.assertEqual(loaded['metadata'].get('spectral_width_hz'), self.sample_metadata['spectral_width_hz'])

    @unittest.skipIf(not H5PY_AVAILABLE, "h5py not installed, skipping .h5 tests")
    def test_load_spectra_h5(self):
        import h5py
        h5_filepath = os.path.join(self.test_dir.name, "test.h5")
        with h5py.File(h5_filepath, 'w') as hf:
            hf.create_dataset('data', data=self.sample_data_array)
            hf.create_dataset('axis', data=self.sample_axis_array)
            meta_group = hf.create_group('metadata_group')
            for k, v in self.sample_metadata.items():
                meta_group.attrs[k] = v

        loaded = load_spectra(h5_filepath, "h5", logger=test_logger)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_almost_equal(loaded['data'], self.sample_data_array.astype(np.complex128)) # load_spectra converts to complex128
        np.testing.assert_array_almost_equal(loaded['axis'], self.sample_axis_array)
        self.assertEqual(loaded['metadata'].get('tx_freq_hz'), self.sample_metadata['tx_freq_hz'])

    def test_load_spectra_unsupported_format(self):
        dummy_filepath = os.path.join(self.test_dir.name, "test.txt")
        with open(dummy_filepath, 'w') as f: f.write("dummy")
        with self.assertRaisesRegex(ValueError, "Unsupported file format: 'txt'"):
            load_spectra(dummy_filepath, "txt", logger=test_logger)

    def test_load_spectra_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_spectra("non_existent_file.mat", "mat", logger=test_logger)

    def test_preprocess_spectra_phase_correction_manual(self):
        # Simple test for phase correction (assumes data is complex frequency domain)
        # This test uses a simplified scenario. Real data would need careful axis handling.
        data = np.array([1+1j, 1+1j, 1+1j, 1+1j], dtype=np.complex64)
        # Create a dummy axis; actual values don't matter much for this specific phase correction test if ph1_deg_per_point is used
        axis = np.arange(len(data))
        metadata = {'transmitter_frequency_hz': 100e6} # Dummy value for potential ph1_deg_per_ppm scaling

        spectra_obj = {'data': data.copy(), 'axis': axis, 'metadata': metadata}

        # ph0 = 90 degrees, ph1 = 0
        # e^(j*pi/2) = j. So (1+1j)*j = j - 1 = -1 + 1j
        processing_params = {
            "phase_correction": {"method": "manual", "ph0_deg": 90.0, "ph1_deg_per_point": 0.0}
        }
        processed_obj = preprocess_spectra(spectra_obj, processing_params, logger=test_logger)
        expected_data = (data * 1j) # Multiplication by exp(j*pi/2)
        np.testing.assert_array_almost_equal(processed_obj['data'], expected_data, decimal=5)

    def test_preprocess_spectra_apodization_exponential(self):
        # Test exponential apodization (assumes time-domain data)
        data = np.ones(100, dtype=np.complex64) # Simple FID (array of ones)
        time_axis = np.arange(100) * 0.001 # 1ms dwell time
        metadata = {'dwell_time_s': 0.001, 'spectral_width_hz': 1000.0}
        spectra_obj = {'data': data.copy(), 'axis': time_axis, 'metadata': metadata}

        lb_hz = 1.0
        processing_params = {
            "apodization": {"function": "exponential", "lb_hz": lb_hz}
        }
        processed_obj = preprocess_spectra(spectra_obj, processing_params, logger=test_logger)

        # Expected decay: exp(-pi * lb_hz * t)
        decay_factor = np.pi * lb_hz * metadata['dwell_time_s']
        expected_apod_func = np.exp(-decay_factor * time_axis)
        expected_data = data * expected_apod_func

        self.assertNotEqual(processed_obj['data'][0], processed_obj['data'][-1], "Data should be apodized")
        np.testing.assert_array_almost_equal(processed_obj['data'], expected_data, decimal=5)


    def test_preprocess_spectra_baseline_poly(self):
        # Test polynomial baseline correction (assumes frequency domain data)
        # Create data with a linear baseline y = 0.1*x + (complex_signal)
        np_axis = np.linspace(-1, 1, 100)
        signal = np.zeros_like(np_axis, dtype=np.complex64) # No actual signal, just baseline
        baseline_real = 0.1 * np.arange(len(np_axis))
        baseline_imag = 0.05 * np.arange(len(np_axis))
        data_with_baseline = signal + baseline_real + 1j*baseline_imag

        spectra_obj = {'data': data_with_baseline.copy(), 'axis': np_axis, 'metadata': {}}
        processing_params = {
            "baseline_correction": {"method": "polynomial", "degree": 1}
        }

        processed_obj = preprocess_spectra(spectra_obj, processing_params, logger=test_logger)

        # After correction, the mean of the data should be close to zero if baseline was removed effectively
        # This is a very rough check. For polynomial, the residuals should not be fittable by same degree poly.
        self.assertTrue(np.mean(processed_obj['data'].real) < np.mean(baseline_real))
        self.assertTrue(np.mean(processed_obj['data'].imag) < np.mean(baseline_imag))
        # A more rigorous test would check if residuals are orthogonal to polynomial basis.
        # For simplicity here, check if the magnitude is reduced.
        self.assertTrue(np.abs(np.mean(processed_obj['data'].real)) < 0.1) # Should be very close to 0
        self.assertTrue(np.abs(np.mean(processed_obj['data'].imag)) < 0.1)


if __name__ == '__main__':
    # Setup logger for testing this file standalone
    # logging.basicConfig(level=logging.DEBUG)
    # test_logger = logging.getLogger("data_io_test_logger")
    unittest.main()
```
