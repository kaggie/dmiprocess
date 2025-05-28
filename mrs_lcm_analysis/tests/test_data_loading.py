import unittest
import numpy as np
import scipy.fft
from mrs_lcm_analysis.lcm_library.data_loading import MRSData

class TestMRSData(unittest.TestCase):
    """Tests for the MRSData class."""

    def setUp(self):
        self.N_POINTS = 128
        self.SAMPLING_FREQ = 1000.0 # Hz
        self.CENTRAL_FREQ = 50.0 # MHz (for ppm conversion, using 50 for easier math)
        self.time_data = np.random.rand(self.N_POINTS) + 1j * np.random.rand(self.N_POINTS)
        # Create a simple, somewhat symmetric spectrum for FFT checks
        self.freq_data_unsh = np.zeros(self.N_POINTS, dtype=complex)
        self.freq_data_unsh[self.N_POINTS//4] = 10 # A peak
        self.freq_data_unsh[3*self.N_POINTS//4] = 10 # Its conjugate for real time data
        self.freq_data_shifted = scipy.fft.fftshift(self.freq_data_unsh)


    def test_creation_time_domain(self):
        """Test creation with time-domain data."""
        mrs_td = MRSData(self.time_data, "time", sampling_frequency=self.SAMPLING_FREQ)
        self.assertTrue(mrs_td.is_time_domain)
        self.assertFalse(mrs_td.is_frequency_domain)
        self.assertEqual(mrs_td.sampling_frequency, self.SAMPLING_FREQ)
        np.testing.assert_array_equal(mrs_td.data_array, self.time_data)

    def test_creation_frequency_domain(self):
        """Test creation with frequency-domain data."""
        mrs_fd = MRSData(self.freq_data_shifted, "frequency", 
                         sampling_frequency=self.SAMPLING_FREQ, 
                         central_frequency=self.CENTRAL_FREQ)
        self.assertTrue(mrs_fd.is_frequency_domain)
        self.assertFalse(mrs_fd.is_time_domain)
        np.testing.assert_array_equal(mrs_fd.data_array, self.freq_data_shifted)

    def test_time_to_frequency_conversion(self):
        """Test get_frequency_domain_data from time-domain input."""
        mrs_td = MRSData(self.time_data, "time", sampling_frequency=self.SAMPLING_FREQ)
        freq_output = mrs_td.get_frequency_domain_data()
        self.assertEqual(freq_output.shape, self.time_data.shape)
        # Basic check: compare with scipy's fft
        expected_freq_output = scipy.fft.fftshift(scipy.fft.fft(self.time_data))
        np.testing.assert_allclose(freq_output, expected_freq_output, atol=1e-9)

    def test_frequency_to_time_conversion(self):
        """Test get_time_domain_data from frequency-domain input."""
        # Use a known spectrum that corresponds to a real time signal
        time_signal_real = np.cos(2 * np.pi * 10 * np.arange(self.N_POINTS) / self.SAMPLING_FREQ)
        spectrum_complex = scipy.fft.fftshift(scipy.fft.fft(time_signal_real))
        
        mrs_fd = MRSData(spectrum_complex, "frequency", sampling_frequency=self.SAMPLING_FREQ)
        time_output = mrs_fd.get_time_domain_data()
        self.assertEqual(time_output.shape, spectrum_complex.shape)
        # Check if output is predominantly real (small imag part due to precision is ok)
        self.assertTrue(np.all(np.abs(time_output.imag) < 1e-9)) 
        np.testing.assert_allclose(time_output.real, time_signal_real, atol=1e-9)


    def test_get_frequency_axis_hz(self):
        """Test get_frequency_axis for 'hz' unit."""
        mrs_fd = MRSData(self.freq_data_shifted, "frequency", sampling_frequency=self.SAMPLING_FREQ)
        hz_axis = mrs_fd.get_frequency_axis(unit='hz')
        self.assertEqual(hz_axis.shape, (self.N_POINTS,))
        self.assertAlmostEqual(hz_axis[0], -self.SAMPLING_FREQ / 2)
        self.assertAlmostEqual(hz_axis[-1], self.SAMPLING_FREQ / 2 - self.SAMPLING_FREQ / self.N_POINTS)

    def test_get_frequency_axis_ppm(self):
        """Test get_frequency_axis for 'ppm' unit."""
        mrs_fd = MRSData(self.freq_data_shifted, "frequency", 
                         sampling_frequency=self.SAMPLING_FREQ, 
                         central_frequency=self.CENTRAL_FREQ) # Using MHz for CFREQ
        ppm_axis = mrs_fd.get_frequency_axis(unit='ppm')
        self.assertEqual(ppm_axis.shape, (self.N_POINTS,))
        
        # Expected ppm range (reversed)
        expected_max_ppm = (self.SAMPLING_FREQ / 2) / self.CENTRAL_FREQ 
        expected_min_ppm = (-self.SAMPLING_FREQ / 2) / self.CENTRAL_FREQ
        
        self.assertAlmostEqual(ppm_axis[0], expected_max_ppm, places=5) 
        self.assertAlmostEqual(ppm_axis[-1], expected_min_ppm, places=5)

    def test_missing_params_errors(self):
        """Test errors for missing required parameters."""
        with self.assertRaisesRegex(ValueError, "sampling_frequency is required for time-domain data"):
            MRSData(self.time_data, "time")
        
        mrs_td = MRSData(self.time_data, "time", sampling_frequency=self.SAMPLING_FREQ)
        with self.assertRaisesRegex(ValueError, "central_frequency is required for ppm scale"):
            mrs_td.get_frequency_axis(unit='ppm')
            
        mrs_fd_no_sf = MRSData(self.freq_data_shifted, "frequency")
        with self.assertRaisesRegex(ValueError, "sampling_frequency is required to generate a frequency axis"):
            mrs_fd_no_sf.get_frequency_axis(unit='hz')


if __name__ == '__main__':
    unittest.main()
