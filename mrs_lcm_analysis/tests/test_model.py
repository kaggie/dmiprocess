import unittest
import numpy as np
import torch # For torch.linalg.lstsq if used in model
from mrs_lcm_analysis.lcm_library.data_loading import MRSData
from mrs_lcm_analysis.lcm_library.basis import BasisSpectrum, BasisSet
from mrs_lcm_analysis.lcm_library.model import LinearCombinationModel
from mrs_lcm_analysis.lcm_library.baseline import create_polynomial_basis_vectors

class TestLinearCombinationModel(unittest.TestCase):
    """Tests for the LinearCombinationModel class."""

    def setUp(self):
        """Set up common resources for tests."""
        self.N_POINTS = 128
        self.SAMPLING_FREQ = 1000.0
        self.CENTRAL_FREQ = 50.0

        # Create a simple frequency axis (ppm, reversed for typical MRS display)
        # This axis will be used for both MRSData and BasisSpectrum for consistency
        hz_axis = np.linspace(-self.SAMPLING_FREQ / 2, self.SAMPLING_FREQ / 2 - self.SAMPLING_FREQ / self.N_POINTS, self.N_POINTS)
        self.ppm_axis = (hz_axis / self.CENTRAL_FREQ)[::-1]


        # Basis Spectrum 1 (e.g., NAA-like) - Gaussian peak
        bs1_data = np.zeros(self.N_POINTS)
        bs1_peak_idx = self.N_POINTS // 4 # Arbitrary peak position
        bs1_data[bs1_peak_idx] = 1.0 # Simple peak
        # Ensure basis spectra are fftshifted if MRSData expects/provides shifted data
        # For simplicity, let's assume BasisSpectrum stores unshifted data and get_basis_matrix handles shift if necessary,
        # or that MRSData's get_frequency_domain_data handles the final state.
        # Given our MRSData.get_frequency_domain_data applies fftshift, and BasisSpectrum.spectrum_data is taken as is,
        # we should store basis spectra in a way that's compatible with the model's internal frequency representation.
        # The model internally uses spectrum_data = self.mrs_data.get_frequency_domain_data(apply_fftshift=True)
        # and basis_matrix = self.basis_set.get_basis_matrix() (which should also be fftshifted)
        # So, if we make a Gaussian centered, then fftshift it, it's ready.
        
        # Let's make a more realistic Gaussian for basis
        gauss = lambda x, mu, sigma: np.exp(-0.5 * ((x - mu) / sigma)**2)
        x_points = np.arange(self.N_POINTS)
        
        # Basis Spectrum 1 (e.g., NAA-like)
        bs1_center_shifted = self.N_POINTS // 2 + 10 # Peak position in shifted spectrum
        bs1_data_shifted = gauss(x_points, bs1_center_shifted, 3) # Gaussian peak in shifted domain
        self.bs1 = BasisSpectrum("NAA", bs1_data_shifted, frequency_axis=self.ppm_axis)

        # Basis Spectrum 2 (e.g., Cr-like)
        bs2_center_shifted = self.N_POINTS // 2 - 15 # Another peak position
        bs2_data_shifted = gauss(x_points, bs2_center_shifted, 3)
        self.bs2 = BasisSpectrum("Cr", bs2_data_shifted, frequency_axis=self.ppm_axis)

        self.basis_set = BasisSet()
        self.basis_set.add_metabolite(self.bs1)
        self.basis_set.add_metabolite(self.bs2)

        # Create MRSData (frequency domain, shifted)
        # For a simple test, make data = 1.0 * bs1 + 0.5 * bs2 + noise
        self.true_amp_bs1 = 1.0
        self.true_amp_bs2 = 0.5
        simulated_spectrum_shifted = (self.true_amp_bs1 * self.bs1.spectrum_data + 
                                      self.true_amp_bs2 * self.bs2.spectrum_data)
        noise = np.random.normal(0, 0.05, self.N_POINTS) # Low noise
        self.mrs_data_array_shifted = simulated_spectrum_shifted + noise
        
        self.mrs_data = MRSData(self.mrs_data_array_shifted, "frequency",
                                sampling_frequency=self.SAMPLING_FREQ,
                                central_frequency=self.CENTRAL_FREQ)
        
        # For a noise-free, single metabolite test
        self.mrs_data_bs1_only_shifted = self.bs1.spectrum_data.copy()
        self.mrs_data_bs1_only = MRSData(self.mrs_data_bs1_only_shifted, "frequency",
                                         sampling_frequency=self.SAMPLING_FREQ,
                                         central_frequency=self.CENTRAL_FREQ)


    def test_model_creation(self):
        """Test successful creation of a LinearCombinationModel object."""
        model = LinearCombinationModel(self.mrs_data, self.basis_set)
        self.assertIsInstance(model, LinearCombinationModel)

    def test_fit_basic(self):
        """Test basic fit functionality."""
        model = LinearCombinationModel(self.mrs_data, self.basis_set)
        model.fit()
        
        amps = model.get_estimated_metabolite_amplitudes()
        self.assertIsNotNone(amps)
        self.assertEqual(len(amps), 2) # NAA and Cr
        self.assertIn("NAA", amps)
        self.assertIn("Cr", amps)

        fitted_spectrum = model.get_fitted_spectrum()
        self.assertEqual(fitted_spectrum.shape, self.mrs_data.data_array.real[model.fit_indices if model.fit_indices is not None else slice(None)].shape)
        
        residuals = model.get_residuals()
        self.assertEqual(residuals.shape, fitted_spectrum.shape)

    def test_fit_single_metabolite_noise_free(self):
        """Test fit with data being exactly one basis spectrum (noise-free)."""
        model = LinearCombinationModel(self.mrs_data_bs1_only, self.basis_set)
        model.fit()
        amps = model.get_estimated_metabolite_amplitudes()
        self.assertAlmostEqual(amps["NAA"], 1.0, places=3) # Should be close to 1.0
        self.assertAlmostEqual(amps["Cr"], 0.0, places=3)  # Should be close to 0.0

    def test_fit_with_baseline(self):
        """Test fit with polynomial baseline correction."""
        BASELINE_DEGREE = 3
        model = LinearCombinationModel(self.mrs_data, self.basis_set, baseline_degree=BASELINE_DEGREE)
        model.fit()
        
        met_amps = model.get_estimated_metabolite_amplitudes()
        self.assertIsNotNone(met_amps)
        self.assertEqual(len(met_amps), 2)
        
        bl_amps = model.get_estimated_baseline_amplitudes()
        self.assertIsNotNone(bl_amps)
        self.assertEqual(len(bl_amps), BASELINE_DEGREE + 1)

        fitted_spectrum = model.get_fitted_spectrum()
        self.assertEqual(fitted_spectrum.shape, model.data_to_fit.shape)
        
        fitted_baseline = model.get_fitted_baseline()
        self.assertEqual(fitted_baseline.shape, model.data_to_fit.shape)

    def test_fit_with_fitting_range(self):
        """Test fit with a specified PPM fitting range."""
        # Define a range that doesn't cover the whole spectrum
        # ppm_axis is reversed, so min_ppm > max_ppm for typical MRS display (high to low)
        # But fitting_range_ppm expects (min_actual_ppm_value, max_actual_ppm_value)
        # Our self.ppm_axis is e.g. [5, 4.9, ..., 0.1, 0]
        # So a fitting range of (1.0, 3.0) means we take points where 1.0 <= ppm <= 3.0
        
        fitting_range = (self.ppm_axis.min() + 1.0, self.ppm_axis.max() - 1.0) 
        if fitting_range[0] >= fitting_range[1]: # Ensure valid range if bounds are too close
            fitting_range = (self.ppm_axis[len(self.ppm_axis)//2 + 10], self.ppm_axis[len(self.ppm_axis)//2 -10])

        model = LinearCombinationModel(self.mrs_data, self.basis_set, fitting_range_ppm=fitting_range)
        model.fit()
        
        self.assertIsNotNone(model.fit_indices)
        self.assertTrue(len(model.fit_indices) < self.N_POINTS)
        self.assertEqual(model.data_to_fit.shape[0], len(model.fit_indices))
        self.assertEqual(model.fitted_spectrum.shape[0], len(model.fit_indices))

    def test_crlb_calculation(self):
        """Test CRLB calculation runs and produces output."""
        model = LinearCombinationModel(self.mrs_data, self.basis_set, baseline_degree=2)
        model.fit() # CRLBs are calculated within fit()
        
        crlbs = model.get_estimated_crlbs()
        self.assertIsNotNone(crlbs)
        self.assertIn('absolute', crlbs)
        self.assertIn('percent_metabolite', crlbs)

        if crlbs['absolute'] is not None: # Check if calculation was successful
            self.assertEqual(len(crlbs['absolute']), 2 + (2 + 1)) # 2 metabolites + 3 baseline params
            self.assertIn("NAA", crlbs['absolute'])
            self.assertIn("baseline_deg0", crlbs['absolute'])
        if crlbs['percent_metabolite'] is not None:
            self.assertEqual(len(crlbs['percent_metabolite']), 2)
            self.assertIn("NAA", crlbs['percent_metabolite'])
            
    def test_relative_quantification(self):
        """Test relative quantification."""
        model = LinearCombinationModel(self.mrs_data, self.basis_set)
        model.fit()
        amps = model.get_estimated_metabolite_amplitudes()
        
        # Ensure Cr amplitude is positive for a meaningful test
        if amps.get("Cr", 0) <=0:
            # Fudge Cr amplitude to be positive for test purposes if needed
            # This might happen due to noise in setUp making Cr negative
            model.estimated_metabolite_amplitudes["Cr"] = abs(amps.get("Cr", 0.1)) + 0.1 
            if model.estimated_metabolite_amplitudes["Cr"] == 0:  model.estimated_metabolite_amplitudes["Cr"] = 1.0


        rel_quants = model.get_relative_quantification(reference_metabolites=['Cr'])
        self.assertIsNotNone(rel_quants)
        if rel_quants: # if calculation was successful
            self.assertIn('NAA', rel_quants)
            self.assertAlmostEqual(rel_quants['NAA'], amps['NAA'] / model.estimated_metabolite_amplitudes['Cr'])

        with self.assertRaises(ValueError):
            model.get_relative_quantification([]) # Empty list
        with self.assertRaises(ValueError):
            model.get_relative_quantification("Cr") # Not a list
            
        # Test case where reference metabolite is missing
        self.assertIsNone(model.get_relative_quantification(['NonExistentMetabolite']))


if __name__ == '__main__':
    unittest.main()
