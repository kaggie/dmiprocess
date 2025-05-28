import unittest
import numpy as np
from mrs_lcm_analysis.lcm_library.basis import BasisSpectrum, BasisSet

class TestBasisSpectrum(unittest.TestCase):
    """Tests for the BasisSpectrum class."""

    def test_creation(self):
        """Test successful creation of a BasisSpectrum object."""
        name = "TestSpec"
        data = np.array([1.0, 2.0, 3.0])
        freq_axis = np.array([0.1, 0.2, 0.3])
        
        bs = BasisSpectrum(name, data, freq_axis)
        self.assertEqual(bs.name, name)
        np.testing.assert_array_equal(bs.spectrum_data, data)
        np.testing.assert_array_equal(bs.frequency_axis, freq_axis)

    def test_attributes_storage(self):
        """Test that name and spectrum_data attributes are correctly stored."""
        name = "NAA"
        data = np.random.rand(10)
        bs = BasisSpectrum(name, data) # No frequency axis
        self.assertEqual(bs.name, name)
        np.testing.assert_array_equal(bs.spectrum_data, data)
        self.assertIsNone(bs.frequency_axis)

    def test_creation_type_errors(self):
        """Test type errors for invalid input."""
        data = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(TypeError):
            BasisSpectrum(123, data) # Invalid name type
        with self.assertRaises(TypeError):
            BasisSpectrum("Test", [1,2,3]) # Invalid data type
        with self.assertRaises(TypeError):
            BasisSpectrum("Test", data, [0.1, 0.2, 0.3]) # Invalid freq_axis type
        with self.assertRaises(ValueError):
             BasisSpectrum("Test", data, np.array([0.1, 0.2])) # Mismatched length


class TestBasisSet(unittest.TestCase):
    """Tests for the BasisSet class."""

    def setUp(self):
        """Set up common resources for tests."""
        self.basis_set = BasisSet()
        self.spec1_data = np.array([1.0, 2.0, 3.0, 4.0])
        self.spec1 = BasisSpectrum("Spec1", self.spec1_data)
        
        self.spec2_data = np.array([5.0, 6.0, 7.0, 8.0])
        self.spec2 = BasisSpectrum("Spec2", self.spec2_data)

    def test_add_and_get_metabolite(self):
        """Test adding a BasisSpectrum and retrieving it."""
        self.basis_set.add_metabolite(self.spec1)
        retrieved_spec = self.basis_set.get_metabolite("Spec1")
        self.assertEqual(retrieved_spec.name, "Spec1")
        np.testing.assert_array_equal(retrieved_spec.spectrum_data, self.spec1_data)

    def test_get_metabolite_names(self):
        """Test retrieving metabolite names."""
        self.basis_set.add_metabolite(self.spec1)
        self.basis_set.add_metabolite(self.spec2)
        names = self.basis_set.get_metabolite_names()
        self.assertIn("Spec1", names)
        self.assertIn("Spec2", names)
        self.assertEqual(len(names), 2)

    def test_get_basis_matrix_single_spectrum(self):
        """Test get_basis_matrix with one spectrum."""
        self.basis_set.add_metabolite(self.spec1)
        matrix = self.basis_set.get_basis_matrix()
        self.assertEqual(matrix.shape, (4, 1))
        np.testing.assert_array_equal(matrix[:, 0], self.spec1_data)

    def test_get_basis_matrix_multiple_spectra(self):
        """Test get_basis_matrix with multiple spectra."""
        self.basis_set.add_metabolite(self.spec1)
        self.basis_set.add_metabolite(self.spec2)
        matrix = self.basis_set.get_basis_matrix(metabolite_names=["Spec1", "Spec2"])
        self.assertEqual(matrix.shape, (4, 2))
        np.testing.assert_array_equal(matrix[:, 0], self.spec1_data)
        np.testing.assert_array_equal(matrix[:, 1], self.spec2_data)
        
        # Test with default order (all metabolites)
        matrix_all = self.basis_set.get_basis_matrix()
        self.assertEqual(matrix_all.shape, (4, 2))
        # Order might vary, so check for columns individually if needed or by name
        # For now, assuming consistent order or that specific name list is used.

    def test_get_basis_matrix_mismatched_lengths(self):
        """Test get_basis_matrix with spectra of different lengths."""
        spec_short_data = np.array([1.0, 2.0])
        spec_short = BasisSpectrum("ShortSpec", spec_short_data)
        self.basis_set.add_metabolite(self.spec1) # length 4
        self.basis_set.add_metabolite(spec_short) # length 2
        
        with self.assertRaises(ValueError):
            self.basis_set.get_basis_matrix()

    def test_get_basis_matrix_empty_set(self):
        """Test get_basis_matrix on an empty set."""
        with self.assertRaises(ValueError):
            self.basis_set.get_basis_matrix()
            
    def test_get_basis_matrix_unknown_name(self):
        """Test get_basis_matrix with an unknown metabolite name."""
        self.basis_set.add_metabolite(self.spec1)
        with self.assertRaises(ValueError):
            self.basis_set.get_basis_matrix(metabolite_names=["Spec1", "UnknownSpec"])


if __name__ == '__main__':
    unittest.main()
