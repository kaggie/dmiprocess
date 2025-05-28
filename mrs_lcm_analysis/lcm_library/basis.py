import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class BasisSpectrum:
    """Represents a single metabolite's basis spectrum."""

    def __init__(self, name: str, spectrum_data: np.ndarray, frequency_axis: Optional[np.ndarray] = None):
        """
        Initializes a BasisSpectrum object.

        Args:
            name (str): The name of the metabolite (e.g., "NAA", "Cr").
            spectrum_data (np.ndarray): A 1D NumPy array of the frequency domain spectrum.
            frequency_axis (np.ndarray, optional): A 1D NumPy array representing the
                                                   frequency axis (e.g., in ppm). Defaults to None.
        """
        if not isinstance(name, str):
            raise TypeError("Metabolite name must be a string.")
        if not isinstance(spectrum_data, np.ndarray) or spectrum_data.ndim != 1:
            raise TypeError("Spectrum data must be a 1D NumPy array.")
        if frequency_axis is not None and (not isinstance(frequency_axis, np.ndarray) or frequency_axis.ndim != 1):
            raise TypeError("Frequency axis must be a 1D NumPy array if provided.")
        if frequency_axis is not None and len(spectrum_data) != len(frequency_axis):
            raise ValueError("Spectrum data and frequency axis must have the same length.")

        self.name = name
        self.spectrum_data = spectrum_data
        self.frequency_axis = frequency_axis

    def plot(self):
        """
        Plots the basis spectrum.

        If frequency_axis is available, it's used for the x-axis.
        Otherwise, a simple index plot is generated.
        """
        plt.figure()
        if self.frequency_axis is not None:
            plt.plot(self.frequency_axis, self.spectrum_data)
            plt.xlabel("Frequency (ppm)")
        else:
            plt.plot(self.spectrum_data)
            plt.xlabel("Index")
        plt.ylabel("Intensity")
        plt.title(f"Basis Spectrum: {self.name}")
        plt.show()

class BasisSet:
    """Manages a collection of BasisSpectrum objects."""

    def __init__(self):
        """Initializes an empty BasisSet."""
        self.metabolites = {}

    def add_metabolite(self, basis_spectrum: BasisSpectrum):
        """
        Adds a BasisSpectrum object to the set.

        Args:
            basis_spectrum (BasisSpectrum): The basis spectrum object to add.
        """
        if not isinstance(basis_spectrum, BasisSpectrum):
            raise TypeError("Input must be a BasisSpectrum object.")
        if basis_spectrum.name in self.metabolites:
            print(f"Warning: Metabolite '{basis_spectrum.name}' already exists and will be overwritten.")
        self.metabolites[basis_spectrum.name] = basis_spectrum

    def get_metabolite(self, name: str) -> BasisSpectrum:
        """
        Retrieves a BasisSpectrum object by its name.

        Args:
            name (str): The name of the metabolite to retrieve.

        Returns:
            BasisSpectrum: The corresponding basis spectrum object.

        Raises:
            KeyError: If the metabolite name is not found.
        """
        if name not in self.metabolites:
            raise KeyError(f"Metabolite '{name}' not found in the basis set.")
        return self.metabolites[name]

    def get_metabolite_names(self) -> list:
        """
        Returns a list of names of all metabolites in the basis set.

        Returns:
            list: A list of metabolite names.
        """
        return list(self.metabolites.keys())

    def get_basis_matrix(self, metabolite_names: list = None) -> np.ndarray:
        """
        Constructs a basis matrix from specified or all metabolite spectra.

        Each column in the matrix corresponds to a metabolite's spectrum.
        All spectra included in the matrix must have the same length.

        Args:
            metabolite_names (list, optional): A list of metabolite names to include
                                               in the matrix. If None, all metabolites
                                               in the basis set are used. Defaults to None.

        Returns:
            np.ndarray: A 2D NumPy array where each column is a spectrum.

        Raises:
            ValueError: If no metabolites are available to form the matrix.
            ValueError: If specified metabolite names are not in the basis set.
            ValueError: If spectra have inconsistent lengths.
        """
        if not self.metabolites:
            raise ValueError("No metabolites in the basis set to form a matrix.")

        names_to_use = metabolite_names
        if names_to_use is None:
            names_to_use = self.get_metabolite_names()

        if not names_to_use:
            raise ValueError("No metabolite names specified or available to form the matrix.")

        # Check if all specified names are valid
        for name in names_to_use:
            if name not in self.metabolites:
                raise ValueError(f"Metabolite '{name}' not found in basis set.")

        spectra_list = [self.metabolites[name].spectrum_data for name in names_to_use]

        # Check for consistent lengths
        first_spectrum_len = len(spectra_list[0])
        for i, spectrum in enumerate(spectra_list):
            if len(spectrum) != first_spectrum_len:
                raise ValueError(
                    f"Spectra have inconsistent lengths. Metabolite '{names_to_use[i]}' has length "
                    f"{len(spectrum)}, expected {first_spectrum_len} (based on '{names_to_use[0]}')."
                )

        # Stack spectra as columns
        basis_matrix = np.column_stack(spectra_list)
        return basis_matrix
