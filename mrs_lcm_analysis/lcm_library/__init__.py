"""
MRS LCM Analysis Library
========================

This library provides tools for Linear Combination Modeling (LCM) of Magnetic
Resonance Spectroscopy (MRS) data.

It includes modules for:
- Data loading and representation (`MRSData`)
- Basis set management (`BasisSpectrum`, `BasisSet`)
- Linear Combination Model fitting (`LinearCombinationModel`)
- Baseline modeling (`create_polynomial_basis_vectors`, `generate_polynomial_baseline`)

This file marks mrs_lcm_analysis/lcm_library as a Python package and exports
key classes and functions for easier access.
"""
from .basis import BasisSpectrum, BasisSet
from .data_loading import MRSData
from .model import LinearCombinationModel
from .baseline import create_polynomial_basis_vectors, generate_polynomial_baseline
