"""
MRS LCM Analysis Library
========================

This library provides tools for Linear Combination Modeling (LCM) of Magnetic
Resonance Spectroscopy (MRS) data.

It includes modules for:
- Data loading and representation (`MRSData`, `load_mrs` from `io` submodule)
- Basis set management (`BasisSpectrum`, `BasisSet`)
- Linear Combination Model fitting (`LinearCombinationModel`)
- Advanced PyTorch-based Linear Combination Model fitting (`AdvancedLinearCombinationModel`)
- Baseline modeling (`create_polynomial_basis_vectors`, `generate_polynomial_baseline`)
- Quantification (`AbsoluteQuantifier`)
- I/O operations for various file formats (see `lcm_library.io`)

This file marks mrs_lcm_analysis/lcm_library as a Python package and exports
key classes and functions for easier access.
"""
from .basis import BasisSpectrum, BasisSet
from .data_loading import MRSData
from .model import LinearCombinationModel
from .advanced_model import AdvancedLinearCombinationModel
from .baseline import create_polynomial_basis_vectors, generate_polynomial_baseline
from .quantification import AbsoluteQuantifier # New import
from . import io # Import the io submodule

# Re-export the main loader function from the io submodule
from .io import load_mrs

__all__ = [
    "BasisSpectrum",
    "BasisSet",
    "MRSData",
    "LinearCombinationModel",
    "AdvancedLinearCombinationModel", 
    "create_polynomial_basis_vectors",
    "generate_polynomial_baseline",
    "AbsoluteQuantifier", # New export
    "load_mrs", 
    "io" 
]
