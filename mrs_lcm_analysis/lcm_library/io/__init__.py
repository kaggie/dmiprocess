"""
MRS Data I/O Submodule
======================

This submodule provides classes and functions for loading MRS data from various
file formats.

Key Components:
---------------
- `BaseMRSLoader`: Abstract base class for all data loaders.
- `NumpyLoader`: Loader for data stored in NumPy's .npy or .npz formats.
- `MatlabLoader`: Loader for data stored in MATLAB .mat format.
- `DicomLoader`: Loader for MRS data stored in DICOM format.
- `SuspectSiemensTwixLoader`: Loader for Siemens TWIX (.dat) files using the 'suspect' library.
- `SuspectPhilipsSdatLoader`: Loader for Philips SDAT (.sdat) files using the 'suspect' library.
- `load_mrs`: A dispatcher function to automatically select the
  appropriate loader for a given file.

Usage:
------
Specific loaders can be imported directly:
```python
from mrs_lcm_analysis.lcm_library.io import SuspectSiemensTwixLoader
# loader = SuspectSiemensTwixLoader()
# mrs_data = loader.load("path/to/my_data.dat") # Requires a .dat file and suspect
```

Or using the main `load_mrs` function:
```python
from mrs_lcm_analysis.lcm_library.io import load_mrs
# mrs_data = load_mrs("path/to/my_data.dat") # Requires a .dat file and suspect
```
"""
import os
from typing import TYPE_CHECKING

from .base_loader import BaseMRSLoader
from .numpy_loader import NumpyLoader
from .matlab_loader import MatlabLoader
from .dicom_loader import DicomLoader
from .suspect_loaders import SuspectSiemensTwixLoader, SuspectPhilipsSdatLoader, SUSPECT_AVAILABLE

if TYPE_CHECKING:
    from mrs_lcm_analysis.lcm_library.data_loading import MRSData


# A simple registry for available loaders.
LOADER_REGISTRY = {
    "numpy": NumpyLoader,
    "npy": NumpyLoader,
    "npz": NumpyLoader,
    "matlab": MatlabLoader,
    "mat": MatlabLoader,
    "dicom": DicomLoader,
    "dcm": DicomLoader,
}

if SUSPECT_AVAILABLE:
    LOADER_REGISTRY["twix"] = SuspectSiemensTwixLoader
    LOADER_REGISTRY["dat"] = SuspectSiemensTwixLoader  # Siemens TWIX
    LOADER_REGISTRY["sdat"] = SuspectPhilipsSdatLoader # Philips SDAT
    # .spar files usually accompany .sdat files; loading is typically initiated with the .sdat file.
    # No separate key for 'spar' as a primary format key, but .sdat loader handles it.
else:
    # Optionally, log that suspect loaders are not available if SUSPECT_AVAILABLE is False
    # import warnings
    # warnings.warn("Suspect library not found, Suspect-based loaders (TWIX, SDAT) will not be registered.", ImportWarning)
    pass


__all__ = [
    "BaseMRSLoader",
    "NumpyLoader",
    "MatlabLoader",
    "DicomLoader",
    "load_mrs",
]

if SUSPECT_AVAILABLE:
    __all__.extend([
        "SuspectSiemensTwixLoader",
        "SuspectPhilipsSdatLoader",
    ])


def load_mrs(filepath: str, format_key: str = None) -> 'MRSData':
    """
    Loads MRS data from the given filepath, attempting to infer format
    or using the specified format.

    Args:
        filepath (str): Path to the MRS data file.
        format_key (str, optional): A key to identify the loader type
                                    (e.g., 'npy', 'mat', 'dicom', 'twix', 'sdat').
                                    If None, format is inferred from extension.

    Returns:
        MRSData: An MRSData object.

    Raises:
        ValueError: If format is not specified and cannot be inferred,
                    or if specified format is not supported.
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
        ImportError: If a required library (e.g., suspect) for the format is not installed.
    """
    from mrs_lcm_analysis.lcm_library.data_loading import MRSData 

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    loader_instance = None
    
    if format_key:
        loader_class = LOADER_REGISTRY.get(format_key.lower())
        if loader_class:
            loader_instance = loader_class()
        else:
            raise ValueError(f"Unsupported format key specified: '{format_key}'. Supported keys: {list(LOADER_REGISTRY.keys())}")
    else:
        _, ext = os.path.splitext(filepath)
        ext_lower = ext.lower().strip('.')
        
        loader_class = LOADER_REGISTRY.get(ext_lower)
        if loader_class:
            loader_instance = loader_class()
        else:
            # Fallback: try all known loaders by calling their can_load method
            for LClass in LOADER_REGISTRY.values(): # Iterate through unique loader classes
                # Avoid redundant checks if multiple extensions map to the same loader
                # if loader_instance and isinstance(loader_instance, LClass): continue
                # This logic needs to be careful not to instantiate unnecessarily or repeatedly
                
                # A simple way for now, assuming can_load is efficient:
                temp_loader = LClass()
                if temp_loader.can_load(filepath):
                    loader_instance = temp_loader
                    break 
            
            if loader_instance is None:
                raise ValueError(f"Could not automatically determine loader for file: {filepath}. "
                                 f"Extension '{ext_lower}' is not explicitly registered. "
                                 f"Please specify format_key or ensure a loader's can_load() method returns True.")

    if loader_instance:
        # Check for SUSPECT_AVAILABLE if the loader is one of the suspect loaders
        # This check is now more specific to the chosen loader_instance
        if isinstance(loader_instance, (SuspectSiemensTwixLoader, SuspectPhilipsSdatLoader)) and not SUSPECT_AVAILABLE:
             raise ImportError(f"The 'suspect' library is required to load '{filepath}' with {type(loader_instance).__name__} but is not installed. "
                               "Please install it, e.g., 'pip install suspect'.")
        return loader_instance.load(filepath)
    else: 
        # This path should ideally not be reached if the logic above is sound.
        raise ValueError(f"No suitable loader found or instantiated for file: {filepath}")

```
