import os
import numpy as np
import scipy.io
import warnings
from typing import TYPE_CHECKING, Optional, Dict, Union, Any, List

from .base_loader import BaseMRSLoader
if TYPE_CHECKING:
    from mrs_lcm_analysis.lcm_library.data_loading import MRSData 
from mrs_lcm_analysis.lcm_library.data_loading import MRSData 

class MatlabLoader(BaseMRSLoader):
    """
    Loads MRS data from MATLAB's .mat file format.
    """

    def load(self, filepath: str) -> 'MRSData':
        """
        Loads MRS data from the given .mat file.

        Expects predefined variable names (keys in the loaded dictionary):
            - 'fid' (required): complex numpy array (1D or 2D, will be reshaped to 1D).
            - 'sw_hz' (required): float or scalar numpy array, spectral width in Hz (sampling frequency).
            - 'f0_mhz' (required): float or scalar numpy array, spectrometer frequency in MHz (central frequency).
            - 'te_ms' (optional): float or scalar numpy array, echo time in ms.
            - 'tr_ms' (optional): float or scalar numpy array, repetition time in ms.
        Other scalar variables or simple strings from the .mat file will be stored in metadata.

        Args:
            filepath (str): The path to the MRS data file.

        Returns:
            MRSData: An MRSData object.

        Raises:
            FileNotFoundError: If the filepath does not exist.
            ValueError: If the file is not a .mat file, or if
                        required keys are missing or have incorrect data types/shapes.
            IOError: If there is an issue reading the file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        _, ext = os.path.splitext(filepath)
        if ext.lower() != '.mat':
            raise ValueError(f"Unsupported file extension '{ext}'. MatlabLoader supports '.mat'.")

        try:
            mat_contents = scipy.io.loadmat(filepath)
        except Exception as e:
            raise IOError(f"Could not load .mat file: {e}")

        required_keys = ['fid', 'sw_hz', 'f0_mhz']
        for key in required_keys:
            if key not in mat_contents:
                raise ValueError(f"Required key '{key}' not found in .mat file: {filepath}")

        fid = mat_contents['fid']
        if not isinstance(fid, np.ndarray) or not np.iscomplexobj(fid):
            raise ValueError(f"Key 'fid' in {filepath} must be a complex NumPy array.")
        
        if fid.ndim == 2:
            if fid.shape[0] == 1 or fid.shape[1] == 1:
                fid = fid.ravel()
            else:
                raise ValueError(f"Key 'fid' in {filepath} is a 2D array that is not a vector (shape: {fid.shape}).")
        elif fid.ndim != 1:
            raise ValueError(f"Key 'fid' in {filepath} must be a 1D or 2D (vector-like) NumPy array. Got ndim: {fid.ndim}.")


        def _extract_scalar_float(key_name: str, contents: dict) -> Optional[float]:
            if key_name not in contents:
                return None
            val = contents[key_name]
            try:
                if isinstance(val, np.ndarray) and val.size == 1: 
                    return float(val.item())
                elif isinstance(val, (float, int)):
                    return float(val)
                else: # Try to convert if it's a string representing a number
                    return float(str(val)) 
            except (ValueError, TypeError) as e:
                warnings.warn(f"Could not convert key '{key_name}' to float in {filepath}. Value: {val}. Error: {e}. Skipping.", UserWarning)
                return None

        sw_hz = _extract_scalar_float('sw_hz', mat_contents)
        if sw_hz is None: 
             raise ValueError(f"Required key 'sw_hz' missing or invalid in {filepath}")
             
        f0_mhz = _extract_scalar_float('f0_mhz', mat_contents)
        if f0_mhz is None: 
            raise ValueError(f"Required key 'f0_mhz' missing or invalid in {filepath}")

        te_ms = _extract_scalar_float('te_ms', mat_contents)
        tr_ms = _extract_scalar_float('tr_ms', mat_contents)
        
        metadata: Dict[str, Union[str, float, int, List[Any], Dict[Any, Any]]] = {'source_format': 'mat'}
        
        # Add any other scalar or simple string variables from the .mat file to metadata
        # Exclude private MATLAB keys starting with '__'
        excluded_keys = ['fid', 'sw_hz', 'f0_mhz', 'te_ms', 'tr_ms', '__header__', '__version__', '__globals__']
        for key in mat_contents.keys():
            if key.startswith('__') or key in excluded_keys:
                continue
            
            item = mat_contents[key]
            if isinstance(item, np.ndarray) and item.size == 1:
                item_val = item.item()
                if isinstance(item_val, (str, int, float, bool)):
                    metadata[key] = item_val
            elif isinstance(item, (str, int, float, bool)): # Handles non-array scalars
                 metadata[key] = item
            elif isinstance(item, np.ndarray) and item.ndim == 1 and item.dtype.kind in 'SU': # String array
                metadata[key] = list(item) if item.size > 1 else item.item() if item.size ==1 else ""
            # Other complex types could be stringified or handled more specifically if needed.

        return MRSData(data_array=fid.astype(np.complex64),
                       data_type="time", 
                       sampling_frequency=sw_hz,
                       central_frequency=f0_mhz,
                       echo_time_ms=te_ms,
                       repetition_time_ms=tr_ms,
                       metadata=metadata)

    def can_load(self, filepath: str) -> bool:
        """
        Checks if this loader can likely load the given file based on its extension.

        Args:
            filepath (str): The path to the MRS data file.

        Returns:
            bool: True if filepath ends with '.mat', False otherwise.
        """
        _, ext = os.path.splitext(filepath)
        return ext.lower() == '.mat'

```
