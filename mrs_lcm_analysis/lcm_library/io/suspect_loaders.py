import os
import numpy as np
import warnings
from typing import TYPE_CHECKING, Optional, Dict, Union, List, Any

from .base_loader import BaseMRSLoader
if TYPE_CHECKING:
    from mrs_lcm_analysis.lcm_library.data_loading import MRSData
from mrs_lcm_analysis.lcm_library.data_loading import MRSData

try:
    import suspect.io
    import suspect # For MRSData type checking
    SUSPECT_AVAILABLE = True
except ImportError:
    SUSPECT_AVAILABLE = False

class SuspectSiemensTwixLoader(BaseMRSLoader):
    """
    Loads Siemens TWIX (.dat) MRS data using the 'suspect' library.
    """

    def can_load(self, filepath: str) -> bool:
        """
        Checks if this loader can likely load the given file based on its extension.
        Args:
            filepath (str): The path to the MRS data file.
        Returns:
            bool: True if filepath ends with '.dat' (case-insensitive), False otherwise.
        """
        _, ext = os.path.splitext(filepath)
        return ext.lower() == '.dat'

    def load(self, filepath: str) -> 'MRSData':
        """
        Loads MRS data from the given Siemens TWIX (.dat) file using 'suspect'.

        Args:
            filepath (str): The path to the TWIX data file.

        Returns:
            MRSData: An MRSData object.

        Raises:
            ImportError: If the 'suspect' library is not installed.
            FileNotFoundError: If the filepath does not exist.
            IOError: If there is an issue reading the file (e.g., suspect fails).
            ValueError: If FID data or critical metadata cannot be extracted or FID is not 1D after processing.
        """
        if not SUSPECT_AVAILABLE:
            raise ImportError("The 'suspect' library is required for SuspectSiemensTwixLoader but is not installed. "
                              "Please install it, e.g., 'pip install suspect'.")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"TWIX file not found: {filepath}")

        try:
            suspect_data_raw = suspect.io.load_twix(filepath)
            if isinstance(suspect_data_raw, list):
                if not suspect_data_raw:
                    raise ValueError("suspect.io.load_twix returned an empty list.")
                if len(suspect_data_raw) > 1:
                    warnings.warn(f"suspect.io.load_twix returned {len(suspect_data_raw)} datasets. Using the first one by default.", UserWarning)
                suspect_mrs_data = suspect_data_raw[0]
            elif isinstance(suspect_mrs_data, suspect.MRSData): # Corrected type check
                suspect_mrs_data = suspect_data_raw
            else:
                raise ValueError(f"suspect.io.load_twix returned an unexpected type: {type(suspect_data_raw)}")
                
        except Exception as e:
            raise IOError(f"Failed to load TWIX file '{filepath}' using suspect.io.load_twix. Error: {e}")

        fid_array = np.asarray(suspect_mrs_data) 

        original_dims = list(suspect_mrs_data.dims) 
        original_shape = fid_array.shape

        if fid_array.ndim > 1:
            warnings.warn(f"Loaded TWIX data is multi-dimensional (shape: {original_shape}, dims: {original_dims}). "
                          "Attempting to extract a single FID.", UserWarning)
            
            processed_dims_count = 0
            if 't' not in original_dims:
                 raise ValueError("Time dimension 't' not found in suspect data dimensions.")
            spectral_axis_index_orig = original_dims.index('t')

            for dim_name in ['average', 'coil', 'channel', 'repetition', 'set', 'slice', 'x', 'y', 'z']: 
                if dim_name in original_dims and fid_array.ndim > 1:
                    current_axis_index = original_dims.index(dim_name) - processed_dims_count
                    if fid_array.shape[current_axis_index] > 1:
                        if dim_name == 'average':
                            fid_array = np.mean(fid_array, axis=current_axis_index)
                            warnings.warn(f"Averaged across '{dim_name}' dimension.", UserWarning)
                        else: 
                            fid_array = fid_array.take(0, axis=current_axis_index)
                            warnings.warn(f"Selected first element from '{dim_name}' dimension.", UserWarning)
                        processed_dims_count += 1
            
            if fid_array.ndim > 1:
                current_time_dim_idx = spectral_axis_index_orig - processed_dims_count 
                slicer = [0] * fid_array.ndim
                slicer[current_time_dim_idx] = slice(None)
                fid_array = fid_array[tuple(slicer)]
                warnings.warn(f"Reduced remaining dimensions to first element to make FID 1D. Final shape: {fid_array.shape}", UserWarning)

        if fid_array.ndim != 1:
             raise ValueError(f"Could not reduce FID to 1D. Final shape: {fid_array.shape}")
        if not np.iscomplexobj(fid_array):
            warnings.warn(f"FID data extracted from {filepath} is not complex. This is unusual. Casting to complex.", UserWarning)
            fid_array = fid_array.astype(np.complex64)

        sw_hz = float(suspect_mrs_data.sw)
        f0_mhz = float(suspect_mrs_data.f0)
        te_ms = float(suspect_mrs_data.te)
        tr_ms_val = getattr(suspect_mrs_data, 'tr', None) # TR might not always be present or None
        if tr_ms_val is None: # Check if TR is None from suspect object's attribute
            tr_val_from_metadata = suspect_mrs_data.metadata.get("TR") # Check header
            if tr_val_from_metadata is not None:
                try:
                    tr_ms_val = float(tr_val_from_metadata)
                except (TypeError, ValueError):
                     warnings.warn(f"Could not parse TR '{tr_val_from_metadata}' from suspect metadata. Setting TR to None.", UserWarning)
                     tr_ms_val = None # Explicitly None
            else:
                warnings.warn("'tr' (Repetition Time) not found or is None in suspect MRSData object or its metadata. Set to None.", UserWarning)
                tr_ms_val = None
        tr_ms = float(tr_ms_val) if tr_ms_val is not None else None
        
        misc_metadata: Dict[str, Union[str, float, int, List[Any], Dict[Any, Any]]] = {'source_format': 'twix'}
        if hasattr(suspect_mrs_data, 'metadata') and isinstance(suspect_mrs_data.metadata, dict):
            for key, value in suspect_mrs_data.metadata.items():
                if key.upper() not in ['TR', 'TE', 'SW', 'F0', 'SW_HZ', 'F0_MHZ', 'TE_MS', 'TR_MS']: # Avoid duplicating core/already handled
                    if isinstance(value, (str, int, float, bool)):
                        misc_metadata[key] = value
                    elif isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in value) :
                        misc_metadata[key] = str(value) 
                    elif isinstance(value, np.ndarray) and value.size == 1:
                        misc_metadata[key] = value.item()
        
        misc_metadata['suspect_original_dims'] = str(suspect_mrs_data.dims)
        misc_metadata['suspect_original_shape'] = str(original_shape)

        return MRSData(data_array=fid_array.astype(np.complex64),
                       data_type="time",
                       sampling_frequency=sw_hz,
                       central_frequency=f0_mhz,
                       echo_time_ms=te_ms,
                       repetition_time_ms=tr_ms,
                       metadata=misc_metadata)


class SuspectPhilipsSdatLoader(BaseMRSLoader):
    """
    Loads Philips SDAT (.sdat) MRS data using the 'suspect' library.
    Assumes a corresponding .SPAR file exists in the same directory.
    """

    def can_load(self, filepath: str) -> bool:
        """
        Checks if this loader can likely load the given file based on its extension
        and the presence of a corresponding .spar file.
        Args:
            filepath (str): The path to the MRS data file (.sdat).
        Returns:
            bool: True if filepath ends with '.sdat' (case-insensitive) and a .spar file exists, False otherwise.
        """
        base, ext = os.path.splitext(filepath)
        if ext.lower() != '.sdat':
            return False
        
        spar_filepath_lower = base + '.spar'
        spar_filepath_upper = base + '.SPAR'
        return os.path.exists(spar_filepath_lower) or os.path.exists(spar_filepath_upper)


    def load(self, filepath: str) -> 'MRSData':
        """
        Loads MRS data from the given Philips SDAT (.sdat) file using 'suspect'.
        The corresponding .spar file must be in the same directory.

        Args:
            filepath (str): The path to the SDAT data file.

        Returns:
            MRSData: An MRSData object.

        Raises:
            ImportError: If the 'suspect' library is not installed.
            FileNotFoundError: If the filepath or its corresponding .spar file does not exist.
            IOError: If there is an issue reading the file (e.g., suspect fails).
            ValueError: If FID data or critical metadata cannot be extracted or FID is not 1D.
        """
        if not SUSPECT_AVAILABLE:
            raise ImportError("The 'suspect' library is required for SuspectPhilipsSdatLoader but is not installed. "
                              "Please install it, e.g., 'pip install suspect'.")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"SDAT file not found: {filepath}")
        
        base, _ = os.path.splitext(filepath)
        spar_filepath_lower = base + '.spar'
        spar_filepath_upper = base + '.SPAR'
        if not (os.path.exists(spar_filepath_lower) or os.path.exists(spar_filepath_upper)):
            raise FileNotFoundError(f".sdat file found ({filepath}), but corresponding .spar/.SPAR file is missing.")

        try:
            suspect_mrs_data = suspect.io.load_sdat(filepath)
        except Exception as e:
            raise IOError(f"Failed to load SDAT file '{filepath}' using suspect.io.load_sdat. Error: {e}")

        fid_array = np.asarray(suspect_mrs_data)
        original_dims = list(suspect_mrs_data.dims)
        original_shape = fid_array.shape

        if fid_array.ndim > 1:
            warnings.warn(f"Loaded SDAT data is multi-dimensional (shape: {original_shape}, dims: {original_dims}). "
                          "Attempting to extract a single FID.", UserWarning)
            
            processed_dims_count = 0 
            if 't' not in original_dims:
                 raise ValueError("Time dimension 't' not found in suspect data dimensions.")
            spectral_axis_index_orig = original_dims.index('t')

            for dim_name in ['average', 'coil', 'channel', 'repetition', 'set', 'slice', 'x', 'y', 'z']: 
                if dim_name in original_dims and fid_array.ndim > 1:
                    current_axis_index = original_dims.index(dim_name) - processed_dims_count
                    if fid_array.shape[current_axis_index] > 1:
                        if dim_name == 'average':
                            fid_array = np.mean(fid_array, axis=current_axis_index)
                            warnings.warn(f"Averaged across '{dim_name}' dimension.", UserWarning)
                        else: 
                            fid_array = fid_array.take(0, axis=current_axis_index)
                            warnings.warn(f"Selected first element from '{dim_name}' dimension.", UserWarning)
                        processed_dims_count += 1
            
            if fid_array.ndim > 1:
                current_time_dim_idx = spectral_axis_index_orig - processed_dims_count
                slicer = [0] * fid_array.ndim
                slicer[current_time_dim_idx] = slice(None)
                fid_array = fid_array[tuple(slicer)]
                warnings.warn(f"Reduced remaining dimensions to first element to make FID 1D. Final shape: {fid_array.shape}", UserWarning)


        if fid_array.ndim != 1:
             raise ValueError(f"Could not reduce FID to 1D. Final shape: {fid_array.shape}")
        if not np.iscomplexobj(fid_array):
            warnings.warn(f"FID data extracted from {filepath} is not complex. This is unusual. Casting to complex.", UserWarning)
            fid_array = fid_array.astype(np.complex64)

        sw_hz = float(suspect_mrs_data.sw)
        f0_mhz = float(suspect_mrs_data.f0)
        te_ms = float(suspect_mrs_data.te)
        tr_ms_val = getattr(suspect_mrs_data, 'tr', None)
        if tr_ms_val is None:
            tr_val_from_metadata = suspect_mrs_data.metadata.get("TR")
            if tr_val_from_metadata is not None:
                try:
                    tr_ms_val = float(tr_val_from_metadata)
                except (TypeError, ValueError):
                     warnings.warn(f"Could not parse TR '{tr_val_from_metadata}' from suspect metadata. Setting TR to None.", UserWarning)
                     tr_ms_val = None
            else:
                warnings.warn("'tr' (Repetition Time) not found or is None in suspect MRSData object or its metadata. Set to None.", UserWarning)
                tr_ms_val = None
        tr_ms = float(tr_ms_val) if tr_ms_val is not None else None
        
        misc_metadata: Dict[str, Union[str, float, int, List[Any], Dict[Any, Any]]] = {'source_format': 'sdat/spar'}
        if hasattr(suspect_mrs_data, 'metadata') and isinstance(suspect_mrs_data.metadata, dict):
            for key, value in suspect_mrs_data.metadata.items():
                if key.upper() not in ['TR', 'TE', 'SW', 'F0', 'SW_HZ', 'F0_MHZ', 'TE_MS', 'TR_MS']: 
                    if isinstance(value, (str, int, float, bool)):
                        misc_metadata[key] = value
                    elif isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in value) :
                        misc_metadata[key] = str(value)
                    elif isinstance(value, np.ndarray) and value.size == 1:
                        misc_metadata[key] = value.item()

        misc_metadata['suspect_original_dims'] = str(suspect_mrs_data.dims)
        misc_metadata['suspect_original_shape'] = str(original_shape)

        return MRSData(data_array=fid_array.astype(np.complex64),
                       data_type="time", 
                       sampling_frequency=sw_hz,
                       central_frequency=f0_mhz,
                       echo_time_ms=te_ms,
                       repetition_time_ms=tr_ms,
                       metadata=misc_metadata)

```
