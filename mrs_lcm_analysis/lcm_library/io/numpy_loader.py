import os
import numpy as np
import warnings
from typing import TYPE_CHECKING, Optional, Dict, Union, Any, List

from .base_loader import BaseMRSLoader
if TYPE_CHECKING:
    from mrs_lcm_analysis.lcm_library.data_loading import MRSData 
from mrs_lcm_analysis.lcm_library.data_loading import MRSData 

class NumpyLoader(BaseMRSLoader):
    """
    Loads MRS data from NumPy's .npy and .npz file formats.
    """

    def load(self, filepath: str) -> 'MRSData':
        """
        Loads MRS data from the given .npy or .npz file.

        For .npy files:
            - Assumes the array is the FID (complex data).
            - Metadata (sampling_frequency, central_frequency, echo_time_ms, repetition_time_ms) 
              will be None and must be set manually later. A warning is issued.

        For .npz files:
            - Expects predefined keys for data and metadata:
                - 'fid' (required): complex numpy array.
                - 'sw_hz' (required): float, spectral width in Hz (sampling frequency).
                - 'f0_mhz' (required): float, spectrometer frequency in MHz (central frequency).
                - 'te_ms' (optional): float, echo time in ms.
                - 'tr_ms' (optional): float, repetition time in ms.
            - Other keys in the .npz file will be stored in the MRSData.metadata dictionary.

        Args:
            filepath (str): The path to the MRS data file.

        Returns:
            MRSData: An MRSData object.

        Raises:
            FileNotFoundError: If the filepath does not exist.
            ValueError: If the file is not a .npy or .npz file, or if
                        a .npz file is missing required keys or has
                        incorrect data types.
            IOError: If there is an issue reading the file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        _, ext = os.path.splitext(filepath)

        if ext.lower() == '.npy':
            try:
                data_array = np.load(filepath)
            except Exception as e:
                raise IOError(f"Could not load .npy file: {e}")
            
            if not np.iscomplexobj(data_array):
                warnings.warn(f"Data in {filepath} is not complex. Assuming it's an FID, but this is unusual.", UserWarning)
            if data_array.ndim != 1:
                warnings.warn(f"Data in {filepath} is not 1D (shape: {data_array.shape}). Attempting to ravel.", UserWarning)
                data_array = data_array.ravel()
                if data_array.ndim != 1:
                    raise ValueError(f"Could not convert data in {filepath} to 1D array.")

            warnings.warn("Loaded from .npy file. Essential metadata (sampling_frequency, central_frequency, "
                          "echo_time_ms, repetition_time_ms) is missing and will be set to None. "
                          "Please populate these manually.", UserWarning)
            
            return MRSData(data_array=data_array, 
                           data_type="time", 
                           sampling_frequency=None, 
                           central_frequency=None,
                           echo_time_ms=None,
                           repetition_time_ms=None,
                           metadata={'source_format': 'npy'})

        elif ext.lower() == '.npz':
            try:
                npz_file = np.load(filepath)
            except Exception as e:
                raise IOError(f"Could not load .npz file: {e}")

            required_keys = ['fid', 'sw_hz', 'f0_mhz']
            for key in required_keys:
                if key not in npz_file:
                    raise ValueError(f"Required key '{key}' not found in .npz file: {filepath}")

            fid = npz_file['fid']
            if not isinstance(fid, np.ndarray) or not np.iscomplexobj(fid) or fid.ndim != 1:
                # Attempt to ravel if it's a vector-like 2D array
                if isinstance(fid, np.ndarray) and np.iscomplexobj(fid) and fid.ndim == 2 and (fid.shape[0] == 1 or fid.shape[1] == 1):
                    fid = fid.ravel()
                else:
                    raise ValueError(f"Key 'fid' in {filepath} must be a 1D complex NumPy array (or a 2D complex vector). Got shape {fid.shape if isinstance(fid, np.ndarray) else type(fid)}.")


            def _get_float_from_npz(key: str, file_obj: np.lib.npyio.NpzFile) -> Optional[float]:
                if key in file_obj:
                    try:
                        return float(file_obj[key].item() if isinstance(file_obj[key], np.ndarray) else file_obj[key])
                    except (ValueError, TypeError):
                        warnings.warn(f"Could not convert '{key}' to float in {filepath}. Skipping.", UserWarning)
                return None

            sw_hz = _get_float_from_npz('sw_hz', npz_file)
            if sw_hz is None: raise ValueError(f"Key 'sw_hz' in {filepath} must be a float.")
            
            f0_mhz = _get_float_from_npz('f0_mhz', npz_file)
            if f0_mhz is None: raise ValueError(f"Key 'f0_mhz' in {filepath} must be a float.")

            te_ms = _get_float_from_npz('te_ms', npz_file)
            tr_ms = _get_float_from_npz('tr_ms', npz_file)
            
            metadata: Dict[str, Union[str, float, int, List[Any], Dict[Any, Any]]] = {'source_format': 'npz'}
            
            # Any other keys in the npz file can be added to metadata too
            for key in npz_file.keys():
                if key not in ['fid', 'sw_hz', 'f0_mhz', 'te_ms', 'tr_ms']:
                    item = npz_file[key]
                    if isinstance(item, np.ndarray) and item.size == 1:
                        try:
                            metadata[key] = item.item() # Store scalar as its Python type
                        except TypeError: # If item.item() fails for some object arrays
                            metadata[key] = str(item) 
                    elif isinstance(item, (str, int, float, bool, list, tuple)):
                         metadata[key] = item
                    else: # For other numpy arrays or complex objects, store as string representation
                        metadata[key] = str(item)


            return MRSData(data_array=fid,
                           data_type="time", 
                           sampling_frequency=sw_hz,
                           central_frequency=f0_mhz,
                           echo_time_ms=te_ms,
                           repetition_time_ms=tr_ms,
                           metadata=metadata)
        else:
            raise ValueError(f"Unsupported file extension '{ext}'. NumpyLoader supports '.npy' and '.npz'.")

    def can_load(self, filepath: str) -> bool:
        """
        Checks if this loader can likely load the given file based on its extension.

        Args:
            filepath (str): The path to the MRS data file.

        Returns:
            bool: True if filepath ends with '.npy' or '.npz', False otherwise.
        """
        _, ext = os.path.splitext(filepath)
        return ext.lower() in ['.npy', '.npz']

```
