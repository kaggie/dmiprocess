import numpy as np
import logging
import os

# Attempt to import optional dependencies
try:
    import scipy.io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import pyspecdata
    PYSPECDATA_AVAILABLE = True
except ImportError:
    PYSPECDATA_AVAILABLE = False

# Default logger if none is provided
default_logger = logging.getLogger(__name__)
default_logger.addHandler(logging.NullHandler())


def load_spectra(file_path, file_format, vendor=None, logger=None):
    """
    Loads spectral data from various file formats.

    Args:
        file_path (str): Path to the data file.
        file_format (str): Format of the file (e.g., 'dicom', 'mat', 'hdf5', 'pyspecdata_varian').
        vendor (str, optional): Vendor information, useful for some formats like DICOM.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        dict: A dictionary containing 'data', 'axis', and 'metadata'.
              Returns None if loading fails.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        ValueError: If file_format is unsupported or required libraries are missing.
        IOError: For general loading issues.
    """
    logger = logger or default_logger

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Attempting to load spectra from '{file_path}' in '{file_format}' format.")

    spectra_object = {"data": None, "axis": None, "metadata": {}}

    if file_format == 'mat':
        if not SCIPY_AVAILABLE:
            logger.error("SciPy is not installed. Cannot load .mat files.")
            raise ValueError("SciPy is required for .mat file loading but not installed.")
        try:
            mat_contents = scipy.io.loadmat(file_path)
            if 'data' in mat_contents and 'axis' in mat_contents:
                spectra_object['data'] = np.asarray(mat_contents['data'])
                spectra_object['axis'] = np.asarray(mat_contents['axis']).squeeze() # Ensure axis is 1D
                if 'metadata' in mat_contents:
                    # Handle metadata appropriately (e.g., if it's a dict or struct)
                    # For simplicity, assuming it's loadable directly or needs specific parsing
                    # This part might need refinement based on actual .mat structure
                    raw_metadata = mat_contents['metadata']
                    if isinstance(raw_metadata, dict):
                        spectra_object['metadata'] = raw_metadata # Already a dict
                    elif hasattr(raw_metadata, 'dtype') and raw_metadata.dtype.names is not None: # MATLAB struct
                        # Squeeze out single-element arrays from struct fields
                        temp_metadata = {}
                        for name in raw_metadata.dtype.names:
                            value = raw_metadata[name][0,0]
                            if isinstance(value, np.ndarray) and value.size == 1:
                                temp_metadata[name] = value.item() # Extract scalar
                            elif isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == 1 and value.shape[1] == 1: # [[item]]
                                temp_metadata[name] = value[0,0]
                            else:
                                temp_metadata[name] = value
                        spectra_object['metadata'] = temp_metadata
                    elif isinstance(raw_metadata, np.ndarray) and raw_metadata.size == 1 and not raw_metadata.dtype.names: # Single item array
                        spectra_object['metadata'] = {'value': raw_metadata.item()}
                    else: # Other cases
                        spectra_object['metadata'] = {'raw_mat_metadata': raw_metadata}
                    logger.info(f"Loaded .mat file. Data shape: {spectra_object['data'].shape}, Axis length: {len(spectra_object['axis'])}")
                else:
                    logger.warning("'.mat' file loaded, but 'metadata' variable not found.")
                # Basic validation of shapes
                if spectra_object['data'].ndim > 1 and spectra_object['data'].shape[-1] != len(spectra_object['axis']):
                     if not (spectra_object['data'].ndim == 1 and len(spectra_object['data']) == len(spectra_object['axis'])): # allow 1D data too
                        logger.warning(f"Data shape {spectra_object['data'].shape} and axis length {len(spectra_object['axis'])} mismatch.")

            else:
                logger.error("'.mat' file loaded, but 'data' or 'axis' variables not found.")
                raise IOError(f"Required variables ('data', 'axis') not found in MAT file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading .mat file '{file_path}': {e}")
            raise IOError(f"Error loading .mat file '{file_path}': {e}")

    elif file_format in ['hdf5', 'h5']:
        if not H5PY_AVAILABLE:
            logger.error("h5py is not installed. Cannot load .hdf5/.h5 files.")
            raise ValueError("h5py is required for HDF5 file loading but not installed.")
        try:
            with h5py.File(file_path, 'r') as hf:
                if 'data' in hf and 'axis' in hf:
                    spectra_object['data'] = np.array(hf['data'][:])
                    spectra_object['axis'] = np.array(hf['axis'][:]).squeeze() # Ensure axis is 1D
                    if 'metadata_group' in hf:
                        metadata_group = hf['metadata_group']
                        for key in metadata_group.attrs:
                            spectra_object['metadata'][key] = metadata_group.attrs[key]
                        for key in metadata_group: # if metadata are stored as datasets in the group
                            spectra_object['metadata'][key] = metadata_group[key][:]
                        logger.info(f"Loaded HDF5 file. Data shape: {spectra_object['data'].shape}, Axis length: {len(spectra_object['axis'])}")
                    else:
                        logger.warning("HDF5 file loaded, but 'metadata_group' not found.")
                     # Basic validation of shapes
                    if spectra_object['data'].ndim > 1 and spectra_object['data'].shape[-1] != len(spectra_object['axis']):
                        if not (spectra_object['data'].ndim == 1 and len(spectra_object['data']) == len(spectra_object['axis'])): # allow 1D data too
                            logger.warning(f"Data shape {spectra_object['data'].shape} and axis length {len(spectra_object['axis'])} mismatch.")
                else:
                    logger.error("HDF5 file loaded, but 'data' or 'axis' datasets not found.")
                    raise IOError(f"Required datasets ('data', 'axis') not found in HDF5 file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading HDF5 file '{file_path}': {e}")
            raise IOError(f"Error loading HDF5 file '{file_path}': {e}")

    elif file_format == 'dicom':
        if not PYSPECDATA_AVAILABLE:
            logger.warning("pyspecdata is not installed. DICOM loading via pyspecdata is not available.")
            # Placeholder: Acknowledge fallback might be needed.
            # For now, we raise an error if pyspecdata is the chosen method but not available.
            raise ValueError("pyspecdata is required for DICOM loading but not installed.")
        try:
            # This is a simplified call; actual pyspecdata usage might be more complex
            # and depend on how pyspecdata handles different DICOM MRS formats.
            # Example: data = pyspecdata.read_dicom(file_path)
            # You'd then need to extract 'data', 'axis', 'metadata' from pyspecdata's object.
            # This is a placeholder and will likely need significant refinement.
            logger.info(f"Attempting to load DICOM with pyspecdata (vendor: {vendor or 'not specified'}).")
            spec_data_obj = pyspecdata.figdata(searchpath=file_path, filename=os.path.basename(file_path), type='dicom') # This is just an example

            # Assuming spec_data_obj is an nddata object from pyspecdata
            # This extraction is highly dependent on pyspecdata's object structure
            if hasattr(spec_data_obj, 'data') and hasattr(spec_data_obj, 'get_ft_prop'):
                spectra_object['data'] = spec_data_obj.data
                # Try to get frequency axis, might need to check if it's time or freq domain
                if 't2' in spec_data_obj.dimlabels: # Example dimlabel for spectral data
                     spectra_object['axis'] = spec_data_obj.getaxis('t2') # or appropriate axis
                else: # Fallback or find another way to get the axis
                    logger.warning("Could not determine primary spectral axis from pyspecdata object dimlabels.")
                    # Create a dummy axis if necessary, or raise error
                    spectra_object['axis'] = np.arange(spec_data_obj.data.shape[-1])


                # Metadata extraction (highly dependent on pyspecdata)
                # Example:
                # spectra_object['metadata']['transmitter_frequency'] = spec_data_obj.get_ft_prop('Bo')
                # spectra_object['metadata']['sweep_width'] = spec_data_obj.get_ft_prop('SW')
                # This is very speculative and needs to be verified with actual pyspecdata usage.

                # For now, storing the whole props might be a start
                if hasattr(spec_data_obj, 'ft_prop'):
                     spectra_object['metadata'].update(spec_data_obj.ft_prop)

                logger.info(f"DICOM file loaded via pyspecdata. Data shape: {spectra_object['data'].shape}")
            else:
                logger.error("Failed to extract data and axis from pyspecdata object for DICOM file.")
                raise IOError(f"Could not parse DICOM data using pyspecdata from {file_path}")

        except Exception as e:
            logger.error(f"Error loading DICOM file '{file_path}' with pyspecdata: {e}")
            logger.warning("DICOM loading with pyspecdata failed. A manual DICOM reader or alternative library might be needed if this issue persists.")
            raise IOError(f"Error loading DICOM file '{file_path}' with pyspecdata: {e}")

    # Example for pyspecdata_varian
    elif file_format == 'pyspecdata_varian':
        if not PYSPECDATA_AVAILABLE:
            logger.warning("pyspecdata is not installed. Varian loading via pyspecdata is not available.")
            raise ValueError("pyspecdata is required for Varian loading but not installed.")
        try:
            logger.info(f"Attempting to load Varian data with pyspecdata from directory: {file_path}")
            # Assuming file_path is the directory containing 'fid' and 'procpar'
            spec_data_obj = pyspecdata.varian.load_fid(file_path) # This is one way, or figdata with type='varian'

            spectra_object['data'] = spec_data_obj.data # Typically time-domain FID
            spectra_object['axis'] = spec_data_obj.getaxis('t2') # Time axis

            # Metadata from procpar (already parsed by pyspecdata)
            if hasattr(spec_data_obj, 'procpar'):
                spectra_object['metadata'].update(spec_data_obj.procpar) # procpar is usually a dict-like object

            logger.info(f"Varian data loaded via pyspecdata. Data shape: {spectra_object['data'].shape}")

        except Exception as e:
            logger.error(f"Error loading Varian data '{file_path}' with pyspecdata: {e}")
            raise IOError(f"Error loading Varian data '{file_path}' with pyspecdata: {e}")

    else:
        logger.error(f"Unsupported file format: '{file_format}'")
        raise ValueError(f"Unsupported file format: '{file_format}'")

    # Final check for data and axis
    if spectra_object['data'] is None or spectra_object['axis'] is None:
        logger.error(f"Data or axis is still None after attempting to load {file_path}. This indicates a loading failure.")
        # This case should ideally be caught by specific loaders, but as a safeguard:
        raise IOError(f"Failed to populate 'data' or 'axis' from file {file_path} with format {file_format}.")

    # Ensure data is complex for subsequent processing if it's spectral data
    # This is a heuristic. For MRS, data is often complex.
    # If it's purely image data from DICOM, this might not apply.
    # For now, let's assume spectral data should be complex.
    if not np.iscomplexobj(spectra_object['data']) and file_format not in ['mat_image_placeholder']: # Add more non-complex formats if any
        logger.info(f"Data for {file_format} is not complex. Converting to complex for consistency (real part only).")
        spectra_object['data'] = spectra_object['data'].astype(np.complex128)


    return spectra_object


def preprocess_spectra(spectra_object, processing_params, logger=None):
    """
    Applies preprocessing steps to the spectral data.

    Args:
        spectra_object (dict): Dictionary from load_spectra (contains 'data', 'axis', 'metadata').
                               The 'data' key will be modified.
        processing_params (dict): Dictionary of processing parameters from config.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        dict: The modified spectra_object.
    """
    logger = logger or default_logger
    data = spectra_object.get('data')
    axis = spectra_object.get('axis')

    if data is None:
        logger.error("Preprocessing error: 'data' is missing in spectra_object.")
        raise ValueError("'data' is missing in spectra_object for preprocessing.")
    if axis is None: # Axis might be needed for some steps
        logger.warning("Preprocessing warning: 'axis' is missing in spectra_object. Some steps might fail or behave unexpectedly.")
        # Create a dummy axis if critical for some functions, though this is not ideal
        # axis = np.arange(data.shape[-1])

    original_dtype = data.dtype

    # Ensure data is float or complex for processing, but try to preserve original type if possible
    if not np.issubdtype(data.dtype, np.floating) and not np.issubdtype(data.dtype, np.complexfloating):
        logger.info(f"Data type is {data.dtype}. Converting to float64 for processing.")
        try:
            data = data.astype(np.float64) # Start with float, convert to complex if phase correction is applied
        except ValueError:
            logger.error("Could not convert data to float64. Aborting preprocessing for this item.")
            return spectra_object # Or raise error


    # 1. Baseline Correction
    if 'baseline_correction' in processing_params:
        params = processing_params['baseline_correction']
        method = params.get('method', 'none').lower()
        logger.info(f"Applying baseline correction using method: {method}")
        if method == 'polynomial':
            degree = params.get('degree', 3)
            # For simplicity, apply to the last dimension (spectral dimension)
            # This assumes data is (..., n_points)
            if data.ndim == 1:
                coeffs = np.polyfit(axis, data.real, degree) # Fit to real part
                baseline = np.polyval(coeffs, axis)
                data -= baseline
                if np.iscomplexobj(original_dtype): # if original was complex, apply to imag too or handle appropriately
                    coeffs_imag = np.polyfit(axis, data.imag, degree)
                    baseline_imag = np.polyval(coeffs_imag, axis)
                    data -= 1j * baseline_imag
            else: # Iterate over higher dimensions if they exist (e.g., for spectral images)
                # Assuming the last axis is the spectral dimension
                for index in np.ndindex(data.shape[:-1]):
                    current_spectrum = data[index]
                    coeffs = np.polyfit(axis, current_spectrum.real, degree)
                    baseline = np.polyval(coeffs, axis)
                    data[index] -= baseline
                    if np.iscomplexobj(original_dtype):
                         coeffs_imag = np.polyfit(axis, current_spectrum.imag, degree)
                         baseline_imag = np.polyval(coeffs_imag, axis)
                         data[index] -= 1j * baseline_imag
            logger.info(f"Polynomial baseline correction (degree {degree}) applied.")
        elif method != 'none':
            logger.warning(f"Baseline correction method '{method}' not implemented. Skipping.")

    # 2. Apodization (typically applied to time-domain data, FID)
    # Assuming data is FID if apodization is requested.
    # If data is frequency domain, inverse FFT, apodize, then FFT back, or apply filter in freq domain.
    # For simplicity here, assume data is time-domain if apodization is called.
    if 'apodization' in processing_params:
        params = processing_params['apodization']
        func_type = params.get('function', 'none').lower()
        logger.info(f"Applying apodization using function: {func_type}")

        if data.ndim == 0: # scalar data cannot be apodized
            logger.warning("Data is scalar, cannot apply apodization. Skipping.")
        elif func_type != 'none':
            # Create time axis for apodization if not directly available or if 'axis' is frequency
            # This is a simplification; proper handling needs knowing if data is time/freq domain
            time_points = data.shape[-1]
            time_vector = np.arange(time_points) # Generic time vector: 0, 1, 2...
            # A more correct time_vector would use dwell time from metadata if available
            # e.g., dwell_time = metadata.get('dwell_time'); time_vector = np.arange(time_points) * dwell_time

            apod_func = np.ones_like(data[..., 0], dtype=float) # Get a representative slice for shape

            if func_type == 'gaussian':
                width_hz = params.get('width_hz', 10.0) # Gaussian broadening in Hz
                # Convert Hz to points if dwell time is known, otherwise relative width
                # This is a simplified conversion factor; true conversion needs dwell time
                # For now, let's assume width is a relative factor if dwell time is missing.
                # A common definition for Gaussian is exp(- (t / (2*sigma))^2 )
                # Or, for line broadening exp(- (pi * LB * t)^2 / (4*ln(2)) ) -> for FWHM
                # Let's use a simpler exp(- (t * factor)^2 ) where factor is derived from width_hz
                # If 'width' is given in points: sigma = width_points
                # If 'width' is in Hz, need SW or dwell time.
                # For now, assume 'width' is a parameter controlling sharpness.
                # Example: sigma = time_points / (width_hz * np.pi) # Highly approximate
                sigma = params.get('sigma_points', time_points / (width_hz * 0.1 if width_hz > 0 else 1.0) ) # heuristic
                apod_values = np.exp(-(time_vector**2) / (2 * sigma**2))
                apod_func = apod_values

            elif func_type in ['lorentzian', 'exponential']:
                lb_hz = params.get('lb_hz', 1.0) # Lorentzian broadening in Hz (exponential decay)
                # Again, conversion to points is needed. factor = pi * lb_hz * dwell_time
                # Simplified: factor = lb_hz / SW_points if SW_points is total width in points
                # Using a decay factor:
                decay_factor = np.pi * lb_hz * (1/(spectra_object['metadata'].get('sweep_width_hz', time_points) / time_points) if time_points >0 else 1) # Approx dwell time
                if spectra_object['metadata'].get('dwell_time_s'): # More accurate
                    decay_factor = np.pi * lb_hz * spectra_object['metadata']['dwell_time_s']

                apod_values = np.exp(-decay_factor * time_vector)
                apod_func = apod_values
            else:
                logger.warning(f"Apodization function '{func_type}' not implemented. Skipping.")

            if apod_func.ndim == 1 and data.ndim > 1: # Apply to last dimension
                 data *= apod_func # Broadcasting
            elif apod_func.shape == data.shape:
                 data *= apod_func
            else:
                logger.warning(f"Could not apply apodization due to shape mismatch: data {data.shape}, apod_func {apod_func.shape}")

            logger.info(f"{func_type.capitalize()} apodization applied.")

    # 3. Phase Correction (assumes data is complex and frequency domain)
    # If data is time domain (FID), it should be FFT'd first.
    # This is a crucial point: ensure data is in freq domain before phasing.
    # For simplicity, we'll assume it IS freq domain if this step is called.
    if 'phase_correction' in processing_params:
        params = processing_params['phase_correction']
        method = params.get('method', 'none').lower()
        logger.info(f"Applying phase correction using method: {method}")

        if not np.iscomplexobj(data):
            logger.warning("Data is not complex. Converting to complex for phase correction (imaginary part will be zero).")
            data = data.astype(np.complex128)

        if method == 'manual':
            ph0_rad = np.deg2rad(params.get('ph0_deg', 0.0)) # Zeroth order phase in degrees
            ph1_rad_per_hz = np.deg2rad(params.get('ph1_deg_per_ppm', 0.0) * spectra_object['metadata'].get('transmitter_frequency_hz', 1.0) * 1e-6) # First order phase, scaled by transmitter freq if given in ppm
            ph1_pivot_hz = params.get('ph1_pivot_hz', (axis[0] + axis[-1]) / 2 if axis is not None and len(axis)>0 else 0) # Pivot point for ph1 in Hz

            if axis is None:
                logger.error("Cannot apply phase correction without an axis. Skipping.")
            else:
                # Ensure axis is correctly scaled if ph1 is per Hz or per ppm
                # Assuming 'axis' is in Hz for this calculation. If it's ppm, it needs conversion.
                # For now, assume axis is frequency in appropriate units for ph1_rad to apply directly or after scaling.
                # If ph1_deg_per_point is given:
                ph1_rad_per_point = np.deg2rad(params.get('ph1_deg_per_point', 0.0))

                # Create phase correction array
                # The ph1 term is often (ph1 * (freq_axis - pivot_point))
                # Or ph1 * normalized_freq_axis where normalized goes from -0.5 to 0.5 or 0 to N-1

                # Simple linear phase ramp based on index if axis is complex
                norm_axis = np.arange(len(axis)) - len(axis)//2 # Centered normalized axis for ph1
                if ph1_rad_per_point != 0: # Prioritize per_point if given
                    phase_correction_array = np.exp(1j * (ph0_rad + ph1_rad_per_point * norm_axis))
                else: # Use frequency-based ph1
                    # Ensure axis is in Hz for ph1_rad_per_hz
                    # This part needs careful handling of units for axis and ph1 term
                    # Assuming axis is in Hz for ph1_rad_per_hz
                    phase_correction_array = np.exp(1j * (ph0_rad + ph1_rad_per_hz * (axis - ph1_pivot_hz)))

                if data.ndim == 1:
                    data *= phase_correction_array
                else: # Apply to last dimension
                    data *= phase_correction_array # Broadcasting
                logger.info(f"Manual phase correction (ph0={params.get('ph0_deg',0)}, ph1 related params used) applied.")
        elif method == 'automatic_search': # Placeholder for more advanced methods
            logger.warning("Automatic phase correction search not yet implemented. Skipping.")
        elif method != 'none':
            logger.warning(f"Phase correction method '{method}' not implemented. Skipping.")

    spectra_object['data'] = data.astype(original_dtype, copy=False) # Try to restore original dtype if no complex conversion happened implicitly
    if np.iscomplexobj(data) and not np.iscomplexobj(original_dtype): # If it became complex
        spectra_object['data'] = data # keep it complex

    logger.info("Preprocessing finished.")
    return spectra_object

if __name__ == '__main__':
    # Setup basic logger for testing
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('data_io_test')

    # --- Create dummy data files ---
    os.makedirs("sample_data", exist_ok=True)

    # Create dummy .mat file
    if SCIPY_AVAILABLE:
        mat_data = {'data': np.array([[1+1j, 2+2j, 3+3j, 4+4j], [5+5j, 6+6j, 7+7j, 8+8j]]),
                    'axis': np.array([10, 20, 30, 40]),
                    'metadata': {'tx_freq_hz': 123.2e6, 'spectral_width_hz': 5000.0, 'name': 'test_mat'}}
        try:
            scipy.io.savemat("sample_data/test.mat", mat_data)
            logger.info("Created sample_data/test.mat")

            # Test loading .mat
            loaded_mat = load_spectra("sample_data/test.mat", "mat", logger=logger)
            if loaded_mat:
                logger.info(f"Successfully loaded sample_data/test.mat. Data shape: {loaded_mat['data'].shape}, Metadata: {loaded_mat['metadata']}")
        except Exception as e:
            logger.error(f"Error with MAT file creation/loading: {e}")
    else:
        logger.warning("SciPy not available, skipping .mat file creation and test.")

    # Create dummy .h5 file
    if H5PY_AVAILABLE:
        try:
            with h5py.File("sample_data/test.h5", "w") as hf:
                hf.create_dataset("data", data=np.array([10., 20., 30., 40., 50.]))
                hf.create_dataset("axis", data=np.array([1., 2., 3., 4., 5.]))
                meta_group = hf.create_group("metadata_group")
                meta_group.attrs['tx_freq_hz'] = 123.2e6
                meta_group.attrs['spectral_width_hz'] = 4000.0
                meta_group.attrs['name'] = 'test_h5'
            logger.info("Created sample_data/test.h5")

            # Test loading .h5
            loaded_h5 = load_spectra("sample_data/test.h5", "h5", logger=logger)
            if loaded_h5:
                logger.info(f"Successfully loaded sample_data/test.h5. Data shape: {loaded_h5['data'].shape}, Metadata: {loaded_h5['metadata']}")
        except Exception as e:
            logger.error(f"Error with HDF5 file creation/loading: {e}")
    else:
        logger.warning("h5py not available, skipping .h5 file creation and test.")

    # --- Test preprocessing ---
    if SCIPY_AVAILABLE and os.path.exists("sample_data/test.mat"): # Use the .mat file for preprocessing test
        logger.info("--- Testing Preprocessing using sample_data/test.mat ---")
        spec_obj = load_spectra("sample_data/test.mat", "mat", logger=logger)

        if spec_obj and spec_obj['data'] is not None:
            # Ensure data is 1D or 2D for these simple tests
            if spec_obj['data'].ndim > 1:
                 test_data_slice = spec_obj['data'][0].copy() # Take first FID if multiple
                 spec_obj_single = {'data': test_data_slice, 'axis': spec_obj['axis'], 'metadata': spec_obj['metadata']}
            else:
                 spec_obj_single = spec_obj.copy()

            logger.info(f"Original data for preprocessing: {spec_obj_single['data']}")

            processing_params_test = {
                "apodization": {"function": "exponential", "lb_hz": 1.0},
                # "baseline_correction": {"method": "polynomial", "degree": 1}, # Baseline on FID is unusual, but for testing
                "phase_correction": {"method": "manual", "ph0_deg": 0, "ph1_deg_per_point": 0} # No phase change for FID test
            }
            # If we want to test phase correction, we'd typically FFT the data first
            # For now, let's assume data is FID for apodization, then FFT, then phase.
            # This example will just apodize.

            # Simulate FID (time domain data) for apodization
            spec_obj_single['metadata']['sweep_width_hz'] = 5000.0 # Hz for lb_hz conversion
            spec_obj_single['metadata']['dwell_time_s'] = 1.0 / spec_obj_single['metadata']['sweep_width_hz']

            processed_obj = preprocess_spectra(spec_obj_single, processing_params_test, logger=logger)
            logger.info(f"Data after apodization: {processed_obj['data']}")

            # Simulate Frequency domain data for phase and baseline
            # This is a mock FFT for testing purposes
            if np.iscomplexobj(processed_obj['data']):
                freq_domain_data = np.fft.fftshift(np.fft.fft(processed_obj['data']))
                freq_axis = np.fft.fftshift(np.fft.fftfreq(len(processed_obj['axis']), d=spec_obj_single['metadata']['dwell_time_s']))

                spec_obj_freq = {'data': freq_domain_data, 'axis': freq_axis, 'metadata': spec_obj_single['metadata']}

                processing_params_freq = {
                    "baseline_correction": {"method": "polynomial", "degree": 1},
                    "phase_correction": {"method": "manual", "ph0_deg": 10, "ph1_deg_per_point": 0.5} # Small phase shift
                }
                processed_obj_freq = preprocess_spectra(spec_obj_freq, processing_params_freq, logger=logger)
                logger.info(f"Data after FFT, baseline, and phase correction: {processed_obj_freq['data']}")
            else:
                logger.warning("Skipping frequency domain processing test as data is not complex after apodization.")
        else:
            logger.warning("Could not load test.mat for preprocessing test or data is None.")
    else:
        logger.warning("SciPy not available or test.mat not found, skipping preprocessing test.")

    logger.info("data_io.py example run finished.")
