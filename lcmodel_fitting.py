import numpy as np
import logging
import re

# Attempt to import from the local mrs_lcm_analysis library
# This structure assumes that 'mrs_lcm_analysis' is in the Python path
# or that this script is run in an environment where it's accessible.
try:
    from mrs_lcm_analysis.lcm_library.basis import BasisSet, BasisSpectrum
    from mrs_lcm_analysis.lcm_library.data_loading import MRSData
    from mrs_lcm_analysis.lcm_library.model import LinearCombinationModel
    MRS_LCM_LIB_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import from mrs_lcm_analysis: {e}. Ensure the library is installed and in PYTHONPATH.")
    MRS_LCM_LIB_AVAILABLE = False
    # Define dummy classes if library not available, to allow CLI to load, but fitting will fail.
    class BasisSet: pass
    class BasisSpectrum: pass
    class MRSData: pass
    class LinearCombinationModel: pass


# Default logger if none is provided
default_logger = logging.getLogger(__name__)
default_logger.addHandler(logging.NullHandler())


def _parse_dot_basis_simplified(file_path, logger=None):
    """
    Simplified parser for LCModel .basis files.
    Extracts metabolite names, HZPPPM, NDATAB, and spectral data.

    Args:
        file_path (str): Path to the .basis file.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        dict: {
            'metabolites': {name: np.array(complex_data)},
            'hzpppm': float,
            'ndatab': int,
            'header_info': dict (other extracted header fields like te, seq)
        }
        Returns None if parsing fails.
    """
    logger = logger or default_logger
    logger.info(f"Parsing .basis file: {file_path}")

    metabolite_data = {}
    header_info = {}
    hzpppm = None
    ndatab = None
    metabolite_names = []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        in_header = False
        in_namelist = False
        current_metab_name = None
        temp_spectra_cols = [] # To hold text columns for current metabolite

        for line_num, line in enumerate(lines):
            line = line.strip()

            if "$BASIS" in line or "$SYSTEM" in line: # Older/alternative header start
                in_header = True
                continue
            if "$END" in line and in_header:
                in_header = False
                if current_metab_name: # Finalize last read metabolite if any
                    if temp_spectra_cols:
                        try:
                            real_parts = [float(val) for val in temp_spectra_cols[0]]
                            imag_parts = [float(val) for val in temp_spectra_cols[1]]
                            metabolite_data[current_metab_name] = np.array(real_parts) + 1j * np.array(imag_parts)
                            if ndatab and len(metabolite_data[current_metab_name]) != ndatab:
                                logger.warning(f"Metabolite {current_metab_name} data points {len(metabolite_data[current_metab_name])} != NDATAB {ndatab}")
                        except ValueError as e:
                            logger.error(f"Error parsing spectral data for {current_metab_name} near line {line_num+1}: {e}")
                            metabolite_data.pop(current_metab_name, None) # Remove partially parsed
                    current_metab_name = None
                    temp_spectra_cols = []
                continue

            if "$NMLIST" in line:
                in_namelist = True
                continue
            if "$END" in line and in_namelist:
                in_namelist = False
                continue

            if in_header:
                if "HZPPPM" in line:
                    match = re.search(r"HZPPPM\s*=\s*([\d\.]+)", line, re.IGNORECASE)
                    if match: hzpppm = float(match.group(1))
                elif "NDATAB" in line: # Number of data points in each basis spectrum
                    match = re.search(r"NDATAB\s*=\s*(\d+)", line, re.IGNORECASE)
                    if match: ndatab = int(match.group(1))
                elif "TE " in line: # Keep TE as TE consistently
                     match = re.search(r"TE\s*=\s*([\d\.]+)", line, re.IGNORECASE)
                     if match: header_info['TE'] = float(match.group(1))
                elif "SEQ" in line:
                     match = re.search(r"SEQ\s*=\s*'?([^']+)'?", line, re.IGNORECASE)
                     if match: header_info['SEQ'] = match.group(1).strip()
                # Add more header extractions if needed (e.g., HDRAP, BADDEG)

                # This is a common way to specify metabolite names and their column in older files
                # METABO_str_1_1_1_1 = 'Ala' could mean Ala is the first spectrum (cols 1,2)
                # For this simplified parser, we rely more on NAMELIST or explicit METABO markers
                # but if NAMELIST is not present, this could be a fallback.

            elif in_namelist:
                # Names are typically quoted, comma-separated
                raw_names = line.replace("'", "").split(',')
                for name in raw_names:
                    name = name.strip()
                    if name: metabolite_names.append(name)

            else: # Data section
                # LCModel .basis files often have Frequencies, Real, Imaginary columns
                # Or sometimes just Real, Imaginary for each metabolite sequentially
                # This simplified parser assumes sequential blocks for each metabolite from NAMELIST

                # A common pattern for metabolite data start:
                # CONC= xxx, HZPPPM= xxx, TE= xxx, METABO= 'metab_name'
                # $BASIS ... $END block contains the actual data points for METABO
                if "METABO=" in line:
                    if current_metab_name and temp_spectra_cols: # Finalize previous metabolite
                        try:
                            real_parts = [float(val) for val in temp_spectra_cols[0]]
                            imag_parts = [float(val) for val in temp_spectra_cols[1]]
                            metabolite_data[current_metab_name] = np.array(real_parts) + 1j * np.array(imag_parts)
                            if ndatab and len(metabolite_data[current_metab_name]) != ndatab:
                                logger.warning(f"Metabolite {current_metab_name} data points {len(metabolite_data[current_metab_name])} != NDATAB {ndatab}")
                        except ValueError as e:
                             logger.error(f"Error parsing spectral data for {current_metab_name} (METABO block): {e}")
                             metabolite_data.pop(current_metab_name, None)

                    match = re.search(r"METABO\s*=\s*'([^']+)'", line, re.IGNORECASE)
                    if match:
                        current_metab_name = match.group(1)
                        if current_metab_name not in metabolite_names: # If NAMELIST was absent or incomplete
                            logger.info(f"Found metabolite '{current_metab_name}' via METABO tag, not in NAMELIST or NAMELIST absent.")
                            metabolite_names.append(current_metab_name)
                        temp_spectra_cols = [[], []] # Reset for new metabolite [real_col, imag_col]
                    else: # METABO line but couldn't parse name
                        current_metab_name = None
                        temp_spectra_cols = []
                    in_header = True # After METABO line, there's usually a $BASIS ... $END block

                elif current_metab_name and not in_header and temp_spectra_cols is not None:
                    # Assumes two columns of data (real, imag) per line for the current metabolite
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            temp_spectra_cols[0].append(parts[0]) # Real
                            temp_spectra_cols[1].append(parts[1]) # Imag
                        except IndexError:
                            logger.warning(f"Could not parse data line for {current_metab_name}: '{line}'")
                    elif line: # Non-empty line that doesn't fit format
                        logger.warning(f"Skipping malformed data line for {current_metab_name}: '{line}'")

        # Finalize the very last metabolite if file ends without $END after its data
        if current_metab_name and temp_spectra_cols and not metabolite_data.get(current_metab_name):
            try:
                real_parts = [float(val) for val in temp_spectra_cols[0]]
                imag_parts = [float(val) for val in temp_spectra_cols[1]]
                metabolite_data[current_metab_name] = np.array(real_parts) + 1j * np.array(imag_parts)
                if ndatab and len(metabolite_data[current_metab_name]) != ndatab:
                     logger.warning(f"Metabolite {current_metab_name} data points {len(metabolite_data[current_metab_name])} != NDATAB {ndatab}")
            except ValueError as e:
                logger.error(f"Error parsing spectral data for last metabolite {current_metab_name}: {e}")
                metabolite_data.pop(current_metab_name, None)


        if not metabolite_data or not metabolite_names:
            logger.error("No metabolite data or names extracted from .basis file.")
            return None
        if hzpppm is None or ndatab is None:
            logger.warning("HZPPPM or NDATAB not found in .basis file header. This might affect axis generation.")
            # Try to infer ndatab from first metabolite if possible
            if ndatab is None and metabolite_data:
                first_metab = metabolite_names[0]
                if first_metab in metabolite_data:
                    ndatab = len(metabolite_data[first_metab])
                    logger.info(f"Inferred NDATAB={ndatab} from first metabolite.")

        # Ensure all listed metabolites in NAMELIST were actually found as data
        final_metabolites = {}
        for name in metabolite_names:
            if name in metabolite_data:
                final_metabolites[name] = metabolite_data[name]
            else:
                logger.warning(f"Metabolite '{name}' from NAMELIST was not found in data blocks.")

        if not final_metabolites:
            logger.error("No metabolites from NAMELIST could be matched with data blocks.")
            return None

        return {
            'metabolites': final_metabolites,
            'hzpppm': hzpppm,
            'ndatab': ndatab,
            'header_info': header_info
        }

    except FileNotFoundError:
        logger.error(f".basis file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error parsing .basis file {file_path}: {e}", exc_info=True)
        return None


def load_basis_set(basis_file_path, central_frequency_mhz, sampling_frequency_hz, logger=None):
    """
    Loads basis spectra from a (simplified) LCModel .basis file.

    Args:
        basis_file_path (str): Path to the .basis file.
        central_frequency_mhz (float): Spectrometer frequency in MHz (for PPM axis).
        sampling_frequency_hz (float): Sampling frequency in Hz (for bandwidth).
        logger (logging.Logger, optional): Logger instance.

    Returns:
        BasisSet: Populated BasisSet object, or None if loading fails.
    """
    logger = logger or default_logger
    if not MRS_LCM_LIB_AVAILABLE:
        logger.error("MRS LCM Analysis library not available. Cannot load basis set.")
        return None

    parsed_basis = _parse_dot_basis_simplified(basis_file_path, logger)

    if not parsed_basis:
        return None

    basis_set = BasisSet()
    ndatab = parsed_basis.get('ndatab')
    hzpppm = parsed_basis.get('hzpppm')

    if ndatab is None:
        logger.error("NDATAB (number of data points) could not be determined from .basis file. Cannot create axes.")
        return None
    if hzpppm is None:
        logger.warning("HZPPPM not found in .basis file. PPM axis might be incorrect if basis is not centered at 0 Hz offset.")
        # Default to a typical reference point if needed, e.g. 4.7 ppm for water, but this is risky.
        # For now, the axis calculation proceeds, but may be off.
    if central_frequency_mhz is None or sampling_frequency_hz is None:
        logger.error("Central frequency (MHz) and sampling frequency (Hz) must be provided to generate PPM axis for basis set.")
        return None

    # Create a frequency axis for the basis spectra
    # .basis files store frequency domain data. The frequency axis is implicit.
    # Typically, the center of the spectrum in .basis is 0 Hz offset.
    # This needs to align with how MRSData calculates its PPM axis.

    # Create a Hz axis centered at 0, with total width = sampling_frequency_hz
    # This assumes the .basis spectra span the full bandwidth.
    basis_hz_axis = np.linspace(-sampling_frequency_hz / 2, sampling_frequency_hz / 2 - sampling_frequency_hz / ndatab, ndatab)

    # Convert this Hz axis to PPM, then reverse for MRS display convention
    # ppm = (frequency_offset_Hz / spectrometer_base_frequency_MHz)
    basis_ppm_axis = (basis_hz_axis / central_frequency_mhz)[::-1] # Reversed for typical MRS display

    for name, data_array in parsed_basis['metabolites'].items():
        if len(data_array) != ndatab:
            logger.warning(f"Metabolite '{name}' has {len(data_array)} points, expected {ndatab} based on NDATAB. Skipping.")
            continue
        # The LinearCombinationModel will fit the real part.
        # BasisSpectrum can store complex, but model uses real.
        basis_spectrum_obj = BasisSpectrum(name=name, spectrum_data=data_array, frequency_axis=basis_ppm_axis)
        basis_set.add_metabolite(basis_spectrum_obj)
        logger.info(f"Added '{name}' to basis set. Data shape: {data_array.shape}")

    if not basis_set.get_metabolite_names():
        logger.error("No metabolites successfully loaded into BasisSet.")
        return None

    logger.info(f"BasisSet loaded with {len(basis_set.get_metabolite_names())} metabolites.")
    return basis_set


def create_mrs_data_object(loaded_spectra_dict, logger=None):
    """
    Creates an MRSData object from the dictionary returned by data_io.load_spectra.

    Args:
        loaded_spectra_dict (dict): Output from data_io.load_spectra.
                                    Expected keys: 'data', 'axis', 'metadata'.
                                    Metadata should contain 'tx_freq_hz' and 'spectral_width_hz'.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        MRSData: Instantiated MRSData object, or None if creation fails.
    """
    logger = logger or default_logger
    if not MRS_LCM_LIB_AVAILABLE:
        logger.error("MRS LCM Analysis library not available. Cannot create MRSData object.")
        return None

    if not loaded_spectra_dict or 'data' not in loaded_spectra_dict or 'metadata' not in loaded_spectra_dict:
        logger.error("Invalid input: loaded_spectra_dict is missing 'data' or 'metadata'.")
        return None

    data_array = loaded_spectra_dict['data']
    metadata = loaded_spectra_dict['metadata']

    # Determine data_type (time or frequency) - this is a simplification
    # load_spectra by default converts to complex. If it's an FID, it's time domain.
    # If it's from a format that's already frequency domain, this needs to be known.
    # For now, assume data from load_spectra is time-domain (FID) if not specified otherwise.
    data_type = metadata.get('data_type', 'time') # TODO: Make this more robust

    sampling_freq = metadata.get('spectral_width_hz')
    # MRSData expects central_frequency in MHz, tx_freq_hz is in Hz
    central_freq_mhz = metadata.get('tx_freq_hz') / 1e6 if metadata.get('tx_freq_hz') else None

    echo_time = metadata.get('echo_time_ms')
    repetition_time = metadata.get('repetition_time_ms')


    if data_array is None:
        logger.error("Data array is None in loaded_spectra_dict.")
        return None
    if data_array.ndim > 1:
        logger.warning(f"MRSData expects 1D data, but got shape {data_array.shape}. Using data from index 0.")
        data_array = data_array[0] # Take the first spectrum/FID if multi-dimensional

    if sampling_freq is None:
        logger.error("Missing 'spectral_width_hz' in metadata, required for MRSData's sampling_frequency.")
        return None
    if central_freq_mhz is None:
        logger.error("Missing 'tx_freq_hz' in metadata, required for MRSData's central_frequency.")
        return None

    try:
        mrs_data_obj = MRSData(
            data_array=data_array,
            data_type=data_type,
            sampling_frequency=sampling_freq,
            central_frequency=central_freq_mhz,
            echo_time_ms=echo_time,
            repetition_time_ms=repetition_time,
            metadata=metadata # Pass along all metadata
        )
        logger.info(f"MRSData object created: {mrs_data_obj}")
        return mrs_data_obj
    except Exception as e:
        logger.error(f"Error creating MRSData object: {e}", exc_info=True)
        return None


def fit_lcmodel_data(mrs_data_obj, basis_set_obj, lcmodel_config, fit_settings, logger=None):
    """
    Performs LCModel-like fitting using LinearCombinationModel.

    Args:
        mrs_data_obj (MRSData): MRSData object.
        basis_set_obj (BasisSet): BasisSet object.
        lcmodel_config (dict): LCModel specific parameters from main config
                               (e.g., fitting_range_ppm, baseline_degree).
        fit_settings (dict): Fitting settings (e.g., use_torch).
        logger (logging.Logger, optional): Logger instance.

    Returns:
        dict: Contains fitting results (amplitudes, CRLBs, fitted_spectrum, etc.), or None if fitting fails.
    """
    logger = logger or default_logger
    if not MRS_LCM_LIB_AVAILABLE:
        logger.error("MRS LCM Analysis library not available. Cannot perform LCModel fitting.")
        return None
    if not isinstance(mrs_data_obj, MRSData):
        logger.error("Invalid mrs_data_obj provided to fit_lcmodel_data.")
        return None
    if not isinstance(basis_set_obj, BasisSet):
        logger.error("Invalid basis_set_obj provided to fit_lcmodel_data.")
        return None


    fitting_range = lcmodel_config.get('fitting_range_ppm')
    baseline_deg = lcmodel_config.get('baseline_degree')
    use_torch_fitting = fit_settings.get('use_torch', False)

    if use_torch_fitting:
        try:
            import torch # Check if torch is available if requested
            logger.info("Torch available, proceeding with use_torch=True if set.")
        except ImportError:
            logger.warning("Torch is not installed, but use_torch=True was requested. Defaulting to NumPy for fitting.")
            use_torch_fitting = False

    logger.info(f"Initializing LinearCombinationModel. Range: {fitting_range}, Baseline deg: {baseline_deg}, Use torch: {use_torch_fitting}")

    try:
        lc_model = LinearCombinationModel(
            mrs_data=mrs_data_obj,
            basis_set=basis_set_obj,
            fitting_range_ppm=fitting_range,
            baseline_degree=baseline_deg
        )

        logger.info("Starting LCModel fit...")
        lc_model.fit(use_torch=use_torch_fitting)
        logger.info("LCModel fitting complete.")

        results = {
            'amplitudes': lc_model.get_estimated_metabolite_amplitudes(),
            'crlbs': lc_model.get_estimated_crlbs(), # This returns a dict with 'absolute' and 'percent_metabolite'
            'fitted_spectrum_total': lc_model.get_fitted_spectrum(),
            'fitted_spectrum_metabolites': lc_model.get_fitted_metabolite_component(),
            'fitted_baseline': lc_model.get_fitted_baseline(),
            'residuals': lc_model.get_residuals(),
            'frequency_axis_fitted': lc_model.frequency_axis_to_fit, # The axis corresponding to above spectra
            'baseline_amplitudes': lc_model.get_estimated_baseline_amplitudes()
        }

        # Optionally, plot the fit (can be controlled by a config flag later)
        # lc_model.plot_fit()

        return results

    except Exception as e:
        logger.error(f"Error during LCModel fitting: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Basic test setup
    logging.basicConfig(level=logging.INFO)
    test_logger = logging.getLogger('lcmodel_fitting_test')

    if not MRS_LCM_LIB_AVAILABLE:
        test_logger.error("MRS LCM Analysis library not available. Cannot run tests for lcmodel_fitting.py.")
    else:
        # 1. Create a dummy .basis file
        dummy_basis_content = """
 $NMLIST
  METAB_A, METAB_B
 $END
 $SYSTEM
  HZPPPM = 123.2  $ Same as FT for MRSData
  NDATAB = 4      $ Number of data points for each metab
 $END
 $BASIS
  CONC=1.0, HZPPPM=123.2, TE=30, METABO='METAB_A'
 $END
  1.0  0.1  $ Real Imag for METAB_A pt1
  2.0  0.2
  1.5  0.15
  0.5  0.05
 $BASIS
  CONC=1.0, HZPPPM=123.2, TE=30, METABO='METAB_B'
 $END
  0.8  -0.1
  1.8  -0.2
  1.2  -0.15
  0.3  -0.05
"""
        dummy_basis_path = "sample_data/dummy_test.basis"
        os.makedirs("sample_data", exist_ok=True)
        with open(dummy_basis_path, "w") as f:
            f.write(dummy_basis_content)
        test_logger.info(f"Created dummy basis file: {dummy_basis_path}")

        # 2. Test _parse_dot_basis_simplified
        parsed_content = _parse_dot_basis_simplified(dummy_basis_path, logger=test_logger)
        if parsed_content:
            test_logger.info(f"Parsed .basis content: Metabolites count {len(parsed_content['metabolites'])}, HZPPPM {parsed_content['hzpppm']}, NDATAB {parsed_content['ndatab']}")

            # 3. Test load_basis_set
            # These would typically come from the MRS data being fitted
            test_central_freq_mhz = 123.2
            test_sampling_freq_hz = parsed_content['ndatab'] * parsed_content['hzpppm'] if parsed_content['hzpppm'] and parsed_content['ndatab'] else 2000.0 # SW = N*dwell_time_hz = N * (hz/pt) ; hz/pt = hzpppm*ppm_range / N
            # A bit circular here. Let's assume a typical SW for testing if basis doesn't define it perfectly for this calc.
            # For a .basis file, data is already frequency domain. The effective SW and central freq are those of the data it will be fit to.
            # The HZPPPM from basis tells us the scaling of its own points if its axis was 0 at center.

            # Let's assume the basis spectra cover 4 ppm width. BW = ppm_width * central_freq_mhz
            # Then sampling_freq_hz = BW.
            # This example uses HZPPPM from basis directly. If basis has 1024 pts and HZPPPM = 2, then it covers 512 Hz.
            # The key is that the basis PPM axis must match the data's PPM axis.
            # For simplicity, assume basis was acquired with same parameters as data.
            # Actual sampling_frequency_hz for basis axis construction:
            num_pts_basis = parsed_content['ndatab']
            # The ppm range of the basis is implicitly defined by its points and hzpppm.
            # Total Hz width of basis = num_pts_basis * (1/dwell_time_of_basis_if_it_were_fid)
            # This part is tricky. Let's use the data's parameters for basis axis generation, assuming they match.

            test_sampling_freq_hz_for_data = 2000.0 # Example for the MRSData object

            basis_set_obj = load_basis_set(dummy_basis_path,
                                           central_frequency_mhz=test_central_freq_mhz,
                                           sampling_frequency_hz=test_sampling_freq_hz_for_data, # Use data's SW
                                           logger=test_logger)
            if basis_set_obj:
                test_logger.info(f"BasisSet loaded with metabolites: {basis_set_obj.get_metabolite_names()}")
                # test_logger.info(f"METAB_A freq axis: {basis_set_obj.get_metabolite('METAB_A').frequency_axis}")


            # 4. Test create_mrs_data_object
            # Create a dummy loaded_spectra_dict (as if from data_io.py)
            num_data_pts = num_pts_basis # Match basis points for this test
            dummy_fid = np.zeros(num_data_pts, dtype=complex) # Simple FID for testing structure
            dummy_fid[0] = num_data_pts # Make it non-zero
            for i in range(1, 5): dummy_fid[i] = num_data_pts / (i*2) * np.exp(1j * i * np.pi/4)


            loaded_spectra = {
                'data': dummy_fid,
                'axis': np.arange(num_data_pts) / test_sampling_freq_hz_for_data, # Time axis
                'metadata': {
                    'tx_freq_hz': test_central_freq_mhz * 1e6,
                    'spectral_width_hz': test_sampling_freq_hz_for_data,
                    'echo_time_ms': 30.0,
                    'data_type': 'time' # Explicitly time domain
                }
            }
            mrs_data_obj = create_mrs_data_object(loaded_spectra, logger=test_logger)

            if mrs_data_obj and basis_set_obj:
                # 5. Test fit_lcmodel_data
                lcmodel_cfg = {'fitting_range_ppm': (0.5, 4.0), 'baseline_degree': 2}
                fit_cfg = {'use_torch': False} # Test with NumPy first

                test_logger.info("--- Attempting LCModel fit ---")
                fit_results = fit_lcmodel_data(mrs_data_obj, basis_set_obj, lcmodel_cfg, fit_cfg, logger=test_logger)

                if fit_results:
                    test_logger.info(f"LCModel fit results - Amplitudes: {fit_results['amplitudes']}")
                    if fit_results['crlbs']:
                         test_logger.info(f"LCModel fit results - CRLBs (%): {fit_results['crlbs'].get('percent_metabolite')}")
                else:
                    test_logger.error("LCModel fitting failed during test.")
            else:
                test_logger.error("MRSData or BasisSet object creation failed. Cannot test fitting.")
        else:
            test_logger.error("Parsing dummy .basis file failed. Cannot proceed with tests.")

        # Clean up dummy file
        # os.remove(dummy_basis_path)
        test_logger.info(f"Test finished. Dummy basis file '{dummy_basis_path}' may remain.")
