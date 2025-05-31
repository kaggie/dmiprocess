import argparse
import logging
import os
import numpy as np
import sys # For parse_cli_args

from config_loader import load_config, validate_config
from data_io import load_spectra, preprocess_spectra
# output_utils and mode-specific model/fitting imports will be done inside run_processing or mode blocks.

def parse_cli_args(argv_list=None):
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="SpectroFit CLI - Batch Processing Enabled")
    parser.add_argument(
        "--mode",
        choices=["oxsa", "lcmodel"],
        required=True,
        help="Processing mode: oxsa or lcmodel",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to a single input data file (optional, overrides config, processes only this file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output directory (if ends with /) or results file prefix (if not). Overrides config settings.",
    )
    parser.add_argument(
        "--prior",
        type=str,
        help="Path to the prior knowledge file (optional, for oxsa mode, overrides config).",
    )
    parser.add_argument(
        "--basis",
        type=str,
        help="Path to the basis set file (optional, for lcmodel mode, overrides config).",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)",
    )
    if argv_list is None: # pragma: no cover (cannot easily test this branch without manipulating sys.argv)
        return parser.parse_args()
    else:
        return parser.parse_args(argv_list)

def run_processing(args, config_data_base):
    """
    Main logic for processing based on parsed args and loaded base configuration.
    """
    logger = logging.getLogger() # Get root logger

    # --- Determine Input Files ---
    input_files_to_process = []
    if args.input:
        input_files_to_process = [args.input]
        logger.info(f"Processing single input file from CLI override: {args.input}")
    else:
        config_input_val = config_data_base.get('input_file')
        if isinstance(config_input_val, str):
            input_files_to_process = [config_input_val]
        elif isinstance(config_input_val, list):
            input_files_to_process = config_input_val
        else:
            logger.error("'input_file' in config is not a valid string or list. Cannot proceed.")
            return False # Indicate failure
        logger.info(f"Processing input files from configuration: {len(input_files_to_process)} file(s).")

    if not input_files_to_process:
        logger.error("No input files to process.")
        return False

    # --- Validate Base Configuration (once before batch) ---
    try:
        logger.info(f"Validating base configuration for mode: {args.mode}")
        validate_config(config_data_base, args.mode)
        logger.info("Base configuration validated successfully.")
    except ValueError as e:
        logger.error(f"Base configuration validation error: {e}")
        return False

    # --- Output Utilities Import ---
    OUTPUT_UTILS_OK = False
    try:
        from output_utils import save_results_to_csv, plot_fit_results
        OUTPUT_UTILS_OK = True
        logger.info("Output utilities loaded successfully.")
    except ImportError as e:
        logger.error(f"Failed to import Output components: {e}. Results will not be saved to files.")

    # --- Batch Processing Loop ---
    successful_files = 0
    failed_files_info = []

    for current_input_file_path in input_files_to_process:
        logger.info(f"--- Processing file: {current_input_file_path} ---")
        config_data_loop = config_data_base

        output_dir_cfg = config_data_loop.get('output_dir', 'spectrofit_results')
        output_prefix_cfg = config_data_loop.get('output_prefix', 'fit')

        if args.output:
            if args.output.endswith(os.path.sep) or os.path.isdir(args.output):
                current_output_dir = args.output
                current_file_specific_prefix = output_prefix_cfg
            else:
                current_output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
                current_file_specific_prefix = os.path.basename(args.output)
        else:
            current_output_dir = output_dir_cfg
            current_file_specific_prefix = output_prefix_cfg

        input_basename = os.path.splitext(os.path.basename(current_input_file_path))[0]
        final_output_prefix_for_file = f"{current_file_specific_prefix}_{input_basename}"

        os.makedirs(current_output_dir, exist_ok=True)
        output_base_filepath_for_file = os.path.join(current_output_dir, final_output_prefix_for_file)
        logger.info(f"Output base for this file: {output_base_filepath_for_file}")

        try:
            spectra_obj = None
            file_format_cfg = config_data_loop.get('file_format')
            current_file_format = file_format_cfg
            if not current_file_format:
                _, ext = os.path.splitext(current_input_file_path)
                current_file_format = ext.lower().replace('.', '')
                if not current_file_format:
                    if os.path.isdir(current_input_file_path) and os.path.exists(os.path.join(current_input_file_path, 'fid')):
                        current_file_format = 'pyspecdata_varian'
                    else:
                        raise ValueError("Could not determine file format and 'file_format' not in config.")
                logger.info(f"Inferred file_format '{current_file_format}' for {current_input_file_path}")

            vendor = config_data_loop.get('vendor')
            logger.info(f"Loading data from '{current_input_file_path}' using format '{current_file_format}'.")
            spectra_obj = load_spectra(current_input_file_path, current_file_format, vendor=vendor, logger=logger)

            if not (spectra_obj and spectra_obj['data'] is not None):
                raise ValueError("Failed to load data or data is empty.")
            logger.info(f"Data loaded successfully. Shape: {spectra_obj['data'].shape}")

            if 'preprocessing_params' in config_data_loop and config_data_loop['preprocessing_params']:
                logger.info("Applying preprocessing...")
                spectra_obj = preprocess_spectra(spectra_obj, config_data_loop['preprocessing_params'], logger=logger)
                logger.info(f"Preprocessing complete. Processed data shape: {spectra_obj['data'].shape}.")
            else:
                logger.info("No preprocessing steps defined.")

            fit_results = None
            if args.mode == 'oxsa':
                try:
                    from oxsa_model import fit_oxsa_model
                except ImportError as e:
                    logger.error(f"OXSA components unavailable (PyTorch missing/corrupted): {e}")
                    raise
                oxsa_params_cfg = config_data_loop.get('mode_specific_params', {}).get('oxsa', {})
                if args.prior : oxsa_params_cfg['prior_knowledge_file'] = args.prior
                if not oxsa_params_cfg: raise ValueError("OXSA params missing in config.")
                num_peaks = oxsa_params_cfg.get('num_peaks')
                initial_params_guess = oxsa_params_cfg.get('initial_params_guess')
                fit_settings = oxsa_params_cfg.get('fit_settings')
                if num_peaks is None or initial_params_guess is None or fit_settings is None:
                    raise ValueError("Missing OXSA params: num_peaks, initial_params_guess, or fit_settings.")
                current_data_for_fitting = spectra_obj['data']
                if current_data_for_fitting.ndim > 1:
                    logger.warning(f"OXSA expects 1D data, got {current_data_for_fitting.shape}. Using index 0.")
                    current_data_for_fitting = current_data_for_fitting[0]
                if spectra_obj.get('axis') is None: raise ValueError("OXSA fitting needs time axis.")
                logger.info(f"Starting OXSA fit for {num_peaks} peak(s)...")
                fit_results = fit_oxsa_model(current_data_for_fitting, spectra_obj['axis'], num_peaks, initial_params_guess, fit_settings, logger)
                logger.info("OXSA fitting complete for current file.")
                logger.info(f"  Final Loss: {fit_results['final_loss']:.6e}")

            elif args.mode == 'lcmodel':
                try:
                    from lcmodel_fitting import load_basis_set, create_mrs_data_object, fit_lcmodel_data, MRS_LCM_LIB_AVAILABLE as LCMODEL_LIB_OK
                except ImportError as e:
                    logger.error(f"LCModel components unavailable: {e}")
                    raise
                if not LCMODEL_LIB_OK: raise ImportError("LCModel library (mrs_lcm_analysis or deps) not loaded.")
                lcmodel_params_cfg = config_data_loop.get('mode_specific_params', {}).get('lcmodel', {})
                if args.basis: lcmodel_params_cfg['basis_file'] = args.basis
                if not lcmodel_params_cfg: raise ValueError("LCModel params missing in config.")
                basis_file_path = lcmodel_params_cfg.get('basis_file')
                fitting_range_ppm = lcmodel_params_cfg.get('fitting_range_ppm')
                baseline_degree = lcmodel_params_cfg.get('baseline_degree')
                lc_fit_settings = lcmodel_params_cfg.get('fit_settings', {})
                if not all([basis_file_path, fitting_range_ppm is not None, baseline_degree is not None]):
                    raise ValueError("Missing LCModel params: basis_file, fitting_range_ppm, or baseline_degree.")
                mrs_data_obj = create_mrs_data_object(spectra_obj, logger=logger)
                if not mrs_data_obj: raise ValueError("Failed to create MRSData object.")
                if mrs_data_obj.central_frequency is None or mrs_data_obj.sampling_frequency is None:
                    raise ValueError("MRSData object missing frequency info for basis loading.")
                logger.info(f"Loading LCModel basis set from: {basis_file_path}")
                basis_set_obj = load_basis_set(basis_file_path, mrs_data_obj.central_frequency, mrs_data_obj.sampling_frequency, logger)
                if not basis_set_obj: raise ValueError("Failed to load LCModel basis set.")
                lc_config_for_fit = {'fitting_range_ppm': fitting_range_ppm, 'baseline_degree': baseline_degree}
                logger.info("Starting LCModel fit...")
                fit_results = fit_lcmodel_data(mrs_data_obj, basis_set_obj, lc_config_for_fit, lc_fit_settings, logger)
                logger.info("LCModel fitting complete for current file.")
                logger.info(f"  Estimated Amplitudes: {fit_results.get('amplitudes')}")

            if fit_results and OUTPUT_UTILS_OK:
                logger.info(f"Saving results for {current_input_file_path}...")
                save_results_to_csv(output_base_filepath_for_file, fit_results, args.mode, logger=logger)
                data_for_plot = mrs_data_obj if args.mode == 'lcmodel' and 'mrs_data_obj' in locals() else spectra_obj
                plot_fit_results(output_base_filepath_for_file, data_for_plot, fit_results, args.mode,
                                 config_data_loop.get('preprocessing_params'), logger=logger)
            elif not OUTPUT_UTILS_OK: logger.warning("Output utilities not available, skipping file saving.")
            elif not fit_results: logger.warning("No fit results to save.")
            successful_files += 1
            logger.info(f"--- Successfully processed: {current_input_file_path} ---")
        except Exception as e_file:
            logger.error(f"--- FAILED processing file: {current_input_file_path} ---")
            logger.error(f"Error: {e_file}", exc_info=args.loglevel.upper() == "DEBUG") # Show traceback if DEBUG
            failed_files_info.append((current_input_file_path, str(e_file)))

    logger.info("\n--- Batch Processing Summary ---")
    logger.info(f"Total files attempted: {len(input_files_to_process)}")
    logger.info(f"Successfully processed: {successful_files}")
    logger.info(f"Failed: {len(failed_files_info)}")
    if failed_files_info:
        logger.warning("Failed files:")
        for f_path, err_msg in failed_files_info:
            logger.warning(f"  - {f_path}: {err_msg}")
    logger.info("--- SpectroFit CLI finished ---")
    return successful_files == len(input_files_to_process) # Return True if all succeeded

def main():
    """CLI entry point."""
    args = parse_cli_args(sys.argv[1:]) # Pass command line arguments (excluding script name)

    # Setup logging based on parsed args
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.basicConfig(level=logging.INFO) # Default level
        logging.warning(f"Invalid log level: {args.loglevel}. Defaulting to INFO.")
    else:
        logging.basicConfig(level=numeric_level)

    logger = logging.getLogger()
    logger.info("Parsed arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    try:
        logger.info(f"Loading configuration from: {args.config}")
        config_data_base = load_config(args.config)
        logger.info("Base configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load base configuration: {e}", exc_info=True)
        sys.exit(1) # Critical error, exit

    # Run the main processing logic
    all_succeeded = run_processing(args, config_data_base)

    if not all_succeeded:
        sys.exit(1) # Exit with error code if any file failed

if __name__ == "__main__":
    main()
