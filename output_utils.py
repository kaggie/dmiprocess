import csv
import logging
import os
import numpy as np

# Default logger if none is provided
default_logger = logging.getLogger(__name__)
default_logger.addHandler(logging.NullHandler())

# Conditional matplotlib import for plotting
try:
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend for saving files
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    default_logger.warning("Matplotlib not found. Plotting functions will be disabled.")
    MATPLOTLIB_AVAILABLE = False


def save_results_to_csv(filepath_prefix, fit_results, mode, logger=None):
    """
    Saves fitting results to a CSV file.

    Args:
        filepath_prefix (str): Base path for the output CSV (e.g., config.output_dir/config.output_prefix).
        fit_results (dict): Dictionary from fit_oxsa_model or fit_lcmodel_data.
        mode (str): 'oxsa' or 'lcmodel'.
        logger (logging.Logger, optional): Logger instance.
    """
    logger = logger or default_logger

    if not fit_results:
        logger.warning(f"No fit results provided for mode {mode}. Skipping CSV output.")
        return

    filename = f"{filepath_prefix}_{mode}_results.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger.info(f"Saving {mode} results to CSV: {filename}")

    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            if mode == 'oxsa':
                params = fit_results.get('fitted_params', {})
                crlbs = fit_results.get('crlbs_absolute', {}) # Flat map: {'a_0': val, ...}
                num_peaks = len(params.get('a', []))

                writer.writerow(["Peak_Index", "Parameter_Type", "Fitted_Value", "CRLB_Absolute"])

                param_keys_ordered = ['a', 'f', 'd', 'phi', 'g']
                for i in range(num_peaks):
                    for key in param_keys_ordered:
                        value = params.get(key, [np.nan]*num_peaks)[i] if i < len(params.get(key, [])) else np.nan
                        crlb_key = f"{key}_{i}"
                        crlb_val = crlbs.get(crlb_key, np.nan) if crlbs else np.nan
                        writer.writerow([i, key, f"{value:.6e}", f"{crlb_val:.6e}"])

            elif mode == 'lcmodel':
                amplitudes = fit_results.get('amplitudes', {}) # Dict: {'Metab1': val, ...}
                crlbs_data = fit_results.get('crlbs', {}) # Dict: {'absolute': {'Metab1':...}, 'percent_metabolite': ...}

                abs_crlbs = crlbs_data.get('absolute', {}) if crlbs_data else {}
                perc_crlbs = crlbs_data.get('percent_metabolite', {}) if crlbs_data else {}

                writer.writerow(["Metabolite", "Amplitude", "CRLB_Absolute", "CRLB_Percent"])
                if not amplitudes:
                    logger.warning("LCModel results provided, but no amplitudes found.")
                    writer.writerow(["No metabolite amplitudes found in results."])

                for name, amp_val in amplitudes.items():
                    abs_crlb_val = abs_crlbs.get(name, np.nan) if abs_crlbs else np.nan
                    perc_crlb_val = perc_crlbs.get(name, np.nan) if perc_crlbs else np.nan
                    writer.writerow([name, f"{amp_val:.6e}", f"{abs_crlb_val:.6e}", f"{perc_crlb_val:.2f}"])

                # Add baseline amplitudes if available
                baseline_amps = fit_results.get('baseline_amplitudes')
                if baseline_amps is not None: # Could be empty list or None
                    writer.writerow([]) # Spacer
                    writer.writerow(["Baseline_Coefficient_Index", "Value"])
                    for i, b_amp in enumerate(baseline_amps):
                        writer.writerow([f"coeff_{i}", f"{b_amp:.6e}"])
            else:
                logger.error(f"Unsupported mode '{mode}' for CSV output.")
                return

        logger.info(f"Successfully saved results to {filename}")

    except IOError as e:
        logger.error(f"Error writing CSV file {filename}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during CSV generation for {mode}: {e}", exc_info=True)


def plot_fit_results(filepath_prefix, mrs_data_obj_or_dict, fit_results, mode, processing_params=None, logger=None):
    """
    Generates and saves plots of the fitting results.

    Args:
        filepath_prefix (str): Base path for saving plot files.
        mrs_data_obj_or_dict: MRSData object (LCModel) or dict from data_io (OXSA).
        fit_results (dict): Dictionary from fitting functions.
        mode (str): 'oxsa' or 'lcmodel'.
        processing_params (dict, optional): Preprocessing parameters from config.
        logger (logging.Logger, optional): Logger instance.
    """
    logger = logger or default_logger
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping plot generation.")
        return
    if not fit_results:
        logger.warning(f"No fit results provided for mode {mode}. Skipping plots.")
        return

    os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
    plot_filename_base = f"{filepath_prefix}_{mode}"

    try:
        if mode == 'lcmodel':
            if not hasattr(mrs_data_obj_or_dict, 'get_frequency_axis'): # Check if it's MRSData like
                logger.error("LCModel plotting requires an MRSData-like object. Skipping plots.")
                return

            data_freq_domain = mrs_data_obj_or_dict.get_frequency_domain_data(apply_fftshift=True).real
            try:
                # Use 'ppm' if central_frequency is available, else 'hz'
                ppm_unit = 'ppm' if mrs_data_obj_or_dict.central_frequency is not None else 'hz'
                freq_axis = mrs_data_obj_or_dict.get_frequency_axis(unit=ppm_unit)
            except ValueError as e: # If axis generation fails (e.g. missing params)
                logger.warning(f"Could not generate frequency axis for LCModel plot ({e}). Plotting against index.")
                freq_axis = np.arange(len(data_freq_domain))
                ppm_unit = 'index'

            fitted_total = fit_results.get('fitted_spectrum_total')
            residuals = fit_results.get('residuals')
            fitted_baseline = fit_results.get('fitted_baseline')
            fitted_metabs_sum = fit_results.get('fitted_spectrum_metabolites')
            freq_axis_fitted = fit_results.get('frequency_axis_fitted') # Axis corresponding to fitted components

            if freq_axis_fitted is None and fitted_total is not None: # Fallback if not in results
                logger.warning("frequency_axis_fitted not in LCModel results, will use MRSData axis for fitted components if lengths match.")
                if len(freq_axis) == len(fitted_total):
                     freq_axis_fitted = freq_axis
                else: # Cannot plot if axes don't match and no specific axis provided
                    logger.error("Cannot plot LCModel fit: fitted components axis unknown or mismatched.")
                    return


            # Plot 1: Data vs. Full Fit, Baseline, Metabolite Sum
            plt.figure(figsize=(12, 7))
            plt.plot(freq_axis_fitted, fitted_total, label='Total Fit', color='red')
            if fitted_baseline is not None:
                plt.plot(freq_axis_fitted, fitted_baseline, label='Fitted Baseline', color='green', linestyle=':')
            if fitted_metabs_sum is not None:
                plt.plot(freq_axis_fitted, fitted_metabs_sum, label='Fitted Metabolites Sum', color='purple', linestyle='--')

            # Determine the original data that corresponds to the fitted range
            # This requires knowing the indices used for fitting if `LinearCombinationModel` doesn't return original data subsegment
            # For now, assume freq_axis_fitted and fitted_total are primary for this plot
            # Original data plotting needs care if only a sub-range was fitted
            # For simplicity, we plot the data that was actually fitted by LinearCombinationModel
            # This data is `lc_model.data_to_fit` which is not directly returned.
            # However, `fitted_total + residuals` should be this data.
            if fitted_total is not None and residuals is not None:
                 original_data_fitted_range = fitted_total + residuals
                 plt.plot(freq_axis_fitted, original_data_fitted_range, label='Original Data (fitted range)', color='gray', alpha=0.7)
            else: # Fallback if residuals not available
                 plt.plot(freq_axis, data_freq_domain, label='Original Data (full range)', color='gray', alpha=0.7)


            plt.xlabel(f"Frequency ({ppm_unit})")
            plt.ylabel("Intensity")
            plt.title(f"LCModel Fit: {os.path.basename(filepath_prefix)}")
            if ppm_unit == 'ppm': plt.xlim(max(freq_axis_fitted), min(freq_axis_fitted)) # Reverse PPM
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.savefig(f"{plot_filename_base}_fit.png")
            plt.close()
            logger.info(f"Saved LCModel fit plot to {plot_filename_base}_fit.png")

            # Plot 2: Residuals
            if residuals is not None:
                plt.figure(figsize=(12, 4))
                plt.plot(freq_axis_fitted, residuals, label='Residuals', color='blue')
                plt.xlabel(f"Frequency ({ppm_unit})")
                plt.ylabel("Intensity")
                plt.title(f"LCModel Residuals: {os.path.basename(filepath_prefix)}")
                if ppm_unit == 'ppm': plt.xlim(max(freq_axis_fitted), min(freq_axis_fitted))
                plt.legend()
                plt.grid(True, alpha=0.5)
                plt.savefig(f"{plot_filename_base}_residuals.png")
                plt.close()
                logger.info(f"Saved LCModel residuals plot to {plot_filename_base}_residuals.png")


        elif mode == 'oxsa':
            original_data_time = mrs_data_obj_or_dict.get('data') # This is a dict from data_io
            time_axis = fit_results.get('time_axis') # Should be added to OXSA results
            fitted_total_time = fit_results.get('fitted_spectrum_total') # Should be added
            residuals_time = fit_results.get('residuals_final') # Should be added

            if original_data_time is None or time_axis is None or fitted_total_time is None or residuals_time is None:
                logger.error("OXSA plotting requires 'data', 'time_axis', 'fitted_spectrum_total', and 'residuals_final'. Skipping plots.")
                return

            # If data has multiple dimensions (e.g. imaging) take first one for plot
            if original_data_time.ndim > 1:
                logger.warning(f"OXSA original data has {original_data_time.ndim} dims, plotting first FID.")
                original_data_time = original_data_time[0]
            if fitted_total_time.ndim > 1: fitted_total_time = fitted_total_time[0]
            if residuals_time.ndim > 1: residuals_time = residuals_time[0]


            # Plot 1: Time domain fit (Real part)
            plt.figure(figsize=(12, 7))
            plt.plot(time_axis, original_data_time.real, label='Original Data (Real)', color='gray', alpha=0.8)
            plt.plot(time_axis, fitted_total_time.real, label='Fitted Model (Real)', color='red')
            plt.xlabel("Time (s)")
            plt.ylabel("Intensity")
            plt.title(f"OXSA Time Domain Fit (Real Part): {os.path.basename(filepath_prefix)}")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.savefig(f"{plot_filename_base}_time_domain_real_fit.png")
            plt.close()
            logger.info(f"Saved OXSA time domain real part fit plot to {plot_filename_base}_time_domain_real_fit.png")

            # Plot 2: Time domain fit (Imaginary part if data is complex)
            if np.iscomplexobj(original_data_time):
                plt.figure(figsize=(12, 7))
                plt.plot(time_axis, original_data_time.imag, label='Original Data (Imag)', color='gray', alpha=0.8)
                plt.plot(time_axis, fitted_total_time.imag, label='Fitted Model (Imag)', color='red')
                plt.xlabel("Time (s)")
                plt.ylabel("Intensity")
                plt.title(f"OXSA Time Domain Fit (Imaginary Part): {os.path.basename(filepath_prefix)}")
                plt.legend()
                plt.grid(True, alpha=0.5)
                plt.savefig(f"{plot_filename_base}_time_domain_imag_fit.png")
                plt.close()
                logger.info(f"Saved OXSA time domain imag part fit plot to {plot_filename_base}_time_domain_imag_fit.png")

            # Plot 3: Residuals (Real part)
            plt.figure(figsize=(12, 4))
            plt.plot(time_axis, residuals_time.real, label='Residuals (Real)', color='blue')
            plt.xlabel("Time (s)")
            plt.ylabel("Intensity")
            plt.title(f"OXSA Residuals (Real Part): {os.path.basename(filepath_prefix)}")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.savefig(f"{plot_filename_base}_time_domain_real_residuals.png")
            plt.close()
            logger.info(f"Saved OXSA time domain real part residuals plot to {plot_filename_base}_time_domain_real_residuals.png")

            # Optional: Frequency domain representation of OXSA fit
            if mrs_data_obj_or_dict.get('metadata'):
                metadata = mrs_data_obj_or_dict.get('metadata')
                sw_hz = metadata.get('spectral_width_hz')
                cf_mhz = metadata.get('tx_freq_hz') / 1e6 if metadata.get('tx_freq_hz') else None

                if sw_hz and cf_mhz:
                    N = len(time_axis)
                    freq_hz = np.fft.fftshift(np.fft.fftfreq(N, d=time_axis[1]-time_axis[0] if N > 1 else 1.0))
                    ppm_axis = (freq_hz / cf_mhz)[::-1]

                    original_spec_freq = np.fft.fftshift(np.fft.fft(original_data_time))
                    fitted_spec_freq = np.fft.fftshift(np.fft.fft(fitted_total_time))

                    plt.figure(figsize=(12,7))
                    plt.plot(ppm_axis, original_spec_freq.real, label="Original Data (Freq Domain, Real)", color='gray', alpha=0.8)
                    plt.plot(ppm_axis, fitted_spec_freq.real, label="Fitted Model (Freq Domain, Real)", color='red')
                    plt.xlabel("Frequency (ppm)")
                    plt.ylabel("Intensity")
                    plt.title(f"OXSA Frequency Domain (Real Part): {os.path.basename(filepath_prefix)}")
                    plt.xlim(max(ppm_axis), min(ppm_axis)) # Reverse PPM
                    plt.legend()
                    plt.grid(True, alpha=0.5)
                    plt.savefig(f"{plot_filename_base}_freq_domain_real_fit.png")
                    plt.close()
                    logger.info(f"Saved OXSA frequency domain real part plot to {plot_filename_base}_freq_domain_real_fit.png")
        else:
            logger.error(f"Unsupported mode '{mode}' for plotting.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during plot generation for {mode}: {e}", exc_info=True)
    finally:
        plt.close('all') # Ensure all figures are closed if any error occurs mid-plot
