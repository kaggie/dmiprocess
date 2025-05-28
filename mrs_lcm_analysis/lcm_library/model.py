import numpy as np
import torch
import matplotlib.pyplot as plt
from .data_loading import MRSData
from .basis import BasisSet, BasisSpectrum

class LinearCombinationModel:
    """
    Performs linear combination modeling of MRS data.
    """

    def __init__(self,
                 mrs_data: MRSData,
                 basis_set: BasisSet,
                 fitting_range_ppm: tuple = None,
                 baseline_degree: int = None):
        """
        Initializes a LinearCombinationModel object.

        Args:
            mrs_data (MRSData): An instance of the MRSData class containing the
                                spectrum to be fitted.
            basis_set (BasisSet): An instance of the BasisSet class containing
                                  the metabolite basis spectra.
            fitting_range_ppm (tuple, optional): A tuple (min_ppm, max_ppm)
                                                 specifying the frequency range
                                                 (in ppm) over which the fitting
                                                 should be performed. If None,
                                                 fits the entire spectrum.
                                                 Note: min_ppm should be less than max_ppm.
            baseline_degree (int, optional): Degree of the polynomial baseline to be
                                             simultaneously fitted. If None, no baseline
                                             is fitted. Defaults to None.
        """
        if not isinstance(mrs_data, MRSData):
            raise TypeError("mrs_data must be an instance of MRSData.")
        if not isinstance(basis_set, BasisSet):
            raise TypeError("basis_set must be an instance of BasisSet.")
        if fitting_range_ppm is not None:
            if not (isinstance(fitting_range_ppm, tuple) and
                    len(fitting_range_ppm) == 2 and
                    all(isinstance(x, (int, float)) for x in fitting_range_ppm) and
                    fitting_range_ppm[0] < fitting_range_ppm[1]):
                raise ValueError("fitting_range_ppm must be a tuple of two numbers (min_ppm, max_ppm) with min_ppm < max_ppm.")
        if baseline_degree is not None and (not isinstance(baseline_degree, int) or baseline_degree < 0):
            raise ValueError("baseline_degree must be a non-negative integer or None.")

        self.mrs_data = mrs_data
        self.basis_set = basis_set
        self.fitting_range_ppm = fitting_range_ppm
        self.baseline_degree = baseline_degree # Store baseline_degree

        # Attributes for storing results
        self.estimated_metabolite_amplitudes = None
        self.all_fitted_amplitudes = None 
        self.fitted_spectrum = None # Total fit (metabolites + baseline)
        self.fitted_metabolite_component = None
        self.fitted_baseline_component = None
        self.residuals = None
        
        # Attributes for data and matrices used in fit (within fitting range)
        self.data_to_fit = None
        self.basis_matrix_to_fit = None # Combined matrix (metabolites + baseline)
        self.metabolite_basis_matrix_to_fit = None
        self.baseline_basis_vectors_to_fit = None
        self.frequency_axis_to_fit = None
        self.fit_indices = None
        self.estimated_baseline_amplitudes = None
        self.estimated_crlbs_absolute = None # For all fitted parameters
        self.estimated_metabolite_crlbs_percent = None # For metabolites only


    def fit(self, use_torch: bool = False):
        """
        Performs the linear combination modeling.

        Args:
            use_torch (bool): If True, use torch.linalg.lstsq. Otherwise, use
                              numpy.linalg.lstsq. Defaults to False.
        """
        # a. Retrieve frequency-domain data
        spectrum_data = self.mrs_data.get_frequency_domain_data(apply_fftshift=True)
        if np.iscomplexobj(spectrum_data):
            spectrum_data = spectrum_data.real # Fit real part

        # b. Retrieve metabolite basis matrix
        basis_metabolite_names = self.basis_set.get_metabolite_names()
        if not basis_metabolite_names and self.baseline_degree is None: # Need metabolites if no baseline
             raise ValueError("Basis set is empty and no baseline requested. Cannot perform fit.")
        
        metabolite_basis_matrix_full = np.array([[]]).reshape(spectrum_data.shape[0], 0) # Empty if no metabolites
        if basis_metabolite_names:
            metabolite_basis_matrix_full = self.basis_set.get_basis_matrix()
            if np.iscomplexobj(metabolite_basis_matrix_full):
                metabolite_basis_matrix_full = metabolite_basis_matrix_full.real

            if spectrum_data.shape[0] != metabolite_basis_matrix_full.shape[0]:
                raise ValueError(
                    f"MRS data points ({spectrum_data.shape[0]}) and metabolite basis spectra "
                    f"points ({metabolite_basis_matrix_full.shape[0]}) do not match."
                )

        # c. Determine frequency axis and apply fitting range to data and metabolite basis
        try:
            full_frequency_axis_ppm = self.mrs_data.get_frequency_axis(unit='ppm')
        except ValueError as e:
            raise ValueError(f"Cannot proceed with fit: Could not get frequency axis. Original error: {e}")

        current_data_to_fit = spectrum_data
        current_metabolite_basis_matrix = metabolite_basis_matrix_full

        # Ensure consistent orientation of frequency axis (descending for PPM)
        # This logic assumes get_frequency_axis returns a reversed (descending) ppm axis
        if not np.all(np.diff(full_frequency_axis_ppm) <= 0):
             # If it's ascending, reverse everything to match typical MRS display logic for range selection
             full_frequency_axis_ppm_sorted = full_frequency_axis_ppm[::-1]
             current_data_to_fit_sorted = current_data_to_fit[::-1]
             current_metabolite_basis_matrix_sorted = current_metabolite_basis_matrix[::-1, :] if basis_metabolite_names else current_metabolite_basis_matrix
        else:
             full_frequency_axis_ppm_sorted = full_frequency_axis_ppm
             current_data_to_fit_sorted = current_data_to_fit
             current_metabolite_basis_matrix_sorted = current_metabolite_basis_matrix


        if self.fitting_range_ppm:
            min_ppm, max_ppm = self.fitting_range_ppm
            self.fit_indices = np.where(
                (full_frequency_axis_ppm_sorted >= min_ppm) & (full_frequency_axis_ppm_sorted <= max_ppm)
            )[0]

            if len(self.fit_indices) == 0:
                raise ValueError(
                    f"No data points found in the specified fitting range "
                    f"({min_ppm:.2f} - {max_ppm:.2f} ppm). Check range and data."
                )
            
            self.data_to_fit = current_data_to_fit_sorted[self.fit_indices]
            self.metabolite_basis_matrix_to_fit = current_metabolite_basis_matrix_sorted[self.fit_indices, :] if basis_metabolite_names else np.array([[]]).reshape(len(self.fit_indices),0)
            self.frequency_axis_to_fit = full_frequency_axis_ppm_sorted[self.fit_indices]
        else:
            self.fit_indices = None 
            self.data_to_fit = current_data_to_fit_sorted
            self.metabolite_basis_matrix_to_fit = current_metabolite_basis_matrix_sorted
            self.frequency_axis_to_fit = full_frequency_axis_ppm_sorted
        
        if self.metabolite_basis_matrix_to_fit.shape[0] == 0 and self.baseline_degree is None:
            raise ValueError("No data points selected by fitting range and no baseline requested.")


        # d. Generate baseline basis vectors if baseline_degree is set
        final_combined_basis_matrix = self.metabolite_basis_matrix_to_fit
        num_metabolites = self.metabolite_basis_matrix_to_fit.shape[1]

        if self.baseline_degree is not None:
            from .baseline import create_polynomial_basis_vectors 
            
            # Normalize frequency axis for polynomial generation for stability (scaled to [-1, 1])
            freq_min = self.frequency_axis_to_fit.min()
            freq_max = self.frequency_axis_to_fit.max()
            if freq_max == freq_min: 
                normalized_freq_axis = np.zeros_like(self.frequency_axis_to_fit)
            else:
                # Ensure it's a copy to avoid modifying self.frequency_axis_to_fit if it's a view
                normalized_freq_axis = 2 * (np.array(self.frequency_axis_to_fit) - freq_min) / (freq_max - freq_min) - 1
            
            self.baseline_basis_vectors_to_fit = create_polynomial_basis_vectors(
                normalized_freq_axis, 
                self.baseline_degree
            )
            if num_metabolites > 0 :
                final_combined_basis_matrix = np.hstack(
                    (self.metabolite_basis_matrix_to_fit, self.baseline_basis_vectors_to_fit)
                )
            else: # Only baseline fitting
                 final_combined_basis_matrix = self.baseline_basis_vectors_to_fit
        
        self.basis_matrix_to_fit = final_combined_basis_matrix 

        if self.basis_matrix_to_fit.shape[1] == 0: # No basis vectors at all
            raise ValueError("No basis functions (metabolite or baseline) to fit.")


        # e. Perform linear least squares fit
        if use_torch:
            data_tensor = torch.from_numpy(self.data_to_fit.astype(np.float32))
            basis_tensor = torch.from_numpy(self.basis_matrix_to_fit.astype(np.float32))
            if basis_tensor.ndim == 1: basis_tensor = basis_tensor.unsqueeze(1)
            if data_tensor.ndim == 1: data_tensor = data_tensor.unsqueeze(1)
            
            # torch.linalg.lstsq expects basis_tensor to be (m,n) and data_tensor (m,k)
            # result.solution will be (n,k)
            solution_tensor = torch.linalg.lstsq(basis_tensor, data_tensor).solution
            all_amplitudes_array = solution_tensor.squeeze().numpy()
            if all_amplitudes_array.ndim == 0: # Handle case of single amplitude
                 all_amplitudes_array = np.array([all_amplitudes_array.item()])

        else:
            all_amplitudes_array, residuals_sum_sq, rank, s = np.linalg.lstsq(
                self.basis_matrix_to_fit, self.data_to_fit, rcond=None
            )
        
        self.all_fitted_amplitudes = all_amplitudes_array

        # f. Store estimated amplitudes and fitted spectrum components
        if basis_metabolite_names:
            self.estimated_metabolite_amplitudes = dict(
                zip(basis_metabolite_names, all_amplitudes_array[:num_metabolites])
            )
            self.fitted_metabolite_component = self.metabolite_basis_matrix_to_fit @ all_amplitudes_array[:num_metabolites]
        else: # No metabolites were in the basis set
            self.estimated_metabolite_amplitudes = {}
            self.fitted_metabolite_component = np.zeros_like(self.data_to_fit)


        if self.baseline_degree is not None and self.baseline_basis_vectors_to_fit is not None:
            self.estimated_baseline_amplitudes = all_amplitudes_array[num_metabolites:]
            self.fitted_baseline_component = self.baseline_basis_vectors_to_fit @ self.estimated_baseline_amplitudes
            self.fitted_spectrum = self.fitted_metabolite_component + self.fitted_baseline_component
        else:
            self.estimated_baseline_amplitudes = None
            self.fitted_baseline_component = np.zeros_like(self.data_to_fit) 
            self.fitted_spectrum = self.fitted_metabolite_component
            
        self.residuals = self.data_to_fit - self.fitted_spectrum

        # g. Calculate Cramer-Rao Lower Bounds (CRLBs)
        # Based on the assumption of Gaussian noise and a linear model.
        # Variances of estimated parameters are diagonal elements of sigma_sq * (X^T X)^-1
        
        num_data_points_fitted = len(self.data_to_fit)
        num_fitted_parameters = self.basis_matrix_to_fit.shape[1]

        if num_data_points_fitted <= num_fitted_parameters:
            print("Warning: CRLBs cannot be reliably calculated. Number of data points is less than or equal to number of parameters.")
            self.estimated_crlbs_absolute = None
            self.estimated_metabolite_crlbs_percent = None
        else:
            # Estimate noise variance (sigma_sq) from residuals
            # Using unbiased estimator: sum(residuals^2) / (N - P)
            sigma_sq = np.sum(self.residuals**2) / (num_data_points_fitted - num_fitted_parameters)

            try:
                # X is the combined basis matrix used for the fit (self.basis_matrix_to_fit)
                # Fisher Information Matrix (FIM) approx X^T X / sigma_sq
                # We need (X^T X)^-1
                xtx = self.basis_matrix_to_fit.T @ self.basis_matrix_to_fit
                
                # Check condition number to warn about potential instability
                condition_number_xtx = np.linalg.cond(xtx)
                if condition_number_xtx > 1e10: # Threshold can be tuned
                    print(f"Warning: High condition number for XTX ({condition_number_xtx:.2e}). CRLBs may be unreliable due to near multicollinearity.")

                xtx_inv = np.linalg.inv(xtx)
                
                # Variances of the estimated amplitudes
                variances = np.diag(sigma_sq * xtx_inv)
                # Ensure variances are non-negative (numerical precision might cause small negatives)
                variances[variances < 0] = 0 
                
                crlbs_absolute_all_params = np.sqrt(variances)
                
                param_names = []
                if basis_metabolite_names:
                    param_names.extend(basis_metabolite_names)
                if self.baseline_degree is not None:
                    param_names.extend([f"baseline_deg{i}" for i in range(self.baseline_degree + 1)])
                
                self.estimated_crlbs_absolute = dict(zip(param_names, crlbs_absolute_all_params))

                # Calculate CRLBs as percentage for metabolites
                self.estimated_metabolite_crlbs_percent = {}
                if basis_metabolite_names: # Only if metabolites were fitted
                    for i, name in enumerate(basis_metabolite_names):
                        amplitude = self.all_fitted_amplitudes[i]
                        if amplitude != 0 and not np.isinf(crlbs_absolute_all_params[i]) and not np.isnan(crlbs_absolute_all_params[i]):
                            self.estimated_metabolite_crlbs_percent[name] = (crlbs_absolute_all_params[i] / abs(amplitude)) * 100
                        else:
                            self.estimated_metabolite_crlbs_percent[name] = np.inf # Or some other indicator for problematic CRLB

            except np.linalg.LinAlgError:
                print("Warning: Could not invert XTX matrix. CRLBs cannot be calculated (singular matrix).")
                self.estimated_crlbs_absolute = None
                self.estimated_metabolite_crlbs_percent = None
            except Exception as e:
                print(f"An error occurred during CRLB calculation: {e}")
                self.estimated_crlbs_absolute = None
                self.estimated_metabolite_crlbs_percent = None


    def get_fitted_spectrum(self) -> np.ndarray:
        """Returns the reconstructed total spectrum (metabolites + baseline) from the last fit."""
        if self.fitted_spectrum is None:
            raise RuntimeError("Fit has not been performed yet. Call fit() first.")
        return self.fitted_spectrum

    def get_residuals(self) -> np.ndarray:
        """Returns the difference between the original MRS data (in the fitted range)
           and the total fitted spectrum."""
        if self.residuals is None:
            raise RuntimeError("Fit has not been performed yet. Call fit() first.")
        return self.residuals

    def get_estimated_metabolite_amplitudes(self) -> dict:
        """Returns the dictionary of estimated metabolite amplitudes."""
        if self.estimated_metabolite_amplitudes is None:
            raise RuntimeError("Fit has not been performed yet. Call fit() first.")
        return self.estimated_metabolite_amplitudes

    def get_estimated_baseline_amplitudes(self) -> np.ndarray:
        """
        Returns the array of estimated baseline polynomial coefficients.
        Returns None if baseline was not fitted.
        """
        if self.baseline_degree is None:
            return None # No baseline was intended to be fit
        if self.estimated_baseline_amplitudes is None:
             raise RuntimeError("Fit has not been performed yet. Call fit() first.")
        return self.estimated_baseline_amplitudes

    def get_fitted_baseline(self) -> np.ndarray:
        """
        Returns the fitted baseline component.
        Returns an array of zeros with the same shape as data_to_fit if no baseline was fitted.
        """
        if self.fitted_baseline_component is None:
            if self.baseline_degree is None: 
                if self.data_to_fit is not None:
                    return np.zeros_like(self.data_to_fit)
                else: # Fit not called, data_to_fit not available
                    raise RuntimeError("Fit has not been performed yet, so data_to_fit is unavailable for baseline shape.")
            else: # Baseline was intended but fit not called or failed before this stage
                raise RuntimeError("Fit has not been performed or baseline component not generated. Call fit() first.")
        return self.fitted_baseline_component
        
    def get_fitted_metabolite_component(self) -> np.ndarray:
        """Returns the sum of all fitted metabolite components."""
        if self.fitted_metabolite_component is None:
            raise RuntimeError("Fit has not been performed yet. Call fit() first.")
        return self.fitted_metabolite_component

    def plot_fit(self, plot_individual_components: bool = True, xlim_ppm: tuple = None):
        """
        Plots the original MRS data, the fitted spectrum, and residuals.
        Also plots the estimated baseline if it was fitted.

        Args:
            plot_individual_components (bool): If True, also plots the scaled
                                               individual metabolite basis spectra.
            xlim_ppm (tuple, optional): A tuple (max_ppm, min_ppm) for x-axis limits.
                                        Note typical MRS display (high to low ppm).
                                        If None, uses the fitted range or full spectrum range.
        """
        if self.data_to_fit is None or self.fitted_spectrum is None or self.residuals is None:
            raise RuntimeError("Fit has not been performed yet. Call fit() first.")

        freq_axis_plot = self.frequency_axis_to_fit
        if freq_axis_plot is None: 
             freq_axis_plot = np.arange(len(self.data_to_fit))


        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True) 
        fig.suptitle("Linear Combination Model Fit", fontsize=16)

        # Plot 1: Original Data, Total Fitted Spectrum, and Fitted Baseline
        axes[0].plot(freq_axis_plot, self.data_to_fit, label="Original Data (fitted range)", color='gray', alpha=0.8)
        axes[0].plot(freq_axis_plot, self.fitted_spectrum, label="Total Fitted Spectrum", color='red', linestyle='-')
        
        # Plot baseline only if it was part of the model
        if self.baseline_degree is not None and self.fitted_baseline_component is not None:
            axes[0].plot(freq_axis_plot, self.fitted_baseline_component, label="Fitted Baseline", color='green', linestyle=':')
        
        # Plot metabolite sum only if there are metabolites
        if self.estimated_metabolite_amplitudes and self.fitted_metabolite_component is not None:
             axes[0].plot(freq_axis_plot, self.fitted_metabolite_component, label="Fitted Metabolites Sum", color='purple', linestyle='-.', alpha=0.7)
        
        axes[0].set_ylabel("Intensity")
        axes[0].legend(fontsize='small')
        axes[0].grid(True, alpha=0.4)

        # Plot 2: Residuals
        axes[1].plot(freq_axis_plot, self.residuals, label="Residuals (Data - Total Fit)", color='blue')
        axes[1].set_ylabel("Intensity")
        axes[1].legend(fontsize='small')
        axes[1].grid(True, alpha=0.4)

        # Plot 3: Original Data vs. Metabolite-only part (Data - Baseline) and Individual Metabolite Components
        if self.baseline_degree is not None and self.fitted_baseline_component is not None:
            data_minus_baseline = self.data_to_fit - self.fitted_baseline_component
            axes[2].plot(freq_axis_plot, data_minus_baseline, label="Data - Est. Baseline", color='lightgray', alpha=0.9)
        else: 
            axes[2].plot(freq_axis_plot, self.data_to_fit, label="Original Data", color='lightgray', alpha=0.9)
            
        if self.estimated_metabolite_amplitudes and self.fitted_metabolite_component is not None:
            axes[2].plot(freq_axis_plot, self.fitted_metabolite_component, label="Summed Metabolite Fit", color='purple', linestyle='-')

        if plot_individual_components and self.estimated_metabolite_amplitudes and self.metabolite_basis_matrix_to_fit is not None:
            if self.metabolite_basis_matrix_to_fit.shape[1] > 0: # Check if there are metabolites
                metabolite_names = self.basis_set.get_metabolite_names() 
                for i in range(self.metabolite_basis_matrix_to_fit.shape[1]):
                    name = metabolite_names[i] 
                    amp = self.estimated_metabolite_amplitudes.get(name, 0) 
                    component = self.metabolite_basis_matrix_to_fit[:, i] * amp
                    axes[2].plot(freq_axis_plot, component, label=f"{name} (scaled)", linestyle=':', alpha=0.8)
        
        axes[2].set_xlabel(f"Frequency ({self.mrs_data.metadata.get('ppm_unit', 'ppm')})" if freq_axis_plot is not None and len(freq_axis_plot)>0 else "Index")
        axes[2].set_ylabel("Intensity")
        axes[2].legend(fontsize='small')
        axes[2].grid(True, alpha=0.4)

        if xlim_ppm:
            axes[2].set_xlim(xlim_ppm[0], xlim_ppm[1])
        elif freq_axis_plot is not None and len(freq_axis_plot) > 0 and isinstance(freq_axis_plot[0], (int, float, np.number)):
            try:
                axes[2].set_xlim(max(freq_axis_plot), min(freq_axis_plot))
            except Exception: 
                pass
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.show()

    def get_relative_quantification(self, reference_metabolites: list) -> dict:
        """
        Calculates metabolite amplitudes relative to a sum of reference metabolites.

        Args:
            reference_metabolites (list): A list of metabolite names to be used
                                          as the reference (e.g., ['Cr', 'PCr']).

        Returns:
            dict: A dictionary where keys are metabolite names and values are
                  their amplitudes relative to the sum of reference metabolite
                  amplitudes. Returns None if reference amplitudes cannot be determined
                  (e.g. reference metabolite not found, or sum is zero/negative).
        
        Raises:
            RuntimeError: If fit has not been performed yet.
            ValueError: If reference_metabolites list is empty or not a list.
        """
        if self.estimated_metabolite_amplitudes is None:
            raise RuntimeError("Fit has not been performed yet. Call fit() first.")
        if not isinstance(reference_metabolites, list) or not reference_metabolites:
            raise ValueError("reference_metabolites must be a non-empty list of metabolite names.")

        sum_reference_amplitude = 0.0
        missing_references = []
        for ref_metab in reference_metabolites:
            amp = self.estimated_metabolite_amplitudes.get(ref_metab)
            if amp is None:
                missing_references.append(ref_metab)
            elif amp > 0: # Only sum positive amplitudes for reference
                sum_reference_amplitude += amp
            else:
                # Optionally, one might still want to sum non-positive amplitudes if that makes sense for specific use cases
                # print(f"Warning: Reference metabolite {ref_metab} has non-positive amplitude ({amp:.2f}). It will not be included in the reference sum if sum becomes zero or negative.")
                # For now, let's sum them, and check the total sum later
                sum_reference_amplitude += amp


        if missing_references:
            print(f"Warning: Reference metabolite(s) {', '.join(missing_references)} not found in estimated amplitudes. Cannot calculate relative quantification.")
            return None
        
        if sum_reference_amplitude <= 0:
            print(f"Warning: Sum of reference metabolite amplitudes ({sum_reference_amplitude:.2f}) is zero or negative. Cannot calculate relative quantification.")
            return None

        relative_amplitudes = {}
        for metab_name, metab_amp in self.estimated_metabolite_amplitudes.items():
            relative_amplitudes[metab_name] = metab_amp / sum_reference_amplitude
        
        return relative_amplitudes

    def get_estimated_crlbs(self) -> dict:
        """
        Returns the estimated Cramer-Rao Lower Bounds (CRLBs) for the fitted parameters.

        The 'absolute' CRLBs are the estimated standard deviations of the parameters.
        The 'percent_metabolite' CRLBs are the absolute CRLB of a metabolite
        divided by its estimated amplitude, expressed as a percentage.

        Returns:
            dict: A dictionary with two keys:
                  'absolute': A dictionary where keys are parameter names (e.g., 'NAA',
                              'Cr', 'baseline_deg0', 'baseline_deg1') and values are
                              their absolute CRLB (estimated standard deviation).
                              This dictionary will be None if CRLBs could not be calculated.
                  'percent_metabolite': A dictionary where keys are metabolite names and
                                        values are their CRLBs as a percentage of their
                                        estimated amplitude. This dictionary will be None
                                        if CRLBs could not be calculated or if no
                                        metabolites were fitted.
        
        Raises:
            RuntimeError: If fit has not been performed yet.
        """
        if self.all_fitted_amplitudes is None: # Check if fit() has been run
            raise RuntimeError("Fit has not been performed yet. Call fit() first.")

        return {
            'absolute': self.estimated_crlbs_absolute,
            'percent_metabolite': self.estimated_metabolite_crlbs_percent
        }
