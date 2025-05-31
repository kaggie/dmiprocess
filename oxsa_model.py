import torch
import torch.nn as nn
import numpy as np
import logging

# Default logger if none is provided
default_logger = logging.getLogger(__name__)
default_logger.addHandler(logging.NullHandler())

class AmaresPytorchModel(nn.Module):
    def __init__(self, num_peaks, initial_params, logger=None):
        """
        Initializes the AMARES PyTorch model.

        Args:
            num_peaks (int): Number of signals (peaks) to model.
            initial_params (dict): Dictionary with initial values for parameters.
                Expected keys: 'a' (amplitudes), 'f' (frequencies_hz),
                               'd' (dampings_hz), 'phi' (phases_rad), 'g' (lorentzian_fraction_g).
                Each value should be a list or 1D NumPy array of length num_peaks.
            logger (logging.Logger, optional): Logger instance.
        """
        super().__init__()
        self.num_peaks = num_peaks
        self.logger = logger or default_logger

        if not all(k in initial_params for k in ['a', 'f', 'd', 'phi', 'g']):
            msg = "Initial params dictionary is missing one or more required keys: 'a', 'f', 'd', 'phi', 'g'."
            self.logger.error(msg)
            raise ValueError(msg)

        try:
            # Ensure inputs are numpy arrays before converting to tensors
            # These parameters are learned during the optimization process.
            self.amplitudes = nn.Parameter(torch.tensor(np.array(initial_params['a'], dtype=np.float32)))
            self.frequencies_hz = nn.Parameter(torch.tensor(np.array(initial_params['f'], dtype=np.float32)))
            self.dampings_hz = nn.Parameter(torch.tensor(np.array(initial_params['d'], dtype=np.float32))) # d_k in Hz
            self.phases_rad = nn.Parameter(torch.tensor(np.array(initial_params['phi'], dtype=np.float32))) # phi_k in radians
            self.lorentzian_fraction_g = nn.Parameter(torch.tensor(np.array(initial_params['g'], dtype=np.float32))) # g_k (0 for Gaussian, 1 for Lorentzian)

            # Validate shapes
            for p_name, p_val in zip(['a', 'f', 'd', 'phi', 'g'],
                                     [self.amplitudes, self.frequencies_hz, self.dampings_hz, self.phases_rad, self.lorentzian_fraction_g]):
                if p_val.shape != (num_peaks,):
                    msg = f"Parameter '{p_name}' has incorrect shape {p_val.shape}, expected ({num_peaks,})."
                    self.logger.error(msg)
                    raise ValueError(msg)
            self.logger.info(f"AmaresPytorchModel initialized with {num_peaks} peaks.")

        except Exception as e:
            self.logger.error(f"Error initializing model parameters: {e}")
            raise

    def forward(self, t):
        """
        Calculates the AMARES model signal in the time domain.
        y_n = sum_k(a_k * exp(j*phi_k) * exp(-d_k * (1-g_k + g_k*t_n) * t_n) * exp(j*2*pi*f_k*t_n))
        Simplified lineshape term: exp(-(L_k + G_k*t_n) * t_n)
        where L_k = d_k * g_k (Lorentzian damping) and G_k = d_k * (1-g_k) (Gaussian damping factor)
        This needs d_k to represent the total decay rate.
        Let's use d_k as the base decay (often Lorentzian part) and a Gaussian part modified by (1-g_k).
        A common formulation for mixed Lorentzian-Gaussian lineshape in time domain:
        S_k(t) = a_k * exp(1j * phi_k) * exp(1j * 2 * pi * f_k * t) * exp(- (pi * L_k * t) - ( (pi * G_k * t)**2 / (4*ln(2)) ) )
        This is complex. The problem statement has a simpler form: exp(-d_k * (1-g_k + g_k*t_n) * t_n)
        This simpler form exp(-d_k * t_n - d_k * g_k * t_n^2 + d_k * g_k * t_n) is unusual.
        Let's re-evaluate the lineshape term from the problem:
        exp(-d_k * (1-g_k + g_k*t_n) * t_n) = exp( -d_k*(1-g_k)*t_n - d_k*g_k*t_n^2 )
        This means d_k*(1-g_k) is the Lorentzian damping rate (Hz) and d_k*g_k is related to Gaussian damping (Hz/s).
        Let: Lorentzian_damping_rate_hz = d_k * (1-g_k)
             Gaussian_damping_factor_hz_per_s = d_k * g_k
        So the exponent term is: -(Lorentzian_damping_rate_hz * t + Gaussian_damping_factor_hz_per_s * t^2)

        Args:
            t (torch.Tensor): Time points tensor. Should be 1D.

        Returns:
            torch.Tensor: Complex model signal tensor.
        """
        if t.ndim != 1:
            msg = f"Time tensor t must be 1D, but got {t.ndim} dimensions."
            self.logger.error(msg)
            raise ValueError(msg)

        # Ensure parameters are used in a way that matches their physical interpretation
        # self.dampings_hz represents d_k from the formula.
        # self.lorentzian_fraction_g represents g_k.

        # Reshape parameters for broadcasting: (num_peaks, 1)
        a = self.amplitudes.unsqueeze(1)
        f_hz = self.frequencies_hz.unsqueeze(1)
        d_hz = self.dampings_hz.unsqueeze(1) # This is d_k
        phi_rad = self.phases_rad.unsqueeze(1)
        g = self.lorentzian_fraction_g.unsqueeze(1) # This is g_k

        # Time tensor t needs to be (1, num_time_points) for broadcasting with parameter tensors
        t_broadcast = t.unsqueeze(0)

        # Complex exponential for phase: exp(j*phi_k)
        complex_phase = torch.exp(1j * phi_rad) # (num_peaks, 1)

        # Damping term: exp( -d_k*(1-g_k)*t_n - d_k*g_k*t_n^2 )
        lorentzian_damping_rate = d_hz * (1 - g) # Effective Lorentzian decay rate in Hz
        gaussian_damping_factor = d_hz * g       # Effective Gaussian decay factor in Hz/s (if t is in seconds)

        decay_exponent = - (lorentzian_damping_rate * t_broadcast + gaussian_damping_factor * t_broadcast**2)
        damping_term = torch.exp(decay_exponent) # (num_peaks, num_time_points)

        # Oscillation term: exp(j*2*pi*f_k*t_n)
        oscillation_term = torch.exp(1j * 2 * np.pi * f_hz * t_broadcast) # (num_peaks, num_time_points)

        # Signal for each peak: a_k * complex_phase_k * damping_term_k(t) * oscillation_term_k(t)
        signal_k = a * complex_phase * damping_term * oscillation_term # (num_peaks, num_time_points)

        # Sum over peaks: sum_k signal_k
        y_pred = torch.sum(signal_k, dim=0) # Sum along the num_peaks dimension

        return y_pred


def fit_oxsa_model(time_domain_data, time_axis, num_peaks, initial_params, fit_config, logger=None):
    """
    Fits the AMARES model to time-domain data using PyTorch.

    Args:
        time_domain_data (np.ndarray): Complex time-domain signal (1D).
        time_axis (np.ndarray): Time points (1D).
        num_peaks (int): Number of peaks to fit.
        initial_params (dict): Initial parameters for AmaresPytorchModel.
        fit_config (dict): Fitting settings (optimizer, num_iterations, learning_rate).
        logger (logging.Logger, optional): Logger instance.

    Returns:
        dict: Contains 'fitted_params' (dict of NumPy arrays) and 'final_loss' (float).
    """
    logger = logger or default_logger
    logger.info("Starting OXSA model fitting.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Convert data to PyTorch tensors
    try:
        t_tensor = torch.tensor(time_axis, dtype=torch.float32).to(device)
        data_tensor = torch.tensor(time_domain_data, dtype=torch.complex64).to(device) # Ensure complex
        if t_tensor.ndim != 1 or data_tensor.ndim != 1:
            raise ValueError("Time axis and data must be 1D arrays.")
        if len(t_tensor) != len(data_tensor):
            raise ValueError("Time axis and data must have the same length.")
        logger.debug(f"Data converted to tensors. Time points: {len(t_tensor)}, Data points: {len(data_tensor)}")
    except Exception as e:
        logger.error(f"Error converting data to PyTorch tensors: {e}")
        raise

    # Initialize model
    model = AmaresPytorchModel(num_peaks, initial_params, logger=logger).to(device)

    # Optimizer
    optimizer_name = fit_config.get('optimizer', 'Adam').lower()
    lr = fit_config.get('learning_rate', 0.01)
    num_iterations = fit_config.get('num_iterations', 100)

    if optimizer_name == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
        logger.info(f"Using LBFGS optimizer with lr={lr}.")
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        logger.info(f"Using Adam optimizer with lr={lr}.")
    else:
        logger.error(f"Unsupported optimizer: {optimizer_name}. Defaulting to Adam.")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Starting optimization for {num_iterations} iterations.")

    for i in range(num_iterations):
        def closure():
            optimizer.zero_grad()
            predicted_signal = model(t_tensor)
            # MSE loss for real and imaginary parts separately
            loss_real = torch.mean((predicted_signal.real - data_tensor.real)**2)
            loss_imag = torch.mean((predicted_signal.imag - data_tensor.imag)**2)
            loss = loss_real + loss_imag # Total loss
            loss.backward()
            if (i + 1) % (num_iterations // 10 if num_iterations > 10 else 1) == 0 or i == num_iterations -1 :
                 logger.info(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6e} (Real: {loss_real.item():.6e}, Imag: {loss_imag.item():.6e})")
            return loss

        if optimizer_name == 'lbfgs':
            optimizer.step(closure)
            # For LBFGS, the loss calculation for logging is done inside the closure
            # To get the final loss after step, we might need one more call if not logged already
            if (i + 1) % (num_iterations // 10 if num_iterations > 10 else 1) != 0 and i != num_iterations -1:
                 # Recalculate loss for logging if not done by closure's log frequency
                 with torch.no_grad():
                    predicted_signal = model(t_tensor)
                    loss_real = torch.mean((predicted_signal.real - data_tensor.real)**2)
                    loss_imag = torch.mean((predicted_signal.imag - data_tensor.imag)**2)
                    loss = loss_real + loss_imag
                    if (i + 1) % (num_iterations // 20 if num_iterations > 20 else 1) == 0 : # Log less frequently here
                        logger.debug(f"LBFGS Iteration {i+1}/{num_iterations}, Current Loss: {loss.item():.6e}")


        else: # Adam or other non-LBFGS
            loss = closure() # Calculate loss, gradients
            optimizer.step() # Update parameters

    # Final loss
    with torch.no_grad():
        predicted_signal = model(t_tensor) # t_tensor is already on device
        loss_real = torch.mean((predicted_signal.real - data_tensor.real)**2) # data_tensor on device
        loss_imag = torch.mean((predicted_signal.imag - data_tensor.imag)**2)
        final_loss = (loss_real + loss_imag).item()
    logger.info(f"Fitting finished. Final Loss: {final_loss:.6e}")

    # Extract fitted parameters
    fitted_params_np = {
        'a': model.amplitudes.detach().cpu().numpy().copy(),
        'f': model.frequencies_hz.detach().cpu().numpy().copy(),
        'd': model.dampings_hz.detach().cpu().numpy().copy(),
        'phi': model.phases_rad.detach().cpu().numpy().copy(),
        'g': model.lorentzian_fraction_g.detach().cpu().numpy().copy()
    }
    logger.debug(f"Fitted parameters (NumPy): {fitted_params_np}")

    # CRLBs will be calculated here.
    crlbs_map = None
    # --- CRLB Calculation ---
    try:
        logger.info("Attempting CRLB calculation...")
        model.eval() # Ensure model is in eval mode for Jacobian

        # 1. Estimate noise variance (sigma_sq)
        with torch.no_grad():
            predicted_signal_final = model(t_tensor) # t_tensor is on device
            residuals_complex = data_tensor - predicted_signal_final # data_tensor is on device
            # Variance of real and imaginary parts of residuals separately
            # Using biased variance (ddof=0) for simplicity, torch.var default. For unbiased, use ddof=1 or N-P.
            # For CRLB, sigma_sq is often sum(residuals^2)/(N-P). Here N = 2*num_time_points (real+imag)
            # P = num_total_fitted_params.
            # Simpler: variance of the complex residual vector elements.
            # Or, average of variances of real and imag parts.
            num_complex_points = residuals_complex.numel()
            num_real_points = 2 * num_complex_points

            # Sum of squared residuals (real and imag separately, then summed)
            # ssr = torch.sum(residuals_complex.real**2) + torch.sum(residuals_complex.imag**2)
            # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # if num_real_points > num_params:
            #    sigma_sq_unbiased = ssr / (num_real_points - num_params)
            # else:
            #    logger.warning("Not enough degrees of freedom for unbiased sigma_sq. Using biased variance.")
            #    sigma_sq_unbiased = ssr / num_real_points # Fallback to biased if P >= N

            # Using torch.var on residuals (real and imag separately)
            sigma_sq_real = torch.var(residuals_complex.real) # default ddof=0 (biased)
            sigma_sq_imag = torch.var(residuals_complex.imag) # default ddof=0 (biased)
            sigma_sq = (sigma_sq_real + sigma_sq_imag) / 2.0

            logger.info(f"Estimated noise variance (sigma_sq, avg of real/imag biased var): {sigma_sq.item():.4e}")
            if sigma_sq < 1e-20:
                logger.warning("Estimated noise variance is very low. CRLBs might be unstable or infinite. Clamping sigma_sq.")
                sigma_sq = torch.tensor(1e-20, device=device, dtype=sigma_sq.dtype)


        # 2. Define the function for Jacobian calculation
        def model_output_for_jacobian_calc(a_param, f_param, d_param, phi_param, g_param):
            # This function must reconstruct the model's output given parameter tensors.
            # It uses the model's structure but with explicit parameters.
            a_r = a_param.unsqueeze(1)
            f_hz_r = f_param.unsqueeze(1)
            d_hz_r = d_param.unsqueeze(1)
            phi_rad_r = phi_param.unsqueeze(1)
            g_r = g_param.unsqueeze(1)
            # t_tensor is from the outer scope, already on the correct device
            t_broadcast_jac = t_tensor.unsqueeze(0)

            complex_phase_jac = torch.exp(1j * phi_rad_r)
            lorentzian_damping_rate_jac = d_hz_r * (1 - g_r)
            gaussian_damping_factor_jac = d_hz_r * g_r
            decay_exponent_jac = - (lorentzian_damping_rate_jac * t_broadcast_jac + gaussian_damping_factor_jac * t_broadcast_jac**2)
            damping_term_jac = torch.exp(decay_exponent_jac)
            oscillation_term_jac = torch.exp(1j * 2 * np.pi * f_hz_r * t_broadcast_jac)
            signal_k_jac = a_r * complex_phase_jac * damping_term_jac * oscillation_term_jac
            y_pred_complex_jac = torch.sum(signal_k_jac, dim=0)

            return torch.cat((y_pred_complex_jac.real, y_pred_complex_jac.imag), dim=0)

        params_for_jacobian_tuple = (
            model.amplitudes.detach().clone().requires_grad_(True),
            model.frequencies_hz.detach().clone().requires_grad_(True),
            model.dampings_hz.detach().clone().requires_grad_(True),
            model.phases_rad.detach().clone().requires_grad_(True),
            model.lorentzian_fraction_g.detach().clone().requires_grad_(True)
        )

        logger.info("Calculating Jacobian using torch.autograd.functional.jacobian...")
        # Ensure using create_graph=False if not needing higher-order derivatives.
        # strict=False might be needed if some parameters don't influence output (not our case here)
        jacobian_tuple_of_tensors = torch.autograd.functional.jacobian(
            model_output_for_jacobian_calc,
            params_for_jacobian_tuple,
            create_graph=False,
            strict=False
        )

        # Detach and turn off grad for original model params if they were affected (should not be by clones)
        # for p in model.parameters(): p.requires_grad_(False) # No, this would break prior training loop logic if any.
        # The clones are what we modified.

        jacobian_matrices_list = []
        total_params_count_check = 0
        for i, jac_param_tensor_block in enumerate(jacobian_tuple_of_tensors):
            # jac_param_tensor_block is d(stacked_output)/d(param_group_i)
            # Shape: (2*N, num_peaks_for_this_param_group)
            if jac_param_tensor_block.ndim == 1: # Should be (2N) for a scalar param, but we have (num_peaks)
                jac_param_tensor_block = jac_param_tensor_block.unsqueeze(1)
            elif jac_param_tensor_block.ndim > 2:
                 logger.warning(f"Jacobian block {i} has unexpected ndim {jac_param_tensor_block.ndim}. Reshaping to 2D.")
                 jac_param_tensor_block = jac_param_tensor_block.reshape(jac_param_tensor_block.shape[0], -1)

            jacobian_matrices_list.append(jac_param_tensor_block)
            total_params_count_check += jac_param_tensor_block.shape[1]

        J = torch.cat(jacobian_matrices_list, dim=1)
        logger.info(f"Jacobian matrix J constructed with shape: {J.shape}")
        # Expected: J shape = (2 * len(t_tensor), num_peaks * 5)

        num_total_model_params = sum(p.numel() for p in params_for_jacobian_tuple)
        if J.shape[1] != num_total_model_params:
            logger.error(f"Jacobian column count {J.shape[1]} mismatch with total model params {num_total_model_params}. CRLB calculation aborted.")
            raise ValueError("Jacobian shape mismatch with total number of parameters.")

        logger.info("Calculating Fisher Information Matrix (FIM)...")
        FIM = (J.t() @ J) / sigma_sq  # J.t() is J_transpose

        logger.info("Calculating Covariance Matrix (inverse of FIM)...")
        covariance_matrix = torch.linalg.inv(FIM) # Use linalg.inv for better stability
        # Check for negative diagonal elements in covariance matrix (can happen with numerical instability)
        diag_cov = torch.diag(covariance_matrix)
        if torch.any(diag_cov < 0):
            logger.warning("Negative values found on the diagonal of the covariance matrix. Clamping to positive for CRLB calculation.")
            diag_cov = torch.clamp(diag_cov, min=1e-20) # Clamp small or negative values
            # This only affects CRLBs, not the full cov matrix if user wants it later
            crlbs_absolute_tensor = torch.sqrt(diag_cov)
        else:
            crlbs_absolute_tensor = torch.sqrt(diag_cov)

        crlbs_absolute_np = crlbs_absolute_tensor.cpu().numpy()

        crlbs_map = {}
        param_names_flat_ordered = []
        param_keys_ordered = ['a', 'f', 'd', 'phi', 'g'] # Order must match params_for_jacobian_tuple
        for key in param_keys_ordered:
            for i in range(num_peaks): # Assuming num_peaks is consistent for all params
                param_names_flat_ordered.append(f"{key}_{i}")

        if len(crlbs_absolute_np) == len(param_names_flat_ordered):
            crlbs_map = dict(zip(param_names_flat_ordered, crlbs_absolute_np))
            logger.info("CRLB calculation successful.")
            logger.debug(f"CRLB map: {crlbs_map}")
        else:
            logger.error(f"Mismatch between number of CRLBs ({len(crlbs_absolute_np)}) and number of parameters ({len(param_names_flat_ordered)}).")
            crlbs_map = None

    except torch.linalg.LinAlgError as e: # Catch PyTorch specific linear algebra errors
        logger.error(f"PyTorch LinAlgError during CRLB calculation (e.g., singular FIM): {e}")
        crlbs_map = None
    except RuntimeError as e:
        logger.error(f"PyTorch RuntimeError during CRLB calculation: {e}")
        crlbs_map = None
    except ValueError as e:
        logger.error(f"ValueError during CRLB calculation: {e}")
        crlbs_map = None
    except Exception as e:
        logger.error(f"An unexpected error occurred during CRLB calculation: {e}", exc_info=True)
        crlbs_map = None

    # Add final model output and residuals to the results
    with torch.no_grad():
        final_predicted_signal_np = model(t_tensor).detach().cpu().numpy().copy() # t_tensor is on device
        final_residuals_np = data_tensor.detach().cpu().numpy().copy() - final_predicted_signal_np # data_tensor is on device
        time_axis_np = t_tensor.detach().cpu().numpy().copy()

    return {
        'fitted_params': fitted_params_np,
        'final_loss': final_loss,
        'crlbs_absolute': crlbs_map,
        'fitted_spectrum_total': final_predicted_signal_np, # Full complex signal
        'residuals_final': final_residuals_np,           # Full complex residuals
        'time_axis': time_axis_np                        # Time axis
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('oxsa_model_test')

    # Example Usage
    num_peaks_test = 1
    # True parameters for a synthetic signal
    true_params = {
        'a': np.array([1.0], dtype=np.float32),
        'f': np.array([50.0], dtype=np.float32),    # Hz
        'd': np.array([10.0], dtype=np.float32),    # Hz (overall decay)
        'phi': np.array([0.0], dtype=np.float32),   # Radians
        'g': np.array([0.5], dtype=np.float32)     # 0.5 means mixed contribution
    }

    # Time axis
    N = 1024 # Number of points
    dt = 0.001 # Sampling interval in seconds (1ms) -> SW = 1000 Hz
    time_axis_test = np.arange(N) * dt

    # Create synthetic data using the model itself (ideal case)
    model_for_synth = AmaresPytorchModel(num_peaks_test, true_params, logger=logger)
    t_tensor_synth = torch.tensor(time_axis_test, dtype=torch.float32)

    # Generate clean synthetic data
    synthetic_data_clean = model_for_synth(t_tensor_synth).detach().cpu().numpy()
    # Add some noise
    noise_level = 0.05
    synthetic_data_noisy = synthetic_data_clean + noise_level * (
        np.random.randn(N) + 1j * np.random.randn(N)
    )

    logger.info(f"Generated synthetic data. Clean max: {np.abs(synthetic_data_clean).max()}, Noisy max: {np.abs(synthetic_data_noisy).max()}")


    # Initial guess for fitting (could be slightly off from true_params)
    initial_params_guess = {
        'a': np.array([0.8], dtype=np.float32),
        'f': np.array([52.0], dtype=np.float32),
        'd': np.array([8.0], dtype=np.float32),
        'phi': np.array([0.1], dtype=np.float32),
        'g': np.array([0.4], dtype=np.float32)
    }

    fit_config_test_adam = {
        'optimizer': 'Adam',
        'num_iterations': 200,
        'learning_rate': 0.1 # Adam often needs higher LR for this type of problem initially
    }

    fit_config_test_lbfgs = {
        'optimizer': 'LBFGS',
        'num_iterations': 50, # LBFGS often needs fewer iterations but each is more costly
        'learning_rate': 0.5 # LBFGS learning rate is more like a line search step size
    }

    logger.info("--- Testing fitting with Adam ---")
    results_adam = fit_oxsa_model(
        synthetic_data_noisy, time_axis_test, num_peaks_test,
        initial_params_guess, fit_config_test_adam, logger=logger
    )
    logger.info(f"Adam - Fitted params: {results_adam['fitted_params']}, Final loss: {results_adam['final_loss']:.6e}")
    for key in true_params:
        logger.info(f"Adam - Param {key}: True={true_params[key][0]:.3f}, Fit={results_adam['fitted_params'][key][0]:.3f}")

    logger.info("\n--- Testing fitting with LBFGS ---")
    # Re-initialize initial_params_guess for a fair comparison if they were modified (they are not in this setup)
    results_lbfgs = fit_oxsa_model(
        synthetic_data_noisy, time_axis_test, num_peaks_test,
        initial_params_guess, fit_config_test_lbfgs, logger=logger
    )
    logger.info(f"LBFGS - Fitted params: {results_lbfgs['fitted_params']}, Final loss: {results_lbfgs['final_loss']:.6e}")
    for key in true_params:
        logger.info(f"LBFGS - Param {key}: True={true_params[key][0]:.3f}, Fit={results_lbfgs['fitted_params'][key][0]:.3f}")
