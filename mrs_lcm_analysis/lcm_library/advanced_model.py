import torch
import torch.nn as nn
import torch.fft
import warnings 
import numpy as np 
from typing import List, Optional, Dict, Union, Any

class AdvancedLinearCombinationModel(nn.Module):
    """
    PyTorch-based Linear Combination Model for MRS data fitting.

    This model allows for fitting metabolite amplitudes, frequency shifts,
    linewidth broadenings, and a polynomial baseline. It is designed to be
    used with PyTorch's autograd for optimization. Parameters like shifts
    and linewidths are handled as 'raw' parameters which are then transformed
    during the forward pass to apply constraints.
    """

    def __init__(self,
                 basis_spectra_tensor: Union[torch.Tensor, np.ndarray], 
                 metabolite_names: List[str],
                 observed_spectrum_tensor: Union[torch.Tensor, np.ndarray], 
                 dt: float, 
                 fitting_mask: Optional[Union[torch.Tensor, np.ndarray]] = None, 
                 initial_params: Optional[Dict[str, Dict[str, float]]] = None,
                 constraints: Optional[Dict[str, float]] = None,
                 baseline_degree: Optional[int] = None,
                 device: Optional[str] = 'cpu'):
        """
        Initializes the AdvancedLinearCombinationModel.

        Args:
            basis_spectra_tensor (Union[torch.Tensor, np.ndarray]): Tensor/array containing 
                the basis spectra, shape (num_points, num_metabolites). 
                Should be frequency-domain, correctly fftshifted, and complex-valued.
                If NumPy array, it's converted. Internal representation is `torch.complex64`.
            metabolite_names (List[str]): List of metabolite names, corresponding
                to the columns in basis_spectra_tensor.
            observed_spectrum_tensor (Union[torch.Tensor, np.ndarray]): Tensor/array 
                containing the observed MRS spectrum (frequency-domain, fftshifted), 
                shape (num_points,). If complex, only the real part will be used.
                If NumPy array, it's converted. Internal representation is `torch.float32`.
            dt (float): Dwell time of the acquisition in seconds. Used for generating
                        frequency and time axes for transformations.
            fitting_mask (Optional[Union[torch.Tensor, np.ndarray]]): A boolean 
                tensor/array of shape (num_points,) indicating which points of the 
                spectrum to include in the loss calculation. If None, all points are used. 
                Defaults to None. If NumPy array, it's converted to a `torch.bool` tensor.
            initial_params (Optional[Dict[str, Dict[str, float]]]): Dictionary for
                initializing parameters. Keys are metabolite names or 'baseline'.
                Values are dicts, e.g., {'amp': 1.0, 'shift_hz': 0.0, 'lw_hz': 2.0}.
                For 'baseline', e.g., {'coeff0': 0.1, 'coeff1': -0.01}.
                These are the desired *constrained* initial values. Inverse transforms
                are applied to set the initial raw parameter values.
            constraints (Optional[Dict[str, float]]): Dictionary specifying constraints
                for parameters, e.g., {'max_shift_hz': 5.0, 'max_lw_hz': 10.0, 'min_lw_hz': 0.1}.
                These will be used to scale transformed raw parameters.
            baseline_degree (Optional[int]): Degree of the polynomial baseline to be fitted.
                If None, no baseline is fitted. Defaults to None.
            device (Optional[str]): Device to move tensors to ('cpu' or 'cuda').
                Defaults to 'cpu'.
        """
        super().__init__()

        self.metabolite_names = metabolite_names
        self.constraints = constraints if constraints is not None else {}
        self.baseline_degree = baseline_degree
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.dt = dt 

        # Convert basis_spectra_tensor to complex64 tensor
        if not isinstance(basis_spectra_tensor, torch.Tensor):
            basis_spectra_tensor = torch.from_numpy(basis_spectra_tensor)
        if not basis_spectra_tensor.is_complex():
            warnings.warn("Basis spectra tensor is not complex. Casting to complex64. "
                          "This assumes the input was meant to be complex or real-valued spectra "
                          "that can be safely cast. If real, ensure it represents the real part of an analytic signal if phase is critical.", UserWarning)
            basis_spectra_tensor = basis_spectra_tensor.to(torch.complex64)
        else: # Already complex, ensure correct dtype
            basis_spectra_tensor = basis_spectra_tensor.to(torch.complex64)


        # Convert observed_spectrum_tensor to float32 tensor, taking real part if complex
        if not isinstance(observed_spectrum_tensor, torch.Tensor):
            observed_spectrum_tensor = torch.from_numpy(observed_spectrum_tensor)
        if observed_spectrum_tensor.is_complex():
            warnings.warn("Observed spectrum tensor is complex. Using only its real part.", UserWarning)
            observed_spectrum_tensor = observed_spectrum_tensor.real
        observed_spectrum_tensor = observed_spectrum_tensor.to(torch.float32)


        if basis_spectra_tensor.ndim != 2:
            raise ValueError("basis_spectra_tensor must be 2D (num_points, num_metabolites).")
        if observed_spectrum_tensor.ndim != 1:
            raise ValueError("observed_spectrum_tensor must be 1D (num_points,).")
        if basis_spectra_tensor.shape[0] != observed_spectrum_tensor.shape[0]:
            raise ValueError("Basis spectra and observed spectrum must have the same number of points.")
        if len(metabolite_names) != basis_spectra_tensor.shape[1]:
            raise ValueError("Number of metabolite names must match the number of basis spectra.")

        self.num_points = basis_spectra_tensor.shape[0]
        self.num_metabolites = len(metabolite_names)

        self.register_buffer('basis_spectra_freq_shifted', basis_spectra_tensor.clone().detach().to(self.device))
        self.register_buffer('observed_spectrum_real_shifted', observed_spectrum_tensor.clone().detach().to(self.device))
        
        if fitting_mask is not None:
            if not isinstance(fitting_mask, torch.Tensor):
                fitting_mask = torch.from_numpy(fitting_mask) 
            if fitting_mask.dtype != torch.bool:
                 warnings.warn("Fitting mask is not boolean. Casting to boolean.", UserWarning)
                 fitting_mask = fitting_mask.bool()
            if fitting_mask.shape[0] != self.num_points:
                raise ValueError("Fitting mask must have the same number of points as the spectra.")
            self.register_buffer('fitting_mask', fitting_mask.clone().detach().to(self.device))
        else:
            self.register_buffer('fitting_mask', torch.ones(self.num_points, dtype=torch.bool, device=self.device))

        initial_params = initial_params if initial_params is not None else {}

        # Amplitudes: raw -> softplus -> final_amplitude
        default_amp = 1.0 
        initial_amp_vals_raw = torch.empty(self.num_metabolites, device=self.device, dtype=torch.float32)
        for i, name in enumerate(self.metabolite_names):
            amp_val = initial_params.get(name, {}).get('amp', default_amp)
            amp_val = max(amp_val, 1e-6) 
            initial_amp_vals_raw[i] = torch.log(torch.exp(torch.tensor(amp_val, device=self.device, dtype=torch.float32)) - 1.0)
        self.amplitudes_raw = nn.Parameter(initial_amp_vals_raw)

        # Frequency Shifts: raw -> tanh -> shift_hz * max_shift_constraint
        self.max_shift_hz_val = self.constraints.get('max_shift_hz', 5.0)
        self.register_buffer('max_shift_hz_tensor', torch.tensor(self.max_shift_hz_val, device=self.device, dtype=torch.float32))
        initial_shift_vals_raw = torch.empty(self.num_metabolites, device=self.device, dtype=torch.float32)
        for i, name in enumerate(self.metabolite_names):
            shift_val = initial_params.get(name, {}).get('shift_hz', 0.0)
            norm_shift = shift_val / self.max_shift_hz_val if self.max_shift_hz_val != 0 else 0.0
            norm_shift_clamped = torch.clamp(torch.tensor(norm_shift, device=self.device, dtype=torch.float32), -0.99999, 0.99999)
            initial_shift_vals_raw[i] = torch.atanh(norm_shift_clamped)
        self.shifts_hz_raw = nn.Parameter(initial_shift_vals_raw)

        # Linewidths: raw -> sigmoid -> min_lw + scaled_sigmoid * (max_lw - min_lw)
        self.min_lw_hz_val = self.constraints.get('min_lw_hz', 0.1)
        self.max_lw_hz_val = self.constraints.get('max_lw_hz', 10.0)
        if self.min_lw_hz_val >= self.max_lw_hz_val: self.max_lw_hz_val = self.min_lw_hz_val + 5.0 
        self.register_buffer('min_lw_hz_tensor', torch.tensor(self.min_lw_hz_val, device=self.device, dtype=torch.float32))
        self.register_buffer('lw_range_tensor', torch.tensor(self.max_lw_hz_val - self.min_lw_hz_val, device=self.device, dtype=torch.float32))
        
        initial_lw_vals_raw = torch.empty(self.num_metabolites, device=self.device, dtype=torch.float32) 
        for i, name in enumerate(self.metabolite_names):
            lw_val = initial_params.get(name, {}).get('lw_hz', self.min_lw_hz_val + self.lw_range_tensor.item() / 2.0)
            norm_lw = (lw_val - self.min_lw_hz_val) / self.lw_range_tensor.item() if self.lw_range_tensor.item() > 1e-6 else 0.5 
            norm_lw_clamped = torch.clamp(torch.tensor(norm_lw, device=self.device, dtype=torch.float32), 1e-7, 1.0 - 1e-7)
            initial_lw_vals_raw[i] = torch.log(norm_lw_clamped / (1.0 - norm_lw_clamped)) 
        self.linewidths_hz_raw = nn.Parameter(initial_lw_vals_raw)
        
        if self.baseline_degree is not None:
            if self.baseline_degree < 0: raise ValueError("Baseline degree cannot be negative.")
            num_bl_coeffs = self.baseline_degree + 1
            initial_bl_coeffs = torch.zeros(num_bl_coeffs, device=self.device, dtype=torch.float32)
            if 'baseline' in initial_params:
                for i in range(num_bl_coeffs):
                    initial_bl_coeffs[i] = initial_params['baseline'].get(f'coeff{i}', 0.0)
            self.baseline_coeffs_raw = nn.Parameter(initial_bl_coeffs)
        else:
            self.register_parameter('baseline_coeffs_raw', None)

        time_axis_np = np.arange(0, self.num_points) * self.dt
        self.register_buffer('time_axis', torch.tensor(time_axis_np, dtype=torch.float32, device=self.device))
        if self.baseline_degree is not None:
             freq_axis_normalized = torch.linspace(-1, 1, self.num_points, device=self.device, dtype=torch.float32)
             self.register_buffer('baseline_poly_terms', 
                                  torch.stack([freq_axis_normalized ** d for d in range(self.baseline_degree + 1)], dim=1))


    def forward(self) -> torch.Tensor:
        """
        Constructs the model spectrum from current parameters.
        
        The input basis spectra (`self.basis_spectra_freq_shifted`) are assumed to be 
        frequency-domain and correctly fftshifted. 
        The output model spectrum is also frequency-domain, fftshifted, real-valued, 
        and masked according to `self.fitting_mask`.
        """
        amplitudes = torch.nn.functional.softplus(self.amplitudes_raw)
        shifts_hz = torch.tanh(self.shifts_hz_raw) * self.max_shift_hz_tensor
        linewidths_hz = self.min_lw_hz_tensor + torch.sigmoid(self.linewidths_hz_raw) * self.lw_range_tensor
        
        basis_spectra_time_domain = torch.fft.ifft(torch.fft.ifftshift(self.basis_spectra_freq_shifted, dim=0), dim=0)
        
        time_axis_exp = self.time_axis.unsqueeze(1) 
        linewidths_hz_exp = linewidths_hz.unsqueeze(0) 
        shifts_hz_exp = shifts_hz.unsqueeze(0) 

        decay_matrix = torch.exp(-time_axis_exp * torch.pi * linewidths_hz_exp) 
        broadened_basis_time = basis_spectra_time_domain * decay_matrix
            
        phase_ramp_matrix = torch.exp(1j * 2 * torch.pi * shifts_hz_exp * time_axis_exp) 
        shifted_broadened_basis_time = broadened_basis_time * phase_ramp_matrix
            
        modified_basis_matrix_freq_shifted = torch.fft.fftshift(torch.fft.fft(shifted_broadened_basis_time, dim=0), dim=0)
        
        metabolite_sum_spectrum_complex = modified_basis_matrix_freq_shifted @ amplitudes.unsqueeze(1)
        metabolite_sum_spectrum_complex = metabolite_sum_spectrum_complex.squeeze(1) 

        model_spectrum_full_complex = metabolite_sum_spectrum_complex
        if self.baseline_coeffs_raw is not None and self.baseline_poly_terms is not None:
            baseline_component = self.baseline_poly_terms @ self.baseline_coeffs_raw
            model_spectrum_full_complex = model_spectrum_full_complex + baseline_component
        
        return model_spectrum_full_complex.real[self.fitting_mask]

    def get_transformed_parameters(self) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Returns a dictionary of the current *constrained* (interpretable) parameter values
        as detached tensors on their original device.
        """
        with torch.no_grad():
            params = {}
            params["amplitudes"] = torch.nn.functional.softplus(self.amplitudes_raw)
            params["shifts_hz"] = torch.tanh(self.shifts_hz_raw) * self.max_shift_hz_tensor
            params["linewidths_hz"] = self.min_lw_hz_tensor + torch.sigmoid(self.linewidths_hz_raw) * self.lw_range_tensor
            if self.baseline_coeffs_raw is not None:
                params["baseline_coeffs"] = self.baseline_coeffs_raw.clone()
        return params
    
    def get_fitted_amplitudes(self) -> Dict[str, float]:
        """Returns the current fitted (constrained) amplitudes as a dictionary (NumPy float values)."""
        amps = torch.nn.functional.softplus(self.amplitudes_raw.detach()).cpu().numpy()
        return {name: float(amps[i]) for i, name in enumerate(self.metabolite_names)}

    def get_fitted_shifts_hz(self) -> Dict[str, float]:
        """Returns the current fitted (constrained) frequency shifts in Hz as a dictionary (NumPy float values)."""
        shifts = (torch.tanh(self.shifts_hz_raw.detach()) * self.max_shift_hz_tensor).cpu().numpy()
        return {name: float(shifts[i]) for i, name in enumerate(self.metabolite_names)}

    def get_fitted_linewidths_hz(self) -> Dict[str, float]:
        """Returns the current fitted (constrained) linewidths in Hz as a dictionary (NumPy float values)."""
        lws = (self.min_lw_hz_tensor + torch.sigmoid(self.linewidths_hz_raw.detach()) * self.lw_range_tensor).cpu().numpy()
        return {name: float(lws[i]) for i, name in enumerate(self.metabolite_names)}

    def get_fitted_baseline_coeffs(self) -> Optional[np.ndarray]:
        """Returns the current fitted baseline coefficients (raw). Returns None if no baseline."""
        if self.baseline_coeffs_raw is not None:
            return self.baseline_coeffs_raw.detach().cpu().numpy()
        return None

    def get_full_model_spectrum(self, real_part: bool = True) -> torch.Tensor:
        """
        Constructs and returns the full model spectrum (all points, not masked).
        
        Args:
            real_part (bool): If True, returns the real part. Otherwise, returns complex spectrum.
                              Defaults to True.
        
        Returns:
            torch.Tensor: The full model spectrum, on the model's device.
        """
        with torch.no_grad(): 
            amplitudes = torch.nn.functional.softplus(self.amplitudes_raw)
            shifts_hz = torch.tanh(self.shifts_hz_raw) * self.max_shift_hz_tensor
            linewidths_hz = self.min_lw_hz_tensor + torch.sigmoid(self.linewidths_hz_raw) * self.lw_range_tensor

            basis_spectra_time_domain = torch.fft.ifft(torch.fft.ifftshift(self.basis_spectra_freq_shifted, dim=0), dim=0)
            
            time_axis_exp = self.time_axis.unsqueeze(1)
            linewidths_hz_exp = linewidths_hz.unsqueeze(0)
            shifts_hz_exp = shifts_hz.unsqueeze(0)

            decay_matrix = torch.exp(-time_axis_exp * torch.pi * linewidths_hz_exp)
            broadened_basis_time = basis_spectra_time_domain * decay_matrix
            
            phase_ramp_matrix = torch.exp(1j * 2 * torch.pi * shifts_hz_exp * time_axis_exp)
            shifted_broadened_basis_time = broadened_basis_time * phase_ramp_matrix
                
            modified_basis_matrix_freq_shifted = torch.fft.fftshift(torch.fft.fft(shifted_broadened_basis_time, dim=0), dim=0)
            
            metabolite_sum_spectrum_complex = modified_basis_matrix_freq_shifted @ amplitudes.unsqueeze(1)
            metabolite_sum_spectrum_complex = metabolite_sum_spectrum_complex.squeeze(1)

            model_spectrum_full_complex = metabolite_sum_spectrum_complex
            if self.baseline_coeffs_raw is not None and self.baseline_poly_terms is not None:
                baseline_component = self.baseline_poly_terms @ self.baseline_coeffs_raw
                model_spectrum_full_complex = model_spectrum_full_complex + baseline_component
            
        return model_spectrum_full_complex.real if real_part else model_spectrum_full_complex


    def fit(self, num_iterations: int = 1000, lr: float = 1e-2, 
            loss_fn_type: str = 'mse', optim_type: str = 'adam', 
            print_loss_every: int = 100, weight_decay: float = 0.0):
        """
        Fits the model to the observed spectrum using PyTorch's autograd.

        Args:
            num_iterations (int): Number of optimization iterations.
            lr (float): Learning rate for the optimizer.
            loss_fn_type (str): Type of loss function ('mse' for Mean Squared Error).
            optim_type (str): Type of optimizer ('adam', 'lbfgs').
            print_loss_every (int): Print loss every N iterations. If <=0, no printing.
            weight_decay (float): L2 penalty (weight decay) for Adam optimizer.
        """
        self.train() 

        if optim_type.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim_type.lower() == 'lbfgs':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=20) 
        else:
            raise ValueError(f"Unsupported optimizer type: {optim_type}")

        if loss_fn_type.lower() == 'mse':
            loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function type: {loss_fn_type}")

        target_spectrum = self.observed_spectrum_real_shifted[self.fitting_mask]

        for i in range(num_iterations):
            def closure():
                optimizer.zero_grad()
                model_output_masked = self.forward() 
                loss = loss_fn(model_output_masked, target_spectrum)
                loss.backward()
                return loss

            if optim_type.lower() == 'lbfgs':
                loss = optimizer.step(closure) 
            else: # Adam
                optimizer.zero_grad()
                model_output_masked = self.forward()
                loss = loss_fn(model_output_masked, target_spectrum)
                loss.backward()
                optimizer.step()
            
            if print_loss_every > 0 and (i % print_loss_every == 0 or i == num_iterations - 1):
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6e}")
        
        self.eval() 
        print("Fitting complete.")
