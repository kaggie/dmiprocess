import torch
import torch.nn.functional as F # Not explicitly used in the provided snippet, but often used with torch
import numpy as np
from scipy.interpolate import BSpline # Used in make_bspline_basis
from scipy.stats import norm # Not explicitly used in the provided snippet, but often used with scipy

# Global DEVICE, can be updated by the main application if necessary.
# For now, mimicking the structure observed.
# Consider passing device as a parameter to functions for better modularity.
DEVICE = torch.device('cpu') 

def phase_correct_gpu(data, method="Zero-order", num_basis=8, lr=0.05, n_iters=200,
                      std_range=4, peak_list=None, half_peak_width=5, degree=3
                     ):

    assert method in ["Zero-order", "First-order", "B-spline", "Fourier"]

    # Ensure data is a PyTorch tensor. If numpy, convert it.
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(DEVICE) # Move to device after conversion
    elif isinstance(data, torch.Tensor):
        data = data.to(DEVICE) # Ensure it's on the correct device
    else:
        raise TypeError("Input data must be a NumPy array or PyTorch tensor.")


    T, F, X, Y, Z = data.shape[:5]
    N = X * Y * Z * T
    # device = data.device # Already set by global DEVICE or data's device
    data_flat = data.view(N, F)

    freq = torch.linspace(0, 1, F, device=DEVICE) # Use global DEVICE
    # freq_batch = freq.unsqueeze(0).repeat(N, 1) # freq_batch not used, freq is used in lambda for First-order

    if method == "Zero-order":
        params = torch.zeros(N, 1, device=DEVICE, requires_grad=True)
        phase_fn = lambda p: p
    elif method == "First-order":
        params = torch.zeros(N, 2, device=DEVICE, requires_grad=True)
        phase_fn = lambda p: p[:, [0]] + p[:, [1]] * freq # Use freq here
    elif method == "B-spline":
        B = make_bspline_basis(F, peak_list, spacing=64, half_peak_width=half_peak_width, degree=degree).to(DEVICE)
        B = B.unsqueeze(0).expand(N, -1, -1)
        params = torch.zeros(N, B.shape[2], device=DEVICE, requires_grad=True)
        phase_fn = lambda p: torch.bmm(B, p.unsqueeze(-1)).squeeze(-1)
    elif method == "Fourier":
        B = make_fourier_basis(F, num_basis).to(DEVICE)
        B = B.unsqueeze(0).expand(N, -1, -1)
        params = torch.zeros(N, B.shape[2], device=DEVICE, requires_grad=True)
        phase_fn = lambda p: torch.bmm(B, p.unsqueeze(-1)).squeeze(-1)
    else:
        raise ValueError("Unsupported method")

    optimizer = torch.optim.Adam([params], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad()

        phi = phase_fn(params)
        s_corr = data_flat * torch.exp(-1j * phi)

        imag = torch.imag(s_corr)
        # The original code might have intended a specific region for loss calculation (e.g. around peaks)
        # For now, using sum over all frequencies as in the original snippet.
        # Consider if std_range was meant to be used here.
        imag_loss = torch.sum(imag ** 2, dim=1)

        loss = imag_loss.mean()
        loss.backward()
        optimizer.step()

    final_phi = phase_fn(params)
    corrected = data_flat * torch.exp(-1j * final_phi)
    corrected_data = corrected.view(T, F, X, Y, Z)
    
    # Detach from graph and convert to numpy for output, as GUI expects numpy
    return corrected_data.cpu().detach().numpy(), params.view(T, -1, X, Y, Z).cpu().detach().numpy()


def make_bspline_basis(num_freqs, peak_list, spacing=64, half_peak_width=10, degree=3):

    peak_bins = [peak for peak in peak_list] if peak_list else []

    knot_bins = list(range(0, num_freqs, spacing))

    for peak in peak_bins:
        local_knots = [peak-half_peak_width, peak+half_peak_width]
        #local_knots = [peak + offset for offset in range(-half_peak_width, half_peak_width + 1)] # Original commented out
        knot_bins.extend(local_knots)

    knot_bins = sorted(set(np.clip(knot_bins, 0, num_freqs - 1)))
    custom_knots = np.array(knot_bins) / (num_freqs - 1) # Normalized knot positions

    # Pad knots for B-spline construction
    full_knots = np.pad(custom_knots, (degree, degree), mode='edge') # 'reflect' or 'symmetric' might also be options
    num_basis = len(full_knots) - degree - 1

    freq_norm = np.linspace(0, 1, num_freqs) # Normalized frequencies for spline evaluation
    basis_list = []
    for i in range(num_basis):
        coeffs = np.zeros(num_basis)
        coeffs[i] = 1.0 # Ensure float for coefficients
        spline = BSpline(full_knots, coeffs, degree, extrapolate=False) # extrapolate=False is safer
        basis_list.append(spline(freq_norm))
    B = np.stack(basis_list, axis=1)
    return torch.tensor(B, dtype=torch.float32)


def make_fourier_basis(num_freqs, num_terms):
    x = torch.linspace(0, 1, num_freqs, device=DEVICE) # Ensure tensor is on the correct device
    basis = [torch.ones_like(x)]
    for k in range(1, num_terms + 1):
        basis.append(torch.sin(2 * np.pi * k * x))
        basis.append(torch.cos(2 * np.pi * k * x))
    B = torch.stack(basis, dim=1)
    return B
