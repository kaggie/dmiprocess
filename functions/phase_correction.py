#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import norm

def phase_correct_gpu(data, method="Zero-order", num_basis=8, lr=0.05, n_iters=200, 
                      std_range=4, peak_list=None, half_peak_width=5, degree=3
                     ):
    
    assert method in ["Zero-order", "First-order", "B-spline", "Fourier"]
    
    data = torch.from_numpy(data)
    T, F, X, Y, Z = data.shape[:5]
    N = X * Y * Z * T
    device = data.device
    data_flat = data.view(N, F)

    freq = torch.linspace(0, 1, F, device=device)
    freq_batch = freq.unsqueeze(0).repeat(N, 1)

    if method == "Zero-order":
        params = torch.zeros(N, 1, device=device, requires_grad=True)
        #def phase_fn(p): return p
        phase_fn = lambda p: p
    elif method == "First-order":
        params = torch.zeros(N, 2, device=device, requires_grad=True)
        #def phase_fn(p): return p[:, [0]] + p[:, [1]] * freq_batch
        phase_fn = lambda p: p[:, [0]] + p[:, [1]] * freq
    elif method == "B-spline":
        B = make_bspline_basis(F, peak_list, spacing=64, half_peak_width=half_peak_width, degree=degree).to(device)
        B = B.unsqueeze(0).expand(N, -1, -1)
        params = torch.zeros(N, B.shape[2], device=device, requires_grad=True)
        #def phase_fn(p): return torch.bmm(B, p.unsqueeze(-1)).squeeze(-1)
        phase_fn = lambda p: torch.bmm(B, p.unsqueeze(-1)).squeeze(-1)
    elif method == "Fourier":
        B = make_fourier_basis(F, num_basis).to(device)
        B = B.unsqueeze(0).expand(N, -1, -1)
        params = torch.zeros(N, B.shape[2], device=device, requires_grad=True)
        #def phase_fn(p): return torch.bmm(B, p.unsqueeze(-1)).squeeze(-1)
        phase_fn = lambda p: torch.bmm(B, p.unsqueeze(-1)).squeeze(-1)
    else:
        raise ValueError("Unsupported method")

    optimizer = torch.optim.Adam([params], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad()
        
        phi = phase_fn(params)
        s_corr = data_flat * torch.exp(-1j * phi)
        
        imag = torch.imag(s_corr)
        imag_loss = torch.sum(imag ** 2, dim=1)

        loss = imag_loss.mean()
        loss.backward()
        optimizer.step()

    final_phi = phase_fn(params)
    corrected = data_flat * torch.exp(-1j * final_phi)
    corrected_data = corrected.view(T, F, X, Y, Z)
    return corrected_data.detach().numpy(), params.view(T, -1, X, Y, Z)


def make_bspline_basis(num_freqs, peak_list, spacing=64, half_peak_width=10, degree=3):

    peak_bins = [peak for peak in peak_list] if peak_list else []

    knot_bins = list(range(0, num_freqs, spacing))
    
    for peak in peak_bins:
        local_knots = [peak-half_peak_width, peak+half_peak_width]
        #local_knots = [peak + offset for offset in range(-half_peak_width, half_peak_width + 1)]
        knot_bins.extend(local_knots)

    knot_bins = sorted(set(np.clip(knot_bins, 0, num_freqs - 1)))
    custom_knots = np.array(knot_bins) / (num_freqs - 1)

    full_knots = np.pad(custom_knots, (degree, degree), mode='edge')
    num_basis = len(full_knots) - degree - 1
    
    freq_norm = np.linspace(0, 1, num_freqs)
    basis_list = []
    for i in range(num_basis):
        coeffs = np.zeros(num_basis)
        coeffs[i] = 1
        spline = BSpline(full_knots, coeffs, degree)
        basis_list.append(spline(freq_norm))
    B = np.stack(basis_list, axis=1)
    return torch.tensor(B, dtype=torch.float32)


def make_fourier_basis(num_freqs, num_terms):
    x = torch.linspace(0, 1, num_freqs)
    basis = [torch.ones_like(x)]
    for k in range(1, num_terms + 1):
        basis.append(torch.sin(2 * np.pi * k * x))
        basis.append(torch.cos(2 * np.pi * k * x))
    B = torch.stack(basis, dim=1)
    return B
