import numpy as np
import torch
import pytest

from preprocessing.corrections import phase_correct_gpu, make_bspline_basis, make_fourier_basis

def test_phase_correct_gpu_zero_order():
    # Use small synthetic data: shape (T, F, X, Y, Z)
    data = (np.random.randn(2, 8, 2, 2, 1) + 1j * np.random.randn(2, 8, 2, 2, 1)).astype(np.complex64)
    corrected, params = phase_correct_gpu(data, method="Zero-order", n_iters=2)
    assert corrected.shape == data.shape
    assert params.shape[0] == 2

def test_make_bspline_basis():
    B = make_bspline_basis(16, peak_list=[4, 8], spacing=4, half_peak_width=2, degree=3)
    assert isinstance(B, torch.Tensor)
    assert B.shape[0] == 16

def test_make_fourier_basis():
    B = make_fourier_basis(16, 3)
    assert isinstance(B, torch.Tensor)
    assert B.shape[0] == 16
