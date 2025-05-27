import numpy as np
import torch
import pytest

from denoising.algorithms import apply_spectral_denoising_batch, PEwithPeak

def test_apply_spectral_denoising_batch_mean_filter():
    # Shape: (T, F, X, Y, Z)
    arr = np.random.rand(1, 16, 1, 1, 1)
    denoised = apply_spectral_denoising_batch(arr, 'Mean Filter', window_size=3)
    assert denoised.shape == (16,)

def test_pewithpeak_forward():
    batch_size = 2
    seq_len = 8
    embed_dim = 4
    x = torch.randn(seq_len, batch_size, embed_dim)
    peak_positions = torch.tensor([[1, 3, -1, -1], [0, 2, -1, -1]])
    pe = PEwithPeak(embed_dim=embed_dim, max_len=seq_len, num_peaks=4)
    out = pe(x, peak_positions)
    assert out.shape == x.shape
