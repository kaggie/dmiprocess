#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pywt
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter
from scipy.signal import savgol_filter, wiener
from skimage.restoration import denoise_tv_chambolle
from sklearn.decomposition import PCA

def apply_spectral_denoising_batch(data, method, **kwargs):
    
    T, F, X, Y, Z = data.shape[:5]
    data = data.transpose(0, 2, 3, 4, 1)
    spectra = data.reshape(-1, F)
    N = spectra.shape[0]

    if method == 'Mean Filter':
        size = kwargs.get('window_size', 5)
        return uniform_filter1d(spectra, size=size, axis=1, mode='nearest')
    
    elif method == 'Median Filter':
        size = kwargs.get('window_size', 5)
        return median_filter(spectra, size=(1, size), mode='nearest')

    elif method == 'Gaussian Filter':
        sigma = kwargs.get('sigma', 1.0)
        return gaussian_filter1d(spectra, sigma=sigma, axis=1, mode='nearest')
    
    elif method == 'Singular Value Decomposition':
        n_components = kwargs.get('num_components', 5)  # How many singular vectors to keep
        U, S, Vh = np.linalg.svd(spectra, full_matrices=False)
        S[n_components:] = 0  # Zero out small singular values
        denoised = (U * S) @ Vh
        return denoised
    
    elif method == 'Principal Component Analysis':
        n_components = kwargs.get('num_components', 5)
        pca = PCA(n_components=n_components)
        denoised = pca.fit_transform(spectra)
        return pca.inverse_transform(denoised)

    elif method == 'Savitzky-Golay Filter':
        window_length = kwargs.get('window_size', 7)
        polyorder = kwargs.get('polyorder', 2)
        # Ensure window_length is odd and less than or equal to F
        window_length = min(window_length, F // 2 * 2 + 1)
        if window_length % 2 == 0:
            window_length += 1
        return np.stack([
            savgol_filter(s, window_length, polyorder, mode='nearest') for s in spectra
        ])

    elif method == 'Wavelet Thresholding':
        #Selects the wavelet type, defaulting to 'db4' (Daubechies 4)
        #Common alternatives: 'haar', 'sym4', 'coif1', etc.
        wavelet = kwargs.get('wavelet', 'db4')        
        threshold = kwargs.get('threshold', 0.04)
        mode = kwargs.get('mode', 'soft')
        #Controls how many levels of wavelet decomposition to perform.
        #If None, automatically choose the maximum level based on signal length and wavelet.
        #Lower levels = more local detail; higher levels = more global smoothing.
        level = kwargs.get('level', None)
        denoised = []
        for s in spectra:
            coeffs = pywt.wavedec(s, wavelet, level=level)
            coeffs_thresh = [pywt.threshold(c, threshold, mode=mode) if i > 0 else c
                             for i, c in enumerate(coeffs)]
            s_denoised = pywt.waverec(coeffs_thresh, wavelet)
            denoised.append(s_denoised[:F])  # Ensure output length matches
        return np.stack(denoised)

    elif method == 'Fourier Filter':
        cutoff = kwargs.get('cutoff_freq', 0.1)  # fraction of frequencies to keep
        denoised = []
        for s in spectra:
            S = np.fft.fft(s)
            freq_cut = int(len(S) * cutoff)
            S[freq_cut:-freq_cut] = 0
            s_denoised = np.fft.ifft(S).real
            denoised.append(s_denoised)
        return np.stack(denoised)

    elif method == 'Total Variation':
        weight = kwargs.get('weight', 0.1)
        return np.stack([
            denoise_tv_chambolle(s, weight=weight) for s in spectra
        ])
    
    elif method == 'Wiener Filter':
        kernel_size = kwargs.get('kernel_size', 5)
        return np.stack([
            wiener(s, mysize=kernel_size) for s in spectra
        ])

    else:
        raise ValueError(f"Unsupported denoising method: {method}")