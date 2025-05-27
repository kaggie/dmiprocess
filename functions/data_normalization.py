#!/usr/bin/env python
# coding: utf-8

import numpy as np

def apply_norm(data, method, bg_data):
    if method == "Max Abs":
        data_norm= data / np.abs(data).max()
    elif method == "Min-Max":
        min_val = data.min()
        max_val = data.max()
        data_norm = (data - min_val) / (max_val - min_val)  
    elif method == "Background Mean Scaling":
        mean = bg_data.mean()
        data_norm = data / mean 
    elif method == "Background Z-score":
        mean = bg_data.mean()
        std = bg_data.std()
        data_norm= (data - mean) / std

    return data_norm

def normalize_data(data, method, mode, bg_data=None, flag_complex=True):   
    if not flag_complex:
        data_norm = apply_norm(data, method, bg_data=bg_data)
    else:
        if mode == "Magnitude Only":
            magnitude = np.abs(data)
            magnitude_bg = np.abs(bg_data) if bg_data is not None else None
            phase = np.angle(data)
            magnitude_norm = apply_norm(magnitude, method, bg_data=magnitude_bg)
            data_norm = magnitude_norm * np.exp(1j * phase)
        elif mode == "Real and Imaginary Separately":
            real = np.real(data)
            real_bg = np.real(bg_data) if bg_data is not None else None
            real_norm = apply_norm(real, method, bg_data=real_bg)
            imag = np.imag(data)
            imag_bg = np.imag(bg_data) if bg_data is not None else None
            imag_norm = apply_norm(imag, method, bg_data=imag_bg)
            data_norm = real_norm + 1j * imag_norm
        elif mode == "Complex as Whole":
            complex_bg = bg_data if bg_data is not None else None
            data_norm = apply_norm(data, method, bg_data=complex_bg)

    return data_norm      
  