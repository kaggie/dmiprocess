import numpy as np

def apply_norm(data, method, bg_data):
    if method == "Max Abs":
        # Ensure denominator is not zero and handle all-zero data
        max_abs_val = np.abs(data).max()
        if max_abs_val == 0:
            data_norm = data # Or handle as an error/specific case
        else:
            data_norm= data / max_abs_val
    elif method == "Min-Max":
        min_val = data.min()
        max_val = data.max()
        if max_val == min_val: # Avoid division by zero if all values are the same
            data_norm = np.zeros_like(data) if max_val == 0 else np.ones_like(data) * data / max_val if max_val != 0 else np.zeros_like(data) # Or handle as an error/specific case
        else:
            data_norm = (data - min_val) / (max_val - min_val)
    elif method == "Background Mean Scaling":
        if bg_data is None:
            raise ValueError("Background data must be provided for Background Mean Scaling.")
        mean = bg_data.mean()
        if mean == 0:
            raise ValueError("Background mean is zero, cannot perform scaling.")
        data_norm = data / mean
    elif method == "Background Z-score":
        if bg_data is None:
            raise ValueError("Background data must be provided for Background Z-score.")
        mean = bg_data.mean()
        std = bg_data.std()
        if std == 0:
            # Handle case where background std is zero (e.g., constant background)
            # data_norm = data - mean # Or raise an error, or return data as is
            raise ValueError("Background standard deviation is zero, cannot perform Z-score.")
        data_norm= (data - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return data_norm

def normalize_data(data, method, mode, bg_data=None, flag_complex=True):
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if bg_data is not None and not isinstance(bg_data, np.ndarray):
        raise TypeError("Background data must be a NumPy array if provided.")

    if not flag_complex:
        data_norm = apply_norm(data.astype(np.float64), method, bg_data=bg_data.astype(np.float64) if bg_data is not None else None) # Ensure float for calculations
    else:
        if mode == "Magnitude Only":
            magnitude = np.abs(data)
            magnitude_bg = np.abs(bg_data) if bg_data is not None else None
            phase = np.angle(data)
            magnitude_norm = apply_norm(magnitude.astype(np.float64), method, bg_data=magnitude_bg.astype(np.float64) if magnitude_bg is not None else None)
            data_norm = magnitude_norm * np.exp(1j * phase)
        elif mode == "Real and Imaginary Separately":
            real_part = np.real(data)
            real_bg = np.real(bg_data) if bg_data is not None else None
            real_norm = apply_norm(real_part.astype(np.float64), method, bg_data=real_bg.astype(np.float64) if real_bg is not None else None)
            
            imag_part = np.imag(data)
            imag_bg = np.imag(bg_data) if bg_data is not None else None
            imag_norm = apply_norm(imag_part.astype(np.float64), method, bg_data=imag_bg.astype(np.float64) if imag_bg is not None else None)
            data_norm = real_norm + 1j * imag_norm
        elif mode == "Complex as Whole":
            # This mode might be problematic for methods like Min-Max or Z-score on complex numbers directly.
            # Max Abs is generally fine. For others, it's ambiguous without specific domain knowledge.
            # The original apply_norm doesn't inherently handle complex numbers for all methods.
            # For now, applying as is, but this might need refinement.
            if method not in ["Max Abs", "Background Mean Scaling"]: # Methods that might be more straightforward for complex
                 print(f"Warning: Applying method '{method}' to complex data as a whole might yield unexpected results.")
            data_norm = apply_norm(data.astype(np.complex128), method, bg_data=bg_data.astype(np.complex128) if bg_data is not None else None)
        else:
            raise ValueError(f"Unknown mode for complex data normalization: {mode}")
            
    return data_norm
