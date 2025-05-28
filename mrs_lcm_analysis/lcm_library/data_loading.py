import numpy as np
import torch
import scipy.fft
import matplotlib.pyplot as plt
from typing import Optional, Dict, Union

class MRSData:
    """
    Represents MRS data, capable of handling both time-domain (FID)
    and frequency-domain (spectrum) data.
    """

    def __init__(self,
                 data_array: np.ndarray,
                 data_type: str,
                 sampling_frequency: Optional[float] = None,
                 central_frequency: Optional[float] = None,
                 metadata: Optional[Dict[str, Union[str, float, int]]] = None):
        """
        Initializes an MRSData object.

        Args:
            data_array (np.ndarray): 1D NumPy array of MRS data.
            data_type (str): "time" or "frequency".
            sampling_frequency (float, optional): Sampling frequency in Hz.
                                                 Required for time-domain data or conversions.
            central_frequency (float, optional): Central frequency in Hz.
                                                 Required for ppm scale generation.
            metadata (dict, optional): Dictionary for other metadata.
        """
        if not isinstance(data_array, np.ndarray) or data_array.ndim != 1:
            raise TypeError("Data array must be a 1D NumPy array.")
        if data_type not in ["time", "frequency"]:
            raise ValueError("data_type must be 'time' or 'frequency'.")
        if data_type == "time" and sampling_frequency is None:
            raise ValueError("sampling_frequency is required for time-domain data.")

        self.data_array = data_array
        self.data_type = data_type.lower()
        self.sampling_frequency = sampling_frequency
        self.central_frequency = central_frequency
        self.metadata = metadata if metadata is not None else {}

    @property
    def is_time_domain(self) -> bool:
        """True if data is time-domain, False otherwise."""
        return self.data_type == "time"

    @property
    def is_frequency_domain(self) -> bool:
        """True if data is frequency-domain, False otherwise."""
        return self.data_type == "frequency"

    def get_time_domain_data(self) -> np.ndarray:
        """
        Returns the time-domain representation of the data.
        Performs iFFT if current data is frequency-domain.
        """
        if self.is_time_domain:
            return self.data_array
        else:
            # Use scipy.fft.ifft for simplicity here
            # PyTorch's ifft expects complex input, ensure data_array is complex
            if np.isrealobj(self.data_array):
                # If it's purely real, it implies it might be a magnitude spectrum
                # or only half of a symmetric spectrum. For a true inverse,
                # we'd need the complex spectrum. Assuming it's from fftshift.
                complex_spectrum = scipy.fft.ifftshift(self.data_array)
            else:
                complex_spectrum = scipy.fft.ifftshift(self.data_array)
            
            time_domain_signal = scipy.fft.ifft(complex_spectrum)
            # If the original frequency domain data was from a real time-domain signal,
            # its iFFT should be real. We take the real part to discard small imaginary
            # parts due to numerical precision.
            return np.real(time_domain_signal)


    def get_frequency_domain_data(self, apply_fftshift: bool = True) -> np.ndarray:
        """
        Returns the frequency-domain representation of the data.
        Performs FFT if current data is time-domain.

        Args:
            apply_fftshift (bool): Whether to apply fftshift to center the spectrum.
                                   Default is True.

        Returns:
            np.ndarray: The frequency domain spectrum.
        """
        if self.is_frequency_domain:
            return self.data_array
        else:
            if self.sampling_frequency is None:
                raise ValueError("sampling_frequency is required to convert time-domain data to frequency-domain.")
            # Use scipy.fft.fft
            spectrum = scipy.fft.fft(self.data_array)
            if apply_fftshift:
                spectrum = scipy.fft.fftshift(spectrum)
            return spectrum

    def get_frequency_axis(self, unit: str = 'ppm') -> np.ndarray:
        """
        Calculates and returns a frequency axis.

        Args:
            unit (str): The desired unit for the frequency axis ('ppm' or 'hz').
                        Defaults to 'ppm'.

        Returns:
            np.ndarray: The frequency axis.

        Raises:
            ValueError: If required parameters (sampling_frequency for 'hz' or 'ppm',
                        central_frequency for 'ppm') are missing.
            ValueError: If an invalid unit is specified.
        """
        if self.sampling_frequency is None:
            raise ValueError("sampling_frequency is required to generate a frequency axis.")

        n_points = len(self.data_array)
        # Frequency axis in Hz, centered at 0 after fftshift
        hz_axis = np.linspace(-self.sampling_frequency / 2, self.sampling_frequency / 2 - self.sampling_frequency / n_points, n_points)

        if unit.lower() == 'hz':
            return hz_axis
        elif unit.lower() == 'ppm':
            if self.central_frequency is None:
                raise ValueError("central_frequency is required for ppm scale.")
            # PPM = ((freq_hz - 0) / central_freq_MHz)
            # The 0 Hz in hz_axis corresponds to the central_frequency (transmitter frequency)
            # A frequency f in hz_axis is f Hz away from the central_frequency.
            # So, the absolute frequency is central_frequency + f.
            # ppm = (absolute_frequency - reference_frequency) / reference_frequency_MHz
            # For MRS, the reference frequency is usually the transmitter frequency (central_frequency)
            # and the scale is (freq_hz_offset_from_center / central_frequency_MHz)
            # The typical MRS convention has higher ppm values for frequencies lower than the water reference.
            # So, ppm = - ( (hz_axis * 1e-6) / (self.central_frequency * 1e-6) ) * 1e6 = -hz_axis / self.central_frequency
            # No, this is simpler: ppm = ( (f_obs - f_ref) / f_ref ) * 1e6
            # f_obs = central_frequency + hz_axis
            # f_ref = central_frequency (for water referencing, typically)
            # ppm = ( (central_frequency + hz_axis) - central_frequency ) / central_frequency * 1e6 = (hz_axis / self.central_frequency) * 1e6
            # To match convention (e.g. NAA at 2.01 ppm, water at 4.7 ppm), higher frequencies are to the left (lower ppm values).
            # So if NAA is at a frequency slightly less than water, its offset from water (0 Hz in hz_axis) is negative.
            # ppm = - (hz_axis / self.central_frequency) * 1e6 + reference_ppm_shift (e.g. 4.7 for water)
            # Let's use a simpler convention for now: (frequency_hz / Larmor_frequency_MHz)
            # The hz_axis is already relative to the central frequency.
            # A common convention in spectroscopy: ppm_value = (frequency_offset_Hz / spectrometer_base_frequency_MHz)
            # And often the scale is reversed.
            ppm_axis = (hz_axis / self.central_frequency) * 1e6
            # MRS spectra are typically displayed with high ppm on left, low ppm on right.
            # This means reversing the ppm axis relative to the frequency axis.
            return ppm_axis[::-1] # Reversing for typical display
        else:
            raise ValueError("Invalid unit for frequency axis. Choose 'ppm' or 'hz'.")

    def plot(self, unit: str = 'ppm', apply_fftshift_to_freq: bool = True):
        """
        Plots the MRS data.

        If time-domain, plots the FID.
        If frequency-domain, plots the spectrum against the calculated frequency axis.

        Args:
            unit (str): The unit for the frequency axis if data is frequency-domain
                        or needs conversion ('ppm' or 'hz'). Defaults to 'ppm'.
            apply_fftshift_to_freq (bool): If converting from time to frequency for plotting,
                                           whether to apply fftshift. Default is True.
        """
        plt.figure()
        if self.is_time_domain:
            n_points = len(self.data_array)
            if self.sampling_frequency:
                time_axis = np.arange(n_points) / self.sampling_frequency
                plt.plot(time_axis, self.data_array.real) # Plot real part for FID
                if np.any(self.data_array.imag): # Plot imaginary if it exists
                    plt.plot(time_axis, self.data_array.imag, label="Imaginary Part")
                plt.xlabel("Time (s)")
            else:
                plt.plot(self.data_array.real)
                if np.any(self.data_array.imag):
                    plt.plot(self.data_array.imag, label="Imaginary Part")
                plt.xlabel("Index")
            plt.ylabel("Signal Intensity")
            plt.title("Time-Domain Data (FID)")
            if np.any(self.data_array.imag): plt.legend()

        elif self.is_frequency_domain:
            spectrum_to_plot = self.data_array
            try:
                freq_axis = self.get_frequency_axis(unit=unit)
                plt.plot(freq_axis, spectrum_to_plot.real) # Plot real part of spectrum
                if np.any(spectrum_to_plot.imag):
                     plt.plot(freq_axis, spectrum_to_plot.imag, label="Imaginary Part", alpha=0.7)
                plt.xlabel(f"Frequency ({unit})")
                if unit == 'ppm': # Reverse ppm axis for typical display
                    plt.xlim(max(freq_axis), min(freq_axis))
            except ValueError as e:
                print(f"Plotting error: {e}. Plotting against index instead.")
                plt.plot(spectrum_to_plot.real)
                if np.any(spectrum_to_plot.imag):
                     plt.plot(spectrum_to_plot.imag, label="Imaginary Part", alpha=0.7)
                plt.xlabel("Index")
            plt.ylabel("Signal Intensity")
            plt.title("Frequency-Domain Data (Spectrum)")
            if np.any(spectrum_to_plot.imag): plt.legend()

        else: # Should not happen given constructor checks
            plt.title("Unknown Data Type")

        plt.grid(True, alpha=0.3)
        plt.show()

    def __repr__(self):
        return (f"MRSData(data_type='{self.data_type}', "
                f"num_points={len(self.data_array)}, "
                f"sampling_freq={self.sampling_frequency} Hz, "
                f"central_freq={self.central_frequency} Hz)")

```
