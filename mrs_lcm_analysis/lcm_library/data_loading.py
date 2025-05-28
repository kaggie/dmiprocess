import numpy as np
import scipy.fft # Keep scipy for FFT, torch is not essential for this class
import matplotlib.pyplot as plt
from typing import Optional, Dict, Union, Any, List # Added Any, List

class MRSData:
    """
    Represents MRS data, capable of handling both time-domain (FID)
    and frequency-domain (spectrum) data.
    """

    def __init__(self,
                 data_array: np.ndarray,
                 data_type: str,
                 sampling_frequency: Optional[float] = None,
                 central_frequency: Optional[float] = None, # This is Larmor/transmitter frequency
                 echo_time_ms: Optional[float] = None,
                 repetition_time_ms: Optional[float] = None,
                 metadata: Optional[Dict[str, Union[str, float, int, List[Any], Dict[Any, Any]]]] = None):
        """
        Initializes an MRSData object.

        Args:
            data_array (np.ndarray): 1D NumPy array of MRS data (FID or spectrum).
            data_type (str): Type of data provided, must be "time" or "frequency".
            sampling_frequency (float, optional): Sampling frequency in Hz.
                                                 Required if data_type is "time" or for
                                                 conversions between domains. Defaults to None.
            central_frequency (float, optional): Spectrometer (Larmor/transmitter) frequency in MHz.
                                                 Required for generating a PPM axis. Defaults to None.
            echo_time_ms (float, optional): Echo time (TE) in milliseconds. Defaults to None.
            repetition_time_ms (float, optional): Repetition time (TR) in milliseconds. Defaults to None.
            metadata (dict, optional): Dictionary for storing any other relevant metadata.
                                       Values can be simple types or lists/dicts. Defaults to None.
        """
        if not isinstance(data_array, np.ndarray) or data_array.ndim != 1:
            raise TypeError("Data array must be a 1D NumPy array.")
        if data_type.lower() not in ["time", "frequency"]:
            raise ValueError("data_type must be 'time' or 'frequency'.")
        if data_type.lower() == "time" and sampling_frequency is None:
            # Allow creation if user intends to set it later, but conversions will fail.
            # Consider a warning instead of an error, or make it strictly required.
            # For now, keeping the strict check as conversions are a core part.
            raise ValueError("sampling_frequency is required for time-domain data if conversions are expected.")
        if sampling_frequency is not None and not isinstance(sampling_frequency, (float, int)):
            raise TypeError("sampling_frequency must be a number (float or int).")
        if central_frequency is not None and not isinstance(central_frequency, (float, int)):
            raise TypeError("central_frequency must be a number (float or int).")
        if echo_time_ms is not None and not isinstance(echo_time_ms, (float, int)):
            raise TypeError("echo_time_ms must be a number (float or int).")
        if repetition_time_ms is not None and not isinstance(repetition_time_ms, (float, int)):
            raise TypeError("repetition_time_ms must be a number (float or int).")


        self.data_array = data_array
        self.data_type = data_type.lower()
        self.sampling_frequency = float(sampling_frequency) if sampling_frequency is not None else None
        self.central_frequency = float(central_frequency) if central_frequency is not None else None
        self.echo_time_ms = float(echo_time_ms) if echo_time_ms is not None else None
        self.repetition_time_ms = float(repetition_time_ms) if repetition_time_ms is not None else None
        self.metadata = metadata if metadata is not None else {}

    @property
    def dwell_time(self) -> Optional[float]:
        """Dwell time in seconds, calculated as 1/sampling_frequency. Returns None if sampling_frequency is not set."""
        if self.sampling_frequency and self.sampling_frequency > 0:
            return 1.0 / self.sampling_frequency
        return None

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
        Assumes frequency domain data was correctly fftshifted if it originated from time domain.
        """
        if self.is_time_domain:
            return self.data_array.copy() # Return a copy to prevent external modification
        else:
            if np.isrealobj(self.data_array):
                # If the stored frequency domain data is purely real, it might represent
                # a magnitude spectrum or only half of a symmetric spectrum.
                # For a true inverse FFT to a real time-domain signal, the input spectrum must be conjugate symmetric.
                # We assume here that if it's real, it has been fftshifted.
                # However, without the complex components, the iFFT might not be what's expected.
                # A common case for real spectrum is after magnitude calculation, which is non-invertible to original FID.
                # For now, proceed with ifftshift and ifft, but this path might be problematic.
                warnings.warn("Attempting iFFT on real frequency-domain data. Ensure it's a correctly preprocessed real part of a complex spectrum.", UserWarning)
                complex_spectrum = scipy.fft.ifftshift(self.data_array)
            else:
                complex_spectrum = scipy.fft.ifftshift(self.data_array)
            
            time_domain_signal = scipy.fft.ifft(complex_spectrum)
            # If the original signal was real, the imaginary part of iFFT should be negligible.
            # It's safer to return the complex signal if there's doubt, or if the input spectrum
            # was not guaranteed to be from a real time signal.
            # For now, if the input was complex, assume the output should be complex.
            # If input was real, then output FID's imag part should be small.
            if np.isrealobj(self.data_array):
                 return np.real(time_domain_signal)
            return time_domain_signal


    def get_frequency_domain_data(self, apply_fftshift: bool = True) -> np.ndarray:
        """
        Returns the frequency-domain representation of the data.
        Performs FFT if current data is time-domain.

        Args:
            apply_fftshift (bool): Whether to apply fftshift to center the spectrum.
                                   Default is True.

        Returns:
            np.ndarray: The frequency domain spectrum (complex).
        """
        if self.is_frequency_domain:
            return self.data_array.copy() # Return a copy
        else:
            if self.sampling_frequency is None:
                raise ValueError("sampling_frequency is required to convert time-domain data to frequency-domain.")
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
                        central_frequency for 'ppm' in MHz) are missing.
            ValueError: If an invalid unit is specified.
        """
        if self.sampling_frequency is None:
            raise ValueError("sampling_frequency is required to generate a frequency axis.")

        n_points = len(self.data_array)
        hz_axis = np.linspace(-self.sampling_frequency / 2, self.sampling_frequency / 2 - self.sampling_frequency / n_points, n_points)

        if unit.lower() == 'hz':
            return hz_axis
        elif unit.lower() == 'ppm':
            if self.central_frequency is None:
                raise ValueError("central_frequency (in MHz) is required for ppm scale.")
            if self.central_frequency == 0:
                raise ValueError("central_frequency cannot be zero for ppm conversion.")
            
            # ppm = (frequency_offset_Hz / spectrometer_base_frequency_MHz)
            # The hz_axis is already the offset from the center.
            # MRS spectra are typically displayed with high ppm on left, low ppm on right (decreasing scale).
            ppm_axis = (hz_axis / self.central_frequency) 
            return ppm_axis[::-1] 
        else:
            raise ValueError("Invalid unit for frequency axis. Choose 'ppm' or 'hz'.")

    def plot(self, unit: str = 'ppm', apply_fftshift_to_freq: bool = True):
        """
        Plots the MRS data.

        If time-domain, plots the FID (real and imaginary parts if complex).
        If frequency-domain, plots the spectrum (real and imaginary parts if complex)
        against the calculated frequency axis.

        Args:
            unit (str): The unit for the frequency axis if data is frequency-domain
                        or needs conversion ('ppm' or 'hz'). Defaults to 'ppm'.
            apply_fftshift_to_freq (bool): If converting from time to frequency for plotting,
                                           whether to apply fftshift. Default is True.
        """
        plt.figure(figsize=(10, 5))
        if self.is_time_domain:
            n_points = len(self.data_array)
            if self.sampling_frequency:
                time_axis = np.arange(n_points) / self.sampling_frequency
                plt.plot(time_axis, self.data_array.real, label="Real Part")
                if np.any(np.iscomplex(self.data_array)): 
                    plt.plot(time_axis, self.data_array.imag, label="Imaginary Part", alpha=0.7)
                plt.xlabel("Time (s)")
            else:
                plt.plot(self.data_array.real, label="Real Part")
                if np.any(np.iscomplex(self.data_array)):
                    plt.plot(self.data_array.imag, label="Imaginary Part", alpha=0.7)
                plt.xlabel("Index")
            plt.ylabel("Signal Intensity")
            plt.title("Time-Domain Data (FID)")
            if np.any(np.iscomplex(self.data_array)): plt.legend()

        elif self.is_frequency_domain:
            spectrum_to_plot = self.data_array
            try:
                freq_axis = self.get_frequency_axis(unit=unit)
                plt.plot(freq_axis, spectrum_to_plot.real, label="Real Part") 
                if np.any(np.iscomplex(spectrum_to_plot)):
                     plt.plot(freq_axis, spectrum_to_plot.imag, label="Imaginary Part", alpha=0.7)
                plt.xlabel(f"Frequency ({unit})")
                if unit == 'ppm': 
                    plt.xlim(max(freq_axis), min(freq_axis)) # Reverse ppm axis for typical display
            except ValueError as e:
                warnings.warn(f"Plotting error: {e}. Plotting against index instead.", UserWarning)
                plt.plot(spectrum_to_plot.real, label="Real Part")
                if np.any(np.iscomplex(spectrum_to_plot)):
                     plt.plot(spectrum_to_plot.imag, label="Imaginary Part", alpha=0.7)
                plt.xlabel("Index")
            plt.ylabel("Signal Intensity")
            plt.title("Frequency-Domain Data (Spectrum)")
            if np.any(np.iscomplex(spectrum_to_plot)): plt.legend()
        else: 
            plt.title("Unknown Data Type")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        repr_str = (f"MRSData(data_type='{self.data_type}', "
                    f"num_points={len(self.data_array)}, "
                    f"sampling_freq={self.sampling_frequency} Hz, "
                    f"central_freq={self.central_frequency} MHz")
        if self.echo_time_ms is not None:
            repr_str += f", TE={self.echo_time_ms} ms"
        if self.repetition_time_ms is not None:
            repr_str += f", TR={self.repetition_time_ms} ms"
        repr_str += ")"
        return repr_str
