# MRS Data I/O Submodule (`mrs_lcm_analysis.lcm_library.io`)

## Purpose

The `io` submodule is designed to handle the loading of Magnetic Resonance Spectroscopy (MRS) data from various file formats into a consistent internal representation, specifically the `MRSData` object provided by the `lcm_library.data_loading` module. This abstraction allows the rest of the `lcm_library` to operate on a standardized data structure, irrespective of the original data source format.

## Core Components

### 1. `BaseMRSLoader` (Abstract Base Class)

Located in `base_loader.py`, `BaseMRSLoader` defines the interface for all concrete MRS data loaders. Key features:

-   **Abstract Method `load(self, filepath: str) -> MRSData`**:
    Each concrete loader (e.g., `NumpyLoader`, `DicomLoader`) must implement this method. It takes a file path as input and is responsible for reading the file, extracting the MRS signal data (time-domain or frequency-domain), and all relevant metadata (e.g., sampling frequency, central frequency, echo time, repetition time, metabolite names if available in headers). It then populates and returns an `MRSData` object.
    This method should raise appropriate errors like `FileNotFoundError`, `IOError` (for general read issues), or `ValueError` (for format inconsistencies or corrupted data).

-   **Concrete Method `can_load(self, filepath: str) -> bool`**:
    This optional method can be overridden by concrete loaders to provide a quick check (e.g., based on file extension or magic numbers) if the loader is likely capable of handling the specified file. It returns `False` by default in the base class. This is not a guarantee of successful loading but can be used by the dispatcher to quickly select a potential loader.

### 2. Concrete Loader Classes (Future Implementation)

For each supported MRS file format, a dedicated loader class will be created, inheriting from `BaseMRSLoader`. Examples:

-   `NumpyLoader`: For loading data stored in NumPy's `.npy` or `.npz` formats.
-   `MatlabLoader`: For loading data from MATLAB `.mat` files (e.g., from other toolboxes).
-   `DicomLoader`: For loading MRS data stored in DICOM format (e.g., Siemens DICOM MRS objects).
-   `TwixLoader`: For Siemens TWIX raw data files.
-   `RdaLoader`: For Siemens RDA files.
-   `SdatLoader`: For Philips SDAT/SPAR files.
-   `PfileLoader`: For GE P-files.
-   `BrukerLoader`: For Bruker ParaVision format.

Each concrete loader will implement the `load` method specific to its format, potentially using external libraries like `scipy.io` (for MATLAB files), `pydicom` (for DICOM), or specialized MRS libraries like `Suspect` (which has robust readers for many formats like TWIX, SDAT, RDA, etc.).

### 3. Main Dispatcher Function: `load_mrs()`

A central function, likely named `load_mrs(filepath: str, format: Optional[str] = None) -> MRSData`, will be implemented in `mrs_lcm_analysis/lcm_library/io/__init__.py`. This function will:

-   Act as a dispatcher for the various concrete loader classes.
-   If `format` is specified (e.g., "dicom", "twix"), it will attempt to use the corresponding loader directly.
-   If `format` is `None`, it may iterate through registered/available loaders, calling their `can_load()` method (or attempting to load directly and catching errors) to automatically determine the correct loader.
-   The goal is to provide a single, user-friendly function to load MRS data.

This `load_mrs` function will be the primary public interface for data loading from this submodule. It is intended to be re-exported by the main `lcm_library.__init__.py` for easier access (e.g., `from lcm_library import load_mrs`).

## Data Output

All successful load operations will return an instance of the `MRSData` class, ensuring that the rest of the analysis pipeline (e.g., `LinearCombinationModel`) receives data in a standardized format. The `MRSData` object will contain:

-   The MRS signal (time or frequency domain).
-   Essential acquisition parameters (sampling frequency, central frequency).
-   Other relevant metadata extracted from the file headers.

## Adding New Loaders

To support a new file format:

1.  Create a new class that inherits from `BaseMRSLoader`.
2.  Implement the `load(self, filepath: str) -> MRSData` method, including data/metadata extraction and `MRSData` object population.
3.  Optionally, override `can_load(self, filepath: str) -> bool` for quick format checking.
4.  Register the new loader with the main `load_mrs` dispatcher function (details of registration TBD, could be a simple list or a more dynamic plugin system).

## Planned Format Support and Libraries

The following formats are planned for eventual support, potentially leveraging these libraries:

-   **NumPy (`.npy`, `.npz`)**: `numpy`
-   **MATLAB (`.mat`)**: `scipy.io.loadmat`
-   **DICOM (`.dcm`)**: `pydicom`, potentially with helpers from `Suspect` or `nibabel`.
-   **Siemens TWIX (`.dat`)**: `Suspect` (e.g., `suspect.io.load_twix`) or custom parsing.
-   **Siemens RDA (`.rda`)**: `Suspect` (e.g., `suspect.io.load_rda`) or custom parsing.
-   **Philips SDAT/SPAR (`.sdat`, `.spar`)**: `Suspect` (e.g., `suspect.io.load_sdat`)
-   **GE P-file (`P*.7`)**: `Suspect` (e.g., `suspect.io.load_pfile`) or `nibabel`.
-   **Bruker ParaVision**: Custom parsing or specialized Bruker libraries if available/permissible.

The use of `Suspect` is highly encouraged for formats it supports due to its specialization in MRS data handling.
