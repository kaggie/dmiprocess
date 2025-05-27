# dmiprocess

A toolbox for processing, denoising, normalizing, and fitting data—especially designed for DMI (Dynamic Molecular Imaging) and spectroscopy data workflows.

## Features

- Data loading from files and folders.
- Classic and deep learning denoising methods.
- Curve and peak fitting (CPU/GPU).
- Data normalization.
- Phase correction.
- Utilities for preprocessing, visualization, and more.

## Installation

Clone the repository and install dependencies (requires Python 3.7+):

```bash
git clone https://github.com/kaggie/dmiprocess.git
cd dmiprocess
pip install -r requirements.txt
```

## Toolbox Structure

- `functions/`: Core processing functions.
- `denoising/`: Additional denoising methods.
- `peak_detection/`: Peak detection routines.
- `visualization/`: Plotting and visualization tools.
- `io/`, `utils/`, `preprocessing/`, etc.: Supporting modules.

## Example Usage

### Load Data

```python
from functions.load_from_file import load_data_from_file

data = load_data_from_file('data_for_test/sample_data.csv')
print(data)
```

### Denoising (Classic)

```python
from functions.classic_denoiser import classic_denoise

denoised = classic_denoise(data, method='gaussian', sigma=1.0)
```

### Denoising (Transformer/UNet)

```python
from functions.denoise_unet_pe import unet_denoise
# from functions.denoise_trans_pe import transformer_denoise

denoised = unet_denoise(data)
```

### Data Normalization

```python
from functions.data_normalization import normalize

normalized = normalize(data, method='minmax')
```

### Curve Fitting

```python
from functions.curve_fitting import fit_curve

fit_params = fit_curve(xdata, ydata, model='gaussian')
```

### Peak Fitting (CPU/GPU)

```python
from functions.peak_fitting import fit_peaks
from functions.peak_fitting_gpu import fit_peaks_gpu

peaks = fit_peaks(ydata)
# Or on GPU:
peaks_gpu = fit_peaks_gpu(ydata)
```

### Phase Correction

```python
from functions.phase_correction import correct_phase

corrected = correct_phase(data)
```

## Running as a Script

You can also use `main.py` as an entry point for batch jobs or workflows:

```bash
python main.py --input data_for_test/sample_data.csv --output results/
```

