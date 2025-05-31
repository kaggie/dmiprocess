# Running Unit Tests

To run the unit tests for the SpectroFit CLI and its modules, navigate to the
project root directory and use the Python `unittest` module's discovery feature:

```bash
python -m unittest discover -s tests -v
```

This command will search for all files named `test_*.py` within the `tests`
directory, execute the test cases defined in them, and provide verbose output.

**Note on Current Environment:**

Due to limitations in the current execution environment (specifically, issues
with installing PyTorch and Matplotlib due to disk space), some tests,
particularly those for `oxsa_model.py`, `lcmodel_fitting.py` (related to actual
fitting with `mrs_lcm_analysis`), and plotting in `output_utils.py`, are
designed as placeholders or test graceful failure paths.

Tests for modules like `config_loader.py`, `data_io.py` (for non-PyTorch dependent
parts), CSV generation in `output_utils.py`, and the overall CLI structure
(`spectrofit_cli.py` using mocks) should run and pass assuming their direct
dependencies (like SciPy, H5Py, PyYAML, NumPy) are correctly installed.
```
