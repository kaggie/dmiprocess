# Running Unit Tests

To run the unit tests for the `lcm_library`, navigate to the root directory of the `mrs_lcm_analysis` project (the directory containing this `tests` folder and the `lcm_library` folder).

Then, execute the following command in your terminal:

```bash
python -m unittest discover ./tests
```

This command will automatically discover and run all test files (named `test_*.py`) within the `tests` directory.

Alternatively, to run a specific test file:
```bash
python -m unittest mrs_lcm_analysis.tests.test_basis # Example for test_basis.py
```

Or a specific test class within a file:
```bash
python -m unittest mrs_lcm_analysis.tests.test_basis.TestBasisSpectrum # Example
```

Or a specific test method:
```bash
python -m unittest mrs_lcm_analysis.tests.test_basis.TestBasisSpectrum.test_creation # Example
```
