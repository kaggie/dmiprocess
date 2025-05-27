from .loaders import load_image
from .loaders import load_data_use_brukerapi # Main function for Bruker folders
from .loaders import recon_from_2dseq # Specific Bruker reconstruction
from .loaders import convert_data_to_npy # The function that uses the above

# Other functions from loaders.py (load_dicom, load_nifti, load_visu_pars, etc.)
# are currently treated as internal helper functions to load_image or load_data_use_brukerapi.
# If they need to be public, they can be added here.
# For now, only exporting the main entry points used by convert_data_to_npy and potentially by GUI directly.

# For direct use in GUI_v3.py's load_saved_npy, it seems it was using np.load directly.
# The original functions.py load_from_file and load_from_folder were placeholders.
# Now, convert_data_to_npy uses load_image and load_data_use_brukerapi.
# The GUI's load_saved_npy loads .npy files directly, which is fine and doesn't need these imports.
# The critical part is that convert_data_to_npy now has its dependencies (load_image, load_data_use_brukerapi) in the same file.

# The GUI's `convert_data_to_npy` button calls the `convert_data_to_npy` function from this module.
# The GUI's `load_saved_npy` button calls `np.load` directly.
# The placeholder `load_from_file` and `load_from_folder` in GUI_v3.py's imports
# were remnants of the old structure and should be removed from GUI_v3.py imports if they are still there.
# The `convert_data_to_npy` in this module is the one that should be used.

# Therefore, the GUI should only need to import `convert_data_to_npy` from `io`.
# However, `convert_data_to_npy` internally calls `load_image` and `load_data_use_brukerapi`.
# If `GUI_v3.py` was *also* calling `load_image` etc. directly (which it was, via `from io import load_from_file, load_from_folder`),
# then those functions should be exported.
# `load_from_file` in the GUI context was `load_image` and `load_from_folder` was `load_data_use_brukerapi` or `recon_from_2dseq`.

# The previous `io/__init__.py` was:
# from .loaders import load_from_file -> this should map to load_image
# from .loaders import load_from_folder -> this should map to the primary Bruker loading logic
# from .loaders import convert_data_to_npy

# Let's align with the actual functions now available and previously expected names if possible.
# `load_from_file` was the old name in `functions.py` that mapped to `load_image`.
# `load_from_folder` was the old name that mapped to Bruker loading.

# For clarity, we'll export the new specific names.
# The GUI will be updated to use these specific names.
# It already imports convert_data_to_npy.
# The `load_saved_npy` in GUI uses `np.load` for .npy files.
# The `convert_data_to_npy` in `io.loaders` now uses `load_image` and `load_data_use_brukerapi`.

# The GUI's `load_saved_npy` method has an internal implementation for loading .npy files.
# The `convert_data_to_npy` function in `io.loaders` is what is called by the "Data2npy" button.
# This function (`io.loaders.convert_data_to_npy`) now correctly calls `load_image` or `load_data_use_brukerapi`.

# The `GUI_v3.py` import section for `io` is:
# `from io import load_from_file, load_from_folder, convert_data_to_npy`
# We need to ensure `io/__init__.py` provides these.
# `load_from_file` should be an alias for `load_image`.
# `load_from_folder` should be an alias for `load_data_use_brukerapi`.

from .loaders import load_image as load_from_file
from .loaders import load_data_use_brukerapi as load_from_folder
# convert_data_to_npy is already correctly named.

# Also exporting the more specific names if direct use is preferred in the future.
# from .loaders import load_image, load_data_use_brukerapi, recon_from_2dseq
# For now, stick to what GUI_v3.py currently expects from the io module.
