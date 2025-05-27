import os
import glob 
import re 
import numpy as np
import pydicom 
import nibabel as nib 
# QFileDialog is removed from here as it will be handled by the GUI
from brukerapi.dataset import Dataset 

def default_log_message(message):
    print(message)

def load_dicom(dicom_dir):
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not dicom_files: 
        raise FileNotFoundError(f"No DICOM files found in directory: {dicom_dir}")
    dicom_files.sort()
    first_dicom = pydicom.dcmread(dicom_files[0])
    image = first_dicom.pixel_array
    image = image[np.newaxis, :, :]
    for dicom_file in dicom_files[1:]:
        dicom = pydicom.dcmread(dicom_file)
        image = np.concatenate((image, dicom.pixel_array[np.newaxis, :, :]), axis=0)
    return image

def load_nifti(nifti_file_path):
    nifti_data = nib.load(nifti_file_path)
    image = nifti_data.get_fdata()
    return image

def load_image(file_path): # file_path can be a DICOM dir or NIFTI/DICOM file
    if os.path.isdir(file_path): 
        if any(f.endswith(".dcm") for f in os.listdir(file_path)):
            return load_dicom(file_path)
        else:
            raise ValueError(f"Directory specified but no DICOM files found: {file_path}")
    elif file_path.endswith(".dcm"): # Single DICOM file
        return load_dicom(os.path.dirname(file_path)) # Load the whole directory
    elif file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
        return load_nifti(file_path)
    else:
        raise ValueError(f"Invalid file type or path: {file_path}")

def load_data_use_brukerapi(data_path, log_message=default_log_message, **kwargs): # Removed unused idx, i and output_path from here
    # Try to load from common Bruker structures.
    # This function aims to return a single NumPy array representing the dataset.
    # It first tries BrukerAPI, then recon_from_2dseq as a fallback.
    
    # Case 1: data_path is a pdata directory (e.g., .../1/pdata/1)
    potential_2dseq_path = os.path.join(data_path, "2dseq")
    if os.path.exists(potential_2dseq_path):
        try:
            dataset = Dataset(potential_2dseq_path)
            data_array = dataset.data 
            log_message(f"Successfully loaded Bruker data via BrukerAPI from: {potential_2dseq_path}")
            return data_array
        except Exception as e_api:
            log_message(f"BrukerAPI failed for {potential_2dseq_path}: {e_api}. Trying recon_from_2dseq.")
            try:
                data_array_recon = recon_from_2dseq(data_path, imaginary=True, log_message=log_message)
                if data_array_recon is not None:
                    log_message(f"Successfully loaded data using recon_from_2dseq from: {data_path}")
                    return data_array_recon
            except Exception as e_recon:
                log_message(f"recon_from_2dseq also failed for {data_path}: {e_recon}")
                # Fall through to other cases or return None

    # Case 2: data_path is a scan directory (e.g., .../1) containing pdata/1, pdata/2 etc.
    # Try to find the first valid pdata directory (e.g., pdata/1)
    for pdata_sub_dir_name in sorted(os.listdir(data_path) if os.path.isdir(data_path) else []):
        if pdata_sub_dir_name.startswith("pdata"): # Or just check for common names like "1", "2" if pdata is not explicit
             pdata_dir = os.path.join(data_path, pdata_sub_dir_name)
             if os.path.isdir(pdata_dir): # Check if it's a directory
                # Try to find a numbered processing directory inside pdata, e.g. "1"
                for proc_no_str in sorted(os.listdir(pdata_dir)):
                    if proc_no_str.isdigit():
                        proc_dir = os.path.join(pdata_dir, proc_no_str)
                        potential_2dseq_path = os.path.join(proc_dir, "2dseq")
                        if os.path.exists(potential_2dseq_path):
                            log_message(f"Found potential Bruker 2dseq at: {potential_2dseq_path}")
                            try:
                                dataset = Dataset(potential_2dseq_path)
                                data_array = dataset.data
                                log_message(f"Successfully loaded Bruker data via BrukerAPI from: {potential_2dseq_path}")
                                return data_array
                            except Exception as e_api_subdir:
                                log_message(f"BrukerAPI failed for {potential_2dseq_path}: {e_api_subdir}. Trying recon_from_2dseq.")
                                try:
                                    data_array_recon = recon_from_2dseq(proc_dir, imaginary=True, log_message=log_message)
                                    if data_array_recon is not None:
                                        log_message(f"Successfully loaded data using recon_from_2dseq from: {proc_dir}")
                                        return data_array_recon
                                except Exception as e_recon_subdir:
                                    log_message(f"recon_from_2dseq also failed for {proc_dir}: {e_recon_subdir}")
                                    # Continue to next potential pdata/proc_no if this one fails
                        # If BrukerAPI and recon_from_2dseq fail for this proc_dir, continue to next proc_no

    log_message(f"Could not load Bruker data from folder: {data_path} using common structures.")
    return None


def load_visu_pars(pdata_path, keywords):
    visu_pars_path = os.path.join(pdata_path, 'visu_pars')
    extracted_values = {key: None for key in keywords}
    if not os.path.exists(visu_pars_path):
        return extracted_values
    with open(visu_pars_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines_visu_pars = file.readlines()
    for p, line in enumerate(lines_visu_pars):
        for keyword in keywords:
            if f'##${keyword}' in line or f'##${keyword.upper()}' in line :
                value_part = line.split('=')[-1].strip()
                if keyword in ["VisuCoreDataSlope", "VisuCoreDataOffs"]:
                    try:
                        val_match = re.search(r'([0-9\.eE\+\-]+)', value_part)
                        if val_match: extracted_values[keyword] = float(val_match.group(1)); continue
                        for next_line_idx in range(p + 1, min(p + 4, len(lines_visu_pars))):
                            next_line = lines_visu_pars[next_line_idx].strip()
                            if not next_line or next_line.startswith('#'): continue
                            val_match_next = re.search(r'([0-9\.eE\+\-]+)', next_line.split(')')[0])
                            if val_match_next: extracted_values[keyword] = float(val_match_next.group(1)); break
                        if extracted_values[keyword] is not None: continue
                    except ValueError: pass
                else: extracted_values[keyword] = value_part 
    return extracted_values

def load_data_type(file_path): # file_path is pdata_dir
    visu_pars_path = os.path.join(file_path, 'visu_pars')
    if not os.path.exists(visu_pars_path): return np.dtype('int32') 
    with open(visu_pars_path, 'r', encoding='utf-8', errors='ignore') as file: lines_visu_pars = file.readlines()
    VisuCoreWordType, VisuCoreByteOrder = None, None
    for line in lines_visu_pars:
        if '##$VisuCoreWordType' in line or '##$VISUCOREWORDTYPE' in line: VisuCoreWordType = line.split('=')[-1].strip()
        elif '##$VisuCoreByteOrder' in line or '##$VISUCOREBYTEORDER' in line: VisuCoreByteOrder = line.split('=')[-1].strip()
    word_map = {'_32BIT_SGN_INT':'int32','_16BIT_SGN_INT':'int16','_8BIT_UNSGN_INT':'uint8','_32BIT_FLOAT':'float32'}
    prec_np = word_map.get(VisuCoreWordType, 'int32')
    byte_map = {'LITTLE_ENDIAN':'<','BIG_ENDIAN':'>'}
    endian_np = byte_map.get(VisuCoreByteOrder, '<' if sys.byteorder == 'little' else '>')
    return np.dtype(endian_np + prec_np)

def load_image_size(file_path): # file_path is pdata_dir
    reco_pars_path = os.path.join(file_path, 'reco')
    if not os.path.exists(reco_pars_path): return []
    with open(reco_pars_path, 'r', encoding='utf-8', errors='ignore') as file: lines_reco_pars = file.readlines()
    slices, repetitions, size = None, None, []
    for p, line in enumerate(lines_reco_pars):
        if '##$RecoObjectsPerRepetition' in line or '##$RECOOBJECTSPERREPETITION' in line: slices = int(line.split('=')[-1].strip())
        elif '##$RecoNumRepetitions' in line or '##$RECONUMREPETITIONS' in line: repetitions = int(line.split('=')[-1].strip())
        elif ('##$RECO_size' in line or '##$RECO_SIZE' in line) and p + 1 < len(lines_reco_pars):
            next_line = lines_reco_pars[p+1].strip()
            if next_line and not next_line.startswith('##$'): size = [int(i) for i in next_line.split()]
    img_size = []
    if size: img_size.extend(size)
    if slices is not None: img_size.append(slices)
    if repetitions is not None: img_size.append(repetitions)
    return img_size

def recon_from_2dseq(pdata_dir_path, imaginary=False, log_message=default_log_message):
    data_path_2dseq = os.path.join(pdata_dir_path, '2dseq')
    if not os.path.exists(data_path_2dseq): log_message(f"2dseq not found: {data_path_2dseq}"); return None
    pars = load_visu_pars(pdata_dir_path, ['VisuCoreDataSlope', 'VisuCoreDataOffs'])
    Slope_val, Offs_val = pars.get('VisuCoreDataSlope'), pars.get('VisuCoreDataOffs')
    Slope = Slope_val[0] if isinstance(Slope_val, list) and Slope_val else (Slope_val if isinstance(Slope_val, float) else 1.0)
    Offs = Offs_val[0] if isinstance(Offs_val, list) and Offs_val else (Offs_val if isinstance(Offs_val, float) else 0.0)
    data_type, image_size = load_data_type(pdata_dir_path), load_image_size(pdata_dir_path)
    if not image_size: log_message(f"Cannot get image size for {pdata_dir_path}"); return None
    with open(data_path_2dseq, 'rb') as file: binary_data = file.read()
    array_data = np.frombuffer(binary_data, dtype=data_type)
    
    expected_elements = np.prod(image_size)
    if imaginary: expected_elements *= 2

    if array_data.size < expected_elements: # Allow if array_data is just real part for complex
        if imaginary and array_data.size * 2 == expected_elements :
             log_message(f"Warning: 2dseq for complex data in {pdata_dir_path} might only contain real part or be half size. Size: {array_data.size}, Expected complex elements: {expected_elements//2}")
             # Proceeding by padding with zeros for imaginary part
             array_data_c = array_data.astype(np.float32) + 0j # Make it complex
        elif array_data.size == expected_elements // 2 and not imaginary: # If asking for real, but it's half size
            log_message(f"Error: Data size {array_data.size} is half of expected {expected_elements} for real data in {pdata_dir_path}.")
            return None
        elif array_data.size != expected_elements : # General mismatch
            log_message(f"Error: Data size {array_data.size} mismatch with expected {expected_elements} in {pdata_dir_path}.")
            return None
    
    if imaginary:
        if array_data.size == expected_elements : # Contains real and imag
            half_len = array_data.size // 2
            array_data_c = array_data[:half_len].astype(np.float32) + 1j * array_data[half_len:].astype(np.float32)
        else: # Assume it's real, make complex (already handled above if padded with zeros)
            array_data_c = array_data.astype(np.float32) + 0j 
    else:
        array_data_c = array_data.astype(np.float32)

    array_data_c = array_data_c * Slope + Offs
    try:
        array_data_reshaped = array_data_c.reshape(image_size[::-1])
    except ValueError as e:
        log_message(f"Error reshaping data in {pdata_dir_path}: {e}. Image size: {image_size}, array elements: {array_data_c.size}")
        return None
    img_recon = array_data_reshaped.transpose(np.arange(len(image_size) -1, -1, -1)).squeeze()
    return img_recon

def convert_data_to_npy(log_message, input_path_selected_by_user, output_npy_file_path):
    """
    Converts data from various formats (DICOM, NIFTI, Bruker) to a NumPy .npy file.
    QFileDialog calls are now handled by the GUI part.
    Args:
        log_message (callable): Function to log messages.
        input_path_selected_by_user (str): Path to the input file or folder.
        output_npy_file_path (str): Path (including filename) where the .npy file should be saved.
    """
    log_func = log_message if callable(log_message) else default_log_message

    if not input_path_selected_by_user:
        log_func("Error: No input folder or file selected by the user.")
        return
    log_func(f"Input path: {input_path_selected_by_user}")

    if not output_npy_file_path:
        log_func("Error: No output .npy file path provided.")
        return
    
    # Ensure output directory exists
    save_dir = os.path.dirname(output_npy_file_path)
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
            log_func(f"Created output directory: {save_dir}")
        except OSError as e:
            log_func(f"Error creating output directory {save_dir}: {e}")
            return

    data_array = None
    try:
        if os.path.isdir(input_path_selected_by_user):
            log_func(f"Attempting to load data from folder: {input_path_selected_by_user}")
            # Try Bruker loading logic first
            data_array = load_data_use_brukerapi(input_path_selected_by_user, log_message=log_func)
            if data_array is None:
                log_func(f"Bruker loading failed or returned None. Attempting to load as DICOM directory.")
                # If Bruker loading fails or returns None, try to load as a directory of DICOMs
                data_array = load_image(input_path_selected_by_user) # load_image handles DICOM dirs
                if data_array is not None:
                     log_func(f"Successfully loaded as DICOM directory from: {input_path_selected_by_user}")
            else:
                log_func(f"Successfully loaded Bruker data from: {input_path_selected_by_user}")

        elif os.path.isfile(input_path_selected_by_user):
            log_func(f"Attempting to load single file: {input_path_selected_by_user}")
            data_array = load_image(input_path_selected_by_user) # Handles .nii, .nii.gz, .dcm
            if data_array is not None:
                log_func(f"Successfully loaded from file: {input_path_selected_by_user}")
        else:
            raise ValueError(f"Invalid input path: {input_path_selected_by_user}. Not a valid file or directory.")

        if data_array is not None:
            np.save(output_npy_file_path, data_array)
            log_func(f"Data saved as NumPy array: {output_npy_file_path}")
            log_func(f"Saved data shape: {data_array.shape}")
        else:
            log_func(f"Error: Could not load data from {input_path_selected_by_user}. No data to save.")
            # GUI part should show QMessageBox to user

    except Exception as e:
        log_func(f"Error in convert_data_to_npy: {str(e)}")
        # GUI part should show QMessageBox to user
        # Re-raise the exception if this function is expected to signal failure to the caller
        raise
