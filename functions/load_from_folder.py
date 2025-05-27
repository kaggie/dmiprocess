#!/usr/bin/env python
# coding: utf-8

import os
import re
import numpy as np
from brukerapi.dataset import Dataset


def load_data_use_brukerapi(data_path, idx=10, i=100, **kwargs):
    """
    Loads data using Bruker API and saves it as .npy files.
    
    Parameters:
        data_path (str): Path to the data folder.
        idx (int, optional): Index for pdata folder. Defaults to 10.
        i (int, optional): Number of scans. Defaults to 100.
        **kwargs:
            output_path (str, optional): Directory to save the output files.
                                       If not provided, saves in the same location as data_path.
    """
    output_path = kwargs.get("output_path", data_path)
    os.makedirs(output_path, exist_ok=True)
    
    for scan_idx in range(idx):
        for scan_i in range(i):
            file_path = os.path.join(data_path, str(scan_i), "pdata", str(scan_idx), "2dseq")
            
            if os.path.exists(file_path):
                try:
                    dataset = Dataset(file_path)
                    data_array = dataset.data
                    shape_str = "x".join(map(str, data_array.shape))
                    file_name = f"{scan_i}_{shape_str}.npy"
                    save_path = os.path.join(output_path, file_name)
                
                    np.save(save_path, data_array)
                    print(f"Saved: {save_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
                
                

def load_visu_pars(pdata_path, keywords):
    """
    Reads specified parameters from the visu_pars file.
    
    Parameters:
        pdata_path (str): Path to the pdata directory containing visu_pars.
        keywords (list): A list of parameter names to extract (e.g., ['VisuCoreDataSlope', 'VisuCoreDataOffs']).
    
    Returns:
        dict: A dictionary containing the extracted values of the specified keywords.
    """
    visu_pars_path = os.path.join(pdata_path, 'visu_pars')
    extracted_values = {key: None for key in keywords} 
    
    # Read visu_pars file
    with open(visu_pars_path, 'rb') as file:
        visu_pars = file.read()
    lines_visu_pars = visu_pars.decode("utf-8").split('\n')
    
    for p, line in enumerate(lines_visu_pars):
        for keyword in keywords:
            if f'##${keyword}' in line:
                value = line.replace(f'##${keyword}=', '').strip()
                
                # Handle numerical extraction cases
                if keyword in ["VisuCoreDataSlope", "VisuCoreDataOffs"]:
                    if p + 1 < len(lines_visu_pars) and lines_visu_pars[p+1].split()[0][0] == '@':
                        extracted_value = re.findall(r'\((.*?)\)', lines_visu_pars[p+1])
                        extracted_values[keyword] = float(extracted_value[0]) if extracted_value else None
                    else:
                        extracted_values[keyword] = float(lines_visu_pars[p+1].split()[0])
                else:
                    extracted_values[keyword] = value  # Store the extracted value as a string
    
    return extracted_values



def load_data_type(file_path):
    """
    Extracts and returns the data type from the visu_pars file in numpy format.
    
    Parameters:
        file_path (str): Path to the file directory containing visu_pars.
        
    Returns:
        np.dtype: The corresponding numpy data type.
    """
    visu_pars_path = os.path.join(file_path, 'visu_pars')
    
    # Read visu_pars file
    with open(visu_pars_path, 'rb') as file:
        visu_pars = file.read()
    lines_visu_pars = visu_pars.decode("utf-8").split('\n')
    
    VisuCoreWordType = None
    VisuCoreByteOrder = None
    
    for line in lines_visu_pars:
        if '##$VisuCoreWordType' in line:
            VisuCoreWordType = line.replace('##$VisuCoreWordType=', '').strip()
        elif '##$VisuCoreByteOrder' in line:
            VisuCoreByteOrder = line.replace('##$VisuCoreByteOrder=', '').strip()
    
    word_type_to_precision = {
        '_32BIT_SGN_INT': 'int32',
        '_16BIT_SGN_INT': 'int16',
        '_8BIT_UNSGN_INT': 'uint8',
        '_32BIT_FLOAT': 'single'
    }
    precision = word_type_to_precision.get(VisuCoreWordType, 'int32')  # Default to 'int32' if not found
    
    byte_order_to_endian = {'littleEndian': 'l','bigEndian': 'b'}
    endian = byte_order_to_endian.get(VisuCoreByteOrder, 'l')  # Default to 'l' if not found
    
    precision_format = {'int32': 'i', 'int16': 'h', 'uint8': 'B', 'single': 'f'}
    endian_format = {'l': '<', 'b': '>'}
    
    data_type_format = endian_format[endian] + precision_format[precision]
    return np.dtype(data_type_format)


def load_image_size(file_path):
    """
    Extracts and returns the image size from the reco_pars file.
    
    Parameters:
        file_path (str): Path to the file directory containing reco_pars.
        
    Returns:
        list: Image size dimensions including slices and repetitions.
    """
    reco_pars_path = os.path.join(file_path, 'reco')
    
    # Read reco_pars file
    with open(reco_pars_path, 'rb') as file:
        reco_pars = file.read()
    lines_reco_pars = reco_pars.decode("utf-8").split('\n')
    
    slices = repetitions = None
    size = []
    
    for p, line in enumerate(lines_reco_pars):
        if '##$RecoObjectsPerRepetition' in line:
            slices = int(line.replace('##$RecoObjectsPerRepetition=', '').strip())
        elif '##$RecoNumRepetitions' in line:
            repetitions = int(line.replace('##$RecoNumRepetitions=', '').strip())
        elif '##$RECO_size' in line and p + 1 < len(lines_reco_pars):
            size = [int(i) for i in lines_reco_pars[p+1].split()]
    
    # Construct image size list
    img_size = size + ([slices] if slices is not None else []) + ([repetitions] if repetitions is not None else [])
    return img_size


def recon_from_2dseq(file_path, imaginary=False):
    """
    Reconstructs an image from a 2dseq file.
    
    Parameters:
        file_path (str): Path to the directory containing 2dseq and parameter files.
        imaginary (bool, optional): Whether to reconstruct complex data with imaginary components. Defaults to False.
    
    Returns:
        np.ndarray: Reconstructed image data.
    """
    data_path = os.path.join(file_path, '2dseq')
    
    # Load necessary parameters
    keywords = ['VisuCoreDataSlope', 'VisuCoreDataOffs', 'VisuCoreFrameType']
    pars = load_visu_pars(file_path, keywords)
    Slope, Offs, Image_type = pars.get('VisuCoreDataSlope'), pars.get('VisuCoreDataOffs'), pars.get('VisuCoreFrameType')
    
    data_type = load_data_type(file_path)
    image_size = load_image_size(file_path)
    
    # Read binary data from 2dseq
    with open(data_path, 'rb') as file:
        data = file.read()
    array_data = np.frombuffer(data, dtype=data_type)
    
    # Handle complex data if imaginary is True
    if imaginary:
        half_length = len(array_data) // 2
        array_data_c = array_data[:half_length] + 1j * array_data[half_length:]
    else:
        array_data_c = array_data
    
    array_data_c = array_data_c*Slope + Offs
    # Reshape the array according to image dimensions
    array_data_reshaped = array_data_c.reshape(image_size[::-1])
    
    # Adjust dimensions and return reconstructed image
    if len(image_size) == 6:
        img_recon = array_data_reshaped.transpose(5, 4, 3, 2, 1, 0).squeeze()
    elif len(image_size) == 5:
        img_recon = array_data_reshaped.transpose(4, 3, 2, 1, 0).squeeze()
    else:
        img_recon = array_data_reshaped
    
    return img_recon
