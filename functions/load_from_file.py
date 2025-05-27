#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy as np
import pydicom
import nibabel as nib

def load_dicom(dicom_dir):
    """
    Load DICOM files from a directory.
    
    Parameters:
        dicom_dir (str): Path to the directory containing DICOM files.
    
    Returns:
        nibabel.nifti1.Nifti1Image: A 3D NIfTI image.
    """
    # Load DICOM files
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    dicom_files.sort()
    
    # Read the first DICOM file
    first_dicom = pydicom.dcmread(dicom_files[0])
    image = first_dicom.pixel_array
    image = image[np.newaxis, :, :]
    
    # Read the rest of the DICOM files
    for dicom_file in dicom_files[1:]:
        dicom = pydicom.dcmread(dicom_file)
        image = np.concatenate((image, dicom.pixel_array[np.newaxis, :, :]), axis=0)
    
    return image

def load_nifti(nifti_file):
    """
    Load a NIfTI file.
    
    Parameters:
        nifti_file (str): Path to the NIfTI file.
    
    Returns:
        nibabel.nifti1.Nifti1Image: A 3D NIfTI image.
    """
    nifti_file = glob.glob(os.path.join(nifti_file, "*.nii*"))
    nifti_file.sort()
    first_nifti = nib.load(nifti_file[0])
    image = first_nifti.get_fdata()
    image = image[np.newaxis, :, :]

    for nifti_file in nifti_file[1:]:
        nifti = nib.load(nifti_file)
        image = np.concatenate((image, nifti.get_fdata()[np.newaxis, :, :]), axis=0)

    return image

def load_image(file_path):
    """
    Load an image from a file.
    
    Parameters:
        file_path (str): Path to the image file.
    
    Returns:
        numpy.ndarray: A 3D NumPy array.
    """
    if file_path.endswith(".dcm"):
        image = load_dicom(file_path)
    elif file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
        image = load_nifti(file_path)
    else:
        raise ValueError("Invalid file type. Must be DICOM or NIfTI.")
    
    return image