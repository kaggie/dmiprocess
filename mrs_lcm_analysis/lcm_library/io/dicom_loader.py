import os
import numpy as np
import pydicom
import warnings
from typing import TYPE_CHECKING, Optional, Dict, Union, Any, List

from .base_loader import BaseMRSLoader
if TYPE_CHECKING:
    from mrs_lcm_analysis.lcm_library.data_loading import MRSData 
from mrs_lcm_analysis.lcm_library.data_loading import MRSData 

class DicomLoader(BaseMRSLoader):
    """
    Loads MRS data from DICOM (.dcm) files.
    
    Attempts to load data from the Spectroscopy SOP Class.
    This implementation focuses on a common way Siemens and Philips store
    single-voxel spectroscopy (SVS) data (e.g., complex float32 interleaved).
    """

    MRS_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.4.2'

    def load(self, filepath: str) -> 'MRSData':
        """
        Loads MRS data from the given DICOM file.

        Extracts FID data and key metadata for MRS analysis.
        Currently assumes single-frame, complex float32 data in SpectroscopyData.

        Args:
            filepath (str): The path to the DICOM MRS data file.

        Returns:
            MRSData: An MRSData object.

        Raises:
            FileNotFoundError: If the filepath does not exist.
            pydicom.errors.InvalidDicomError: If the file is not a valid DICOM file.
            ValueError: If critical MRS data or metadata is missing or has an
                        unexpected format.
            IOError: If there is an issue reading the file beyond DICOM format errors.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            dicom_dataset = pydicom.dcmread(filepath)
        except pydicom.errors.InvalidDicomError as e:
            raise ValueError(f"Not a valid DICOM file or could not parse: {filepath}. Error: {e}")
        except Exception as e: 
            raise IOError(f"Could not read DICOM file: {filepath}. Error: {e}")

        sop_class_uid = dicom_dataset.get("SOPClassUID")
        if sop_class_uid != self.MRS_SOP_CLASS_UID:
            warnings.warn(f"File {filepath} has SOPClassUID {sop_class_uid}, which may not be an MRS object. Attempting to load anyway.", UserWarning)

        metadata: Dict[str, Union[str, float, int, List[Any], Dict[Any, Any]]] = {'source_format': 'dicom'}
        
        f0_mhz_val = dicom_dataset.get("TransmitterFrequency", dicom_dataset.get("ImagingFrequency", None))
        if f0_mhz_val is not None:
            f0_mhz = float(f0_mhz_val)
        else:
            raise ValueError("Critical metadata 'TransmitterFrequency' (0018,9098) or 'ImagingFrequency' (0018,0084) not found.")
        
        sw_hz_val = dicom_dataset.get("SpectralWidth", None) 
        if sw_hz_val is not None:
            sw_hz = float(sw_hz_val)
        else:
            raise ValueError("Critical metadata 'SpectralWidth' (0018,9052) not found.")
            
        num_points_header_val = dicom_dataset.get("SpectroscopyAcquisitionDataColumns", None) 
        num_points_header = int(num_points_header_val) if num_points_header_val is not None else None

        te_ms_val = dicom_dataset.get("EffectiveEchoTime", dicom_dataset.get("EchoTime", None))
        te_ms = float(te_ms_val) if te_ms_val is not None else None
        if te_ms is None: warnings.warn("Echo Time not found (EffectiveEchoTime/EchoTime). Will be None.", UserWarning)

        tr_ms_val = dicom_dataset.get("RepetitionTime", None)
        tr_ms = float(tr_ms_val) if tr_ms_val is not None else None
        if tr_ms is None: warnings.warn("Repetition Time (0018,0080) not found. Will be None.", UserWarning)
            
        # Store other DICOM tags in metadata
        for tag in ["PatientName", "StudyDate", "SeriesDescription", "SequenceName", "ProtocolName", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength"]:
            value = dicom_dataset.get(tag, None)
            if value is not None:
                metadata[tag] = str(value) # Ensure string representation for simplicity in metadata dict

        if "SpectroscopyData" not in dicom_dataset: 
            raise ValueError("Tag 'SpectroscopyData' (7FE1,1010) not found in DICOM file.")
        
        spectroscopy_data_element = dicom_dataset[0x7FE1, 0x1010]
        raw_fid_data = spectroscopy_data_element.value
        
        number_of_frames = int(dicom_dataset.get("NumberOfFrames", 1))
        if number_of_frames > 1:
            warnings.warn(f"DICOM file contains {number_of_frames} frames. Only the first frame will be loaded.", UserWarning)

        try:
            fid_array_float32 = np.frombuffer(raw_fid_data, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Could not interpret SpectroscopyData as float32. Error: {e}")

        if fid_array_float32.size == 0:
            raise ValueError("SpectroscopyData is empty.")

        expected_raw_values_per_frame = fid_array_float32.size // number_of_frames
        
        if expected_raw_values_per_frame % 2 != 0:
            raise ValueError(f"Raw data points per frame ({expected_raw_values_per_frame}) is not even. Cannot form complex FID.")
        num_complex_points_from_data = expected_raw_values_per_frame // 2

        num_points_to_use: int
        if num_points_header is None:
            num_points_to_use = num_complex_points_from_data
            metadata['num_points_inferred_from_data'] = num_points_to_use
        elif num_points_header != num_complex_points_from_data:
            warnings.warn(f"Number of points from header (0018,9126) ({num_points_header}) does not match data length "
                          f"({num_complex_points_from_data} complex points). Using data length for FID extraction.", UserWarning)
            num_points_to_use = num_complex_points_from_data
            metadata['num_points_adjusted_from_data'] = num_points_to_use
        else:
            num_points_to_use = num_points_header
        
        if num_points_to_use == 0:
            raise ValueError("Zero points determined for FID data.")

        raw_frame_data = fid_array_float32[:num_points_to_use * 2]
        fid_complex = raw_frame_data[0::2] + 1j * raw_frame_data[1::2]
        
        if len(fid_complex) != num_points_to_use:
            raise ValueError(f"Mismatch after parsing complex FID: Expected {num_points_to_use} points, got {len(fid_complex)}.")

        return MRSData(data_array=fid_complex,
                       data_type="time", 
                       sampling_frequency=sw_hz,
                       central_frequency=f0_mhz,
                       echo_time_ms=te_ms,
                       repetition_time_ms=tr_ms,
                       metadata=metadata)

    def can_load(self, filepath: str) -> bool:
        """
        Checks if this loader can likely load the given file.
        Checks for .dcm extension or if pydicom can read it and it's an MRS SOP class.
        """
        _, ext = os.path.splitext(filepath)
        is_dcm_ext = ext.lower() == '.dcm'

        try:
            dcm_peek = pydicom.dcmread(filepath, specific_tags=['SOPClassUID'], stop_before_pixels=True)
            if dcm_peek.SOPClassUID == self.MRS_SOP_CLASS_UID:
                return True
            elif is_dcm_ext: 
                return True 
        except Exception:
            return False 
        return is_dcm_ext
```
