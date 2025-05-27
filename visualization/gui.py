import os
import sys # For sys.exit, QApplication
import cv2
import numpy as np
import torch
import nibabel as nib
import glob # For load_saved_npy series loading
import re # For load_saved_npy series loading

import pyqtgraph as pg
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.console import ConsoleWidget # MainApp uses this
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QLabel, QMainWindow, QFileDialog, 
                             QVBoxLayout, QWidget, QPlainTextEdit, QPushButton, 
                             QHBoxLayout, QLineEdit, QRadioButton, QButtonGroup,
                             QDialog, QComboBox, QFormLayout, QGridLayout, QStackedLayout,
                             QCheckBox, QMessageBox, QFormLayout, QScrollArea, QGroupBox,QSlider
                            )

# Import functions from refactored modules
from io import convert_data_to_npy 
from preprocessing import phase_correct_gpu, normalize_data
from denoising import apply_spectral_denoising_batch, UNet1DWithPEPeak, TransformerDenoiser
from peak_detection import fit_volume_gpu, LinearModel, ExpModel, BiExpModel, BBModel, model_fitting

# Import plotters module
from . import plotters 
# Import config settings
from config.settings import get_device, set_device


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DMI Data Processing Toolbox")
        self.setGeometry(100, 100, 1200, 800)

        self.I = None 
        self.I_norm = None
        self.I_corrected = None
        self.I_denoised = None
        self.I_fitted = None
        self.data_before_fitting = None 
        self.structural_image = None 
        
        self.active_windows = []

        self.area = DockArea()
        self.setCentralWidget(self.area)

        self.load_dock = Dock("Load Image", size=(500, 400))
        self.phase_dock = Dock("Phase Correction", size=(500, 400))
        self.denoise_dock = Dock("Denoising", size=(500, 400))
        self.peak_dock = Dock("Peak Fitting", size=(500, 400))
        self.console_dock = Dock("Console", size=(800, 200))

        self.area.addDock(self.load_dock, "left")
        self.area.addDock(self.phase_dock, "right")
        self.area.addDock(self.denoise_dock, "bottom", self.load_dock)
        self.area.addDock(self.peak_dock, "bottom", self.phase_dock)
        self.area.addDock(self.console_dock, "bottom")

        self.load_widget, self.load_plot_widgets = self.create_section("Load Image", ["Data2npy", "Load", "Overlay", "Plot Spectrum", "Show Image", "Clear Display"])
        self.phase_widget, self.phase_plot_widgets = self.create_section("Phase Correction", ["Device", "Normalize", "Comparison", "Apply Phase Correction",
                                                                                       "Plot Spectrum", "Show Image", "Clear Display"])
        self.denoise_widget, self.denoise_plot_widgets = self.create_section("Denoising", ["Comparison", "Apply Denoising", 
                                                                                   "Plot Spectrum", "Show Image", "Clear Display"])
        self.peak_widget, self.peak_plot_widgets = self.create_section("Peak Fitting", ["Apply Peak Fitting", "Display Fitting Results","Data Analysis", "Save"])

        self.load_dock.addWidget(self.load_widget)
        self.phase_dock.addWidget(self.phase_widget)
        self.denoise_dock.addWidget(self.denoise_widget)
        self.peak_dock.addWidget(self.peak_widget)

        self.console = ConsoleWidget()
        self.console_dock.addWidget(self.console)

        self.log_out = QPlainTextEdit()
        self.log_out.setReadOnly(True)
        self.log_out.setPlaceholderText("Log messages will appear here...")
        self.console_dock.addWidget(self.log_out)
        
        self.log_message(f"Application started. Initial device: {get_device()}") # Use get_device()
        self.connect_buttons()

    def log_message(self, message):
        self.log_out.appendPlainText(message)
        print(message)

    def create_section(self, title, buttons):
        widget = QWidget()
        layout = QHBoxLayout() 
        button_layout = QVBoxLayout()
        for btn_text in buttons:
            button = QPushButton(btn_text)
            button.setFixedWidth(150)
            button_layout.addWidget(button)
            if btn_text == "Data2npy":
                self.save_path_input = QLineEdit()
                self.save_path_input.setPlaceholderText("Enter path to save .npy file")
                self.save_path_input.setText("GUI/data/data.npy") 
                button_layout.addWidget(self.save_path_input)
            if btn_text == "Load":
                self.file_extension_input = QLineEdit() 
                self.file_extension_input.setPlaceholderText("Enter series base (e.g. image_ for image_0.npy)")
                self.file_extension_input.setText("image") 
                button_layout.addWidget(self.file_extension_input)
        layout.addLayout(button_layout)

        plot_widgets = {}
        container = QWidget()
        container_layout = QStackedLayout(container)
        
        plot_widgets["spectrum"] = pg.PlotWidget()
        plot_widgets["spectrum"].setMinimumSize(500, 300)
        container_layout.addWidget(plot_widgets["spectrum"])

        if title == "Peak Fitting":
            plot_widgets["image_wrapper"] = QWidget()
            plot_widgets["image_wrapper"].setMinimumSize(500,300)
            container_layout.addWidget(plot_widgets["image_wrapper"])
        else:
            plot_widgets["image"] = pg.GraphicsLayoutWidget()
            plot_widgets["image"].setMinimumSize(500, 300)
            container_layout.addWidget(plot_widgets["image"])
            
        plot_widgets["container"] = container_layout
        layout.addWidget(container)
        widget.setLayout(layout)
        return widget, plot_widgets

    def _handle_data_to_npy(self):
        output_npy_path = self.save_path_input.text().strip()
        if not output_npy_path:
            self.log_message("Error: Output .npy file path is not specified.")
            QMessageBox.warning(self, "Path Error", "Please specify the output .npy file path.")
            return
        if not output_npy_path.endswith(".npy"):
            output_npy_path += ".npy"
            self.log_message(f"Output path modified to: {output_npy_path}")

        input_path = QFileDialog.getExistingDirectory(self, "Select Input Data Folder (Bruker, DICOM series)")
        if not input_path: 
            input_path, _ = QFileDialog.getOpenFileName(self, "Select Input Data File (NIfTI, single DICOM)", "", "All Files (*);;NIFTI (*.nii *.nii.gz);;DICOM (*.dcm)")
        
        if not input_path: 
            self.log_message("Data to NPY conversion cancelled: No input path selected.")
            return
            
        self.log_message(f"Input path selected for .npy conversion: {input_path}")
        self.log_message(f"Output .npy file will be: {output_npy_path}")
        
        try:
            convert_data_to_npy(self.log_message, input_path, output_npy_path)
        except Exception as e:
            self.log_message(f"Error during Data2npy operation: {str(e)}")
            QMessageBox.critical(self, "Conversion Error", f"An error occurred during .npy conversion: {str(e)}")


    def connect_buttons(self):
        load_buttons = self.load_widget.findChildren(QPushButton)
        for btn in load_buttons:
            if btn.text() == "Data2npy":
                btn.clicked.connect(self._handle_data_to_npy) 
            elif btn.text() == "Load":
                btn.clicked.connect(self.load_saved_npy)
            elif btn.text() == "Overlay":
                btn.clicked.connect(lambda: self.open_overlay_settings(self.I))
            elif btn.text() == "Plot Spectrum":
                btn.clicked.connect(lambda: self.open_spectrum_settings(self.I, self.load_plot_widgets))
            elif btn.text() == "Show Image":
                btn.clicked.connect(lambda: self.open_image_settings(self.I, self.load_plot_widgets))
            elif btn.text() == "Clear Display":
                btn.clicked.connect(lambda: self.clear_display(self.load_plot_widgets))

        phase_buttons = self.phase_widget.findChildren(QPushButton)
        for btn in phase_buttons:   
            if btn.text() == "Device": btn.clicked.connect(self.detect_device)
            elif btn.text() == "Normalize": btn.clicked.connect(self.open_normalization_settings)
            elif btn.text() == "Comparison": btn.clicked.connect(self.open_comparison_settings)
            elif btn.text() == "Apply Phase Correction": btn.clicked.connect(lambda: self.open_phase_correction_settings(self.I_norm))
            elif btn.text() == "Plot Spectrum": btn.clicked.connect(lambda: self.open_spectrum_settings(self.I_corrected, self.phase_plot_widgets))
            elif btn.text() == "Show Image": btn.clicked.connect(lambda: self.open_image_settings(self.I_corrected, self.phase_plot_widgets))
            elif btn.text() == "Clear Display": btn.clicked.connect(lambda: self.clear_display(self.phase_plot_widgets))

        denoise_buttons = self.denoise_widget.findChildren(QPushButton)
        for btn in denoise_buttons:
            if btn.text() == "Comparison": btn.clicked.connect(lambda: self.open_denoise_comparison_settings(self.I_corrected))
            elif btn.text() == "Apply Denoising": btn.clicked.connect(lambda: self.open_denoising_settings(self.I_corrected))
            elif btn.text() == "Plot Spectrum": btn.clicked.connect(lambda: self.open_spectrum_settings(self.I_denoised, self.denoise_plot_widgets))
            elif btn.text() == "Show Image": btn.clicked.connect(lambda: self.open_image_settings(self.I_denoised, self.denoise_plot_widgets))
            elif btn.text() == "Clear Display": btn.clicked.connect(lambda: self.clear_display(self.denoise_plot_widgets))

        peak_buttons = self.peak_widget.findChildren(QPushButton)
        for btn in peak_buttons:
            if btn.text() == "Apply Peak Fitting": btn.clicked.connect(lambda: self.open_peak_fitting_settings(self.I_denoised))
            elif btn.text() == "Display Fitting Results": btn.clicked.connect(lambda: self.display_fitting_results(self.I_fitted, self.peak_plot_widgets, self.data_before_fitting))
            elif btn.text() == "Data Analysis": btn.clicked.connect(lambda: self.open_data_analysis_settings(self.I_fitted, self.I))


    def clear_display(self,target_plot_widgets):
        try:
            if "spectrum" in target_plot_widgets: target_plot_widgets["spectrum"].clear()
            if "image" in target_plot_widgets and hasattr(target_plot_widgets["image"], 'clear'): 
                target_plot_widgets["image"].clear()
                if hasattr(target_plot_widgets["image"], 'setCentralItem'): target_plot_widgets["image"].setCentralItem(None)
            elif "image_wrapper" in target_plot_widgets and target_plot_widgets["image_wrapper"].layout() is not None:
                layout = target_plot_widgets["image_wrapper"].layout()
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget: widget.deleteLater()
            self.log_message("Cleared display.")
        except Exception as e: self.log_message(f"Could not clear display: {e}")

    def detect_device(self):
        new_device = set_device() # Calls set_device from config.settings, auto-detects
        self.log_message(f"Device set to: {new_device}")


    def load_saved_npy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select NumPy File or First File in Series (e.g., image_0.npy)", "", "NumPy Array (*.npy);;All Files (*)")
        if not path:
            dir_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing .npy Series")
            if not dir_path: self.log_message("Error: No file or directory selected."); return
            path = dir_path 
        self.log_message(f"Selected path for .npy loading: {path}")
        try:
            data_array = None
            if os.path.isfile(path) and path.endswith(".npy"):
                base_name, _ = os.path.splitext(path)
                series_match = re.match(r'(.*?)_(\d+)$', base_name) 
                if series_match:
                    name_prefix = series_match.group(1)
                    dir_path = os.path.dirname(path)
                    file_pattern = os.path.join(dir_path, f"{os.path.basename(name_prefix)}_*.npy")
                    file_paths = sorted(glob.glob(file_pattern))
                    if len(file_paths) > 1:
                        data_list = [np.load(fp) for fp in file_paths]
                        data_array = np.array(data_list)
                        self.log_message(f"Loaded {len(file_paths)} files as series, stacked shape: {data_array.shape}")
                    else: 
                        data_array = np.load(path)
                        self.log_message(f"Data loaded as single file: {path}, Shape: {data_array.shape}")
                else: 
                    data_array = np.load(path)
                    self.log_message(f"Data loaded as single file: {path}, Shape: {data_array.shape}")
            elif os.path.isdir(path): 
                file_paths = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npy")])
                if not file_paths: raise FileNotFoundError(f"No .npy files in directory: {path}")
                data_list = [np.load(fp) for fp in file_paths]
                data_array = np.array(data_list)
                self.log_message(f"Loaded {len(file_paths)} .npy files from directory, stacked shape: {data_array.shape}")
            else: raise ValueError("Invalid selection. Must be .npy or directory of .npy files.")
            self.I = data_array
        except Exception as e: self.log_message(f"Error loading .npy: {e}")
        
    def open_overlay_settings(self, dmi_data):
        if dmi_data is None: self.log_message("Error: Load DMI data first for overlay."); return
        overlay_win = OverlayWindow(dmi_data, current_structural=self.structural_image, parent=self)
        if overlay_win.exec(): 
            self.structural_image = overlay_win.structural_image 
        self.active_windows.append(overlay_win)

    def open_spectrum_settings(self, data_to_plot, target_plot_widgets):
        if data_to_plot is None: self.log_message("Error: No data for spectrum plot."); return
        spec_len = data_to_plot.shape[1] if data_to_plot.ndim > 1 else data_to_plot.shape[0]
        spectrum_win = SpectrumSettingsWindow(self, spectrum_length=spec_len)
        if spectrum_win.exec():
            params = spectrum_win.get_parameters()
            self.log_message(f"Spectrum Params: {params}")
            plotters.plot_spectrum_detailed(data_to_plot, target_plot_widgets, params, self.log_message)

    def open_image_settings(self, data_to_show, target_plot_widgets):
        if data_to_show is None: self.log_message("Error: No data for image display."); return
        image_win = ImageSettingsWindow(self, image_shape=data_to_show.shape)
        if image_win.exec():
            params = image_win.get_parameters()
            self.log_message(f"Image Params: {params}")
            freq_fallback_shape = self.I.shape if self.I is not None and self.I.ndim > 1 else data_to_show.shape
            plotters.show_image_detailed(data_to_show, target_plot_widgets, params, self.log_message, 
                                          full_data_freq_dim_len=freq_fallback_shape[1] if len(freq_fallback_shape)>1 else 0)

    def open_phase_correction_settings(self,data_to_correct):
        if data_to_correct is None: self.log_message("Error: self.I_norm is None."); return
        phase_win = PhaseCorrectionSettingsWindow(self)
        if phase_win.exec():
            params = phase_win.get_parameters()
            self.log_message(f"Phase Correction Params: {params}")
            self.apply_phase_correction(data_to_correct, params)

    def apply_phase_correction(self,data_to_correct, params): 
        data_complex = data_to_correct
        if data_complex.shape[-1] == 2: 
             data_complex = data_complex[..., 0] + 1j * data_complex[..., 1]
        original_shape = data_complex.shape
        data_complex_padded = data_complex 
        if len(original_shape) < 5 and data_complex.ndim > 0 : 
            squeezed_data = data_complex.squeeze()
            current_dims = len(squeezed_data.shape)
            if current_dims < 5 and current_dims > 0 : 
                 reshape_dims = (1,) * (5 - current_dims)
                 data_complex_padded = squeezed_data.reshape(squeezed_data.shape + reshape_dims)
            elif current_dims == 0: self.log_message("Error: Cannot apply phase correction to scalar data."); return
            else: data_complex_padded = squeezed_data if current_dims == 5 else data_complex 
        elif data_complex.ndim == 0: self.log_message("Error: Cannot apply phase correction to scalar data."); return
        try:
            # Pass get_device() for device-aware operations in phase_correct_gpu
            corrected_data_padded, _ = phase_correct_gpu(data_complex_padded, device=get_device(), **params)
            self.I_corrected = corrected_data_padded.reshape(original_shape) 
            self.log_message(f"Phase correction applied. I_corrected shape: {self.I_corrected.shape}")
        except Exception as e: self.log_message(f"Error phase correction: {e}")


    def open_normalization_settings(self):
        if self.I is None: self.log_message("Error: Load data first."); return
        norm_win = NormalizationSettingsWindow(self)
        if norm_win.exec():
            params = norm_win.get_parameters()
            self.log_message(f"Normalization Params: {params}")
            self.normalize_image(params)
    
    def normalize_image(self, params):
        # ... (normalization logic remains, ensure it doesn't use global DEVICE directly if not needed)
        # For this subtask, focus is on DEVICE usage in GUI, not necessarily deep in backend if already flexible.
        # Assuming normalize_data in preprocessing.normalization doesn't directly use a global DEVICE.
        method, mode, bg_region_str = params["method"], params["mode"], params["bg_region"]
        data_to_norm = self.I
        data_complex, flag_complex = data_to_norm, False
        if data_to_norm.ndim > 0 and data_to_norm.shape[-1] == 2: # Check ndim > 0 for safety
            data_complex = data_to_norm[...,0] + 1j*data_to_norm[...,1]; flag_complex = True
        original_shape = data_complex.shape
        current_dims = len(original_shape)
        data_for_norm = data_complex
        if current_dims < 5 and current_dims > 0: 
            reshape_dims = (1,) * (5-current_dims)
            data_for_norm = data_complex.reshape(original_shape + reshape_dims)
        elif current_dims == 0: self.log_message("Error: Cannot normalize scalar data."); return
        bg_data_slice = None
        if bg_region_str:
            try:
                slices = []
                for item in bg_region_str.split(','):
                    item = item.strip()
                    if item == ':': slices.append(slice(None))
                    elif ':' in item: start,end = map(int, item.split(':')); slices.append(slice(start,end))
                    else: slices.append(int(item))
                bg_data_slice = data_for_norm[tuple(slices[:data_for_norm.ndim])] 
            except Exception as e: self.log_message(f"Error parsing BG region: {e}. Using None for BG.")
        try:
            data_norm_result_padded = normalize_data(data_for_norm, method, mode, bg_data_slice, flag_complex)
            self.I_norm = data_norm_result_padded.reshape(original_shape) 
            self.log_message(f"Normalization applied. I_norm shape: {self.I_norm.shape}")
        except Exception as e: self.log_message(f"Error normalization: {e}")
                
    def open_comparison_settings(self):
        if self.I_norm is None: self.log_message("Error: I_norm is None."); return
        compare_win = ComparisonSettingsWindow(self.I_norm, self.apply_single_spectrum_phase_correction, parent=self)
        self.active_windows.append(compare_win); compare_win.show()


    def apply_single_spectrum_phase_correction(self, spectrum_data, method, params_dict_str_vals):
        params = {
            "lr": float(params_dict_str_vals.get("learning_rate", 0.05)), 
            "n_iters": int(params_dict_str_vals.get("num_iterations", 200)), 
            "num_basis": int(params_dict_str_vals.get("num_fourier_basis", 8)), 
            "half_peak_width": int(params_dict_str_vals.get("half_peak_width", 5)),
            "degree": int(params_dict_str_vals.get("b-spline_degree", 3)), 
            "peak_list": None
        }
        peak_list_str = params_dict_str_vals.get("peak_list_(comma-sep,_e.g._10,20)", "None") 
        if peak_list_str.lower() != "none" and peak_list_str:
            try: params["peak_list"] = [float(p.strip()) for p in peak_list_str.split(',')]
            except: params["peak_list"] = None
        
        data_complex_single = np.zeros((1, spectrum_data.shape[0], 1, 1, 1), dtype=np.complex64)
        data_complex_single[0, :, 0, 0, 0] = spectrum_data
        try:
            # Pass get_device() for device-aware operations in phase_correct_gpu
            corrected_spectrum, _ = phase_correct_gpu(data_complex_single, device=get_device(), method=method, **params)
            return corrected_spectrum.squeeze()
        except Exception as e: self.log_message(f"Single spectrum phase correction ({method}) failed: {e}"); return None


    def open_denoise_comparison_settings(self,data_to_compare):
        if data_to_compare is None: self.log_message("Error: self.I_corrected is None."); return
        denoise_compare_win = DenoiseComparisonSettingsWindow(data_to_compare, self.apply_single_spectrum_denoising, parent=self)
        self.active_windows.append(denoise_compare_win); denoise_compare_win.show()

    def apply_single_spectrum_denoising(self, spectrum_data, method, params_dict_str_vals):
        F_len = spectrum_data.shape[0]
        data_5d = np.zeros((1, F_len, 1, 1, 1), dtype=spectrum_data.dtype) 
        data_5d[0, :, 0, 0, 0] = spectrum_data
        internal_params = {}
        for k, v_str in params_dict_str_vals.items():
            try: v_float = float(v_str); internal_params[k] = int(v_float) if v_float.is_integer() else v_float
            except ValueError: internal_params[k] = v_str 
        try:
            if method == "UNet Model" or method == "Transform Model":
                spec_reshaped = np.abs(spectrum_data).reshape(1, 1, F_len)
                current_F = spec_reshaped.shape[2]
                MODEL_EXPECTED_LEN = 256 
                if current_F != MODEL_EXPECTED_LEN:
                    pad_width = ((0,0),(0,0),(0, MODEL_EXPECTED_LEN - current_F))
                    spec_padded = np.pad(spec_reshaped, pad_width, mode='constant')
                else: spec_padded = spec_reshaped
                spec_tensor = torch.from_numpy(spec_padded).to(get_device()).to(torch.float32) # Use get_device()
                peaks_parsed = [int(p.strip()) for p in internal_params.get("peak_list","").split(',') if p.strip()]
                peaks_tensor = torch.tensor(peaks_parsed, dtype=torch.long).to(get_device()).unsqueeze(0) # Use get_device()
                model_instance = None
                if method == "UNet Model": model_instance = UNet1DWithPEPeak(in_channels=1, out_channels=1).to(get_device()) # Use get_device()
                else: model_instance = TransformerDenoiser(num_layers=4).to(get_device()) # Use get_device()
                model_instance.load_state_dict(torch.load(internal_params["model_path"], map_location=get_device())) # Use get_device()
                model_instance.eval()
                with torch.no_grad(): denoised_tensor = model_instance(spec_tensor, peaks_tensor).cpu().numpy()
                return denoised_tensor[0,0,:current_F] 
            else: 
                # apply_spectral_denoising_batch does not use DEVICE
                return apply_spectral_denoising_batch(data_5d, method, **internal_params)
        except Exception as e: self.log_message(f"Single spectrum denoising ({method}) failed: {e}"); return None


    def open_denoising_settings(self, data_to_denoise):
        if data_to_denoise is None: self.log_message("Error: self.I_corrected is None."); return
        denoise_win = DenoiseSettingsWindow(self)
        if denoise_win.exec():
            method, params = denoise_win.get_parameters()
            self.log_message(f"Denoising method: {method}, Params: {params}")
            self.apply_denoising(data_to_denoise, method, **params)

    def apply_denoising(self, data_to_denoise, method, params_dict_str_vals):
        if data_to_denoise is None: self.log_message("Error: No data for apply_denoising."); return
        T, F, X, Y, Z = data_to_denoise.shape[:5]
        internal_params = {} 
        for k, v_str in params_dict_str_vals.items():
            try: v_float = float(v_str); internal_params[k] = int(v_float) if v_float.is_integer() else v_float
            except ValueError: internal_params[k] = v_str
        try:
            denoised_result = None
            if method == "UNet Model" or method == "Transform Model":
                data_permuted = data_to_denoise.transpose(0,2,3,4,1) 
                data_reshaped = np.abs(data_permuted).reshape(-1, 1, F) 
                N_spectra = data_reshaped.shape[0]
                peaks_parsed = [int(p.strip()) for p in internal_params.get("peak_list","").split(',') if p.strip()]
                peaks_tensor_batch = torch.tensor(peaks_parsed,dtype=torch.long).to(get_device()).unsqueeze(0).expand(N_spectra, -1) # Use get_device()
                MODEL_EXPECTED_LEN = 256
                if F != MODEL_EXPECTED_LEN:
                    pad_width = ((0,0),(0,0),(0, MODEL_EXPECTED_LEN - F))
                    data_padded = np.pad(data_reshaped, pad_width, mode='constant')
                else: data_padded = data_reshaped
                data_tensor = torch.from_numpy(data_padded).to(get_device()).to(torch.float32) # Use get_device()
                model_instance = None
                if method == "UNet Model": model_instance = UNet1DWithPEPeak(in_channels=1, out_channels=1).to(get_device()) # Use get_device()
                else: model_instance = TransformerDenoiser(num_layers=4).to(get_device()) # Use get_device()
                model_instance.load_state_dict(torch.load(internal_params["model_path"], map_location=get_device())) # Use get_device()
                model_instance.eval()
                with torch.no_grad(): denoised_full = model_instance(data_tensor, peaks_tensor_batch).cpu().numpy()
                denoised_unpadded = denoised_full[:,:,:F] 
                denoised_result = denoised_unpadded.reshape(T,X,Y,Z,F).transpose(0,4,1,2,3) 
            else: # Classic denoisers do not use DEVICE
                denoised_result = apply_spectral_denoising_batch(data_to_denoise, method, **internal_params)
            self.I_denoised = denoised_result
            self.log_message(f"Denoising applied ({method}). I_denoised shape: {self.I_denoised.shape}")
        except Exception as e: self.log_message(f"Error applying batch denoising ({method}): {e}")


    def open_peak_fitting_settings(self, data_to_fit):
        if data_to_fit is None: self.log_message("Error: self.I_denoised is None."); return
        T, F, X, Y, Z = data_to_fit.shape[:5]
        peak_fit_win = PeakFittingSettingsWindow(self)
        if peak_fit_win.exec():
            params = peak_fit_win.get_parameters() 
            self.log_message(f"Peak Fitting Params: {params}")
            value_type = params.pop("value_type", "Magnitude")
            data_for_fitting = np.abs(data_to_fit) if value_type == "Magnitude" else np.abs(np.real(data_to_fit))
            self.data_before_fitting = data_for_fitting.copy()
            try:
                param_peak = params.pop("param_peak") 
                param_gamma = params.pop("param_gamma") 
                # fit_volume_gpu uses get_device() internally if it needs it (though current impl doesn't explicitly show it)
                fitted, components, a, gamma, bg = fit_volume_gpu(data_for_fitting, np.arange(F), 
                                                                  param_peak, param_gamma, **params)
                components_transposed = components.transpose(3,5,0,1,2,4) 
                self.I_fitted = {'fitted_data': fitted, 'separate_peaks': components_transposed,
                                 'amplitude': a, 'gamma': gamma, 'bg': bg}
                self.log_message(f"Peak fitting completed. Fitted: {fitted.shape}, Components: {components_transposed.shape}")
            except Exception as e: self.log_message(f"Error peak fitting: {e}")

    def display_fitting_results(self, fitted_data_dict, target_plot_widgets, raw_data_for_spectrum_plot):
        if fitted_data_dict is None: self.log_message("Error: self.I_fitted is None."); return
        
        display_settings_win = PeakFittingDisplayWindow(self)
        if display_settings_win.exec():
            display_type, display_params = display_settings_win.get_parameters()
            self.log_message(f"Fitting Display Params: {display_params}")

            if display_type == "Spectrum":
                if raw_data_for_spectrum_plot is None:
                    self.log_message("Error: data_before_fitting is None for spectrum plot.")
                    return
                plotters.plot_fitted_spectrum(
                    target_plot_widget=target_plot_widgets['spectrum'], 
                    container_widget=target_plot_widgets['container'], 
                    fitted_data_dict=fitted_data_dict, 
                    raw_data_for_spectrum_plot=raw_data_for_spectrum_plot, 
                    coord_params=display_params, 
                    log_message_func=self.log_message
                )
            elif display_type == "Image Overlay":
                image_wrapper = target_plot_widgets.get("image_wrapper")
                if image_wrapper is None:
                     self.log_message("Error: image_wrapper not found for Peak Fitting.")
                     return
                target_plot_widgets["container"].setCurrentWidget(image_wrapper)
                plotters.plot_peak_fitting_image_overlay(
                    image_wrapper_widget=image_wrapper,
                    fitted_data_dict=fitted_data_dict,
                    structural_image_data=self.structural_image, 
                    display_params=display_params,
                    log_message_func=self.log_message,
                    q_file_dialog_parent=self 
                )


    def open_data_analysis_settings(self, fitted_data_dict, raw_data_for_analysis): 
        if fitted_data_dict is None: self.log_message("Error: self.I_fitted is None."); return
        if raw_data_for_analysis is None: self.log_message("Error: self.I is None."); return
        
        analysis_win = DataAnalysisWindow(fitted_data_dict, raw_data_for_analysis, parent=self)
        self.active_windows.append(analysis_win); analysis_win.show()


# --- All QDialog and QWidget subclasses ---
# (SpectrumSettingsWindow, ImageSettingsWindow, etc. remain unchanged from previous visualization/gui.py content)
# ... (The rest of the QDialog/QWidget classes: SpectrumSettingsWindow, ImageSettingsWindow, etc.)
# For brevity, the unchanged parts of these classes are not repeated here but are assumed to be part of the file.
# The critical change is the `DEVICE` handling in MainApp and how it's passed or accessed by other parts.

class SpectrumSettingsWindow(QDialog):
    def __init__(self, parent=None, spectrum_length=512):
        super().__init__(parent)
        self.setWindowTitle("Spectrum Settings")
        layout = QFormLayout()
        self.value_type_combo = QComboBox()
        self.value_type_combo.addItems(["Magnitude", "Imaginary", "Real"])
        layout.addRow(QLabel("Value Type:"), self.value_type_combo)
        self.abs_checkbox = QCheckBox("Show Absolute Value")
        layout.addRow(self.abs_checkbox)
        self.display_method_combo = QComboBox()
        self.display_method_combo.addItems(["Average", "Maximum"])
        layout.addRow(QLabel("Time Series Display Method:"), self.display_method_combo)
        self.index_input = QLineEdit()
        self.index_input.setPlaceholderText("Enter indices (comma-separated)")
        layout.addRow(QLabel("Time Series Index:"), self.index_input)
        self.freq_range_input = QLineEdit()
        self.freq_range_input.setPlaceholderText(f"Enter frequency range (default: 0:{spectrum_length})")
        self.freq_range_input.setText(f"0:{spectrum_length}")
        layout.addRow(QLabel("Frequency Range:"), self.freq_range_input)
        self.confirm_button = QPushButton("OK")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)
        self.setLayout(layout)

    def get_parameters(self):
        return {
            "value_type": self.value_type_combo.currentText(),
            "display_method": self.display_method_combo.currentText(),
            "time_series_index": self.index_input.text(),
            "frequency_range": self.freq_range_input.text(),
            "show_abs": self.abs_checkbox.isChecked()
        }
    
class ImageSettingsWindow(QDialog):
    def __init__(self, parent=None, image_shape=(21,250,10,10,10,2)): 
        super().__init__(parent)
        self.setWindowTitle("Image Data Settings")
        layout = QFormLayout()
        self.value_type_combo = QComboBox()
        self.value_type_combo.addItems(["Magnitude", "Imaginary", "Real"])
        layout.addRow(QLabel("Value Type:"), self.value_type_combo)
        self.abs_checkbox = QCheckBox("Show Absolute Value")
        layout.addRow(self.abs_checkbox)
        self.negative_checkbox = QCheckBox("Show Negative Value")
        layout.addRow(self.negative_checkbox)
        self.display_method_combo = QComboBox()
        self.display_method_combo.addItems(["Average", "Maximum"])
        layout.addRow(QLabel("Time Series Display Method:"), self.display_method_combo)
        self.index_input = QLineEdit()
        self.index_input.setText("1,3,5,7,9,11") 
        layout.addRow(QLabel("Time Series Index:"), self.index_input)
        self.freq_range_input = QLineEdit()
        self.freq_range_input.setPlaceholderText(f"Enter frequency range (default: 0:{image_shape[1]})")
        self.freq_range_input.setText(f"0:{image_shape[1]}")
        layout.addRow(QLabel("Frequency Range:"), self.freq_range_input)
        self.slice_input = QLineEdit()
        self.slice_input.setPlaceholderText(f"dim,slices (e.g., 4,{image_shape[4]//2 if len(image_shape) > 4 else 'N/A'})")
        layout.addRow(QLabel("Slice Selection:"), self.slice_input)
        self.roi_input = QLineEdit()
        self.roi_input.setPlaceholderText("Enter ROI coordinates (e.g., 4:6,5:8 or 4,5)")
        layout.addRow(QLabel("Region of Interest:"), self.roi_input)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "gray", "plasma", "inferno", "magma"])
        layout.addRow(QLabel("Colormap:"), self.colormap_combo)
        self.confirm_button = QPushButton("OK")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)
        self.setLayout(layout)

    def get_parameters(self):
        return {
            "value_type": self.value_type_combo.currentText(),
            "display_method": self.display_method_combo.currentText(),
            "time_series_index": self.index_input.text(),
            "frequency_range": self.freq_range_input.text(),
            "slice_selection": self.slice_input.text(),
            "roi": self.roi_input.text(),
            "colormap": self.colormap_combo.currentText(),
            "show_abs": self.abs_checkbox.isChecked(),
            "show_negative": self.negative_checkbox.isChecked()
        }

class OverlayWindow(QDialog): 
    def __init__(self, dmi_image_data, current_structural=None, parent=None): 
        super().__init__(parent) 
        self.setWindowTitle("Overlay Structural Image")
        self.setMinimumSize(800, 600)
        self.dmi_image0 = dmi_image_data
        self.structural_image = current_structural 
        self.dmi_image_processed = None 
        self.parent_app = parent 
        layout = QVBoxLayout()
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Select structural image (.npy or .nii)")
        if self.structural_image is not None: self.file_input.setText("[Using previously loaded/selected image]")
        browse_button = QPushButton("Browse"); browse_button.clicked.connect(self.browse_structural_image)
        file_layout.addWidget(QLabel("Structural Image Path:")); file_layout.addWidget(self.file_input); file_layout.addWidget(browse_button)
        layout.addLayout(file_layout)
        self.time_index_input = QLineEdit("0"); layout.addWidget(QLabel("DMI Time Index:")); layout.addWidget(self.time_index_input)
        self.freq_range_input = QLineEdit("0:40"); layout.addWidget(QLabel("Frequency Range (start:end):")); layout.addWidget(self.freq_range_input)
        self.apply_button = QPushButton("Prepare/Update Overlay"); self.apply_button.clicked.connect(self.prepare_overlay); layout.addWidget(self.apply_button)
        self.plot_widget = pg.GraphicsLayoutWidget(); self.view = self.plot_widget.addViewBox(); self.view_items = []; self.view.setAspectLocked(True); layout.addWidget(self.plot_widget)
        slider_layout = QHBoxLayout()
        self.dmi_slider = QSlider(Qt.Orientation.Horizontal); self.dmi_slider.valueChanged.connect(self.update_plot_overlay_call); self.dmi_index_label = QLabel("0"); self.dmi_slider.valueChanged.connect(lambda val: self.dmi_index_label.setText(str(val)))
        slider_layout.addWidget(QLabel("DMI Slice:")); slider_layout.addWidget(self.dmi_slider); slider_layout.addWidget(self.dmi_index_label)
        self.struct_slider = QSlider(Qt.Orientation.Horizontal); self.struct_slider.valueChanged.connect(self.update_plot_overlay_call); self.struct_index_label = QLabel("0"); self.struct_slider.valueChanged.connect(lambda val: self.struct_index_label.setText(str(val)))
        slider_layout.addWidget(QLabel("Structural Slice:")); slider_layout.addWidget(self.struct_slider); slider_layout.addWidget(self.struct_index_label)
        layout.addLayout(slider_layout)
        self.ok_button = QPushButton("OK"); self.ok_button.clicked.connect(self.accept); layout.addWidget(self.ok_button)
        self.setLayout(layout)
        if self.structural_image is not None: self.prepare_overlay()

    def browse_structural_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Structural Image", "", "Images (*.npy *.nii *.nii.gz)")
        if file_path:
            self.file_input.setText(file_path)
            try:
                self.structural_image = np.load(file_path) if file_path.endswith(".npy") else nib.load(file_path).get_fdata()
                self.log_message(f"Structural image loaded from: {file_path}")
                self.prepare_overlay() 
            except Exception as e: self.log_message(f"Error loading structural image: {e}")

    def log_message(self, message): 
        if self.parent_app and hasattr(self.parent_app, 'log_message'): self.parent_app.log_message(message)
        else: print(message)

    def prepare_overlay(self):
        current_path_in_field = self.file_input.text().strip()
        if current_path_in_field and (self.structural_image is None or not current_path_in_field == "[Using previously loaded/selected image]"):
            if os.path.exists(current_path_in_field):
                try:
                    self.structural_image = np.load(current_path_in_field) if current_path_in_field.endswith(".npy") else nib.load(current_path_in_field).get_fdata()
                    self.log_message(f"Structural image (re)loaded from: {current_path_in_field}")
                except Exception as e: self.log_message(f"Error loading structural image from path '{current_path_in_field}': {e}"); return
            else: self.log_message(f"Error: Path for structural image does not exist: {current_path_in_field}"); return
        if self.structural_image is None: self.log_message("Error: Structural image is not available."); return
        try:
            time_idx = int(self.time_index_input.text().strip())
            freq_range_str = self.freq_range_input.text().strip()
            start, end = map(int, freq_range_str.split(":"))
            dmi_data_to_process = self.dmi_image0
            if dmi_data_to_process.shape[-1] == 2: dmi_data_to_process = dmi_data_to_process[..., 0] + 1j * dmi_data_to_process[..., 1]
            if not (0 <= time_idx < dmi_data_to_process.shape[0]): self.log_message(f"Error: Time index {time_idx} out of bounds."); return
            dmi_map_3d = np.abs(dmi_data_to_process[time_idx, start:end,...].squeeze().max(axis=0))
            if dmi_map_3d.ndim < 3: self.log_message("Error: Processed DMI data for overlay is not 3D."); return
            target_shape = self.structural_image.shape[:2]
            resized_slices = [cv2.resize(dmi_map_3d[:,:,z], (target_shape[1],target_shape[0]), interpolation=cv2.INTER_NEAREST) for z in range(dmi_map_3d.shape[2])]
            self.dmi_image_processed = np.stack(resized_slices, axis=2)
            self.struct_slider.setMaximum(self.structural_image.shape[2] - 1); self.struct_slider.setValue(self.structural_image.shape[2] // 2)
            self.dmi_slider.setMaximum(self.dmi_image_processed.shape[2] - 1); self.dmi_slider.setValue(self.dmi_image_processed.shape[2] // 2)
            self.update_plot_overlay_call()
            self.log_message("Overlay prepared/updated.")
        except Exception as e: self.log_message(f"Error in overlay prep: {e}")

    def update_plot_overlay_call(self): 
        if self.structural_image is None or self.dmi_image_processed is None: return
        if not self.view_items: 
            self.struct_img_item = pg.ImageItem(); self.dmi_img_item = pg.ImageItem()
            self.view.addItem(self.struct_img_item); self.view.addItem(self.dmi_img_item)
            self.view_items.extend([self.struct_img_item, self.dmi_img_item])
        s_idx, d_idx = self.struct_slider.value(), self.dmi_slider.value()
        structural_slice = self.structural_image[:,:,s_idx]
        dmi_slice = self.dmi_image_processed[:,:,d_idx]
        plotters.plot_structural_dmi_overlay(self.view, self.struct_img_item, self.dmi_img_item, structural_slice, dmi_slice, 0.5, "viridis", self.log_message)

class PhaseCorrectionSettingsWindow(QDialog): # ... (content as before)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phase Correction Settings")
        layout = QFormLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Zero-order", "First-order", "B-spline", "Fourier"])
        layout.addRow(QLabel("Select Phase Correction Method:"),self.method_combo)
        self.lr_input = QLineEdit("0.05")
        layout.addRow(QLabel("Learning Rate:"), self.lr_input)
        self.iter_input = QLineEdit("200")
        layout.addRow(QLabel("Number of Iterations:"), self.iter_input)
        self.fourier_basis_input = QLineEdit("8")
        layout.addRow(QLabel("Number of Fourier Basis:"), self.fourier_basis_input)
        self.peak_list_input = QLineEdit()
        self.peak_list_input.setPlaceholderText("e.g., 1,2,3")
        layout.addRow(QLabel("Peak List (B-spline):"), self.peak_list_input)
        self.peak_width_input = QLineEdit("5")
        layout.addRow(QLabel("Half Peak Width (B-spline):"), self.peak_width_input)
        self.bspline_degree_input = QLineEdit("3")
        layout.addRow(QLabel("B-spline Degree:"), self.bspline_degree_input)
        self.confirm_button = QPushButton("OK")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)
        self.setLayout(layout)
        self.method_combo.currentTextChanged.connect(self.update_visibility)
        self.update_visibility()

    def update_visibility(self):
        method = self.method_combo.currentText()
        is_fourier = (method == "Fourier")
        is_bspline = (method == "B-spline")
        self.fourier_basis_input.setVisible(is_fourier)
        self.peak_list_input.setVisible(is_bspline)
        self.peak_width_input.setVisible(is_bspline)
        self.bspline_degree_input.setVisible(is_bspline)

    def get_parameters(self):
        peak_list_val = None
        if self.peak_list_input.text():
            try: peak_list_val = [int(p.strip()) for p in self.peak_list_input.text().split(",")]
            except ValueError: 
                if self.parent() and hasattr(self.parent(), 'log_message'): self.parent().log_message("Error parsing peak list.")
        return {
            "method": self.method_combo.currentText(), "lr": float(self.lr_input.text()),
            "n_iters": int(self.iter_input.text()), "num_basis": int(self.fourier_basis_input.text()),
            "peak_list": peak_list_val, "half_peak_width": int(self.peak_width_input.text()),
            "degree": int(self.bspline_degree_input.text())
        }

class NormalizationSettingsWindow(QDialog): # ... (content as before)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Normalization Method")
        layout = QFormLayout()
        layout.addWidget(QLabel("Choose a normalization method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Max Abs", "Min-Max", "Background Z-score", "Background Mean Scaling"])
        layout.addWidget(self.method_combo)
        self.bg_label = QLabel("Background Region (e.g., 0,0:50,:,:,0):") 
        self.bg_input = QLineEdit("0,0:50,:,:,0") 
        layout.addWidget(self.bg_label)
        layout.addWidget(self.bg_input)
        layout.addWidget(QLabel("For complex data normalization:"))
        self.complex_mode_combo = QComboBox()
        self.complex_mode_combo.addItems(["Magnitude Only", "Real and Imaginary Separately","Complex as Whole"])
        layout.addWidget(self.complex_mode_combo)
        self.confirm_button = QPushButton("Apply")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)
        self.setLayout(layout)
        self.method_combo.currentTextChanged.connect(self.update_visibility)
        self.update_visibility()
    
    def update_visibility(self):
        show_bg = "Background" in self.method_combo.currentText()
        self.bg_label.setVisible(show_bg)
        self.bg_input.setVisible(show_bg)

    def get_parameters(self):
        return {
            "method": self.method_combo.currentText(),
            "mode": self.complex_mode_combo.currentText(),
            "bg_region": self.bg_input.text() if "Background" in self.method_combo.currentText() else None
        }

class ComparisonSettingsWindow(QWidget): # ... (content as before, ensure parent is passed to constructor if needed)
    def __init__(self, data_to_compare, apply_phase_func_callback, parent=None): 
        super().__init__(parent)
        self.setWindowTitle("Phase Correction Comparison")
        self.data = data_to_compare 
        self.apply_phase_func = apply_phase_func_callback 
        self.parent_app = parent 
        self.methods = ["Zero-order", "First-order", "B-spline", "Fourier"]
        self.method_checkboxes = {}
        self.method_param_forms = {}
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("Enter coordinate (t,x,y,z) e.g., 0,5,5,0 :"))
        self.coord_input = QLineEdit("0,0,0,0") 
        main_layout.addWidget(self.coord_input)
        main_layout.addWidget(QLabel("Select display type:"))
        self.display_mode = QComboBox()
        self.display_mode.addItems(["Magnitude", "Real", "Imaginary"])
        main_layout.addWidget(self.display_mode)
        self.abs_checkbox = QCheckBox("Show Absolute Value (for Real/Imaginary)")
        main_layout.addWidget(self.abs_checkbox)
        self.negative_checkbox = QCheckBox("Show Negative Value (for Real/Imaginary)")
        main_layout.addWidget(self.negative_checkbox)
        method_group = QGroupBox("Select Phase Correction Methods")
        method_layout_scroll = QVBoxLayout() 
        for method in self.methods:
            cb = QCheckBox(method)
            cb.stateChanged.connect(self.update_parameter_forms)
            self.method_checkboxes[method] = cb
            method_layout_scroll.addWidget(cb)
            form = QFormLayout()
            container = QWidget()
            container.setLayout(form)
            container.setVisible(False)
            self.method_param_forms[method] = (container, form)
            method_layout_scroll.addWidget(container)
        method_group.setLayout(method_layout_scroll)
        scroll = QScrollArea()
        scroll.setWidget(method_group)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)
        self.apply_button = QPushButton("Apply & Plot Comparison")
        self.apply_button.clicked.connect(self.run_comparison_plot) 
        main_layout.addWidget(self.apply_button)
        self.plot_widget = pg.PlotWidget(title="Phase Correction Comparison") # This is the target for the plotter
        main_layout.addWidget(self.plot_widget)
        self.setLayout(main_layout)
        self.update_parameter_forms() 

    def log_message(self, message): 
        if self.parent_app and hasattr(self.parent_app, 'log_message'):
            self.parent_app.log_message(message)
        else: print(message)

    def update_parameter_forms(self, _state=None): 
        for method, checkbox in self.method_checkboxes.items():
            container, form = self.method_param_forms[method]
            if checkbox.isChecked():
                container.setVisible(True)
                if form.rowCount() == 0: 
                    self.populate_form(method, form)
            else:
                container.setVisible(False)

    def populate_form(self, method, form):
        form.addRow("Learning Rate:", QLineEdit("0.05"))
        form.addRow("Num Iterations:", QLineEdit("200"))
        if method == "Fourier":
            form.addRow("Num Fourier Basis:", QLineEdit("8"))
        elif method == "B-spline":
            form.addRow("Peak List (comma-sep, e.g. 10,20):", QLineEdit("None"))
            form.addRow("Half Peak Width:", QLineEdit("5"))
            form.addRow("B-spline Degree:", QLineEdit("3"))
    
    def run_comparison_plot(self): 
        coord_str = self.coord_input.text()
        try:
            t, x, y, z = map(int, coord_str.split(","))
            spectrum_original = self.data[t, :, x, y, z].copy() 
        except Exception as e:
            self.log_message(f"Invalid coordinate input: {coord_str}. Error: {e}")
            QMessageBox.warning(self, "Invalid Input", f"Coordinate format is invalid: {e}")
            return
        
        corrected_spectra_results = {}
        for method_name, cb in self.method_checkboxes.items(): 
            if cb.isChecked():
                _, form_layout = self.method_param_forms[method_name] 
                params_dict = {} 
                for i in range(form_layout.rowCount()):
                    label_widget = form_layout.itemAt(i, QFormLayout.ItemRole.LabelRole).widget()
                    value_widget = form_layout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget()
                    # Robust key generation
                    label_text = label_widget.text().strip(':').lower().replace(' ', '_')
                    label_text = label_text.replace('(b-spline)','').replace('(comma-sep,_e.g._10,20)','').replace('num_fourier_basis','num_fourier_basis') # ensure consistency
                    params_dict[label_text] = value_widget.text()
                
                corrected_spec = self.apply_phase_func(spectrum_original.copy(), method_name, params_dict)
                if corrected_spec is not None:
                    corrected_spectra_results[method_name] = corrected_spec
        
        display_params = {
            "display_type": self.display_mode.currentText(),
            "show_abs": self.abs_checkbox.isChecked(),
            "show_negative": self.negative_checkbox.isChecked()
        }
        plotters.plot_comparison_spectra(self.plot_widget, spectrum_original, corrected_spectra_results, display_params, self.log_message)


class DenoiseComparisonSettingsWindow(QWidget): 
    def __init__(self, data_to_compare, apply_denoising_func_callback, parent=None): 
        super().__init__(parent)
        self.setWindowTitle("Denoising Comparison")
        self.data = data_to_compare 
        self.apply_denoising_func = apply_denoising_func_callback 
        self.parent_app = parent
        self.methods = ["Mean Filter", "Median Filter", "Gaussian Filter", 
                        "Singular Value Decomposition", "Principal Component Analysis", 
                        "Savitzky-Golay Filter", "Wavelet Thresholding", "Fourier Filter",
                        "Total Variation", "Wiener Filter", "UNet Model","Transform Model"]
        self.method_blocks = []
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Enter coordinate (t,x,y,z) e.g., 0,5,5,0 :"))
        self.coord_input = QLineEdit("0,0,0,0")
        layout.addWidget(self.coord_input)
        layout.addWidget(QLabel("Select data type to denoise:"))
        self.value_type = QComboBox()
        self.value_type.addItems(["Magnitude", "Real"]) 
        layout.addWidget(self.value_type)
        self.method_area_widget = QWidget() 
        self.method_area_layout = QVBoxLayout(self.method_area_widget) 
        self.add_method_block() 
        self.add_method_btn = QPushButton("Add Another Method for Comparison")
        self.add_method_btn.clicked.connect(self.add_method_block)
        layout.addWidget(self.add_method_btn)
        scroll = QScrollArea()
        scroll.setWidget(self.method_area_widget) 
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(200) 
        layout.addWidget(scroll)
        self.show_original_checkbox = QCheckBox("Show Original Noisy Spectrum")
        layout.addWidget(self.show_original_checkbox)
        self.abs_checkbox = QCheckBox("Show Absolute Value (if Real)") 
        layout.addWidget(self.abs_checkbox)
        self.negative_checkbox = QCheckBox("Show Negative Value (if Real)") 
        layout.addWidget(self.negative_checkbox)
        self.apply_button = QPushButton("Apply & Plot Comparison")
        self.apply_button.clicked.connect(self.run_comparison_plot) 
        layout.addWidget(self.apply_button)
        self.plot_widget = pg.PlotWidget(title="Denoising Comparison")
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)     

    def log_message(self, message): 
        if self.parent_app and hasattr(self.parent_app, 'log_message'):
            self.parent_app.log_message(message)
        else: print(message)

    def add_method_block(self):
        block_widget = QWidget() 
        method_layout = QFormLayout(block_widget) 
        method_selector = QComboBox(); method_selector.addItems(self.methods)
        method_layout.addRow("Method:", method_selector)
        param_inputs_dict = {} 
        def update_params_for_block():
            while method_layout.rowCount() > 1: method_layout.removeRow(1)
            param_inputs_dict.clear() 
            current_method = method_selector.currentText()
            if current_method in ["Mean Filter", "Median Filter"]: le = QLineEdit("5"); param_inputs_dict["window_size"] = le; method_layout.addRow("Window Size:", le)
            elif current_method == "Gaussian Filter": le = QLineEdit("1.0"); param_inputs_dict["sigma"] = le; method_layout.addRow("Sigma:", le)
            elif current_method in ["Singular Value Decomposition", "Principal Component Analysis"]: le = QLineEdit("5"); param_inputs_dict["num_components"] = le; method_layout.addRow("Num Components:", le)
            elif current_method == "Savitzky-Golay Filter":
                le1 = QLineEdit("9"); param_inputs_dict["window_size"] = le1; method_layout.addRow("Window Size:", le1)
                le2 = QLineEdit("3"); param_inputs_dict["polyorder"] = le2; method_layout.addRow("Poly Order:", le2)
            elif current_method == "Wavelet Thresholding":
                le1 = QLineEdit("db4"); param_inputs_dict["wavelet"] = le1; method_layout.addRow("Wavelet Type:", le1)
                le2 = QLineEdit("0.04"); param_inputs_dict["threshold"] = le2; method_layout.addRow("Threshold:", le2)
            elif current_method in ["UNet Model", "Transform Model"]:
                path_le = QLineEdit(); path_le.setPlaceholderText("Path to .pth weight file"); param_inputs_dict["model_path"] = path_le
                browse_btn = QPushButton("Browse"); browse_btn.clicked.connect(lambda c, le=path_le: self._browse_model_path(le))
                path_layout = QHBoxLayout(); path_layout.addWidget(path_le); path_layout.addWidget(browse_btn)
                method_layout.addRow("Model Path:", path_layout)
                peaks_le = QLineEdit("125,135,150,160"); param_inputs_dict["peak_list"] = peaks_le; method_layout.addRow("Peak List (comma-sep):", peaks_le)
            else: le = QLineEdit(); le.setPlaceholderText("optional"); param_inputs_dict["param"] = le; method_layout.addRow("Parameter:", le)
        method_selector.currentTextChanged.connect(update_params_for_block)
        update_params_for_block()
        self.method_area_layout.addWidget(block_widget)
        self.method_blocks.append({'selector': method_selector, 'params_inputs': param_inputs_dict, 'widget': block_widget})

    def _browse_model_path(self, line_edit_widget):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Weight File", "", "Model Files (*.pth)")
        if file_path: line_edit_widget.setText(file_path)

    def run_comparison_plot(self): 
        coord_str = self.coord_input.text()
        try:
            t, x, y, z = map(int, coord_str.split(","))
            spectrum_original = self.data[t, :, x, y, z].copy() 
        except Exception as e:
            self.log_message(f"Invalid coordinate for denoising comparison: {e}"); QMessageBox.warning(self, "Invalid Input", f"Coordinate error: {e}"); return
        
        denoised_spectra_results = {} # method_label -> denoised_spectrum
        for block in self.method_blocks:
            method_name = block['selector'].currentText()
            params_for_func = {}
            for key, qlineedit in block['params_inputs'].items():
                text = qlineedit.text()
                if text:
                    try: val = float(text); params_for_func[key] = int(val) if val.is_integer() else val
                    except ValueError: params_for_func[key] = text
            
            input_spec_for_method = np.abs(spectrum_original.copy()) if self.value_type.currentText() == "Magnitude" else np.real(spectrum_original.copy())
            
            denoised_spec = self.apply_denoising_func(input_spec_for_method, method_name, params_for_func)
            if denoised_spec is not None:
                param_str = ", ".join(f"{k}={v}" for k,v in params_for_func.items() if k!='model_path') 
                denoised_spectra_results[f"{method_name} ({param_str[:20]})"] = denoised_spec 
        
        display_params = {
            "show_original": self.show_original_checkbox.isChecked(),
            "value_type": self.value_type.currentText(), 
            "show_abs": self.abs_checkbox.isChecked(), 
            "show_negative": self.negative_checkbox.isChecked() 
        }
        plotters.plot_denoise_comparison_spectra(self.plot_widget, spectrum_original, denoised_spectra_results, display_params, self.log_message)


class DenoiseSettingsWindow(QDialog): 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply Denoising to Volume Data") 
        self.methods = ["Mean Filter", "Median Filter", "Gaussian Filter", 
                        "Singular Value Decomposition", "Principal Component Analysis",
                        "Savitzky-Golay Filter", "Wavelet Thresholding", "Fourier Filter",
                        "Total Variation", "Wiener Filter", "UNet Model", "Transform Model"]
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select denoising method:"))
        self.method_selector = QComboBox()
        self.method_selector.addItems(self.methods)
        layout.addWidget(self.method_selector)
        self.param_form = QFormLayout()
        self.param_inputs = {} 
        layout.addLayout(self.param_form)
        self.method_selector.currentTextChanged.connect(self.update_param_form_fields) 
        self.update_param_form_fields() 
        self.confirm_button = QPushButton("OK")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)
        self.setLayout(layout)

    def _browse_model_path(self, line_edit_widget): 
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Path", "", "Model Files (*.pth)")
        if file_path: line_edit_widget.setText(file_path)

    def update_param_form_fields(self): 
        while self.param_form.rowCount() > 0 : self.param_form.removeRow(0)
        self.param_inputs.clear()
        method = self.method_selector.currentText()
        if method in ["Mean Filter", "Median Filter"]: le = QLineEdit("5"); self.param_inputs["window_size"] = le; self.param_form.addRow("Window Size:", le)
        elif method == "Gaussian Filter": le = QLineEdit("1.0"); self.param_inputs["sigma"] = le; self.param_form.addRow("Sigma:", le)
        elif method in ["Singular Value Decomposition", "Principal Component Analysis"]: le = QLineEdit("5"); self.param_inputs["num_components"] = le; self.param_form.addRow("Num Components:", le)
        elif method == "Savitzky-Golay Filter":
            le1 = QLineEdit("9"); self.param_inputs["window_size"] = le1; self.param_form.addRow("Window Size:", le1)
            le2 = QLineEdit("3"); self.param_inputs["polyorder"] = le2; self.param_form.addRow("Poly Order:", le2)
        elif method == "Wavelet Thresholding":
            le1 = QLineEdit("db4"); self.param_inputs["wavelet"] = le1; self.param_form.addRow("Wavelet Type:", le1)
            le2 = QLineEdit("0.04"); self.param_inputs["threshold"] = le2; self.param_form.addRow("Threshold:", le2)
        elif method in ["UNet Model", "Transform Model"]:
            path_le = QLineEdit(); path_le.setPlaceholderText("Path to .pth weight file"); self.param_inputs["model_path"] = path_le
            browse_btn = QPushButton("Browse"); browse_btn.clicked.connect(lambda c, le=path_le: self._browse_model_path(le))
            path_layout = QHBoxLayout(); path_layout.addWidget(path_le); path_layout.addWidget(browse_btn)
            self.param_form.addRow("Model Path:", path_layout)
            peaks_le = QLineEdit("125,135,150,160"); self.param_inputs["peak_list"] = peaks_le; self.param_form.addRow("Peak List (comma-sep):", peaks_le)
        else: le = QLineEdit(); le.setPlaceholderText("optional param"); self.param_inputs["param"] = le; self.param_form.addRow("Parameter:", le)

    def get_parameters(self):
        method = self.method_selector.currentText(); params = {}
        for k, qle_widget in self.param_inputs.items():
            txt = qle_widget.text()
            if txt: try: v_f = float(txt); params[k] = int(v_f) if v_f.is_integer() else v_f 
                    except ValueError: params[k] = txt 
        return method, params

class PeakFittingSettingsWindow(QDialog): 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply Peak Fitting to Volume Data") 
        layout = QVBoxLayout(); self.param_inputs = {}
        layout.addWidget(QLabel("Select data type for fitting (usually Magnitude or Real part of corrected data):"))
        self.value_type_selector = QComboBox(); self.value_type_selector.addItems(["Magnitude", "Real"]); layout.addWidget(self.value_type_selector)
        self.param_inputs["param_peak"] = QLineEdit("120,130,150,160") 
        self.param_inputs["param_gamma"] = QLineEdit("10,10,10,10")
        self.param_inputs["min_gamma"] = QLineEdit("5"); self.param_inputs["max_gamma"] = QLineEdit("20")
        self.param_inputs["peak_shift_limit"] = QLineEdit("2"); self.param_inputs["num_peaks"] = QLineEdit("4") 
        self.param_inputs["epochs"] = QLineEdit("1500"); self.param_inputs["lr"] = QLineEdit("0.01"); self.param_inputs["batch_size"] = QLineEdit("100")
        form = QFormLayout()
        form.addRow("Initial Peak Positions (comma-separated):", self.param_inputs["param_peak"])
        form.addRow("Initial Gamma Values (comma-separated):", self.param_inputs["param_gamma"])
        form.addRow("Min Gamma:", self.param_inputs["min_gamma"]); form.addRow("Max Gamma:", self.param_inputs["max_gamma"])
        form.addRow("Peak Shift Limit:", self.param_inputs["peak_shift_limit"])
        form.addRow("Number of Peaks (auto from peak positions):", self.param_inputs["num_peaks"])
        self.param_inputs["num_peaks"].setReadOnly(True); self.param_inputs["param_peak"].textChanged.connect(self._update_num_peaks)
        form.addRow("Epochs:", self.param_inputs["epochs"]); form.addRow("Learning Rate:", self.param_inputs["lr"]); form.addRow("Batch Size:", self.param_inputs["batch_size"])
        layout.addLayout(form)
        self.confirm_button = QPushButton("Apply"); self.confirm_button.clicked.connect(self.accept); layout.addWidget(self.confirm_button)
        self.setLayout(layout); self._update_num_peaks()

    def _update_num_peaks(self):
        try: num = len([float(p.strip()) for p in self.param_inputs["param_peak"].text().split(',') if p.strip()]); self.param_inputs["num_peaks"].setText(str(num))
        except: self.param_inputs["num_peaks"].setText("Invalid")

    def get_parameters(self):
        params_out = {"value_type": self.value_type_selector.currentText()}
        for key, box in self.param_inputs.items():
            val_str = box.text().strip()
            if not val_str: continue
            if key in ["param_peak", "param_gamma"]:
                try: params_out[key] = [float(x.strip()) for x in val_str.split(',')]
                except ValueError: params_out[key] = [] 
            else:
                try: num = float(val_str); params_out[key] = int(num) if num.is_integer() else num
                except ValueError: params_out[key] = val_str 
        if "param_peak" in params_out: params_out["num_peaks"] = len(params_out["param_peak"])
        return params_out

class PeakFittingDisplayWindow(QDialog): 
    def __init__(self, parent=None): 
        super().__init__(parent)
        self.parent_app = parent 
        self.setWindowTitle("Display Peak Fitting Results")
        layout = QVBoxLayout()
        self.display_type_selector = QComboBox()
        self.display_type_selector.addItems(["Spectrum", "Image Overlay"]) 
        self.display_type_selector.currentTextChanged.connect(self.update_form_fields) 
        layout.addWidget(QLabel("Select Display Type:"))
        layout.addWidget(self.display_type_selector)
        self.form_layout = QFormLayout(); layout.addLayout(self.form_layout)
        self.plot_button = QPushButton("OK / Plot"); self.plot_button.clicked.connect(self.accept); layout.addWidget(self.plot_button)
        self.setLayout(layout); self.update_form_fields("Spectrum") 

    def update_form_fields(self, display_type): 
        while self.form_layout.rowCount(): self.form_layout.removeRow(0)
        if display_type == "Spectrum":
            self.coord_input = QLineEdit("1,7,7,4"); self.form_layout.addRow("Voxel Coordinate [t,x,y,z]:", self.coord_input)
        elif display_type == "Image Overlay":
            self.peak_index_input = QLineEdit("0"); self.form_layout.addRow("Peak Index (0-indexed):", self.peak_index_input)
            self.dimension_input = QLineEdit("2"); self.form_layout.addRow("Slice Dimension (0:X, 1:Y, 2:Z):", self.dimension_input)
            self.time_input = QLineEdit("0"); self.form_layout.addRow("Time Index for DMI map:", self.time_input)

    def get_parameters(self): 
        params = {}; display_type = self.display_type_selector.currentText()
        if display_type == "Spectrum": params["coord"] = self.coord_input.text()
        elif display_type == "Image Overlay":
            params["peak_index"] = self.peak_index_input.text()
            params["slice_dim"] = self.dimension_input.text()
            params["time_idx"] = self.time_input.text()
        return display_type, params

class DataAnalysisWindow(QWidget): 
    def __init__(self, fitted_data_dict, raw_data_volume, parent=None): 
        super().__init__(parent)
        self.parent_app = parent 
        self.setWindowTitle("DMI Data Analysis & Curve Fitting"); self.setMinimumSize(700, 500)
        self.fitted_data = fitted_data_dict
        if raw_data_volume.shape[-1] == 2: self.raw_data_magnitude = np.abs(raw_data_volume[...,0] + 1j*raw_data_volume[...,1])
        else: self.raw_data_magnitude = np.abs(raw_data_volume)
        self.bg_level_estimate = np.mean(self.fitted_data.get("bg", 0.0))
        _comp_raw = self.fitted_data.get("separate_peaks") 
        if _comp_raw is not None: self.peak_maps = _comp_raw.transpose(3,5,0,1,2,4) 
        else: self.peak_maps = None; self.parent_app.log_message("Warning: 'separate_peaks' not found for DataAnalysisWindow.")
        layout = QVBoxLayout()
        if self.bg_level_estimate is not None: layout.addWidget(QLabel(f"Estimated Background Level (mean): {self.bg_level_estimate:.4f}"))
        try:
            noise_region = np.concatenate([self.raw_data_magnitude[:, :5, ...].flatten(), self.raw_data_magnitude[:, -5:, ...].flatten()])
            layout.addWidget(QLabel(f"Raw Data Noise Mean (ends of spectrum): {np.mean(noise_region):.4f}"))
            layout.addWidget(QLabel(f"Raw Data Noise Std (ends of spectrum): {np.std(noise_region):.4f}"))
        except Exception as e: layout.addWidget(QLabel(f"Noise Estimation Failed: {str(e)}"))
        form_layout = QFormLayout()
        self.coord_input = QLineEdit("0,0,0"); form_layout.addRow(QLabel("Select Voxel Coordinate (X,Y,Z):"), self.coord_input)
        self.peak_idx_input = QLineEdit("0"); form_layout.addRow(QLabel("Select Peak Index (0-indexed):"), self.peak_idx_input)
        self.start_time_input = QLineEdit("0"); form_layout.addRow(QLabel("Select Start Time Point for curve:"), self.start_time_input)
        layout.addLayout(form_layout)
        plot_btn = QPushButton("Plot Temporal Dynamics of Selected Peak"); plot_btn.clicked.connect(self.plot_temporal_data_curve); layout.addWidget(plot_btn)
        self.plot_widget_temporal = pg.PlotWidget(title="Temporal Curve of Peak Amplitude"); layout.addWidget(self.plot_widget_temporal)
        self.fit_selector = QComboBox(); self.fit_selector.addItems(["Linear", "Exponential", "BiExponential", "BBFunction"]); layout.addWidget(QLabel("Curve Fitting Model:")); layout.addWidget(self.fit_selector)
        self.fit_result_label = QLabel("Fitting Result: "); layout.addWidget(self.fit_result_label)
        fit_btn = QPushButton("Fit Curve to Plotted Temporal Data"); fit_btn.clicked.connect(self.apply_temporal_curve_fitting); layout.addWidget(fit_btn)
        self.setLayout(layout); self.current_signal_for_fitting = None 

    def log_message(self, message): 
        if self.parent_app and hasattr(self.parent_app, 'log_message'): self.parent_app.log_message(message)
        else: print(message)

    def plot_temporal_data_curve(self): 
        if self.peak_maps is None: self.log_message("Error: Peak maps not available."); return
        try:
            x, y, z = map(int, self.coord_input.text().split(","))
            peak_idx = int(self.peak_idx_input.text()); start_t = int(self.start_time_input.text())
            amplitudes_volume = self.fitted_data.get('amplitude') 
            if amplitudes_volume is None: self.log_message("Error: 'amplitude' data not found."); self.current_signal_for_fitting = None; return
            if not (0<=x<amplitudes_volume.shape[0] and 0<=y<amplitudes_volume.shape[1] and 0<=z<amplitudes_volume.shape[2]): self.log_message("Error: Voxel coordinate out of bounds."); return
            if not (0<=peak_idx<amplitudes_volume.shape[4]): self.log_message(f"Error: Peak index {peak_idx} out of bounds."); return
            
            self.current_signal_for_fitting = amplitudes_volume[x,y,z,start_t:,peak_idx]
            time_axis = np.arange(start_t, amplitudes_volume.shape[3])
            plotters.plot_temporal_curve(self.plot_widget_temporal, time_axis[:len(self.current_signal_for_fitting)], self.current_signal_for_fitting,
                                         title=f"Peak {peak_idx} at ({x},{y},{z})", name="Temporal Signal", log_message_func=self.log_message)
        except Exception as e: self.log_message(f"Error plotting temporal curve: {e}"); self.current_signal_for_fitting = None

    def apply_temporal_curve_fitting(self): 
        if self.current_signal_for_fitting is None or len(self.current_signal_for_fitting) == 0:
            self.fit_result_label.setText("No curve to fit."); self.log_message("Error: No data for curve fitting."); return
        # Use get_device() from config.settings
        current_device = get_device() 
        start_t = int(self.start_time_input.text()); num_points = len(self.current_signal_for_fitting)
        x_axis_data = np.arange(start_t, start_t + num_points); y_axis_data = self.current_signal_for_fitting
        model_type_str = self.fit_selector.currentText()
        try:
            model_instance = None
            if model_type_str == "Linear": model_instance = LinearModel(y_axis_data)
            elif model_type_str == "Exponential": model_instance = ExpModel(y_axis_data)
            elif model_type_str == "BiExponential": model_instance = BiExpModel(y_axis_data)
            elif model_type_str == "BBFunction": model_instance = BBModel(y_axis_data)
            else: self.fit_result_label.setText("Invalid model selection."); return
            
            x_fit, y_fit, params_list = model_fitting(x_axis_data, y_axis_data, model_instance, device=current_device) # Pass current_device
            
            plotters.plot_curve_fitting_results(self.plot_widget_temporal, x_fit, y_fit, model_type_str, self.log_message)
            param_str = ", ".join([f"{p:.4f}" for p in params_list])
            self.fit_result_label.setText(f"Fitting Result ({model_type_str}): {param_str}")
            self.log_message(f"Curve fitting successful with {model_type_str}. Params: {param_str}")
        except Exception as e: self.fit_result_label.setText(f"Fitting failed: {e}"); self.log_message(f"Curve fitting failed: {e}")

[end of visualization/gui.py]
