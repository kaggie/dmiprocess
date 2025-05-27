import os
import sys
import cv2
import numpy as np
import torch
import nibabel as nib

import pyqtgraph as pg
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.console import ConsoleWidget
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QApplication, QLabel, QMainWindow, QFileDialog, 
                             QVBoxLayout, QWidget, QPlainTextEdit, QPushButton, 
                             QHBoxLayout, QLineEdit, QRadioButton, QButtonGroup,
                             QDialog, QComboBox, QFormLayout, QGridLayout, QStackedLayout,
                             QCheckBox, QMessageBox, QFormLayout, QScrollArea, QGroupBox,QSlider
                            )
from functions import (load_from_file, load_from_folder, phase_correction, 
                       data_normalization, classic_denoiser, peak_fitting_gpu, curve_fitting,
                       denoise_unet_pe, denoise_trans_pe
                       )

DEVICE = torch.device('cpu')

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DMI Data Processing Toolbox")
        self.setGeometry(100, 100, 1200, 800)

        # Create DockArea for flexible layout
        self.area = DockArea()
        self.setCentralWidget(self.area)

        # Create docks for each section
        self.load_dock = Dock("Load Image", size=(500, 400))
        self.phase_dock = Dock("Phase Correction", size=(500, 400))
        self.denoise_dock = Dock("Denoising", size=(500, 400))
        self.peak_dock = Dock("Peak Fitting", size=(500, 400))
        self.console_dock = Dock("Console", size=(800, 200))

        # Add docks to the layout
        self.area.addDock(self.load_dock, "left")
        self.area.addDock(self.phase_dock, "right")
        self.area.addDock(self.denoise_dock, "bottom", self.load_dock)
        self.area.addDock(self.peak_dock, "bottom", self.phase_dock)
        self.area.addDock(self.console_dock, "bottom")

        # Add widgets to each dock
        self.load_widget, self.load_plot = self.create_section("Load Image", ["Data2npy", "Load", "Overlay", "Plot Spectrum", "Show Image", "Clear Display"])
        self.phase_widget, self.phase_plot = self.create_section("Phase Correction", ["Device", "Normalize", "Comparison", "Apply Phase Correction",
                                                                                       "Plot Spectrum", "Show Image", "Clear Display"])
        self.denoise_widget, self.denoise_plot = self.create_section("Denoising", ["Comparison", "Apply Denoising", 
                                                                                   "Plot Spectrum", "Show Image", "Clear Display"])
        self.peak_widget, self.peak_plot = self.create_section("Peak Fitting", ["Apply Peak Fitting", "Display Fitting Results","Data Analysis", "Save"])

        self.load_dock.addWidget(self.load_widget)
        self.phase_dock.addWidget(self.phase_widget)
        self.denoise_dock.addWidget(self.denoise_widget)
        self.peak_dock.addWidget(self.peak_widget)

        # Add a Console Widget for real-time debugging/logging
        self.console = ConsoleWidget()
        self.console_dock.addWidget(self.console)

        # Logging Panel
        self.log_out = QPlainTextEdit()
        self.log_out.setReadOnly(True)  # Make it read-only
        self.log_out.setPlaceholderText("Log messages will appear here...")
        self.console_dock.addWidget(self.log_out)
        
        self.log_message("Application started.")

        # Connect button actions
        self.connect_buttons()

    def log_message(self, message):
        """ Append a message to the log output """
        self.log_out.appendPlainText(message)
        print(message)
        #self.console.pushText(f"{message}\n")

    def create_section(self, title, buttons):
        """ Create a section where all buttons are on the left of the display widget. """
        widget = QWidget()
        layout = QHBoxLayout()  # Horizontal layout

        # Button layout (left side)
        button_layout = QVBoxLayout()
        for btn_text in buttons:
            button = QPushButton(btn_text)
            button.setFixedWidth(150)
            button_layout.addWidget(button)

            # If this is the "data2npy" button, add the save path input field right below it
            if btn_text == "Data2npy":
                self.save_path_input = QLineEdit()
                self.save_path_input.setPlaceholderText("Enter path to save .npy file")
                self.save_path_input.setText("GUI/data/data_npy")  # Default value
                button_layout.addWidget(self.save_path_input)
            # If this is the "Load Image" section, add the save path input field below the buttons
            if btn_text == "Load":
                self.file_extension_input = QLineEdit()
                self.file_extension_input.setPlaceholderText("Enter file extension")
                self.file_extension_input.setText("250x10x10x10x2.npy")  # Default value
                button_layout.addWidget(self.file_extension_input)

        layout.addLayout(button_layout)  # Add button layout first (left)

        if title in ["Load Image", "Phase Correction", "Denoising"]:
            container = QWidget()
            container_layout = QStackedLayout(container)
            spectrum_plot = pg.PlotWidget()
            spectrum_plot.setMinimumSize(500, 300)
            container_layout.addWidget(spectrum_plot)
            image_plot = pg.GraphicsLayoutWidget()
            image_plot.setMinimumSize(500, 300)
            container_layout.addWidget(image_plot)
            layout.addWidget(container)
            widget.setLayout(layout)
            return widget, {"spectrum": spectrum_plot, "image": image_plot, 
                            "container": container_layout}
        elif title == "Peak Fitting":
            container = QWidget()
            container_layout = QStackedLayout(container)
            spectrum_plot = pg.PlotWidget()
            spectrum_plot.setMinimumSize(500, 300)
            container_layout.addWidget(spectrum_plot)

            image_wrapper = QWidget()
            image_wrapper_layout = QVBoxLayout(image_wrapper)
            image_plot = pg.ImageView()  
            image_plot.setMinimumSize(500, 300)
            colormap = pg.colormap.get("viridis")
            image_plot.setColorMap(colormap)
            image_wrapper_layout.addWidget(image_plot)
            image_wrapper.setLayout(image_wrapper_layout)
            container_layout.addWidget(image_wrapper)

            layout.addWidget(container)
            widget.setLayout(layout)
            return widget, {"spectrum": spectrum_plot, "image": image_wrapper, 
                            "image_view": image_plot,"container": container_layout}
        else:
            plot_widget = pg.PlotWidget()
            plot_widget.setMinimumSize(500, 300)
            layout.addWidget(plot_widget)
            widget.setLayout(layout)
            return widget, plot_widget

    def connect_buttons(self):
        """ Connect buttons to their respective functions. """
        load_buttons = self.load_widget.findChildren(QPushButton)
        for btn in load_buttons:
            if btn.text() == "Data2npy":
                btn.clicked.connect(self.convert_data_to_npy)
            elif btn.text() == "Load":
                btn.clicked.connect(self.load_saved_npy)
            elif btn.text() == "Overlay":
                btn.clicked.connect(lambda: self.open_overlay_settings(self.I))
            elif btn.text() == "Plot Spectrum":
                btn.clicked.connect(lambda: self.open_spectrum_settings(self.I, self.load_plot))
            elif btn.text() == "Show Image":
                btn.clicked.connect(lambda: self.open_image_settings(self.I, self.load_plot))
            elif btn.text() == "Clear Display":
                btn.clicked.connect(lambda: self.clear_display(self.load_plot))

        phase_buttons = self.phase_widget.findChildren(QPushButton)
        for btn in phase_buttons:   
            if btn.text() == "Device":
                btn.clicked.connect(self.detect_device)
            elif btn.text() == "Normalize":
                btn.clicked.connect(self.open_normalization_settings)
            elif btn.text() == "Comparison":
                btn.clicked.connect(self.open_comparison_settings)
            elif btn.text() == "Apply Phase Correction":
                btn.clicked.connect(lambda: self.open_phase_correction_settings(self.I_norm))
            elif btn.text() == "Plot Spectrum":
                btn.clicked.connect(lambda: self.open_spectrum_settings(self.I_corrected, self.phase_plot))
            elif btn.text() == "Show Image":
                btn.clicked.connect(lambda: self.open_image_settings(self.I_corrected, self.phase_plot))
            elif btn.text() == "Clear Display":
                btn.clicked.connect(lambda: self.clear_display(self.phase_plot))

        denoise_buttons = self.denoise_widget.findChildren(QPushButton)
        for btn in denoise_buttons:
            if btn.text() == "Comparison":
                btn.clicked.connect(lambda: self.open_denoise_comparison_settings(self.I_corrected))
            elif btn.text() == "Apply Denoising":
                btn.clicked.connect(lambda: self.open_denoising_settings(self.I_corrected))
            elif btn.text() == "Plot Spectrum":
                btn.clicked.connect(lambda: self.open_spectrum_settings(self.I_denoised, self.denoise_plot))
            elif btn.text() == "Show Image":
                #btn.clicked.connect(lambda: self.open_image_settings(self.I_fitted['fitted_data'], self.denoise_plot))
                btn.clicked.connect(lambda: self.open_image_settings(self.I_denoised, self.denoise_plot))
            elif btn.text() == "Clear Display":
                btn.clicked.connect(lambda: self.clear_display(self.denoise_plot))

        peak_buttons = self.peak_widget.findChildren(QPushButton)
        for btn in peak_buttons:
            if btn.text() == "Apply Peak Fitting":
                btn.clicked.connect(lambda: self.open_peak_fitting_settings(self.I_denoised))
            elif btn.text() == "Display Fitting Results":
                btn.clicked.connect(lambda: self.display_fitting_results(self.I_fitted, self.peak_plot, self.data_before_fitting))
            elif btn.text() == "Data Analysis":
                btn.clicked.connect(lambda: self.open_data_analysis_settings(self.I_fitted, self.I))
            #elif btn.text() == "Save":
            #    btn.clicked.connect(self.save_results)


    def clear_display(self,target_plot):
        """ Clear all image/spectrum displays from the Load Image dock """
        # Try removing central item if set (GraphicsLayout, ViewBox, etc.)
        try:
            if isinstance(target_plot, dict):
                # Clear both views
                target_plot["spectrum"].clear()
                target_plot["image"].clear()
                target_plot["image"].setCentralItem(None)
                self.log_message("Cleared both spectrum and image views.")
            else:
                # Fallback for single widget
                target_plot.clear()
                self.log_message("Cleared single display.")
        except Exception as e:
            self.log_message(f"Could not remove central item: {e}")

    def detect_device(self):
        """ Detect and return the available GPU device """
        global DEVICE
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_message(f"Detected device: {DEVICE}")

    def convert_data_to_npy(self):
        """ Open data and save as NumPy array """
        path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Data") or \
            QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;DICOM (*.dcm);;NIFTI (*.nii *.nii.gz)")[0]

        if not path:
            self.log_message("Error: No folder or file selected.")
            return
        self.log_message(f"Selected path: {path}")

        save_path = self.save_path_input.text().strip()
        if not save_path:
            self.log_message("Error: No save path provided.")
            return
        # Ensure save_path is absolute; if not, assume it's in the current working directory
        if not os.path.isabs(save_path):
            save_path = os.path.join(os.getcwd(), save_path)
        # Ensure the directory exists, create it if not
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            self.log_message(f"Created new directory: {save_dir}")

        try:
            if os.path.isdir(path):
                # Try loading with method1 first
                try:
                    data_array = load_from_folder.load_data_use_brukerapi(path,output_path=save_path)
                    if data_array is None:
                        raise ValueError("No data found in the specified folder.")
                    else:
                        self.log_message("Loaded using Brukerapi.")
                except Exception as e1:
                    self.log_message(f"Brukerapi failed: {str(e1)}. Trying visualization using home-written method.")
                    try:
                        data_array = load_from_folder.recon_from_2dseq(path)
                        self.log_message("Loaded using home-written method.")
                    except Exception as e2:
                        self.log_message(f"Loading failed: {e1}, {e2}")
                        raise RuntimeError(f"Loading failed: {e1}, {e2}")

            elif os.path.isfile(path):
                data_array = load_from_file.load_image(path)
                self.log_message("Loaded from file.")
            else:
                raise ValueError("Invalid selection. Must be a folder or a DICOM or NIfTI file.")

            np.save(save_path, data_array)
            self.log_message(f"Data saved as NumPy array: {save_path}")

        except Exception as e:
            self.log_message(f"Error: {str(e)}")

    def load_saved_npy(self):
        """ Load saved NumPy data """
            
        file_extension = self.file_extension_input.text().strip()
        if not file_extension:
            self.log_message("Error: No file extension provided.")
            return

        path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Data") or \
            QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*);;NumPy Array (*.npy)")[0]

        if not path:
            self.log_message("Error: No folder or file selected.")
            return
        self.log_message(f"Selected path: {path}")

        try:
            if os.path.isdir(path):
                file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(file_extension)]
                file_paths.sort()
                if not file_paths:
                    raise FileNotFoundError(f"No files found with extension: {file_extension}")
                
                data = []
                for file_path in file_paths:
                    data_npy = np.load(file_path)
                    data.append(data_npy)
                data_array = np.array(data)
                self.log_message(f"Loaded {len(file_paths)} files, stacked shape: {data_array.shape}")
            elif os.path.isfile(path):
                data_array = np.load(path)
                self.log_message(f"Data loaded successfully: {path}")
                self.log_message(f"Data shape: {data_array.shape}")
            else:
                raise ValueError("Invalid selection. Must be a folder or a valid .npy file.")

        except Exception as e:
            self.log_message(f"Error: {str(e)}")
        
        self.I = data_array  # Store loaded data for plotting

    def open_overlay_settings(self, data):
        if data is None:
            self.log_message("Error: No data loaded.")
            return
        """ Open the parameter selection window when 'Overlay' is clicked. """
        self.overlay_window = OverlayWindow(data)
        self.overlay_window.show()

    def open_spectrum_settings(self, data, target_plot):
        """ Open the parameter selection window when 'Plot Spectra' is clicked. """
        length = data.shape[1]
        self.spectrum_window = SpectrumSettingsWindow(self, spectrum_length=length)
        if self.spectrum_window.exec():  # Wait for user to close window
            params = self.spectrum_window.get_parameters()  # Retrieve parameters
            self.log_message(f"Selected Parameters: {params}")
            self.plot_spectrum(data, target_plot)  # Plot the spectrum after settings are selected

    def plot_spectrum(self,data, target_plot):
        """ Plot the image data inside the main window"""
        params = self.spectrum_window.get_parameters()  # Get user selections
        value_type = params["value_type"]
        display_method = params["display_method"]
        index_input = params["time_series_index"]
        freq_range_input = params["frequency_range"]
        abs_value = params["show_abs"]

        # Convert frequency range input to start and end indices
        try:
            freq_start, freq_end = map(int, freq_range_input.split(":"))
        except ValueError:
            self.log_message("Invalid frequency range input. Using full range.")
            freq_start, freq_end = 0, data.shape[1]  # Default full range

        freq_start = max(0, min(freq_start, data.shape[1] - 1))
        freq_end = max(0, min(freq_end, data.shape[1]))

        # Convert index input into a list of indices/ranges
        indices = []
        for item in index_input.split(","):
            item = item.strip()
            if ":" in item:  # If range a:b is provided
                start, end = map(int, item.split(":"))
                indices.append((start, end))  # Store as tuple for range processing
            else:
                indices.append(int(item))  # Store single index

        # Convert Data to complex form if the last dimension is 2
        if data.shape[-1] == 2:
            data = data[..., 0] + 1j * data[..., 1]
            self.log_message("Converted I to complex form: I = I[...,0] + 1j * I[...,1]")

        # Apply value type selection
        if value_type == "Magnitude":
            data = np.abs(data)
        elif value_type == "Imaginary":
            data = np.imag(data)
        elif value_type == "Real":
            data = np.abs(np.real(data)) if abs_value else np.real(data)

        if isinstance(target_plot, dict):
            target_plot["container"].setCurrentWidget(target_plot["spectrum"])
            plot_widget = target_plot["spectrum"]
        else:
            plot_widget = target_plot
        
        plot_widget.clear()
    
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']  # Predefined colors
        color_idx = 0
        plot_widget.addLegend()

        x = np.arange(freq_start, freq_end)  # Frequency range for x-axis
        
        # Loop through indices and plot
        for idx in indices:
            if isinstance(idx, tuple):  # If range a:b
                if display_method == "Average":
                    spectrum = data[idx[0]:idx[1], freq_start:freq_end].mean(axis=0)
                else:  # Maximum
                    spectrum = data[idx[0]:idx[1], freq_start:freq_end].max(axis=0)
            else:  # Single index
                spectrum = data[idx, freq_start:freq_end]

            color = pg.mkPen(colors[color_idx % len(colors)], width=1)  # Cycle through colors
            plot_widget.plot(x, spectrum, pen=color, name=f"Index {idx}")
            color_idx += 1

        plot_widget.setLabel("bottom", "Frequency")
        plot_widget.setLabel("left", "Signal Intensity")
        plot_widget.setTitle(f"Spectra - {value_type} ({display_method})")

        self.log_message(f"Plotted spectra in 'Load Image' window with parameters: {params}")

    def open_image_settings(self, data, target_plot):
        """ Open parameter selection window when 'Image' is clicked. """
        self.image_window = ImageSettingsWindow(self,image_shape=data.shape)  # Create a new instance every time
        if self.image_window.exec():  # Wait for user to close window
            params = self.image_window.get_parameters()  # Retrieve parameters
            self.log_message(f"Selected Parameters: {params}")
            self.show_image(data, target_plot)  # Plot the spectrum after settings are selected

    def show_image(self, data, target_plot):
        """ Show the image data inside the main window"""
        params = self.image_window.get_parameters()
        value_type = params["value_type"]
        display_method = params["display_method"]
        index_input = params["time_series_index"]
        freq_range_input = params["frequency_range"]
        slice_selection_input = params["slice_selection"]
        roi_input = params["roi"]
        colormap_name = params["colormap"]
        abs_value = params["show_abs"]
        negative_value = params["show_negative"]

        # Convert frequency range input to start and end indices
        try:
            freq_start, freq_end = map(int, freq_range_input.split(":"))
        except ValueError:
            self.log_message("Invalid frequency range input. Using full range.")
            freq_start, freq_end = 0, self.I.shape[1]  # Default full range

        freq_start = max(0, min(freq_start, self.I.shape[1] - 1))
        freq_end = max(0, min(freq_end, self.I.shape[1]))   

        # Convert index input into a list of indices/ranges
        indices = []
        for item in index_input.split(","):
            item = item.strip()
            if ":" in item:  # If range a:b is provided
                start, end = map(int, item.split(":"))
                indices.append((start, end))  # Store as tuple for range processing
            else:
                indices.append(int(item))  # Store single index

        # Convert slice selection input
        try:
            slice_dim, slice_num = map(int, slice_selection_input.split(","))
            if slice_dim < 2 or slice_dim >= len(data.shape):
                raise ValueError(f"Invalid slice dimension {slice_dim}. Must be between 2 and {len(data.shape)-1}.")
        except ValueError:
            self.log_message("Invalid slice selection input. Expected format: 'dim,slice'.")
            return
        # Convert Data to complex form if the last dimension is 2
        if data.shape[-1] == 2:
            data = data[..., 0] + 1j * data[..., 1]
            self.log_message("Converted I to complex form: I = I[...,0] + 1j * I[...,1]")
        
        # Apply value type selection
        if value_type == "Magnitude":
            data = np.abs(data)
        elif value_type == "Imaginary":
            data = -np.imag(data) if negative_value else np.imag(data)
        elif value_type == "Real":
            data = -np.real(data) if negative_value else np.real(data)            
            data = np.abs(np.real(data)) if abs_value else np.real(data)

        # Get the correct plot widget and clear it
        #plot_widget = self.load_plot

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']  # Predefined colors
        color_idx = 0
        #plot_widget.addLegend()

        if roi_input:  # If ROI is provided, extract and display the selected region
            
            if isinstance(target_plot, dict):
                target_plot["container"].setCurrentWidget(target_plot["spectrum"])
                plot_widget = target_plot["spectrum"]
            else:
                plot_widget = target_plot
        
            plot_widget.clear()
            plot_widget.addLegend()

            try:
                rois = []
                for item in roi_input.split(","):
                    item = item.strip()
                    if ":" in item:
                        start, end = map(int, item.split(":"))
                        rois.append(slice(start, end))
                    else:
                        rois.append(int(item))

                if len(rois) != 2:  # Ensure enough dimensions are specified
                    raise ValueError("ROI selection must match spatial dimensions.")

                data = np.moveaxis(data, slice_dim, 0)  # Move selected dim to visualization axis
                data_slice = data[slice_num,:].squeeze()  # Extract the selected slice

                roi_h,roi_w = rois[0],rois[1]
                if isinstance(roi_w, tuple):
                    roi_slice = data_slice[:,:,roi_w[0]:roi_w[1], :].mean(axis=2)
                else:
                    roi_slice = data_slice[:,:,roi_w, :]
                if isinstance(roi_h, tuple):
                    roi_data = roi_slice[:,:,roi_h[0]:roi_h[1]].mean(axis=2)
                else:
                    roi_data = roi_slice[:,:,roi_h]
                
                roi_data = roi_data.squeeze()[:, freq_start:freq_end]  # Extract the ROI data
               
                x = np.arange(freq_start, freq_end)  # Frequency range for x-axis
        
                # Loop through indices and plot
                for idx in indices:
                    if isinstance(idx, tuple):  # If range a:b
                        if display_method == "Average":
                            spectrum_data = roi_data[idx[0]:idx[1], :].mean(axis=0)
                        else:  # Maximum
                            spectrum_data = roi_data[idx[0]:idx[1], :].max(axis=0)
                    else:  # Single index
                        spectrum_data = roi_data[idx, :]

                    color = pg.mkPen(colors[color_idx % len(colors)], width=1)  # Cycle through colors
                    plot_widget.plot(x, spectrum_data, pen=color, name=f"Index {idx}")
                    color_idx += 1

                plot_widget.setLabel("bottom", "Frequency")
                plot_widget.setLabel("left", "Signal Intensity")
                plot_widget.setTitle(f"Spectra - {value_type} ({display_method})")
                self.log_message(f"Plotted ROI {roi_input} in 'Load Image' window")

            except Exception as e:
                self.log_message(f"Error processing ROI: {str(e)}")
                return
        else:
            num_subplots = min(len(indices), 9)

            if isinstance(target_plot, dict):
                target_plot["container"].setCurrentWidget(target_plot["image"])
                image_widget = target_plot["image"]
            else:
                image_widget = target_plot

            image_widget.clear()
            #plot_widget.clear()

            grid_layout = pg.GraphicsLayout()
            #plot_widget.setCentralItem(grid_layout)
            image_widget.setCentralItem(grid_layout)

            data_map = data[:,freq_start:freq_end,...] # Extract the selected frequency range
            map_max = data_map.max()

            # Loop through indices and plot images
            for i, idx in enumerate(indices[:num_subplots]):
                img_view = pg.ImageView()
                colormap = pg.colormap.get(colormap_name)
                img_view.setColorMap(colormap)

                if isinstance(idx, tuple):  # If range a:b
                    if display_method == "Average":
                        image_data = data[idx[0]:idx[1], freq_start:freq_end, :].mean(axis=0)
                    else: # Maximum
                        image_data = data[idx[0]:idx[1], freq_start:freq_end, :].max(axis=0)
                else:  # Single index
                    image_data = data[idx, freq_start:freq_end, :]
                
                image_data = image_data.squeeze().max(axis=0) # Max intensity projection
                # Move slice dimension to the third axis (for proper visualization)
                image_data = np.moveaxis(image_data, slice_dim-2, 2)  # Move selected dim to visualization axis
                image_slice = image_data[:, :, slice_num] # Extract the selected slice
                image_slice = np.rot90(image_slice, k=-1)
                #image_slice = image_slice[::-1,::-1].T # Transpose for correct orientation

                img_view.setImage(image_slice, levels=(data.min(), map_max))
                #img_view.setImage(image_slice, levels=(image_slice.min(), image_slice.max()))
                
                grid_layout.addItem(pg.ViewBox(), i // 3, i % 3)
                grid_layout.getItem(i // 3, i % 3).addItem(img_view.getImageItem())
                grid_layout.getItem(i // 3, i % 3).setAspectLocked(True)
                self.log_message(f"Displaying image at index {idx}, slice {slice_num} along dimension {slice_dim} with colormap {colormap}")

    def open_phase_correction_settings(self,data):
        """ Open the parameter selection window when 'Apply phase correction' is clicked. """
        self.phase_correction_window = PhaseCorrectionSettingsWindow(self)
        if self.phase_correction_window.exec():  # Wait for user to close window
            params = self.phase_correction_window.get_parameters()
            self.log_message(f"Selected Parameters: {params}")
            self.apply_phase_correction(data)  # Apply phase correction after settings are selected

    def apply_phase_correction(self,data):
        """ Apply phase correction to the loaded data """
        params = self.phase_correction_window.get_parameters()
        method = params["method"]
        lr = params["lr"]
        n_iters = params["n_iters"]
        num_basis = params["num_basis"]
        peak_list = params["peak_list"]
        half_peak_width = params["half_peak_width"]
        degree = params["degree"]

        # Convert Data to complex form if the last dimension is 2
        if data.shape[-1] == 2:
            data_complex = data[..., 0] + 1j * data[..., 1]
            self.log_message("Create I_complex: I_complex = I[...,0] + 1j * I[...,1]")
        else:
            data_complex = data

        if len(data_complex.shape) != 5: # Check if the data is in the correct shape
            data_complex = data_complex.squeeze()
            data_complex = data_complex[...,None,None,None]

        try:
            corrected_data, _ = phase_correction.phase_correct_gpu(data_complex, method=method, lr=lr, n_iters=n_iters,
                                                 num_basis=num_basis, peak_list=peak_list,
                                                 half_peak_width=half_peak_width, degree=degree)
            self.log_message(f"Phase correction applied using method: {method}. Creat I_corrected.")
            self.I_corrected = corrected_data # Store corrected data for plotting
        except Exception as e:
            self.log_message(f"Error applying phase correction: {str(e)}")

    def open_normalization_settings(self):
        self.norm_window = NormalizationSettingsWindow(self)
        if self.norm_window.exec():
            params = self.norm_window.get_parameters()
            self.log_message(f"Selected Normalization Parameters: {params}")
            self.normalize_image()
    
    def normalize_image(self):
        params = self.norm_window.get_parameters()
        method = params["method"]
        mode = params["mode"]
        bg = params["bg_region"]       
        self.log_message(f"Applying {method} normalization to {mode}.")

        if self.I is None:
            self.log_message("Error: No data loaded.")
            return
        
        data = self.I
        if data.shape[-1] == 2:
            real = data[..., 0]
            imag = data[..., 1]
            data_complex = real + 1j * imag
            flag_complex = True
        else:
            data_complex = data.squeeze()
            flag_complex = False

        if len(data_complex.shape) != 5: # Check if the data is in the correct shape
            data_complex = data_complex.squeeze()
            data_complex = data_complex[...,None,None,None]

        if bg:
            bg_region = []
            for item in bg.split(","):
                item = item.strip()
                if ":" in item:
                    start, end = map(int, item.split(":"))
                    bg_region.append(slice(start, end))
                else:
                    bg_region.append(int(item))

            if len(bg_region) != len(data_complex.shape):
                self.log_message("Error: Background region must match data dimensions.")
                return

        bg_data = data_complex[tuple(bg_region)] if bg else None

        data_norm = data_normalization.normalize_data(data_complex, method, mode, bg_data, flag_complex)

        self.I_norm = data_norm
        self.log_message(f"Generate normalized data: I_norm. Shape: {data_norm.shape}")
                
    def open_comparison_settings(self):
        if self.I_norm is None:
            self.log_message("Error: No Normalized data loaded.")
            return
        self.compare_window = ComparisonSettingsWindow(self.I_norm, self.apply_single_spectrum_phase_correction)
        self.compare_window.show()

    def apply_single_spectrum_phase_correction(self, spectrum, method, params):
        data_complex = np.zeros((1, spectrum.shape[0], 1, 1, 1), dtype=np.complex64)
        data_complex[0, :, 0, 0, 0] = spectrum

        lr = float(params.get("learning_rate", 0.05))
        n_iters = int(params.get("num_iterations", 200))
        num_basis = int(params.get("num_fourier_basis", 8))
        half_peak_width = int(params.get("half_peak_width", 5))
        degree = int(params.get("b-spline_degree", 3))
        peak_list_str = params.get("peak_list", "None")        
        if peak_list_str.lower() == "none":
            peak_list = None
        else:
            try:
                peak_list = [float(p.strip()) for p in peak_list_str.split(',')]
            except:
                peak_list = None
        
        try:
            corrected_spectrum, _ = phase_correction.phase_correct_gpu(data_complex,method=method,lr=lr,
                                                                   n_iters=n_iters,num_basis=num_basis,peak_list=peak_list,
                                                                   half_peak_width=half_peak_width,degree=degree
                                                                   )
        except Exception as e:
            self.log_message(f"[{method}] phase correction failed: {str(e)}")
            return None

        return corrected_spectrum.squeeze()

    def open_denoise_comparison_settings(self,data):
        if data is None:
            self.log_message("Error: No Phase Corrected data loaded.")
            return
        self.denoise_window = DenoiseComparisonSettingsWindow(data, self.apply_single_spectrum_denoising)
        self.denoise_window.show()

    def apply_single_spectrum_denoising(self, spectrum, method, params):
        spectrum = np.asarray(spectrum, dtype=np.float32)
        data = np.zeros((1, spectrum.shape[0], 1, 1, 1), dtype=np.float32)
        data[0, :, 0, 0, 0] = spectrum

        try:
            if method == "UNet Model":
                model_path = params.get("model_path")
                peak_list = params.get("peak_list")
                peaks = [int(p.strip()) for p in peak_list.split(',')]
                peaks_tensor = torch.tensor(peaks,dtype=torch.long).to(DEVICE).unsqueeze(0).unsqueeze(0)
                model = denoise_unet_pe.UNet1DWithPEPeak(in_channels=1, out_channels=1)
                model.to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                with torch.no_grad():
                    spectrum = np.abs(spectrum).reshape(1, 1, -1)
                    F = spectrum.shape[2]
                    if F != 256:
                        pad_width = ((0, 0), (0, 0), (0, 256-F))
                        spectrum = np.pad(spectrum, pad_width, mode='constant', constant_values=0)
                        spectrum_tensor = torch.from_numpy(spectrum).to(DEVICE).to(torch.float32)
                        denoised_spectrum = model(spectrum_tensor,peaks_tensor).cpu().numpy()
                        denoised_spectrum = denoised_spectrum[:,:,:F].reshape(1, -1)
            elif method == "Transform Model":
                model_path = params.get("model_path")
                peak_list = params.get("peak_list")
                peaks = [int(p.strip()) for p in peak_list.split(',')]
                peaks_tensor = torch.tensor(peaks,dtype=torch.long).to(DEVICE).unsqueeze(0).unsqueeze(0)
                model = denoise_trans_pe.TransformerDenoiser(num_layers=4)
                model.to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                with torch.no_grad():
                    spectrum = np.abs(spectrum).reshape(1, 1, -1)
                    F = spectrum.shape[2]
                    if F != 256:
                        pad_width = ((0, 0), (0, 0), (0, 256-F))
                        spectrum = np.pad(spectrum, pad_width, mode='constant', constant_values=0)
                        spectrum_tensor = torch.from_numpy(spectrum).to(DEVICE).to(torch.float32)
                        denoised_spectrum = model(spectrum_tensor,peaks_tensor).cpu().numpy()
                        denoised_spectrum = denoised_spectrum[:,:,:F].reshape(1, -1)
            else:
                denoised_spectrum = classic_denoiser.apply_spectral_denoising_batch(data, method, **params)
        except Exception as e:
            self.log_message(f"[{method}] denoising failed: {str(e)}")
            return None

        return denoised_spectrum.squeeze()

    def open_denoising_settings(self, data):
        if data is None:
            self.log_message("Error: No Phase Corrected data loaded.")
            return
        self.denoise_window = DenoiseSettingsWindow(self)
        if self.denoise_window.exec():
            method, params = self.denoise_window.get_parameters()
            self.log_message(f"Denoising method: {method}, Parameters: {params}")
            self.apply_denoising(data, method, **params)

    def apply_denoising(self, data, method, **params):
        if data is None:
            self.log_message("Error: No Phase Corrected data loaded.")
            return
        
        T, F, X, Y, Z = data.shape[:5]
        try:
            if method == "UNet Model":
                model_path = params.get("model_path")
                peak_list = params.get("peak_list")
                peaks = [int(p.strip()) for p in peak_list.split(',')]
                peaks_tensor = torch.tensor(peaks,dtype=torch.long).to(DEVICE).unsqueeze(0).unsqueeze(0)
                model = denoise_unet_pe.UNet1DWithPEPeak(in_channels=1, out_channels=1)
                model.to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                with torch.no_grad():
                    data  = data.transpose(0, 2, 3, 4, 1) 
                    data = np.abs(data).reshape(-1, 1, F)
                    peaks_tensor = peaks_tensor.expand(data.shape[0], -1, -1)  # Expand peaks_tensor to match data shape
                    if F != 256:
                        pad_width = ((0, 0), (0, 0), (0, 256-F))
                        data = np.pad(data, pad_width, mode='constant', constant_values=0)
                        data_tensor = torch.from_numpy(data).to(DEVICE).to(torch.float32)
                        denoised_data = model(data_tensor, peaks_tensor).cpu().numpy()
                        denoised_data = denoised_data[:,:,:F].reshape(T, X, Y, Z, -1).transpose(0, 4, 1, 2, 3)
            elif method == "Transform Model":
                model_path = params.get("model_path")
                peak_list = params.get("peak_list")
                peaks = [int(p.strip()) for p in peak_list.split(',')]
                peaks_tensor = torch.tensor(peaks,dtype=torch.long).to(DEVICE).unsqueeze(0).unsqueeze(0)
                model = denoise_trans_pe.TransformerDenoiser(num_layers=4)
                model.to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                with torch.no_grad():
                    data  = data.transpose(0, 2, 3, 4, 1) 
                    data = np.abs(data).reshape(-1, 1, F)
                    peaks_tensor = peaks_tensor.expand(data.shape[0], -1, -1)  # Expand peaks_tensor to match data shape
                    if F != 256:
                        pad_width = ((0, 0), (0, 0), (0, 256-F))
                        data = np.pad(data, pad_width, mode='constant', constant_values=0) # Pad to 256
                        data_tensor = torch.from_numpy(data).to(DEVICE).to(torch.float32)
                        denoised_data = model(data_tensor, peaks_tensor).cpu().numpy()
                        denoised_data = denoised_data[:,:,:F].reshape(T, X, Y, Z, -1).transpose(0, 4, 1, 2, 3)
            else:
                denoised_data = classic_denoiser.apply_spectral_denoising_batch(data, method, **params)
            self.log_message(f"Denoising applied using {method}.")
            denoised_data = denoised_data.reshape(T, X, Y, Z, F).transpose(0, 4, 1, 2, 3)
            self.I_denoised = denoised_data
            self.log_message(f"I_denoised created, with a shape of {denoised_data.shape}.")
        except Exception as e:
            self.log_message(f"Error applying denoising: {str(e)}")

    def open_peak_fitting_settings(self, data):
        if data is None:
            self.log_message("Error: No Denoised Corrected data loaded.")
            return
        T, F, X, Y, Z = data.shape[:5]

        self.peak_window = PeakFittingSettingsWindow(self)
        if self.peak_window.exec():
            params = self.peak_window.get_parameters()
            self.log_message(f"Peak Fitting Parameters: {params}")
            
            value_type = params.pop("value_type", "Magnitude") # Default to Magnitude
            data = np.abs(data) if value_type == "Magnitude" else np.abs(np.real(data))
            self.data_before_fitting = data.copy()

            try:
                param_peak = params.pop("param_peak")
                param_gamma = params.pop("param_gamma")

                fitted, components, a, gamma, bg = peak_fitting_gpu.fit_volume_gpu(data, np.arange(F), param_peak,
                                                                                   param_gamma, **params)

                components = components.reshape(X, Y, Z, T, -1, F).transpose(3, 5, 0, 1, 2, 4)

                self.log_message(f"Peak fitting completed. Fitted shape: {fitted.shape}, Components shape: {components.shape}")
                self.I_fitted = {'fitted_data': fitted, 'separate_peaks': components,
                                 'amplitude': a, 'gamma': gamma, 'bg': bg}

            except Exception as e:
                self.log_message(f"Error applying peak fitting: {str(e)}")

    def display_fitting_results(self, data, target_plot, raw_data):
        if data is None:
            self.log_message("Error: No Peak Fitted data loaded.")
            return
        raw_data = raw_data.transpose(0, 2, 3, 4, 1)
        fitted = data['fitted_data']
        fitted = fitted.transpose(0, 2, 3, 4, 1)
        components = data['separate_peaks']
        components = components.transpose(0, 2, 3, 4, 1, 5)

        self.display_window = PeakFittingDisplayWindow(self)
        if self.display_window.exec():
            results = self.display_window.get_parameters()
            if results:
                display_type, params = results
        self.log_message(f"Display Parameters: {params}")
        
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']  # Predefined colors
        color_idx = 0

        try:
            if display_type == "Spectrum":

                if isinstance(target_plot, dict):
                    target_plot["container"].setCurrentWidget(target_plot["spectrum"])
                    plot_widget = target_plot["spectrum"]
                else:
                    plot_widget = target_plot
                plot_widget.clear()
                plot_widget.addLegend()

                coord = tuple(map(int, params["coord"].split(",")))
                idx = np.ravel_multi_index(coord, fitted.shape[:-1])
                spectrum_raw = raw_data.reshape(-1, fitted.shape[-1])[idx]
                plot_widget.plot(spectrum_raw, pen=pg.mkPen("gray"), name="Raw spectrum")
                spectrum_fit = fitted.reshape(-1, fitted.shape[-1])[idx]
                plot_widget.plot(spectrum_fit, pen=pg.mkPen("y"), name="Fitted spectrum")

                for i in range(components.shape[-1]):
                    spectrum_peak = components[..., i].reshape(-1, fitted.shape[-1])[idx]
                    color = pg.mkPen(colors[color_idx % len(colors)], width=1)
                    plot_widget.plot(spectrum_peak, pen=color, name=f"Peak {i+1}")
                    color_idx += 1

                plot_widget.setLabel("bottom", "Frequency")
                plot_widget.setLabel("left", "Signal Intensity")
                plot_widget.setTitle("Peak Fitting Results")
                self.log_message(f"Displayed peak fitting results at {coord}")

            elif display_type == "Image":
                if isinstance(target_plot, dict):
                    target_plot["container"].setCurrentWidget(target_plot["image"])
                    image_wrapper = target_plot["image"]
                    image_widget = target_plot["image_view"]
                else:
                    image_wrapper = target_plot
                    image_widget = target_plot

                layout = image_wrapper.layout()
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()

                # Structural image loading
                if not hasattr(self, 'structural_image') or self.structural_image is None:
                    file_path, _ = QFileDialog.getOpenFileName(None, "Select Structural Image", "", "Images (*.npy *.nii *.nii.gz)")
                    if not file_path:
                        self.log_message("No structural image selected.")
                        return
                    self.structural_image = np.load(file_path) if file_path.endswith(".npy") else nib.load(file_path).get_fdata() # type: ignore
                structural = self.structural_image
                
                # Prepare DMI image
                peak_idx = int(params["peak_index"])
                dim = int(params["slice_dim"])
                time_idx = int(params["time_idx"])
                #print(f"Peak index: {peak_idx}, Dimension: {dim}, Time index: {time_idx}")
                #slice_idx = int(params["slice_idx"])
                peak_map_t = components[..., peak_idx].max(axis=4) # [T, X, Y, Z]
                peak_map = peak_map_t[time_idx, :, :, :].squeeze()  # [X, Y, Z]
                peak_map = np.moveaxis(peak_map, dim, -1) 

                # Resize DMI to match structural
                target_shape = structural.shape[:2]
                resized_slices = []
                for z in range(peak_map.shape[-1]):
                    resized = cv2.resize(peak_map[:, :, z], (target_shape[0], target_shape[1]), interpolation=cv2.INTER_AREA)
                    resized_slices.append(resized)
                dmi_resized = np.stack(resized_slices, axis=2)

                self.struct_slider = QSlider(Qt.Orientation.Horizontal)
                self.struct_slider.setMinimum(0)
                self.struct_slider.setMaximum(structural.shape[2] - 1)
                self.struct_slider.setValue(structural.shape[2] // 2)

                self.dmi_slider = QSlider(Qt.Orientation.Horizontal)
                self.dmi_slider.setMinimum(0)
                self.dmi_slider.setMaximum(dmi_resized.shape[2] - 1)
                self.dmi_slider.setValue(dmi_resized.shape[2] // 2)

                self.opacity_selector = QSlider(Qt.Orientation.Horizontal)
                self.opacity_selector.setMinimum(0)
                self.opacity_selector.setMaximum(100)
                self.opacity_selector.setValue(60)

                self.colormap_selector = QComboBox()
                self.colormap_selector.addItems(["viridis", "plasma", "magma", "cividis"])

                view = pg.ViewBox()
                view.setAspectLocked(True)
                struct_item = pg.ImageItem()
                dmi_item = pg.ImageItem()
                view.addItem(struct_item)
                view.addItem(dmi_item)

                def update_overlay():
                    struct_idx = self.struct_slider.value()
                    dmi_idx = self.dmi_slider.value()

                    struct_slice = structural[:, :, struct_idx]
                    struct_slice = np.rot90(struct_slice, k=-1)
                    dmi_slice = dmi_resized[:, :, dmi_idx]
                    dmi_slice = np.rot90(dmi_slice, k=-1)

                    struct_item.setImage(struct_slice, autoLevels=True)
                    dmi_item.setImage(dmi_slice, autoLevels=True)

                    name = self.colormap_selector.currentText()
                    cmap = pg.colormap.get(name)
                    if cmap is not None:
                        dmi_item.setLookupTable(cmap.getLookupTable())

                    dmi_item.setLevels((0, np.max(dmi_resized)))  # Set levels for DMI image
                    dmi_item.setOpacity(self.opacity_selector.value() / 100.0)
                
                self.struct_slider.valueChanged.connect(update_overlay)
                self.dmi_slider.valueChanged.connect(update_overlay)
                self.opacity_selector.valueChanged.connect(update_overlay)
                self.colormap_selector.currentTextChanged.connect(update_overlay)

                overlay_plot = pg.GraphicsLayoutWidget()
                overlay_plot.setMinimumSize(300, 300)
                overlay_plot.addItem(view)
                layout.addWidget(overlay_plot)
                layout.addWidget(QLabel("Structural Slice:"))
                layout.addWidget(self.struct_slider)
                layout.addWidget(QLabel("DMI Slice:"))
                layout.addWidget(self.dmi_slider)
                layout.addWidget(QLabel("Overlay Opacity:"))
                layout.addWidget(self.opacity_selector)
                layout.addWidget(QLabel("Colormap:"))
                layout.addWidget(self.colormap_selector)

                update_overlay()
                self.log_message(f"Displayed peak fitting results with structural overlay.")

        except Exception as e:
            self.log_message( f"Display Error: {str(e)}")

    def open_data_analysis_settings(self, fitted_data, raw_data):
        if fitted_data is None:
            self.log_message("Error: No data loaded.")
            return
        if raw_data is None:
            self.log_message("Error: No raw data loaded.")
            return
        self.data_analysis_window = DataAnalysisWindow(fitted_data, raw_data)
        self.data_analysis_window.show()

class SpectrumSettingsWindow(QDialog):
    def __init__(self, parent=None, spectrum_length=512):
        super().__init__(parent)
        self.setWindowTitle("Spectrum Settings")

        layout = QFormLayout()

        # Value Type selection
        self.value_type_combo = QComboBox()
        self.value_type_combo.addItems(["Magnitude", "Imaginary", "Real"])
        layout.addRow(QLabel("Value Type:"), self.value_type_combo)

        # Absolute value toggle
        self.abs_checkbox = QCheckBox("Show Absolute Value")
        layout.addRow(self.abs_checkbox)

        # Time Series Display Method selection
        self.display_method_combo = QComboBox()
        self.display_method_combo.addItems(["Average", "Maximum"])
        layout.addRow(QLabel("Time Series Display Method:"), self.display_method_combo)

        # Time Series Index input
        self.index_input = QLineEdit()
        self.index_input.setPlaceholderText("Enter indices (comma-separated)")
        layout.addRow(QLabel("Time Series Index:"), self.index_input)

        # Frequency Range input
        self.freq_range_input = QLineEdit()
        self.freq_range_input.setPlaceholderText(f"Enter frequency range (default: 0:{spectrum_length})")
        self.freq_range_input.setText(f"0:{spectrum_length}")  # Default range
        layout.addRow(QLabel("Frequency Range:"), self.freq_range_input)

        # Confirm button
        self.confirm_button = QPushButton("OK")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def get_parameters(self):
        """ Return selected parameters """
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

        # Value Type selection
        self.value_type_combo = QComboBox()
        self.value_type_combo.addItems(["Magnitude", "Imaginary", "Real"])
        layout.addRow(QLabel("Value Type:"), self.value_type_combo)

        # Absolute value toggle
        self.abs_checkbox = QCheckBox("Show Absolute Value")
        layout.addRow(self.abs_checkbox)

        self.negative_checkbox = QCheckBox("Show Negative Value")
        layout.addRow(self.negative_checkbox)

        # Time Series Display Method selection
        self.display_method_combo = QComboBox()
        self.display_method_combo.addItems(["Average", "Maximum"])
        layout.addRow(QLabel("Time Series Display Method:"), self.display_method_combo)

        # Time Series Index input
        self.index_input = QLineEdit()
        self.index_input.setText("1,3,5,7,9,11")
        layout.addRow(QLabel("Time Series Index:"), self.index_input)

        # Frequency Range input
        self.freq_range_input = QLineEdit()
        self.freq_range_input.setPlaceholderText(f"Enter frequency range (default: 0:{image_shape[1]})")
        self.freq_range_input.setText(f"0:{image_shape[1]}")  # Default range
        layout.addRow(QLabel("Frequency Range:"), self.freq_range_input)

        # Slice Index input
        self.slice_input = QLineEdit()
        self.slice_input.setPlaceholderText(f"dim,slices (e.g., 4,{image_shape[4]//2})")
        layout.addRow(QLabel("Slice Selection:"), self.slice_input)

        # Region of Interest (ROI) input
        self.roi_input = QLineEdit()
        self.roi_input.setPlaceholderText("Enter ROI coordinates (e.g., 4:6,5:8 or 4,5)")
        layout.addRow(QLabel("Region of Interest:"), self.roi_input)

        # Colormap selection
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "gray", "plasma", "inferno", "magma"])
        layout.addRow(QLabel("Colormap:"), self.colormap_combo)

        # Confirm button
        self.confirm_button = QPushButton("OK")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def get_parameters(self):
        """ Return selected parameters """
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

class OverlayWindow(QWidget):
    def __init__(self, dmi_image, parent=None):
        super(OverlayWindow, self).__init__(parent)
        self.setWindowTitle("Overlay Structural Image")
        self.setMinimumSize(800, 600)

        self.dmi_image0 = dmi_image  # Assume shape [T, F, X, Y, Z]
        self.structural_image = None

        layout = QVBoxLayout()

        # Input fields
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Select structural image (.npy or .nii)")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_structural_image)
        file_layout.addWidget(QLabel("Structural Image Path:"))
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(browse_button)
        layout.addLayout(file_layout)

        self.time_index_input = QLineEdit()
        self.time_index_input.setPlaceholderText("Enter DMI time index (e.g., 1)")
        layout.addWidget(QLabel("DMI Time Index:"))
        layout.addWidget(self.time_index_input)

        self.freq_range_input = QLineEdit()
        self.freq_range_input.setPlaceholderText("Enter frequency range (e.g., 0:40)")
        layout.addWidget(QLabel("Frequency Range (start:end):"))
        layout.addWidget(self.freq_range_input)

        # Apply button
        self.load_button = QPushButton("Apply Overlay")
        self.load_button.clicked.connect(self.prepare_overlay)
        layout.addWidget(self.load_button)

        # Display area
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.view = self.plot_widget.addViewBox() # type: ignore
        self.view_items = [] 
        self.view.setAspectLocked(True)
        layout.addWidget(self.plot_widget)

        # Sliders
        slider_layout = QHBoxLayout()
        self.dmi_slider = QSlider(Qt.Orientation.Horizontal)
        self.dmi_slider.valueChanged.connect(self.update_overlay)
        self.dmi_index_label = QLabel("0")
        self.dmi_slider.valueChanged.connect(lambda val: self.dmi_index_label.setText(str(val)))
        slider_layout.addWidget(QLabel("DMI Slice:"))
        slider_layout.addWidget(self.dmi_slider)
        slider_layout.addWidget(self.dmi_index_label)

        self.struct_slider = QSlider(Qt.Orientation.Horizontal)
        self.struct_slider.valueChanged.connect(self.update_overlay)
        self.struct_index_label = QLabel("0")
        self.struct_slider.valueChanged.connect(lambda val: self.struct_index_label.setText(str(val)))
        slider_layout.addWidget(QLabel("Structural Slice:"))
        slider_layout.addWidget(self.struct_slider)
        slider_layout.addWidget(self.struct_index_label)

        layout.addLayout(slider_layout)
        self.setLayout(layout)

    def browse_structural_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Structural Image", "", "Images (*.npy *.nii *.nii.gz)")
        if file_path:
            self.file_input.setText(file_path)
    
    def prepare_overlay(self):
        """ Load structural image and prepare overlay """
        path = self.file_input.text().strip()
        if path:
            self.structural_image = np.load(path) if path.endswith(".npy") else nib.load(path).get_fdata() # type: ignore

            # Get DMI time index and frequency range
            try:
                time_idx = int(self.time_index_input.text().strip())
                freq_range = self.freq_range_input.text().strip()
                start, end = map(int, freq_range.split(":"))

                if self.dmi_image0.shape[-1] == 2:
                    self.dmi_image0 = self.dmi_image0[..., 0] + 1j * self.dmi_image0[..., 1]

                self.dmi_image = np.abs(self.dmi_image0[time_idx, start:end,...].squeeze().max(axis=0))  # [X, Y, Z]
                #import cv2
                target_shape = self.structural_image.shape[:2]  # (X, Y)
                resized_slices = []
                for z in range(self.dmi_image.shape[2]):
                    resized = cv2.resize(self.dmi_image[:, :, z], (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
                    resized_slices.append(resized)
                self.dmi_image = np.stack(resized_slices, axis=2)
            except Exception as e:
                print("Error in overlay preparation:", e)

            self.struct_slider.setMaximum(self.structural_image.shape[2] - 1)
            self.struct_slider.setValue(self.structural_image.shape[2] // 2)
            self.dmi_slider.setMaximum(self.dmi_image.shape[2] - 1)
            self.dmi_slider.setValue(self.dmi_image.shape[2] // 2)
            self.update_overlay()

    def update_overlay(self):
        if self.structural_image is None:
            return

        struct_idx = self.struct_slider.value()
        dmi_idx = self.dmi_slider.value()

        struct_slice = self.structural_image[:, :, struct_idx]
        struct_slice = np.rot90(struct_slice, k=-1)  # Rotate for correct orientation
        dmi_slice = self.dmi_image[:, :, dmi_idx]
        dmi_slice = np.rot90(dmi_slice, k=-1)
        
        #self.view.clear()
        for item in self.view_items:
            self.view.removeItem(item)
        self.view_items.clear()

        struct_item = pg.ImageItem(struct_slice)
        dmi_item = pg.ImageItem(dmi_slice)
        dmi_item.setLookupTable(pg.colormap.get("viridis").getLookupTable()) #type: ignore
        dmi_item.setLevels((0, np.max(self.dmi_image)))  # Set levels for DMI image
        dmi_item.setOpacity(0.5)

        self.view.addItem(struct_item)
        #self.view_items.append(struct_item)
        self.view.addItem(dmi_item)
        #self.view_items.append(dmi_item)
        self.view_items.extend([struct_item, dmi_item])

class PhaseCorrectionSettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phase Correction Settings")

        layout = QFormLayout()

        # Method selection
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Zero-order", "First-order", "B-spline", "Fourier"])
        layout.addRow(QLabel("Select Phase Correction Method:"),self.method_combo)

        # Common parameters
        self.lr_input = QLineEdit("Learning Rate:")
        self.lr_input.setText(f"0.05")
        layout.addRow(QLabel("Learning Rate:"), self.lr_input)

        self.iter_input = QLineEdit("Number of Iterations:")
        self.iter_input.setText(f"200")
        layout.addRow(QLabel("Number of Iterations:"), self.iter_input)

        # Fourier-specific
        #self.fourier_label = QLabel("↓ Fourier-specific Parameters ↓")
        self.fourier_basis_input = QLineEdit("Number of Fourier Basis:")
        self.fourier_basis_input.setText(f"8")
        layout.addRow(QLabel("Number of Fourier Basis:"), self.fourier_basis_input)

        # B-spline-specific
        #self.bspline_label = QLabel("↓ B-spline-specific Parameters ↓")
        self.peak_list_input = QLineEdit()
        self.peak_width_input = QLineEdit("Half Peak Width:")
        self.bspline_degree_input = QLineEdit("B-spline Degree:")
        self.peak_list_input.setPlaceholderText(f"Enter Peak Index (1,2,3,...)")
        self.peak_width_input.setText(f"5")
        self.bspline_degree_input.setText(f"3")
        layout.addRow(QLabel("Peak List:"), self.peak_list_input)
        layout.addRow(QLabel("Half Peak Width:"), self.peak_width_input)
        layout.addRow(QLabel("B-spline Degree:"), self.bspline_degree_input)

        # Confirm button
        self.confirm_button = QPushButton("OK")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

        # Update visibility based on method
        self.method_combo.currentTextChanged.connect(self.update_visibility)
        self.update_visibility()

    def update_visibility(self):
        method = self.method_combo.currentText()
        #self.fourier_label.setVisible(method == "Fourier")
        self.fourier_basis_input.setVisible(method == "Fourier")
        #self.bspline_label.setVisible(method == "B-spline")
        self.peak_list_input.setVisible(method == "B-spline")
        self.peak_width_input.setVisible(method == "B-spline")
        self.bspline_degree_input.setVisible(method == "B-spline")

    def get_parameters(self):
        """ Return selected parameters """
        return {
            "method": self.method_combo.currentText(),
            "lr": float(self.lr_input.text()),
            "n_iters": int(self.iter_input.text()),
            "num_basis": int(self.fourier_basis_input.text()),
            "peak_list": list(map(int, self.peak_list_input.text().split(","))) if self.peak_list_input.text() else None,
            "half_peak_width": int(self.peak_width_input.text()),
            "degree": int(self.bspline_degree_input.text())
        }

class NormalizationSettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Normalization Method")

        layout = QFormLayout()

        layout.addWidget(QLabel("Choose a normalization method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Max Abs", "Min-Max", "Background Z-score", "Background Mean Scaling"])
        layout.addWidget(self.method_combo)

        self.bg_label = QLabel("Background Region")
        self.bg_input = QLineEdit()
        self.bg_input.setText("0, 0:50, 0:10, 0:10, 0:1")
        self.bg_input.setVisible(False)
        self.bg_label.setVisible(False)
        layout.addWidget(self.bg_label)
        layout.addWidget(self.bg_input)

        layout.addWidget(QLabel("For complex data normalization:"))
        self.complex_mode_combo = QComboBox()
        self.complex_mode_combo.addItems(["Magnitude Only", 
                                          "Real and Imaginary Separately","Complex as Whole"
                                          ])
        layout.addWidget(self.complex_mode_combo)

        self.confirm_button = QPushButton("Apply")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

        self.method_combo.currentTextChanged.connect(self.update_visibility)
        self.update_visibility()
    
    def update_visibility(self):
        show_bg = self.method_combo.currentText() == "Background Mean Scaling" or self.method_combo.currentText() == "Background Z-score"
        self.bg_label.setVisible(show_bg)
        self.bg_input.setVisible(show_bg)

    def get_parameters(self):
        """ Return selected parameters """
        return {
            "method": self.method_combo.currentText(),
            "mode": self.complex_mode_combo.currentText(),
            "bg_region": self.bg_input.text()
        }

class ComparisonSettingsWindow(QWidget):
    def __init__(self, data, apply_phase_func):
        super().__init__()
        self.setWindowTitle("Phase Correction Comparison")
        self.data = data
        self.apply_phase_func = apply_phase_func

        self.methods = ["Zero-order", "First-order", "B-spline", "Fourier"]
        self.method_checkboxes = {}
        self.method_param_forms = {}

        main_layout = QVBoxLayout()

        # Coordinate input
        main_layout.addWidget(QLabel("Enter coordinate (e.g., t, x, y, z):"))
        self.coord_input = QLineEdit()
        main_layout.addWidget(self.coord_input)

        # Display option
        main_layout.addWidget(QLabel("Select display type:"))
        self.display_mode = QComboBox()
        self.display_mode.addItems(["Magnitude", "Real", "Imaginary"])
        main_layout.addWidget(self.display_mode)

        # Absolute value toggle
        self.abs_checkbox = QCheckBox("Show Absolute Value")
        main_layout.addWidget(self.abs_checkbox)

        self.negative_checkbox = QCheckBox("Show Negative Value")
        main_layout.addWidget(self.negative_checkbox)

        # Method checkboxes + parameter forms
        method_group = QGroupBox("Select Phase Correction Methods")
        method_layout = QVBoxLayout()

        for method in self.methods:
            cb = QCheckBox(method)
            cb.stateChanged.connect(self.update_parameter_forms)
            self.method_checkboxes[method] = cb
            method_layout.addWidget(cb)

            # Create and hide parameter forms
            form = QFormLayout()
            container = QWidget()
            container.setLayout(form)
            container.setVisible(False)
            self.method_param_forms[method] = (container, form)
            method_layout.addWidget(container)

        method_group.setLayout(method_layout)
        scroll = QScrollArea()
        scroll.setWidget(method_group)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.run_comparison)
        main_layout.addWidget(self.apply_button)

        # Plot area
        self.plot_widget = pg.PlotWidget(title="Phase Correction Comparison")
        main_layout.addWidget(self.plot_widget)

        self.setLayout(main_layout)

    def update_parameter_forms(self):
        for method, checkbox in self.method_checkboxes.items():
            container, form = self.method_param_forms[method]
            if checkbox.isChecked():
                container.setVisible(True)
                if form.rowCount() == 0:
                    self.populate_form(method, form)
            else:
                container.setVisible(False)

    def populate_form(self, method, form):
        if method == "Zero-order":
            form.addRow("Learning Rate:", QLineEdit("0.05"))
            form.addRow("Num Iterations:", QLineEdit("200"))
        elif method == "First-order":
            form.addRow("Learning Rate:", QLineEdit("0.05"))
            form.addRow("Num Iterations:", QLineEdit("200"))
        elif method == "Fourier":
            form.addRow("Num Fourier Basis:", QLineEdit("8"))
        elif method == "B-spline":
            form.addRow("Peak List:", QLineEdit("None"))
            form.addRow("Half Peak Width:", QLineEdit("5"))
            form.addRow("B-spline Degree:", QLineEdit("3"))

    def run_comparison(self):
        coord_str = self.coord_input.text()
        try:
            coord = tuple(map(int, coord_str.split(",")))
        except:
            QMessageBox.warning(self, "Invalid Input", "Coordinate format is invalid.")
            return
        spectrum = self.data[coord[0], :, coord[1], coord[2], coord[3]]

        self.plot_widget.clear()

        show_abs = self.abs_checkbox.isChecked()
        show_negative = self.negative_checkbox.isChecked()
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']  # Predefined colors
        color_idx = 0
        self.plot_widget.addLegend()

        display_type = self.display_mode.currentText()
        if display_type == "Magnitude":
            self.plot_widget.plot(np.abs(spectrum), pen=pg.mkPen("gray"), name="Original")
        elif display_type == "Real":
            spectrum_plot = np.abs(np.real(spectrum)) if show_abs else np.real(spectrum)
            spectrum_plot = -spectrum_plot if show_negative else spectrum_plot
            self.plot_widget.plot(spectrum_plot, pen=pg.mkPen("gray"), name="Original")
        elif display_type == "Imaginary":
            spectrum_plot = -np.imag(spectrum) if show_negative else np.imag(spectrum)
            self.plot_widget.plot(spectrum_plot, pen=pg.mkPen("gray"), name="Original")

        for method, cb in self.method_checkboxes.items():
            if cb.isChecked():
                _, form = self.method_param_forms[method]
                params = {}
                for i in range(form.rowCount()):
                    label = form.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().text().strip(':')
                    value = form.itemAt(i, QFormLayout.ItemRole.FieldRole).widget().text()
                    params[label.lower().replace(' ', '_')] = value

                corrected = self.apply_phase_func(spectrum.copy(), method, params)
                color = pg.mkPen(colors[color_idx % len(colors)], width=1)
                if corrected is not None:
                    if display_type == "Magnitude":
                        self.plot_widget.plot(np.abs(corrected), pen=color, name=method)
                    elif display_type == "Real":
                        data_plot = np.abs(np.real(corrected)) if show_abs else np.real(corrected)
                        data_plot = -data_plot if show_negative else data_plot
                        self.plot_widget.plot(data_plot, pen=color, name=method)
                    elif display_type == "Imaginary":
                        data_plot = -np.imag(corrected) if show_negative else np.imag(corrected)
                        self.plot_widget.plot(data_plot, pen=color, name=method)
                color_idx += 1
        self.plot_widget.setLabel("bottom", "Frequency")
        self.plot_widget.setLabel("left", "Normalized Signal Intensity")

class DenoiseComparisonSettingsWindow(QWidget):
    def __init__(self, data, apply_denoising_func):
        super().__init__()
        self.setWindowTitle("Denoising Comparison")
        self.data = data
        self.apply_denoising_func = apply_denoising_func

        self.methods = ["Mean Filter", "Median Filter", "Gaussian Filter", 
                        "Singular Value Decomposition", "Principal Component Analysis", 
                        "Savitzky-Golay Filter", "Wavelet Thresholding", "Fourier Filter",
                        "Total Variation", "Wiener Filter", "UNet Model","Transform Model"]
        self.method_blocks = []

        layout = QVBoxLayout()

        # Coordinate input
        layout.addWidget(QLabel("Enter coordinate (e.g., t, x, y, z):"))
        self.coord_input = QLineEdit()
        layout.addWidget(self.coord_input)

        # Value type selector
        layout.addWidget(QLabel("Select data type to denoise:"))
        self.value_type = QComboBox()
        self.value_type.addItems(["Magnitude", "Real"])
        layout.addWidget(self.value_type)

        # Add multiple method blocks
        self.method_area = QVBoxLayout()
        self.add_method_block()  # Add one by default
        self.add_method_btn = QPushButton("Add Another Method")
        self.add_method_btn.clicked.connect(self.add_method_block)
        layout.addWidget(self.add_method_btn)

        container = QWidget()
        container.setLayout(self.method_area)
        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(200)
        layout.addWidget(scroll)

        # Show original option
        self.show_original_checkbox = QCheckBox("Show Original Noisy Spectrum")
        layout.addWidget(self.show_original_checkbox)

        # Absolute value toggle
        self.abs_checkbox = QCheckBox("Show Absolute Value")
        layout.addWidget(self.abs_checkbox)

        self.negative_checkbox = QCheckBox("Show Negative Value")
        layout.addWidget(self.negative_checkbox)

        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.run_comparison)
        layout.addWidget(self.apply_button)

        # Plot area
        self.plot_widget = pg.PlotWidget(title="Denoising Comparison")
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)     

    def add_method_block(self):
        method_layout = QFormLayout()
        method_selector = QComboBox()
        method_selector.addItems(self.methods)
        method_layout.addRow("Method:", method_selector)

        param_inputs = {}
        container = QWidget()
        container.setLayout(method_layout)
        self.method_area.addWidget(container)

        def update_param_hint():
            # Clear existing dynamic fields
            while method_layout.rowCount() > 1:
                method_layout.removeRow(1)
                param_inputs.clear()

            method = method_selector.currentText()
            if method in ["Mean Filter", "Median Filter"]:
                label = QLabel("Window Size:")
                input_box = QLineEdit()
                input_box.setText("5")
                method_layout.addRow(label, input_box)
                param_inputs["window_size"] = input_box

            elif method == "Gaussian Filter":
                label = QLabel("Sigma:")
                input_box = QLineEdit()
                input_box.setText("1.0")
                method_layout.addRow(label, input_box)
                param_inputs["sigma"] = input_box

            elif method in ["Singular Value Decomposition", "Principal Component Analysis"]:
                label = QLabel("Number of Components:")
                input_box = QLineEdit()
                input_box.setText("5")
                method_layout.addRow(label, input_box)
                param_inputs["num_components"] = input_box

            elif method == "Savitzky-Golay Filter":
                label1 = QLabel("Window Size:")
                input1 = QLineEdit()
                label2 = QLabel("Poly Order:")
                input2 = QLineEdit()
                input1.setText("9")
                input2.setText("3")
                method_layout.addRow(label1, input1)
                method_layout.addRow(label2, input2)
                param_inputs["window_size"] = input1
                param_inputs["polyorder"] = input2

            elif method == "Wavelet Thresholding":
                label1 = QLabel("Wavelet Type:")
                input1 = QLineEdit()
                label2 = QLabel("Threshold:")
                input2 = QLineEdit()
                input1.setText("db1")
                input2.setText("0.04")
                method_layout.addRow(label1, input1)
                method_layout.addRow(label2, input2)
                param_inputs["wavelet"] = input1
                param_inputs["threshold"] = input2

            elif method == "Fourier Filter":
                label = QLabel("Cutoff Frequency:")
                input_box = QLineEdit()
                input_box.setText("0.25")
                method_layout.addRow(label, input_box)
                param_inputs["cutoff_freq"] = input_box

            elif method == "Total Variation":
                label = QLabel("Weight:")
                input_box = QLineEdit()
                input_box.setText("0.1")
                method_layout.addRow(label, input_box)
                param_inputs["weight"] = input_box

            elif method == "Wiener Filter":
                label = QLabel("Kernel Size:")
                input_box = QLineEdit()
                input_box.setText("5")
                method_layout.addRow(label, input_box)
                param_inputs["kernel_size"] = input_box

            elif method in ["UNet Model", "Transform Model"]:
                label = QLabel("Model Path:")
                input_box = QLineEdit()
                input_box.setPlaceholderText("Path to weight file")

                def browse_model():
                    file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Weight File", "", "Model Files (*.pth)")
                    if file_path:
                        input_box.setText(file_path)

                browse_button = QPushButton("Browse")
                browse_button.clicked.connect(browse_model)

                file_select_layout = QHBoxLayout()
                file_select_layout.addWidget(input_box)
                file_select_layout.addWidget(browse_button)
                container = QWidget()
                container.setLayout(file_select_layout)

                method_layout.addRow(label, container)
                param_inputs["model_path"] = input_box

                peak_label = QLabel("Peak List:")
                peak_input = QLineEdit()
                peak_input.setText("125,135,150,160")
                method_layout.addRow(peak_label, peak_input)
                param_inputs["peak_list"] = peak_input

            else:
                label = QLabel("Parameter:")
                input_box = QLineEdit()
                input_box.setPlaceholderText("optional")
                method_layout.addRow(label, input_box)
                param_inputs["param"] = input_box

        method_selector.currentTextChanged.connect(update_param_hint)
        update_param_hint()
        self.method_blocks.append((method_selector, param_inputs))

    def run_comparison(self):
        coord_str = self.coord_input.text()
        try:
            coord = tuple(map(int, coord_str.split(",")))
        except:
            QMessageBox.warning(self, "Invalid Input", "Coordinate format is invalid.")
            return
        spectrum = self.data[coord[0], :, coord[1], coord[2], coord[3]]

        self.plot_widget.clear()
        show_original = self.show_original_checkbox.isChecked()

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']
        color_idx = 0
        self.plot_widget.addLegend()

        value_type = self.value_type.currentText()
        show_abs = self.abs_checkbox.isChecked()
        show_negative = self.negative_checkbox.isChecked()

        if show_original:
            if value_type == "Magnitude":
                self.plot_widget.plot(np.abs(spectrum), pen=pg.mkPen("gray"), name="Original")
            else:
                spectrum_plot = np.abs(np.real(spectrum)) if show_abs else np.real(spectrum)
                spectrum_plot = -spectrum_plot if show_negative else spectrum_plot              
                self.plot_widget.plot(spectrum_plot, pen=pg.mkPen("gray"), name="Original")

        for method_selector, param_inputs in self.method_blocks:
            method = method_selector.currentText()
            try:
                if value_type == "Magnitude":
                    input_spectrum = np.abs(spectrum)
                else:
                    input_spectrum = -np.real(spectrum) if show_negative else np.real(spectrum)
                    #input_spectrum = np.real(spectrum)

                params = {}
                for key, input_box in param_inputs.items():
                    text = input_box.text()
                    if text:
                        try:
                            val = float(text)
                            if val.is_integer():
                                val = int(val)
                            params[key] = val
                        except:
                            params[key] = text

                denoised = self.apply_denoising_func(input_spectrum.copy(), method, params)
                label = method + " (" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"
                color = pg.mkPen(colors[color_idx % len(colors)], width=1)
                if denoised is not None:
                    denoised = np.abs(denoised) if show_abs else denoised
                    denoised = -denoised if show_negative else denoised
                    self.plot_widget.plot(denoised, pen=color, name=label)
                color_idx += 1
            except Exception as e:
                QMessageBox.warning(self, f"[{method}] Error", str(e))

        self.plot_widget.setLabel("bottom", "Frequency")
        self.plot_widget.setLabel("left", "Denoised Signal Intensity")

class DenoiseSettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply Denoising to Phase-Corrected Data")

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

        self.method_selector.currentTextChanged.connect(self.update_param_form)
        self.update_param_form()

        # Confirm button
        self.confirm_button = QPushButton("OK")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def update_param_form(self):
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self.param_inputs.clear()

        method = self.method_selector.currentText()

        if method in ["Mean Filter", "Median Filter"]:
            box = QLineEdit("Window Size:")
            box.setText("5")
            self.param_form.addRow(QLabel("Window Size:"), box)
            self.param_inputs["window_size"] = box

        elif method == "Gaussian Filter":
            box = QLineEdit("Sigma:")
            box.setText("1.0")
            self.param_form.addRow(QLabel("Sigma:"), box)
            self.param_inputs["sigma"] = box
        
        elif method in ["Singular Value Decomposition", "Principal Component Analysis"]:
            box = QLineEdit("Number of Components:")
            box.setText("5")
            self.param_form.addRow(QLabel("Number of Components:"), box)
            self.param_inputs["num_components"] = box

        elif method == "Savitzky-Golay Filter":
            box1 = QLineEdit("Window Size:")
            box2 = QLineEdit("Poly Order:")
            box1.setText("9")
            box2.setText("3")
            self.param_form.addRow(QLabel("Window Size:"), box1)
            self.param_form.addRow(QLabel("Poly Order:"), box2)
            self.param_inputs["window_size"] = box1
            self.param_inputs["polyorder"] = box2

        elif method == "Wavelet Thresholding":
            box1 = QLineEdit("Wavelet Type:")
            box2 = QLineEdit("Threshold:")
            box1.setText("db1")
            box2.setText("0.04")
            self.param_form.addRow(QLabel("Wavelet Type:"), box1)
            self.param_form.addRow(QLabel("Threshold:"), box2)
            self.param_inputs["wavelet"] = box1
            self.param_inputs["threshold"] = box2

        elif method == "Fourier Filter":
            box = QLineEdit("Cutoff Frequency:")
            box.setText("0.25")
            self.param_form.addRow(QLabel("Cutoff Frequency:"), box)
            self.param_inputs["cutoff_freq"] = box

        elif method == "Total Variation":
            box = QLineEdit("Weight:")
            box.setText("0.1")
            self.param_form.addRow(QLabel("Weight:"), box)
            self.param_inputs["weight"] = box
        
        elif method == "Wiener Filter":
            box = QLineEdit("Kernel Size:")
            box.setText("5")
            self.param_form.addRow(QLabel("Kernel Size:"), box)
            self.param_inputs["kernel_size"] = box

        elif method in ["UNet Model", "Transform Model"]:
            box = QLineEdit()
            box.setPlaceholderText("Select model path (.pth)")

            def browse_model_path():
                file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Path", "", "Model Files (*.pth)")
                if file_path:
                    box.setText(file_path)

            browse_button = QPushButton("Browse")
            browse_button.clicked.connect(browse_model_path)

            self.param_form.addRow(QLabel("Model Path:"), box)
            self.param_form.addRow(browse_button)
            self.param_inputs["model_path"] = box

            peak_label = QLabel("Peak Locations (comma separated):")
            peak_input = QLineEdit()
            peak_input.setPlaceholderText("e.g., 125, 135, 150, 160")
            self.param_form.addRow(peak_label, peak_input)
            self.param_inputs["peak_list"] = peak_input

        else:
            box = QLineEdit("Parameter:")
            self.param_form.addRow(QLabel("Parameter:"), box)
            self.param_inputs["param"] = box

    def get_parameters(self):
        method = self.method_selector.currentText()
        params = {}
        for k, box in self.param_inputs.items():
            text = box.text()
            if text:
                try:
                    val = float(text)
                    if val.is_integer():
                        val = int(val)
                    params[k] = val
                except:
                    params[k] = text
        return method, params

class PeakFittingSettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply Peak Fitting")

        layout = QVBoxLayout()
        self.param_inputs = {}

        # Value Type selection
        layout.addWidget(QLabel("Select data type:"))
        self.value_type_selector = QComboBox()
        self.value_type_selector.addItems(["Magnitude", "Real"])
        layout.addWidget(self.value_type_selector)

        # Define inputs with default values
        self.param_inputs["param_peak"] = QLineEdit()
        self.param_inputs["param_gamma"] = QLineEdit("10,10,10,10")
        self.param_inputs["min_gamma"] = QLineEdit("5")
        self.param_inputs["max_gamma"] = QLineEdit("20")
        self.param_inputs["peak_shift_limit"] = QLineEdit("2")
        self.param_inputs["num_peaks"] = QLineEdit("4")
        self.param_inputs["epochs"] = QLineEdit("1500")
        self.param_inputs["lr"] = QLineEdit("0.01")
        self.param_inputs["batch_size"] = QLineEdit("100")

        layout.addWidget(QLabel("Enter Peak Fitting Parameters:"))
        form = QFormLayout()
        form.addRow("Initial Peak Positions (comma-separated):", self.param_inputs["param_peak"])
        form.addRow("Initial Gamma Values (comma-separated):", self.param_inputs["param_gamma"])
        form.addRow("Min Gamma:", self.param_inputs["min_gamma"])
        form.addRow("Max Gamma:", self.param_inputs["max_gamma"])
        form.addRow("Peak Shift Limit:", self.param_inputs["peak_shift_limit"])
        form.addRow("Number of Peaks:", self.param_inputs["num_peaks"])
        form.addRow("Epochs:", self.param_inputs["epochs"])
        form.addRow("Learning Rate:", self.param_inputs["lr"])
        form.addRow("Batch Size:", self.param_inputs["batch_size"])
        layout.addLayout(form)

        # Confirm button
        self.confirm_button = QPushButton("Apply")
        self.confirm_button.clicked.connect(self.accept)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)
    
    def get_parameters(self):
        from typing import Any
        params: dict[str, Any] = {}
        params = {"value_type": self.value_type_selector.currentText()}
        for key, box in self.param_inputs.items():
            val = box.text().strip()
            if not val:
                continue
            if key in ["param_peak", "param_gamma"]:
                try:
                    # Convert comma-separated string into list of floats
                    params[key] = [float(x.strip()) for x in val.split(',')]
                except Exception as e:
                    params[key] = val  # fallback in case of error
            else:
                try:
                    num = float(val)
                    if num.is_integer():
                        num = int(num)
                    params[key] = num
                except:
                    params[key] = val
        return params

class PeakFittingDisplayWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Display Peak Fitting Results")

        layout = QVBoxLayout()

        self.display_type_selector = QComboBox()
        self.display_type_selector.addItems(["Spectrum", "Image"])
        self.display_type_selector.currentTextChanged.connect(self.update_form)

        layout.addWidget(QLabel("Select Display Type:"))
        layout.addWidget(self.display_type_selector)

        self.form_layout = QFormLayout()
        layout.addLayout(self.form_layout)

        self.plot_button = QPushButton("OK")
        self.plot_button.clicked.connect(self.accept)
        layout.addWidget(self.plot_button)

        self.setLayout(layout)
        self.update_form("Spectrum")

    def update_form(self, display_type):
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)

        if display_type == "Spectrum":
            self.coord_input = QLineEdit()
            self.coord_input.setPlaceholderText("e.g., 1,7,7,4")
            self.form_layout.addRow("Coordinate [t,x,y,z]", self.coord_input)

        elif display_type == "Image":
            self.peak_index_input = QLineEdit()
            self.dimension_input = QLineEdit()
            self.time_input = QLineEdit()
            self.peak_index_input.setText("1")
            self.dimension_input.setText("2") # 0: x, 1: y, 2: z
            self.time_input.setText("1")
            self.form_layout.addRow("Peak Index", self.peak_index_input)
            self.form_layout.addRow("Dimension to Slice", self.dimension_input)
            self.form_layout.addRow("Time Index", self.time_input)

    def get_parameters(self):
        params = {}
        display_type = self.display_type_selector.currentText()

        if display_type == "Spectrum":
            params["coord"] = self.coord_input.text()
            return display_type, params
        elif display_type == "Image":
            params["peak_index"] = self.peak_index_input.text()
            params["slice_dim"] = self.dimension_input.text()
            params["time_idx"] = self.time_input.text()
            return display_type, params

class DataAnalysisWindow(QWidget):
    def __init__(self, fitted_data, raw_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DMI Data Analysis")
        self.setMinimumSize(700, 500)
        self.fitted_data = fitted_data
        self.raw_data = raw_data[...,0] + 1j * raw_data[...,1] if raw_data.shape[-1] == 2 else raw_data
        self.raw_data = np.abs(self.raw_data)

        self.bg_level = fitted_data.get("bg", None)
        self.bg_level = np.mean(self.bg_level) if self.bg_level is not None else None
        self.peak_maps = fitted_data.get("separate_peaks", None)

        layout = QVBoxLayout()

        if self.bg_level is not None:
            layout.addWidget(QLabel(f"Estimated Background Level: {float(self.bg_level):.4f}"))

        try:
            noise_region = np.concatenate([self.raw_data[..., 0], self.raw_data[..., -1]], axis=0)
            layout.addWidget(QLabel(f"Noise Mean: {np.mean(noise_region):.4f}"))
            layout.addWidget(QLabel(f"Noise Std: {np.std(noise_region):.4f}"))
        except Exception as e:
            layout.addWidget(QLabel(f"Noise Estimation Failed: {str(e)}"))

        self.coord_input = QLineEdit()
        self.coord_input.setPlaceholderText("Enter coordinate (X,Y,Z)")
        layout.addWidget(QLabel("Select Voxel Coordinate:"))
        layout.addWidget(self.coord_input)

        self.peak_input = QLineEdit()
        self.peak_input.setPlaceholderText("Enter peak index (e.g., 1)")
        layout.addWidget(QLabel("Select Peak Index:"))
        layout.addWidget(self.peak_input)

        self.start_point_input = QLineEdit()
        self.start_point_input.setPlaceholderText("Enter start point (e.g., 0)")
        layout.addWidget(QLabel("Select Start Point:"))
        layout.addWidget(self.start_point_input)

        plot_btn = QPushButton("Plot Temporal Dynamics")
        plot_btn.clicked.connect(self.plot_temporal_curve)
        layout.addWidget(plot_btn)

        self.plot_widget = pg.PlotWidget(title="Temporal Curve")
        layout.addWidget(self.plot_widget)

        self.fit_selector = QComboBox()
        self.fit_selector.addItems(["Linear", "Exponential", "BiExponential", "BBFunction"])
        layout.addWidget(QLabel("Curve Fitting Model:"))
        layout.addWidget(self.fit_selector)

        self.fit_result_label = QLabel("Fitting Result:")
        layout.addWidget(self.fit_result_label)

        fit_btn = QPushButton("Curve Fitting")
        fit_btn.clicked.connect(self.apply_curve_fitting)
        layout.addWidget(fit_btn)

        self.setLayout(layout)

    def plot_temporal_curve(self):
        self.plot_widget.clear()
        try:
            coord = tuple(map(int, self.coord_input.text().split(",")))
            peak_idx = int(self.peak_input.text())
            self.start_point = int(self.start_point_input.text())
            x_range = np.arange(self.start_point, self.raw_data.shape[0])
            curve = self.peak_maps[self.start_point:, :, coord[0], coord[1], coord[2], peak_idx].squeeze()
            self.current_signal = curve.max(axis=1)
            self.plot_widget.plot(x_range, self.current_signal, pen='y', symbol='o', name="Temporal Signal")
        except Exception as e:
            self.plot_widget.setTitle(f"Error: {str(e)}")

    def apply_curve_fitting(self):
        if not hasattr(self, "current_signal"):
            self.fit_result_label.setText("No curve to fit. Please plot first.")
            return
        
        global DEVICE

        x = np.arange(self.start_point, self.raw_data.shape[0])
        y = self.current_signal
        
        model_type = self.fit_selector.currentText()
        try:
            if model_type == "Linear":
                model = curve_fitting.LinearModel(y)
            elif model_type == "Exponential":
                model = curve_fitting.ExpModel(y)
            elif model_type == "BiExponential":
                model = curve_fitting.BiExpModel(y)
            elif model_type == "BBFunction":
                model = curve_fitting.BBModel(y)
            else:
                self.fit_result_label.setText("Invalid model selection.")
                return

            x_fit, y_fit, params = curve_fitting.model_fitting(x, y, model, device=DEVICE)
            self.plot_widget.plot(x_fit, y_fit, pen='r', name="Fitted")
            self.fit_result_label.setText("Fitting Result: " + ", ".join(f"{v:.4f}" for v in params))

        except Exception as e:
            self.fit_result_label.setText(f"Fitting failed: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())  
