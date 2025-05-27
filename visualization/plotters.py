import numpy as np
import pyqtgraph as pg
import cv2 # For resizing in image overlay functions
from PyQt6.QtWidgets import QFileDialog, QLabel, QSlider, QComboBox, QVBoxLayout, QHBoxLayout, QWidget # For plot_peak_fitting_image_overlay
from PyQt6.QtCore import Qt # For QSlider orientation
# from ..io import load_image # Example if needed, though better to pass data

# Placeholder for log_message, can be replaced by a proper logging setup
def default_log_message(message):
    print(message)

def plot_spectrum_detailed(data_to_plot, target_plot_widgets, params, log_message_func=default_log_message):
    """
    Plots spectra based on provided parameters into target pyqtgraph widgets.
    
    Args:
        data_to_plot (np.ndarray): The data array to plot spectra from.
        target_plot_widgets (dict): A dict containing 'spectrum' (pg.PlotWidget)
                                   and 'container' (QStackedLayout).
        params (dict): Dictionary of parameters from SpectrumSettingsWindow.
        log_message_func (callable): Function to log messages.
    """
    if data_to_plot is None:
        log_message_func("Error: No data provided for spectrum plotting.")
        return

    log_message_func(f"Plotting spectrum with params: {params} for data shape {data_to_plot.shape}")

    value_type = params["value_type"]
    display_method = params["display_method"]
    index_input = params["time_series_index"]
    freq_range_input = params["frequency_range"]
    abs_value = params["show_abs"]

    current_data = data_to_plot

    try:
        freq_start, freq_end = map(int, freq_range_input.split(":"))
        freq_start = max(0, min(freq_start, current_data.shape[1] - 1))
        freq_end = max(freq_start + 1, min(freq_end, current_data.shape[1])) # Ensure freq_end > freq_start
    except ValueError:
        log_message_func("Invalid frequency range input. Using full range.")
        freq_start, freq_end = 0, current_data.shape[1]

    indices_to_plot = []
    if index_input:
        for item in index_input.split(","):
            item = item.strip()
            if ":" in item:
                start, end = map(int, item.split(":"))
                start = max(0, min(start, current_data.shape[0]-1))
                end = max(start + 1, min(end, current_data.shape[0]))
                indices_to_plot.append((start, end))
            else:
                idx_val = int(item)
                idx_val = max(0, min(idx_val, current_data.shape[0]-1))
                indices_to_plot.append(idx_val)
    else:
        indices_to_plot.append((0, current_data.shape[0]))
        log_message_func("No time series index. Plotting average/max over all time points.")


    processed_data = current_data.copy()
    if processed_data.ndim > 1 and processed_data.shape[-1] == 2: 
        processed_data = processed_data[..., 0] + 1j * processed_data[..., 1]
        log_message_func("Data converted to complex form.")

    if value_type == "Magnitude": processed_data = np.abs(processed_data)
    elif value_type == "Imaginary": processed_data = np.imag(processed_data)
    elif value_type == "Real":
        processed_data = np.real(processed_data)
        if abs_value: processed_data = np.abs(processed_data)
    
    plot_widget = target_plot_widgets["spectrum"]
    if "container" in target_plot_widgets:
        target_plot_widgets["container"].setCurrentWidget(plot_widget)
    
    plot_widget.clear()
    plot_widget.addLegend()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']
    color_idx = 0
    x_axis = np.arange(freq_start, freq_end)

    for idx_item in indices_to_plot:
        spectrum_to_display = None
        legend_name = f"Index {idx_item}"
        
        if processed_data.ndim < 2:
            log_message_func(f"Error: Data for spectrum plot has {processed_data.ndim} dims. Need >= 2.")
            continue

        current_selection = processed_data
        if isinstance(idx_item, tuple): 
            start, end = idx_item
            current_selection = current_selection[start:end, ...]
            if display_method == "Average": spectrum_to_display = current_selection.mean(axis=0)
            else: spectrum_to_display = current_selection.max(axis=0)
            legend_name = f"{display_method} {start}:{end}"
        else: 
            spectrum_to_display = current_selection[idx_item, ...]
            legend_name = f"Index {idx_item}"
        
        spectrum_to_display = spectrum_to_display[freq_start:freq_end, ...]
        if spectrum_to_display.ndim > 1:
            spectrum_to_display = np.mean(spectrum_to_display, axis=tuple(range(1, spectrum_to_display.ndim)))

        if spectrum_to_display is not None:
            pen_color = pg.mkPen(colors[color_idx % len(colors)], width=1)
            plot_widget.plot(x_axis, spectrum_to_display, pen=pen_color, name=legend_name)
            color_idx += 1

    plot_widget.setLabel("bottom", "Frequency Index")
    plot_widget.setLabel("left", "Signal Intensity")
    plot_widget.setTitle(f"Spectra - {value_type} ({display_method})")
    log_message_func("Spectrum plotting complete.")


def show_image_detailed(data_to_show, target_plot_widgets, params, log_message_func=default_log_message, full_data_freq_dim_len=None):
    if data_to_show is None:
        log_message_func("Error: No data provided for image display.")
        return

    log_message_func(f"Showing image with params: {params} for data shape {data_to_show.shape}")

    value_type = params["value_type"]
    display_method = params["display_method"] 
    index_input = params["time_series_index"] 
    freq_range_input = params["frequency_range"]
    slice_selection_input = params["slice_selection"] 
    roi_input = params["roi"] 
    colormap_name = params["colormap"]
    abs_value = params["show_abs"]
    show_negative = params["show_negative"]

    current_data = data_to_show.copy() 

    if current_data.shape[-1] == 2: 
        current_data = current_data[..., 0] + 1j * current_data[..., 1]
        log_message_func("Data converted to complex from real/imag parts.")

    if value_type == "Magnitude": processed_data = np.abs(current_data)
    elif value_type == "Imaginary": 
        processed_data = np.imag(current_data)
        if show_negative: processed_data = -processed_data
    elif value_type == "Real":
        processed_data = np.real(current_data)
        if show_negative: processed_data = -processed_data 
        if abs_value: processed_data = np.abs(processed_data) 
    else:
        processed_data = current_data 

    freq_dim_len = processed_data.shape[1] if processed_data.ndim > 1 else (full_data_freq_dim_len if full_data_freq_dim_len is not None else 1)
    try:
        freq_start, freq_end = map(int, freq_range_input.split(":"))
        freq_start = max(0, min(freq_start, freq_dim_len - 1))
        freq_end = max(freq_start + 1, min(freq_end, freq_dim_len))
    except ValueError:
        freq_start, freq_end = 0, freq_dim_len
    
    slicing_for_freq = [slice(None)] * processed_data.ndim
    if processed_data.ndim > 1: slicing_for_freq[1] = slice(freq_start, freq_end)
    processed_data_freq_selected = processed_data[tuple(slicing_for_freq)]
    log_message_func(f"Applied frequency range: {freq_start}:{freq_end}")
    
    indices_to_plot = []
    if index_input: 
        for item in index_input.split(","):
            item = item.strip()
            if ":" in item:
                start, end = map(int, item.split(":"))
                start = max(0, min(start, processed_data_freq_selected.shape[0]-1))
                end = max(start+1, min(end, processed_data_freq_selected.shape[0]))
                indices_to_plot.append((start, end))
            else:
                idx_val = int(item)
                idx_val = max(0, min(idx_val, processed_data_freq_selected.shape[0]-1))
                indices_to_plot.append(idx_val)
    else: indices_to_plot.append(0) 

    slice_dim_orig, slice_num = -1, 0
    if slice_selection_input:
        try:
            dim_str, num_str = slice_selection_input.split(",")
            slice_dim_orig = int(dim_str) 
            slice_num = int(num_str)
        except ValueError: log_message_func("Invalid slice selection. Using defaults.")
    
    if roi_input: 
        plot_widget = target_plot_widgets["spectrum"] 
        if "container" in target_plot_widgets: target_plot_widgets["container"].setCurrentWidget(plot_widget)
        plot_widget.clear(); plot_widget.addLegend()
        try:
            rois_parsed = []
            for item in roi_input.split(","):
                item = item.strip()
                if ":" in item: start, end = map(int, item.split(":")); rois_parsed.append(slice(start, end))
                else: rois_parsed.append(int(item))
            
            if len(rois_parsed) != processed_data_freq_selected.ndim - 2:
                log_message_func(f"ROI dims error. Data spatial dims: {processed_data_freq_selected.ndim-2}, ROI: {len(rois_parsed)}")
                return
            
            roi_slices = [slice(None), slice(None)] + rois_parsed 
            roi_data_tf_spatial = processed_data_freq_selected[tuple(roi_slices)]
            spatial_avg_axes = tuple(range(2, roi_data_tf_spatial.ndim))
            roi_data_tf = np.mean(roi_data_tf_spatial, axis=spatial_avg_axes) if spatial_avg_axes else roi_data_tf_spatial
            
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']; color_idx = 0
            x_axis_spec = np.arange(roi_data_tf.shape[1])

            for idx_item in indices_to_plot:
                spec_roi, name_roi = None, ""
                if isinstance(idx_item, tuple):
                    start, end = idx_item
                    spec_roi = roi_data_tf[start:end].mean(axis=0) if display_method == "Average" else roi_data_tf[start:end].max(axis=0)
                    name_roi = f"ROI {display_method} {start}:{end}"
                else: spec_roi = roi_data_tf[idx_item]; name_roi = f"ROI Index {idx_item}"
                
                if spec_roi is not None:
                    plot_widget.plot(x_axis_spec, spec_roi, pen=pg.mkPen(colors[color_idx % len(colors)]), name=name_roi)
                    color_idx+=1
            plot_widget.setTitle(f"ROI Spectra - {value_type}"); log_message_func("ROI spectrum plotted.")
        except Exception as e: log_message_func(f"Error ROI plot: {e}")
        return

    image_gl_widget = target_plot_widgets["image"] 
    if "container" in target_plot_widgets: target_plot_widgets["container"].setCurrentWidget(image_gl_widget)
    image_gl_widget.clear()
    
    max_plots = 9; num_plots = min(len(indices_to_plot), max_plots)
    cols = min(3, num_plots); rows = (num_plots + cols - 1) // cols if cols > 0 else 1
    
    temp_images_to_plot = []
    for i, idx_item in enumerate(indices_to_plot[:num_plots]):
        data_after_time_agg = None
        if isinstance(idx_item, tuple):
            start, end = idx_item
            data_after_time_agg = processed_data_freq_selected[start:end,...].mean(axis=0) if display_method == "Average" else processed_data_freq_selected[start:end,...].max(axis=0)
        else: data_after_time_agg = processed_data_freq_selected[idx_item,...]
        image_spatial = data_after_time_agg.max(axis=0) 
        
        if slice_dim_orig != -1: 
            slice_dim_adj = slice_dim_orig - 2 
            if 0 <= slice_dim_adj < image_spatial.ndim:
                slice_num_adj = min(slice_num, image_spatial.shape[slice_dim_adj]-1)
                img_slice = np.take(image_spatial, slice_num_adj, axis=slice_dim_adj)
                temp_images_to_plot.append(np.rot90(img_slice, k=-1))
            else: temp_images_to_plot.append(np.zeros((10,10))); log_message_func("Slice dim error.")
        else: 
            default_slice_axis = image_spatial.ndim -1
            if default_slice_axis >= 0:
                default_slice_num = image_spatial.shape[default_slice_axis] // 2
                img_slice = np.take(image_spatial, default_slice_num, axis=default_slice_axis)
                temp_images_to_plot.append(np.rot90(img_slice, k=-1))
            else: temp_images_to_plot.append(image_spatial.reshape(1,1)) 
    
    if temp_images_to_plot:
        valid_images = [img for img in temp_images_to_plot if img.size > 0]
        max_val = np.max([img.max() for img in valid_images]) if valid_images else 1.0
        min_val = np.min([img.min() for img in valid_images]) if valid_images else 0.0

    for i, img_slice in enumerate(temp_images_to_plot):
        if img_slice.size == 0: continue 
        vb = image_gl_widget.addViewBox(row=i // cols, col=i % cols); vb.setAspectLocked(True)
        img_item = pg.ImageItem(img_slice)
        cmap = pg.colormap.get(colormap_name); img_item.setColorMap(cmap)
        img_item.setLevels([min_val, max_val if max_val > min_val else min_val + 1e-6]) 
        vb.addItem(img_item)
    log_message_func("Image grid display complete.")


def plot_fitted_spectrum(target_plot_widget, container_widget, fitted_data_dict, raw_data_for_spectrum_plot, coord_params, log_message_func=default_log_message):
    if not all(k in fitted_data_dict for k in ['fitted_data', 'separate_peaks']):
        log_message_func("Error: Fitted data is incomplete for spectrum plot.")
        return
    if raw_data_for_spectrum_plot is None:
        log_message_func("Error: Raw data spectrum is missing for plot.")
        return

    try:
        t, x, y, z = map(int, coord_params["coord"].split(","))
    except Exception as e:
        log_message_func(f"Error parsing coordinates '{coord_params['coord']}': {e}")
        return

    try:
        raw_spec = raw_data_for_spectrum_plot[t, :, x, y, z]
        fitted_spec = fitted_data_dict['fitted_data'][t, :, x, y, z]
        components_spec = fitted_data_dict['separate_peaks'][t, :, x, y, z, :] # (F, P)
        num_peaks = components_spec.shape[1]
    except IndexError as e:
        log_message_func(f"Error indexing data for voxel ({t},{x},{y},{z}): {e}. Check data shapes.")
        return

    if container_widget:
        container_widget.setCurrentWidget(target_plot_widget)
    
    target_plot_widget.clear()
    target_plot_widget.addLegend()
    
    target_plot_widget.plot(raw_spec, pen=pg.mkPen("gray"), name="Raw spectrum")
    target_plot_widget.plot(fitted_spec, pen=pg.mkPen("y"), name="Fitted spectrum")

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']
    for i in range(num_peaks):
        target_plot_widget.plot(components_spec[:, i], pen=pg.mkPen(colors[i % len(colors)]), name=f"Peak {i+1}")

    target_plot_widget.setLabel("bottom", "Frequency Index")
    target_plot_widget.setLabel("left", "Signal Intensity")
    target_plot_widget.setTitle(f"Peak Fitting Results at Voxel ({t},{x},{y},{z})")
    log_message_func(f"Displayed fitted spectrum at voxel ({t},{x},{y},{z}).")


def plot_peak_fitting_image_overlay(image_wrapper_widget, fitted_data_dict, structural_image_data, 
                                    display_params, log_message_func, q_file_dialog_parent):
    log_message_func("Attempting to display peak fitting image overlay.")

    layout = image_wrapper_widget.layout()
    if layout is None: layout = QVBoxLayout(image_wrapper_widget)
    while layout.count():
        item = layout.takeAt(0); widget = item.widget(); 
        if widget: widget.deleteLater()

    current_structural_image = structural_image_data
    if current_structural_image is None and q_file_dialog_parent:
        file_path_struct, _ = QFileDialog.getOpenFileName(q_file_dialog_parent, "Select Structural Image", "", "Images (*.npy *.nii *.nii.gz)")
        if file_path_struct:
            try:
                current_structural_image = np.load(file_path_struct) if file_path_struct.endswith(".npy") else nib.load(file_path_struct).get_fdata()
                if hasattr(q_file_dialog_parent, 'structural_image'): q_file_dialog_parent.structural_image = current_structural_image
                log_message_func(f"Structural image loaded: {file_path_struct}")
            except Exception as e:
                log_message_func(f"Failed to load structural image: {e}")
                return
        else: log_message_func("No structural image selected for overlay."); return
    elif current_structural_image is None: log_message_func("Structural image data not provided."); return

    components = fitted_data_dict.get('separate_peaks') 
    if components is None: log_message_func("Error: 'separate_peaks' not found."); return

    try:
        peak_idx = int(display_params["peak_index"])
        slice_dim_spatial = int(display_params["slice_dim"]) 
        time_idx = int(display_params["time_idx"])
        peak_map_spatial = components[time_idx, :, :, :, :, peak_idx].max(axis=0) 
        
        if current_structural_image.ndim > 3: 
            current_structural_image = current_structural_image[..., current_structural_image.shape[-1]//2].squeeze() 
        if current_structural_image.ndim < 3: log_message_func("Structural image < 3D."); return

        view_box = pg.ViewBox(); view_box.setAspectLocked(True)
        struct_img_item = pg.ImageItem(); dmi_img_item = pg.ImageItem()
        view_box.addItem(struct_img_item); view_box.addItem(dmi_img_item)
        
        plot_widget_overlay = pg.GraphicsLayoutWidget(); plot_widget_overlay.addItem(view_box)
        layout.addWidget(plot_widget_overlay)

        struct_max_slice = current_structural_image.shape[slice_dim_spatial] - 1
        dmi_max_slice = peak_map_spatial.shape[slice_dim_spatial] - 1
        struct_slider = QSlider(Qt.Orientation.Horizontal)
        if struct_max_slice >=0: struct_slider.setRange(0, struct_max_slice)
        struct_slider.setValue(min(struct_max_slice // 2, struct_max_slice) if struct_max_slice >=0 else 0)
        dmi_slider = QSlider(Qt.Orientation.Horizontal)
        if dmi_max_slice >=0: dmi_slider.setRange(0, dmi_max_slice)
        dmi_slider.setValue(min(dmi_max_slice // 2, dmi_max_slice) if dmi_max_slice >=0 else 0)
        opacity_slider = QSlider(Qt.Orientation.Horizontal); opacity_slider.setRange(0,100); opacity_slider.setValue(60)
        colormap_combo = QComboBox(); colormap_combo.addItems(["viridis", "plasma", "magma", "cividis", "gray"])
        form_controls = QFormLayout()
        form_controls.addRow("Structural Slice:", struct_slider)
        form_controls.addRow("DMI Peak Map Slice:", dmi_slider)
        form_controls.addRow("DMI Opacity:", opacity_slider)
        form_controls.addRow("DMI Colormap:", colormap_combo)
        layout.addLayout(form_controls)
        
        image_wrapper_widget._dynamic_controls = [struct_slider, dmi_slider, opacity_slider, colormap_combo, plot_widget_overlay]

        def update_display_overlay():
            s_idx, d_idx = struct_slider.value(), dmi_slider.value()
            s_slice = np.take(current_structural_image, s_idx, axis=slice_dim_spatial)
            d_slice_raw = np.take(peak_map_spatial, d_idx, axis=slice_dim_spatial)
            s_slice_disp = np.rot90(s_slice, k=-1)
            d_slice_resized = cv2.resize(d_slice_raw, (s_slice_disp.shape[1], s_slice_disp.shape[0]), interpolation=cv2.INTER_NEAREST)
            d_slice_disp = np.rot90(d_slice_resized, k=-1)
            struct_img_item.setImage(s_slice_disp, autoLevels=True)
            dmi_img_item.setImage(d_slice_disp, autoLevels=True)
            cmap_obj = pg.colormap.get(colormap_combo.currentText())
            if cmap_obj: dmi_img_item.setLookupTable(cmap_obj.getLookupTable())
            dmi_max_val = np.max(d_slice_disp)
            dmi_img_item.setLevels([0, dmi_max_val if dmi_max_val > 0 else 1.0]) 
            dmi_img_item.setOpacity(opacity_slider.value() / 100.0)

        struct_slider.valueChanged.connect(update_display_overlay)
        dmi_slider.valueChanged.connect(update_display_overlay)
        opacity_slider.valueChanged.connect(update_display_overlay)
        colormap_combo.currentTextChanged.connect(update_display_overlay)
        
        update_display_overlay()
        log_message_func("Peak fitting image overlay UI created and displayed.")
    except Exception as e:
        log_message_func(f"Error in plot_peak_fitting_image_overlay: {str(e)}")


def plot_structural_dmi_overlay(view_box, struct_img_item, dmi_img_item,
                                structural_slice_data, dmi_slice_data, 
                                dmi_opacity, dmi_colormap_name, log_message_func=default_log_message):
    try:
        struct_img_item.setImage(np.rot90(structural_slice_data, k=-1), autoLevels=True)
        dmi_slice_rotated = np.rot90(dmi_slice_data, k=-1)
        dmi_img_item.setImage(dmi_slice_rotated, autoLevels=True)
        cmap = pg.colormap.get(dmi_colormap_name)
        if cmap: dmi_img_item.setLookupTable(cmap.getLookupTable())
        dmi_max = np.max(dmi_slice_rotated)
        dmi_img_item.setLevels([0, dmi_max if dmi_max > 0 else 1.0])
        dmi_img_item.setOpacity(dmi_opacity)
    except Exception as e:
        log_message_func(f"Error updating overlay plot: {e}")

def plot_comparison_spectra(plot_widget, original_spectrum_data, corrected_spectra_map, 
                            display_params, log_message_func=default_log_message):
    plot_widget.clear()
    plot_widget.addLegend()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] 
    color_idx = 0

    display_type = display_params['display_type']
    show_abs = display_params['show_abs']
    show_negative = display_params['show_negative']

    def _process_for_display(spec_data):
        if spec_data is None: return None
        processed_spec = None
        if display_type == "Magnitude": processed_spec = np.abs(spec_data)
        elif display_type == "Real": processed_spec = np.real(spec_data)
        elif display_type == "Imaginary": processed_spec = np.imag(spec_data)
        
        if display_type in ["Real", "Imaginary"]:
            if show_abs: processed_spec = np.abs(processed_spec)
            if show_negative: processed_spec = -processed_spec 
        return processed_spec

    plot_data_original = _process_for_display(original_spectrum_data)
    if plot_data_original is not None:
        plot_widget.plot(plot_data_original, pen=pg.mkPen("gray", width=2), name="Original")

    for method_name, corrected_spec_data in corrected_spectra_map.items():
        plot_data_corrected = _process_for_display(corrected_spec_data)
        if plot_data_corrected is not None:
            pen_color = pg.mkPen(colors[color_idx % len(colors)], width=1)
            plot_widget.plot(plot_data_corrected, pen=pen_color, name=method_name)
            color_idx += 1
            
    plot_widget.setLabel("bottom", "Frequency Index")
    plot_widget.setLabel("left", "Signal Intensity")
    plot_widget.setTitle(f"Comparison - {display_type}")
    log_message_func("Comparison spectra plotted.")


def plot_denoise_comparison_spectra(plot_widget, original_spectrum_data, denoised_spectra_map,
                                    display_params, log_message_func=default_log_message):
    plot_widget.clear()
    plot_widget.addLegend()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'purple', 'orange', 'pink'] 
    color_idx = 0

    show_original = display_params['show_original']
    value_type = display_params['value_type'] 
    show_abs = display_params['show_abs']
    show_negative = display_params['show_negative']

    def _process_for_display(spec_data, is_original_data=False): # Renamed flag
        if spec_data is None: return None
        
        processed_spec = spec_data
        if is_original_data: # Original spectrum is complex, others are already processed
            if value_type == "Magnitude": processed_spec = np.abs(spec_data)
            elif value_type == "Real": processed_spec = np.real(spec_data)
        
        # Apply abs/negative for Real type, regardless of original or denoised
        if value_type == "Real": 
            if show_abs: processed_spec = np.abs(processed_spec)
            if show_negative: processed_spec = -processed_spec
        return processed_spec

    if show_original:
        plot_data_original = _process_for_display(original_spectrum_data, is_original_data=True)
        if plot_data_original is not None:
            plot_widget.plot(plot_data_original, pen=pg.mkPen("gray", width=2), name="Original")

    for method_label, denoised_spec_data in denoised_spectra_map.items():
        # Denoised data is already magnitude or real based on 'value_type' used in apply_single_spectrum_denoising
        # So, is_original_data=False. Processing here is just for abs/negative display options for 'Real' type.
        plot_data_denoised = _process_for_display(denoised_spec_data, is_original_data=False) 
        if plot_data_denoised is not None:
            pen_color = pg.mkPen(colors[color_idx % len(colors)], width=1)
            plot_widget.plot(plot_data_denoised, pen=pen_color, name=method_label)
            color_idx += 1
            
    plot_widget.setLabel("bottom", "Frequency Index")
    plot_widget.setLabel("left", "Signal Intensity")
    plot_widget.setTitle(f"Denoising Comparison - {value_type}")
    log_message_func("Denoise comparison spectra plotted.")

def plot_temporal_curve(plot_widget, x_axis_data, y_axis_data, title="Temporal Curve", pen='y', symbol='o', name="Temporal Signal", log_message_func=default_log_message):
    """
    Plots a single temporal curve.
    Args:
        plot_widget (pg.PlotWidget): The widget to plot on.
        x_axis_data (np.ndarray): X-axis data (e.g., time points).
        y_axis_data (np.ndarray): Y-axis data (e.g., signal amplitude).
        title (str): Title for the plot.
        pen (str or dict): Pen for the plot line.
        symbol (str): Symbol for plot points.
        name (str): Name for the legend.
        log_message_func (callable): Logging function.
    """
    try:
        plot_widget.clear() # Clear previous plots on this specific widget
        plot_widget.plot(x_axis_data, y_axis_data, pen=pen, symbol=symbol, name=name)
        plot_widget.setTitle(title)
        plot_widget.setLabel("bottom", "Time Point / Index")
        plot_widget.setLabel("left", "Signal Value")
        log_message_func(f"Plotted '{name}' on '{title}'.")
    except Exception as e:
        log_message_func(f"Error in plot_temporal_curve: {e}")
        if hasattr(plot_widget, 'setTitle'): plot_widget.setTitle(f"Error: {e}")


def plot_curve_fitting_results(plot_widget, x_fit_data, y_fit_data, model_name, log_message_func=default_log_message):
    """
    Adds a fitted curve to an existing plot.
    Args:
        plot_widget (pg.PlotWidget): The widget containing the original plot.
        x_fit_data (np.ndarray): X-axis data for the fitted curve.
        y_fit_data (np.ndarray): Y-axis data for the fitted curve.
        model_name (str): Name of the fitted model for the legend.
        log_message_func (callable): Logging function.
    """
    try:
        plot_widget.plot(x_fit_data, y_fit_data, pen='r', name=f"Fitted {model_name}")
        # Assuming legend is already added by the initial plot function (e.g., plot_temporal_curve)
        log_message_func(f"Added fitted curve for '{model_name}' to the plot.")
    except Exception as e:
        log_message_func(f"Error in plot_curve_fitting_results: {e}")

[end of visualization/plotters.py]
