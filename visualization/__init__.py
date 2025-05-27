from .gui import MainApp
# Exporting other UI components (dialogs, specific windows) is optional
# and depends on whether they need to be directly accessed from outside this package.
# For now, only MainApp is essential for running the application.

# Exporting plotter functions is also optional. If they are only used by gui.py internally,
# then no need to export them here. If they are intended as a utility library for plotting,
# then they could be exported. For now, keeping it minimal.
# from .plotters import plot_spectrum_detailed, show_image_detailed # etc.
