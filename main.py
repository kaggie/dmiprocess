import sys
from PyQt6.QtWidgets import QApplication

# Import MainApp from its new location in the visualization package
from visualization import MainApp

# All other specific imports (numpy, torch, other submodules like io, preprocessing, etc.)
# are now handled within their respective modules (gui.py, plotters.py, algorithms.py, etc.).
# The main.py script is now only responsible for launching the application.

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # The MainApp class in visualization.gui now handles all initial setup of the UI
    # and its internal components.
    window = MainApp()
    window.show()
    
    sys.exit(app.exec())
