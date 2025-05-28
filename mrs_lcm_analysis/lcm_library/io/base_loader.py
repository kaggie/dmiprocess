from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mrs_lcm_analysis.lcm_library.data_loading import MRSData # For type hinting

class BaseMRSLoader(ABC):
    """
    Abstract base class for MRS data loaders.

    Concrete implementations of this class are responsible for loading
    MRS data from specific file formats into an MRSData object.
    """

    def __init__(self):
        """Initializes the loader."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> 'MRSData':
        """
        Loads MRS data from the given filepath.

        Args:
            filepath (str): The path to the MRS data file.

        Returns:
            MRSData: An MRSData object populated with the loaded data
                     and relevant metadata.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
            FileNotFoundError: If the filepath does not exist.
            IOError: If there is an issue reading the file.
            ValueError: If the file format is incorrect or data is corrupted.
        """
        raise NotImplementedError

    def can_load(self, filepath: str) -> bool:
        """
        Checks if this loader can likely load the given file.

        This is typically a quick check based on file extension or magic numbers.
        It is not a guarantee of successful loading, as the file might still be
        corrupt or not adhere to the expected internal format.

        Args:
            filepath (str): The path to the MRS data file.

        Returns:
            bool: True if the loader believes it can handle the file, False otherwise.
        """
        return False # Default implementation, to be overridden by concrete loaders
