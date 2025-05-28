import numpy as np

def create_polynomial_basis_vectors(frequency_axis: np.ndarray, degree: int) -> np.ndarray:
    """
    Creates a matrix of polynomial basis vectors.

    Each column in the matrix corresponds to a polynomial term
    (x^0, x^1, x^2, ..., x^degree) evaluated at each point on the
    frequency_axis.

    Args:
        frequency_axis (np.ndarray): The frequency axis (e.g., in ppm or Hz).
                                     The behavior of polynomials can be sensitive
                                     to the range of this axis. Consider normalizing
                                     it (e.g., to [-1, 1] or [0, 1]) before passing
                                     it to this function if a wide range is used.
        degree (int): The maximum degree of the polynomial. (e.g., if degree=2,
                      basis vectors for x^0, x^1, x^2 will be generated).

    Returns:
        np.ndarray: A 2D NumPy array (len(frequency_axis), degree + 1) where
                    each column is a polynomial basis vector.

    Raises:
        ValueError: If degree is negative.
    """
    if degree < 0:
        raise ValueError("Polynomial degree cannot be negative.")
    if not isinstance(frequency_axis, np.ndarray) or frequency_axis.ndim != 1:
        raise TypeError("frequency_axis must be a 1D NumPy array.")

    num_points = len(frequency_axis)
    # Using np.vander for a more direct and potentially stable way to generate polynomial terms
    # np.vander(x, N) generates columns [x^(N-1), x^(N-2), ..., x^0]
    # We want [x^0, x^1, ..., x^degree], so we'll use N = degree + 1 and then flip columns.
    
    # Using np.polynomial.polynomial.polyvander is more numerically stable
    # for higher degrees than direct computation of powers.
    # It generates columns [x^0, x^1, ..., x^degree] directly.
    # Note: The caller should consider normalizing the frequency_axis (e.g., to [-1, 1]) 
    # if its scale is very large or if high-degree polynomials are used, 
    # to improve numerical stability of the subsequent least-squares fitting.
    basis_matrix = np.polynomial.polynomial.polyvander(frequency_axis, degree)
            
    return basis_matrix

def generate_polynomial_baseline(frequency_axis: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    Generates a polynomial baseline given coefficients.

    The degree of the polynomial is inferred from the length of coefficients - 1.
    
    Args:
        frequency_axis (np.ndarray): The frequency axis.
        coefficients (np.ndarray): A 1D array of polynomial coefficients
                                   [c0, c1, c2, ..., cn], where c0 is for x^0,
                                   c1 for x^1, and so on.
    
    Returns:
        np.ndarray: The generated polynomial baseline signal.
    """
    if not isinstance(coefficients, np.ndarray) or coefficients.ndim != 1:
        raise TypeError("Coefficients must be a 1D NumPy array.")
    if len(coefficients) == 0:
        return np.zeros_like(frequency_axis)
        
    degree = len(coefficients) - 1
    basis_vectors = create_polynomial_basis_vectors(frequency_axis, degree)
    
    # Ensure coefficients is a column vector for matrix multiplication if needed,
    # or simply use dot product correctly.
    baseline = np.dot(basis_vectors, coefficients)
    return baseline

```
