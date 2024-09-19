import numpy as np
from numpy.linalg import multi_dot, norm
from numpy.random import MT19937, Generator
from typing import Optional

def normalizeUV(U: np.ndarray, V: np.ndarray):
    """
    Normalize U and V by the Euclidean length of the columns of U.

    U is normalized by dividing every entry by the Euclidean length of its corresponding column. 
    V is normalized by multiplying every entry by the Euclidean length of the corresponding column in the original U.
    This ensures that the product U @ V.T remains the same.

    Parameters
    ----------
    U : np.ndarray
        An m-by-k matrix.
    V : np.ndarray
        An n-by-k matrix.

    Returns
    -------
    U : np.ndarray
        Normalized U.
    V : np.ndarray
        Normalized V.
    """

    col_lengths = np.maximum(1e-15, np.sqrt(np.sum(np.square(U), axis=0)))

    return U / col_lengths, V * col_lengths

def calcObjective(X: np.ndarray, U: np.ndarray, V: np.ndarray, L: np.ndarray) -> float:
    """
    Give the value of the objective function based on Euclidean distance.

    Parameters
    ----------
    X : np.ndarray
        The m-by-n data matrix.
    U : np.ndarray
        The m-by-k left factor of X.
    V : np.ndarray
        The n-by-k transpose of the right factor of X.
    L : np.ndarray
        The n-by-n graph Laplacian, already multiplied with the regularization parameter l.

    Returns
    -------
    objective_val : float
        The value of the objective function.
    """

    return norm(X - U @ V.T) + np.trace(multi_dot([V.T, L, V]))
    # return np.sum(np.square(X - U @ V.T)) + np.trace(multi_dot([V.T, L, V]))

def gnmf(
        X: np.ndarray, 
        k: int, 
        W: np.ndarray, 
        l: float, 
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
        U: Optional[np.ndarray] = None,
        V: Optional[np.ndarray] = None):
    """
    Calculate graph-regularized non-negative matrix factorisation such that X approximately equals U @ V.T (Cai, 2011).

    Parameters
    ----------
    X : np.ndarray
        The m-by-n data matrix, where m represents the number of features and n gives the number of data points.
    k : int
        The number of latent dimensions.
    W : np.ndarray
        The n-by-n weight matrix of the affinity graph.
    l : float
        Regularization parameter that controls the smoothness of the factorization. Has to be equal to or greater than 0. 
    max_iter : int
        Maximum number of iterations. Must be specified if tolerance is not specified.
    tolerance : float
        Improvement threshold of objective function, below which optimization is stopped. Must be specified if max_iter is not specified.
    U : np.ndarray
        Optional. m-by-k matrix to be used as a starting estimate for U.
    V : np.ndarray
        Optional. n-by-k matrix to be used as a starting estimate for V.

    Returns
    -------
    U : np.ndarray
        The m-by-k left factor of X.
    V : np.ndarray
        The n-by-k transpose of the right factor of X.
    obj_history : list
        The history of the objective function values.

    Raises
    ------
    ValueError
        Raises an error if l < 0 or neither max_iter nor tolerance is specified.
    """

    if l < 0:
        raise ValueError("l has to be equal to or larger than 0.")
    
    if max_iter is None:
        if tolerance is None:
            raise ValueError("Either max_iter or tolerance needs to be specified.")
        
        max_iter = np.inf

    if tolerance is None:
        tolerance = -np.inf

    # rng = np.random.default_rng()
    rng = Generator(MT19937(5489))

    # Number of features, number of data points
    m, n = X.shape

    # Calculate graph Laplacian
    W = l * W
    D = np.diag(W.sum(axis=0)) # TODO: Use sparse matrix
    L = D - W

    # Initialize U and V
    if U is None:
        U = np.abs(rng.random(size=(m, k)))

    if V is None:
        V = np.abs(rng.random(size=(n, k)))

    U, V = normalizeUV(U, V)
    meanFit = 10 * calcObjective(X, U, V, L)
    meanFitRatio = 0.1
    maxErr = 1

    # Value of the objective function
    obj_val = np.inf
    diff = np.inf

    # To record objective function values
    obj_history = []

    i = 0

    while i < max_iter and diff > tolerance:
        # Update V. Multiplication by l has already been done above
        denom = multi_dot([V, U.T, U]) + D @ V
        V = V * ((X.T @ U + W @ V) / np.maximum(1e-10, denom))

        # Update U
        denom = multi_dot([U, V.T, V])
        U = U * ((X @ V) / np.maximum(1e-10, denom))

        new_obj_val = calcObjective(X, U, V, L)

        diff = obj_val - new_obj_val
        obj_val = new_obj_val

        obj_history.append(obj_val)

        print(f"Iteration {i} - Objective value: {obj_val:10.2f}", end='\r')

        # Funky interpolation
        meanFit = meanFitRatio * meanFit + (1 - meanFitRatio) * new_obj_val
        maxErr = (meanFit - new_obj_val) / meanFit

        # print(f"maxErr: {maxErr:10.5f}", end='\r')

        i += 1

    U, V = normalizeUV(U, V)

    return U, V, obj_history