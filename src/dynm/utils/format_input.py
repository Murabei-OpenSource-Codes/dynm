"""Auxiliary functions for input transformation."""
import numpy as np
import pandas as pd


def set_X_dict(mod, nobs: int, X: dict = {}):
    """Fill missing regression and transfer-function input arrays.

    Args:
        mod:
            Model instance used to infer transfer-function shape.
        nobs (int):
            Number of time observations.
        X (dict):
            Optional covariate dict. Missing keys are filled.

    Returns:
        dict:
            Copy of ``X`` with both expected keys present.
    """
    copy_X = X.copy()

    # Organize transfer function values
    if X.get('regression') is None:
        x = np.array([None] * (nobs + 1)).reshape(-1, 1)
        copy_X['regression'] = x

    if X.get('transfer_function') is None:
        ntfm = mod.dnm.transfer_function_model.ntfm
        ngamma = mod.dnm.transfer_function_model.gamma_order
        z = np.array([None] * nobs * ntfm * ngamma).reshape(nobs, ntfm, ngamma)
        copy_X['transfer_function'] = z

    return copy_X


def compute_lagged_values(X: np.array, lags: int):
    """Build a lagged covariate array for transfer-function inputs.

    Args:
        X (np.ndarray):
            Observed covariates with shape (nobs, ntfm).
        lags (int):
            Number of lags to include, including lag 0.

    Returns:
        np.ndarray:
            Array with shape (nobs, ntfm, lags).
    """
    nobs = X.shape[0]
    ntfm = X.shape[1]

    np_X = np.ones([nobs, ntfm, lags])
    for i in range(ntfm):
        for j in range(lags):
            shift_x = pd.Series(X[:, i]).shift(j).fillna(0).to_numpy()
            np_X[:, i, j] = shift_x
    return np_X
