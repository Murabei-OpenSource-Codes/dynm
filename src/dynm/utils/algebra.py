"""Auxiliary methods for kalman filter update."""
import numpy as np


def _build_W_diagonal(mod, P: np.array):
    """Build a diagonal evolution covariance from a discount.

    Args:
        mod:
            Submodel with a scalar or diagonal ``discount``.
        P (np.ndarray):
            Prior covariance of the evolved state, ``G C G.T``.

    Returns:
        np.ndarray:
            Evolution covariance ``W``.
    """
    p = P.shape[0]
    discount_matrix = np.ones([p, p])
    np.fill_diagonal(discount_matrix, 1 / mod.discount)

    W = P * discount_matrix - P

    return W


def _build_W_complete(mod, P: np.array):
    """Build a full evolution covariance from a discount.

    Args:
        mod:
            Submodel with a scalar ``discount``.
        P (np.ndarray):
            Prior covariance of the evolved state, ``G C G.T``.

    Returns:
        np.ndarray:
            Evolution covariance ``W``.
    """
    p = P.shape[0]
    discount_matrix = np.ones([p, p]) / mod.discount

    W = P * discount_matrix - P

    return W


def _calc_predictive_mean_and_var(F: np.array, a: np.array,
                                  R: np.array, s: float):
    """Compute one-step predictive mean and variance.

    Args:
        F (np.ndarray):
            Regression vector.
        a (np.ndarray):
            Prior mean of the state.
        R (np.ndarray):
            Prior covariance of the state.
        s (float):
            Observational variance.

    Returns:
        tuple:
            Flattened predictive mean ``f`` and variance ``q``.
    """
    f = F.T @ a
    q = F.T @ R @ F + s
    return np.ravel(f), np.ravel(q)


def _build_Gnonlinear(m: np.array, order: int):
    """Build the nonlinear evolution matrix for AR or TF blocks.

    Args:
        m (np.ndarray):
            State mean with response then decay blocks.
        order (int):
            Autoregressive or transfer-function order.

    Returns:
        np.ndarray:
            Evolution matrix of shape ``(2 * order, 2 * order)``.

    Raises:
        ValueError:
            If the decay and response blocks have different lengths.
    """
    response_block_index = np.arange(0, order)
    decay_block_index = np.arange(order, 2 * order)

    diag_order = np.identity(order)

    response_block = m[response_block_index, 0]
    decay_block = m[decay_block_index, 0]

    if len(decay_block) != len(response_block):
        raise ValueError(
            "Decay and response blocks differ in length")

    diag_decay_block = np.identity(order)[:order - 1, :]
    diag_response_block = np.diag(response_block)[1:, :] * 0

    nonlinear_block = np.block([[decay_block, response_block],
                               [diag_decay_block, diag_response_block],
                               [0 * diag_order, diag_order]])
    nonlinear_block = nonlinear_block.reshape(
        2 * order, 2 * order)

    return nonlinear_block
