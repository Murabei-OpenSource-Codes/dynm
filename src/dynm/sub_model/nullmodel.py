"""Class for null model."""
import numpy as np


class NullModel():
    """Empty structural block used when a component is absent."""

    def __init__(self):
        """Create a null model with empty state arrays."""
        self.order = 0
        self.lambda_order = 0
        self.gamma_order = 0
        self.ntrend = 0
        self.nregn = 0
        self.nseas = 0
        self.ntfm = 0
        self.m = np.array([np.array([])]).reshape(-1, 1)
        self.C = np.empty([0, 0])

        self.F = self._build_F()
        self.G = self._build_G()

    def _update_F(self, x: np.array = None):
        """Return an empty regression vector.

        Args:
            x (np.ndarray):
                Unused covariate. Kept for interface consistency.

        Returns:
            np.ndarray:
                Empty column vector.
        """
        return np.empty([0, 0]).reshape(-1, 1)

    def _build_F(self):
        """Build an empty regression vector.

        Returns:
            np.ndarray:
                Empty column vector.
        """
        return np.empty([0, 0]).reshape(-1, 1)

    def _build_G(self, x: float = None):
        """Build an empty evolution matrix.

        Args:
            x (float):
                Unused covariate. Kept for interface consistency.

        Returns:
            np.ndarray:
                Empty square matrix.
        """
        return np.empty([0, 0])

    def _build_h(self):
        """Build an empty nonlinear offset.

        Returns:
            np.ndarray:
                Empty column vector.
        """
        return np.empty([0, 0]).reshape(-1, 1)

    def _build_P(self, G: np.array = None):
        """Build an empty prior covariance.

        Args:
            G (np.ndarray):
                Unused evolution matrix. Kept for interface consistency.

        Returns:
            np.ndarray:
                Empty square matrix.
        """
        return np.empty([0, 0])

    def _build_W(self, P: np.array):
        """Build an empty evolution covariance.

        Args:
            P (np.ndarray):
                Unused prior covariance. Kept for interface consistency.

        Returns:
            np.ndarray:
                Empty square matrix.
        """
        return np.empty([0, 0])
