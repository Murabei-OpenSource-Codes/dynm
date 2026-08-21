"""Regression in State Space form."""
import numpy as np
from dynm.utils.algebra import _build_W_complete


class Regression():
    """Class for defining regression model in state space form."""

    def __init__(self,
                 m0: np.ndarray,
                 C0: np.ndarray,
                 nregn: int,
                 discount: float = .998,
                 W: np.ndarray = None):
        """Define a regression block.

        Args:
            m0 (np.ndarray):
                Prior mean of the regression coefficients.
            C0 (np.ndarray):
                Prior covariance of the regression coefficients.
            nregn (int):
                Number of regressors.
            discount (float):
                Discount factor used when ``W`` is unknown.
                Defaults to 0.998.
            W (np.ndarray):
                Optional known evolution covariance. Defaults to None.
        """
        self.nregn = nregn
        self.discount = discount

        self.m = m0.reshape(-1, 1)
        self.C = C0

        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        self.F = self._build_F(x=0)
        self.G = self._build_G()

    def _build_F(self, x: np.array):
        """Build the regression vector from covariates.

        Args:
            x (np.ndarray):
                Covariate values, one per regressor.

        Returns:
            np.ndarray:
                Column vector of length ``nregn``.
        """
        nregn = self.nregn
        F = np.ones(nregn) * x
        return F.reshape(-1, 1)

    def _build_G(self):
        """Build the regression evolution matrix.

        Returns:
            np.ndarray:
                Identity of size ``nregn``.
        """
        nregn = self.nregn
        G = np.identity(nregn)
        return G

    def _update_F(self, x: np.array = None):
        """Update the regression vector in place.

        Args:
            x (np.ndarray):
                Covariate values for the current time.

        Returns:
            np.ndarray:
                Updated ``F``.
        """
        F = self.F
        F[:, 0] = np.ravel(x)
        return F

    def _build_P(self):
        """Build the evolved prior covariance ``G C G.T``.

        Returns:
            np.ndarray:
                Prior covariance of the evolved state.
        """
        return self.G @ self.C @ self.G.T

    def _build_W(self, P: np.array):
        """Build the evolution covariance.

        Args:
            P (np.ndarray):
                Evolved prior covariance ``G C G.T``.

        Returns:
            np.ndarray:
                Known ``W`` or a discounted estimate from ``P``.
        """
        if self.estimate_W:
            W = _build_W_complete(mod=self, P=P)
        else:
            W = self.W
        return W
