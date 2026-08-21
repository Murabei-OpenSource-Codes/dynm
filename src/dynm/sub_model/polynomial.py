"""Polynomial model in State Space form."""
import numpy as np
from dynm.utils.algebra import _build_W_diagonal


class Polynomial():
    """Class for defining polynomial model in state space form."""

    def __init__(self,
                 m0: np.ndarray,
                 C0: np.ndarray,
                 ntrend: int,
                 discount: float = .98,
                 W: np.ndarray = None):
        """Define a polynomial trend block.

        Args:
            m0 (np.ndarray):
                Prior mean of the trend state.
            C0 (np.ndarray):
                Prior covariance of the trend state.
            ntrend (int):
                Number of trend components (level, optionally slope).
            discount (float):
                Discount factor used when ``W`` is unknown.
                Defaults to 0.98.
            W (np.ndarray):
                Optional known evolution covariance. Defaults to None.
        """
        self.ntrend = ntrend
        self.m = m0.reshape(-1, 1)  # Validar entrada de dimensões
        self.C = C0

        self.discount = discount

        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        self.F = self._build_F()
        self.G = self._build_G()

    def _build_F(self):
        """Build the polynomial regression vector.

        Returns:
            np.ndarray:
                Column vector of length ``ntrend``.
        """
        ntrend = self.ntrend
        F = np.ones(ntrend)

        if ntrend == 2:
            F[1] = 0

        return F.reshape(-1, 1)

    def _build_G(self):
        """Build the polynomial evolution matrix.

        Returns:
            np.ndarray:
                Identity, or local-linear-trend companion if order 2.
        """
        ntrend = self.ntrend
        G = np.identity(ntrend)

        if ntrend == 2:
            G[0, 1] = 1

        return G

    def _update_F(self, x: np.array = None):
        """Return the current regression vector.

        Args:
            x (np.ndarray):
                Unused covariate. Kept for interface consistency.

        Returns:
            np.ndarray:
                Current ``F``.
        """
        F = self.F
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
            W = _build_W_diagonal(mod=self, P=P)
        else:
            W = self.W
        return W
