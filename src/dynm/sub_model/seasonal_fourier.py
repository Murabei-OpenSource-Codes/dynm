"""Dynamic Linear Model with transfer function."""
import numpy as np
from dynm.utils.algebra import _build_W_complete


class SeasonalFourier():
    """Class for defining seasonal Fourier model in state space form."""

    def __init__(self,
                 m0: np.ndarray,
                 C0: np.ndarray,
                 seas_period: int = None,
                 seas_harm_components: list = None,
                 discount: float = .998,
                 W: np.ndarray = None):
        """Define a Fourier seasonal block.

        Args:
            m0 (np.ndarray):
                Prior mean of the seasonal state.
            C0 (np.ndarray):
                Prior covariance of the seasonal state.
            seas_period (int):
                Seasonal period. Defaults to None.
            seas_harm_components (list):
                Harmonic indices included in the seasonal block.
                Defaults to None.
            discount (float):
                Discount factor used when ``W`` is unknown.
                Defaults to 0.998.
            W (np.ndarray):
                Optional known evolution covariance. Defaults to None.
        """
        self.nseas = 2 * len(seas_harm_components)
        self.seas_period = seas_period
        self.seas_harm_components = seas_harm_components
        self.discount = discount

        self.m = m0.reshape(-1, 1)
        self.C = C0

        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        self.F = self._build_F()
        self.G = self._build_G()

    def _build_F(self):
        """Build the seasonal regression vector.

        Returns:
            np.ndarray:
                Column vector with ones on cosine positions.
        """
        seas_harm_components = self.seas_harm_components

        p = len(seas_harm_components)
        n = 2 * p

        F = np.zeros([n, 1])
        F[0:n:2] = 1

        return F.reshape(-1, 1)

    def _build_G(self):
        """Build the Fourier evolution matrix.

        Returns:
            np.ndarray:
                Block-diagonal rotation matrices per harmonic.
        """
        seas_period = self.seas_period
        seas_harm_components = self.seas_harm_components

        p = len(seas_harm_components)
        n = 2 * p
        G = np.zeros([n, n])

        for j in range(p):
            c = np.cos(2 * np.pi * seas_harm_components[j] / seas_period)
            s = np.sin(2 * np.pi * seas_harm_components[j] / seas_period)
            idx = 2 * j
            G[idx:(idx + 2), idx:(idx + 2)] = np.array([[c, s], [-s, c]])

        return G

    def _update_F(self, x: np.array = None):
        """Return the current seasonal regression vector.

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
            W = _build_W_complete(mod=self, P=P)
        else:
            W = self.W
        return W
