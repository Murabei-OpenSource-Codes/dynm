"""Utils functions."""
import numpy as np
import copy
from dynm.utils.algebra import _build_Gnonlinear, _build_W_diagonal


class AutoRegressive():
    """Autoregressive block in state-space form."""

    def __init__(self,
                 m0: np.ndarray,
                 C0: np.ndarray,
                 order: int,
                 discount: float = None,
                 W: np.ndarray = None,
                 V: float = None):
        """Define an autoregressive block.

        Args:
            m0 (np.ndarray):
                Prior mean of the AR state.
            C0 (np.ndarray):
                Prior covariance of the AR state.
            order (int):
                Autoregressive order.
            discount (float):
                Discount factor used when ``W`` is unknown.
                Defaults to None.
            W (np.ndarray):
                Optional known evolution covariance. Defaults to None.
            V (float):
                Optional known observational variance. Defaults to None.
        """
        self.order = order
        self.m = m0.reshape(-1, 1)
        self.C = C0

        if V is None:
            self.n = 1
            self.d = 1
            self.s = 1
            self.estimate_V = True
        else:
            self.s = V
            self.estimate_V = False

        self.discount = discount
        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        self.F = self._build_F()
        self.G = self._build_G()

        # Get index for blocks
        block_idx = np.cumsum([order, order])
        self.index_dict = {
            'response': np.arange(0, block_idx[0]),
            'decay': np.arange(block_idx[0], block_idx[1])
        }

    def _build_F(self, x: np.array = None):
        """Build the AR regression vector.

        Args:
            x (np.ndarray):
                Unused covariate. Kept for interface consistency.

        Returns:
            np.ndarray:
                Column vector with a leading one.
        """
        F = np.zeros(2 * self.order)
        F[0] = 1
        return F.reshape(-1, 1)

    def _build_G(self):
        """Build the nonlinear AR evolution matrix.

        Returns:
            np.ndarray:
                Evolution matrix from the current state mean.
        """
        m = self.m
        order = self.order
        G = _build_Gnonlinear(m=m, order=order)
        return G

    def _build_h(self):
        """Build the nonlinear offset for the AR evolution.

        Returns:
            np.ndarray:
                Offset ``h`` such that ``a = G m + h``.
        """
        G_ = copy.deepcopy(self.G)
        idx = np.ix_(self.index_dict.get('response'),
                     self.index_dict.get('decay'))

        G_[idx] = G_[idx] * 0.0

        m = self.m.T
        h = (G_ - self.G) @ m.T

        return h

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
            W[1:self.order, 1:self.order] = W[1:self.order, 1:self.order] * 0.0
            W[0, 0] = self.s
        else:
            W = self.W
        return W
