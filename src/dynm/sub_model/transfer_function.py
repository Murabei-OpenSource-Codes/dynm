"""Utils functions."""
import numpy as np
import copy
from dynm.utils.algebra import _build_Gnonlinear, _build_W_diagonal
from scipy.linalg import block_diag


class TransferFunction():
    """Transfer-function block in state-space form."""

    def __init__(self,
                 m0: np.ndarray,
                 C0: np.ndarray,
                 gamma_order: int,
                 lambda_order: int,
                 ntfm: int,
                 discount: np.ndarray = None,
                 W: np.ndarray = None):
        """Define a transfer-function block.

        Args:
            m0 (np.ndarray):
                Prior mean of the transfer-function state.
            C0 (np.ndarray):
                Prior covariance of the transfer-function state.
            gamma_order (int):
                Order of the pulse (input) polynomial.
            lambda_order (int):
                Order of the decay (output) polynomial.
            ntfm (int):
                Number of transfer-function series.
            discount (np.ndarray):
                Discount vector used when ``W`` is unknown.
                Defaults to None.
            W (np.ndarray):
                Optional known evolution covariance. Defaults to None.
        """
        self.gamma_order = gamma_order
        self.lambda_order = lambda_order
        self.ntfm = ntfm
        self.m = m0.reshape(-1, 1)
        self.C = C0

        self.discount = discount

        if W is None:
            self.estimate_W = True
        else:
            self.W = W
            self.estimate_W = False

        # Get index for blocks
        self.index_dict = {}

        for n in range(ntfm):
            block_idx = n * (2 * lambda_order + gamma_order) + \
                np.cumsum([lambda_order, lambda_order, gamma_order])
            self.index_dict[n] = {
                'all': np.arange(n * (2 * lambda_order + gamma_order),
                                 block_idx[2]),
                'response': np.arange(n * (2 * lambda_order + gamma_order),
                                      block_idx[0]),
                'decay': np.arange(block_idx[0], block_idx[1]),
                'pulse': np.arange(block_idx[1], block_idx[2])}

        # Build F and G
        self.F = self._build_F()
        self.G = self._build_G(x=np.zeros([ntfm, self.gamma_order]))

    def _build_F(self):
        """Build the stacked transfer-function regression vector.

        Returns:
            np.ndarray:
                Column vector with a leading one in each series block.
        """
        F = np.array([])

        for i in range(self.ntfm):
            Fi = np.zeros(2 * self.lambda_order + self.gamma_order)
            Fi[0] = 1

            F = np.hstack((F, Fi))

        return F.reshape(-1, 1)

    def _build_G(self, x: np.array):
        """Build the transfer-function evolution matrix.

        Args:
            x (np.ndarray):
                Pulse inputs with shape ``(ntfm, gamma_order)``.

        Returns:
            np.ndarray:
                Block-diagonal evolution matrix.
        """
        m = self.m
        lambda_order = self.lambda_order
        ntfm = self.ntfm

        G = np.empty([0, 0])
        for n in range(ntfm):
            idx_ = np.concatenate((
                self.index_dict.get(n).get('response'),
                self.index_dict.get(n).get('decay'),
                self.index_dict.get(n).get('pulse')))

            m_ = m[idx_]
            Gi = _build_Gnonlinear(m=m_.reshape(-1, 1),
                                   order=lambda_order)

            Hn = np.zeros([Gi.shape[0], self.gamma_order])
            for o in range(self.gamma_order):
                xn = np.ravel(x[n, o])[0]
                Hn[0, o] = xn

            In = np.identity(self.gamma_order)
            Gn = np.block([[Gi, Hn], [Hn.T * 0, In]])
            G = block_diag(G, Gn)

        return G

    def _build_h(self):
        """Build the nonlinear offset for the transfer-function evolution.

        Returns:
            np.ndarray:
                Offset ``h`` such that ``a = G m + h``.
        """
        ntfm = self.ntfm
        G_ = copy.deepcopy(self.G)

        for n in range(ntfm):
            idx = np.ix_(self.index_dict.get(n).get('response'),
                         self.index_dict.get(n).get('decay'))
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

            for n in range(self.ntfm):
                idx = np.ix_(self.index_dict.get(n).get('response')[1:],
                             self.index_dict.get(n).get('response')[1:])

                W[idx] = W[idx] * 0.0
        else:
            W = self.W
        return W
