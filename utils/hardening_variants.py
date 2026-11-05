"""Hardening laws of the sensitivity study (Appendix F of the paper).

Ground truth: Swift law R(p) = K (e0 + p)^n with K = 417.501 MPa, e0 = 0.00457, n = 0.22194 (utils/bench_models.py).
All callables accept floats or torch tensors.
"""
import numpy as np
import torch

K0, E00, N0 = 417.501, 0.00457, 0.22194
R0 = K0 * E00 ** N0                     # initial yield stress R(0) of the ground truth
EPS_U = N0 - E00                        # uniform elongation (Considere) of the Swift law in true strain


class Swift:
    def __init__(self, K=K0, e0=E00, n=N0):
        self.K, self.e0, self.n = float(K), float(e0), float(n)

    def __call__(self, p):
        p = torch.as_tensor(p, dtype=torch.float32)
        return self.K * (self.e0 + p) ** self.n

    def describe(self):
        return dict(law="Swift", K=self.K, e0=self.e0, n=self.n, R0=self.K * self.e0 ** self.n)


class Voce:
    """R(p) = A - B exp(-C p), that is, R(0) + Q [1 - exp(-C p)] with A = R(0) + Q and B = Q."""

    def __init__(self, A, B, C):
        self.A, self.B, self.C = float(A), float(B), float(C)

    def __call__(self, p):
        p = torch.as_tensor(p, dtype=torch.float32)
        return self.A - self.B * torch.exp(-self.C * p)

    def describe(self):
        return dict(law="Voce", A=self.A, B=self.B, C=self.C, R0=self.A - self.B)


def swift_scaled(c):
    """Hardening law scaled by the factor c, R -> c R."""
    return Swift(K=c * K0)


def swift_exponent(factor):
    """Exponent n -> factor * n with the initial yield stress R(0) kept equal to the ground truth."""
    n = factor * N0
    return Swift(K=R0 / E00 ** n, n=n)


def voce_fitted_to_swift(p_max=EPS_U, n_points=400):
    """Voce law with the initial yield stress R(0) of the ground truth, and Q and C fitted by least squares to the
    ground-truth Swift law on [0, p_max] (default: up to the uniform elongation)."""
    from scipy.optimize import least_squares
    p = np.linspace(0.0, p_max, n_points)
    r = K0 * (E00 + p) ** N0
    res = least_squares(lambda x: R0 + x[0] * (1.0 - np.exp(-x[1] * p)) - r, x0=[200.0, 10.0])
    Q, C = res.x
    return Voce(A=R0 + Q, B=Q, C=C)
