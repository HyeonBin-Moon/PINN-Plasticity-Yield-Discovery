from utils.common_library import *

# ======================================================================
# Yield functions
# ======================================================================

def yield_VM(x):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    s11, s22, s12 = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    return (0.5 * (2.0*s11**2 + 2.0*s22**2 - 2.0*s11*s22 + 6.0*s12**2)).sqrt()


def yield_Hill1948(x):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    s11, s22, s12 = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    Fh, Gh, Hh, Nh = 1.3251, 1.0730, 0.8799, 2.2014

    return (0.5 * ((Gh + Hh)*s11**2 +
                   (Fh + Hh)*s22**2 -
                   2.0*Hh*s11*s22 +
                   2.0*Nh*s12**2)).sqrt()


def yield_Yld2000_2d(x):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    a = 6.0

    alpha = torch.tensor(
        [0.9835, 1.1182, 0.7435, 0.8517,
         0.8879, 0.6511, 0.9790, 1.0810],
        dtype=x.dtype,
        device=x.device
    )

    T = torch.tensor(
        [[2.0/3.0, -1.0/3.0, 0.0],
         [-1.0/3.0, 2.0/3.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=x.dtype,
        device=x.device
    )

    C1 = torch.zeros(3, 3, dtype=x.dtype, device=x.device)
    C1[0, 0], C1[1, 1], C1[2, 2] = alpha[0], alpha[1], alpha[6]

    C2 = torch.zeros(3, 3, dtype=x.dtype, device=x.device)
    C2[0, 0] = (4.0 * alpha[4] - alpha[2]) / 3.0
    C2[0, 1] = (2.0 * alpha[5] - 2.0 * alpha[3]) / 3.0
    C2[1, 0] = (2.0 * alpha[2] - 2.0 * alpha[4]) / 3.0
    C2[1, 1] = (4.0 * alpha[3] - alpha[5]) / 3.0
    C2[2, 2] = alpha[7]

    X = x @ (C1 @ T).T
    Y = x @ (C2 @ T).T

    X1, X2, X3 = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    Y1, Y2, Y3 = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]

    rx = ((0.5*(X1 - X2))**2 + X3**2).sqrt()
    ry = ((0.5*(Y1 - Y2))**2 + Y3**2).sqrt()

    eta1 = 0.5*(X1 + X2) + rx
    eta2 = 0.5*(X1 + X2) - rx

    xi1 = 0.5*(Y1 + Y2) + ry
    xi2 = 0.5*(Y1 + Y2) - ry

    phi = (eta1 - eta2).abs()**a + \
          (2.0*xi1 + xi2).abs()**a + \
          (xi1 + 2.0*xi2).abs()**a

    return (0.5 * phi) ** (1.0/a)


def hardening_law(PE_eq, K=417.501, e0=0.00457, n=0.22194):
    return K * (e0 + PE_eq) ** n
