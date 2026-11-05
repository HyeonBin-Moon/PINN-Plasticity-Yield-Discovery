"""Convexity constraint of the polynomial surrogate (Section 2.4, Eq. (31), Algorithm 3).

Each homogeneous polynomial P_m of the surrogate is required to be SOS-convex,

    y^T Hess P_m(sigma) y = z_m^T G_m z_m,   G_m positive definite,

where z_m collects the products of the components of y with the monomials of degree (m-2)/2. The identity is
linear in the coefficients of P_m and in G_m, so that the constraint is a semidefinite condition. The fit
follows Algorithm 3: the unconstrained Adam result is projected onto the constraint set term by term, and then
refined by Gauss--Newton steps with Levenberg--Marquardt damping under the constraint. After the fit, a Gram
matrix is recomputed from the final coefficients for every degree to verify the constraint.

The fit is carried out in the normalized stress sigma / sigma_Y. Because P_m is homogeneous of degree m, the
coefficients are the same in the normalized and in the physical stress, and the target value is one.

Requires cvxpy with the Clarabel solver (pip install cvxpy clarabel).
"""
import numpy as np

try:
    import cvxpy as cp
except ImportError as error:          # pragma: no cover - reported to the user
    raise ImportError("The convexity constraint requires cvxpy and clarabel: pip install cvxpy clarabel") from error

SOS_SETTINGS = dict(
    gram_margin=1.0e-6,           # G_m >= gram_margin * I
    gauss_newton_steps=150,       # N_GN of Algorithm 3
    initial_damping=1.0e-6,       # kappa
    damping_increase=10.0,
    damping_decrease=0.3,
    relative_tolerance=1.0e-5,    # stop when the relative decrease of the MSE falls below this value
    solver="CLARABEL",
)


def gram_system(term_map, m):
    """Matrices (M, L) and size nz such that M @ vec(G) == L @ a encodes y^T Hess P_m y == z^T G z."""
    k = (m - 2) // 2
    gammas = [(i, j, k - i - j) for i in range(k + 1) for j in range(k + 1 - i)]
    zidx = [(i, g) for i in range(3) for g in gammas]
    nz = len(zidx)
    keys = {}

    def key(beta, i, j):
        kk = (tuple(int(b) for b in beta), (min(i, j), max(i, j)))
        if kk not in keys:
            keys[kk] = len(keys)
        return keys[kk]

    gram_entries = []
    for a, (ia, ga) in enumerate(zidx):
        for b, (ib, gb) in enumerate(zidx):
            gram_entries.append((key(np.add(ga, gb), ia, ib), a * nz + b))
    hess_entries = []
    for col, alpha in enumerate(term_map[m]):
        for i in range(3):
            for j in range(i, 3):
                e = list(alpha)
                coef = e[i]; e[i] -= 1
                if coef == 0:
                    continue
                coef *= e[j]; e[j] -= 1
                if coef == 0:
                    continue
                hess_entries.append((key(e, i, j), col, coef * (1 if i == j else 2)))
    M = np.zeros((len(keys), nz * nz))
    L = np.zeros((len(keys), len(term_map[m])))
    for r, c in gram_entries:
        M[r, c] += 1
    for r, c, v in hess_entries:
        L[r, c] += v
    return M, L, nz


def monomials(points, term_map):
    """Monomial features (float64) of every degree at points shaped (N, 3)."""
    return {m: np.prod(points[:, None, :] ** np.asarray(term_map[m])[None, :, :], axis=2) for m in term_map}


class SOSConvexFit:
    """Algorithm 3 after the Adam step: projection and constrained Gauss--Newton refinement."""

    def __init__(self, term_map, settings=None):
        self.term_map = term_map
        self.degrees = tuple(sorted(term_map))
        self.settings = dict(SOS_SETTINGS, **(settings or {}))
        self.systems = {m: gram_system(term_map, m) for m in self.degrees}

    def _constraints(self, coefficients):
        constraints = []
        for m in self.degrees:
            M, L, nz = self.systems[m]
            G = cp.Variable((nz, nz), symmetric=True)
            constraints += [M @ cp.reshape(G, (nz * nz,), order="C") == L @ coefficients[m],
                            G >> self.settings["gram_margin"] * np.eye(nz)]
        return constraints

    def residual(self, coefficients, features):
        radicands = {m: features[m] @ coefficients[m] for m in self.degrees}
        if not all(np.all(radicands[m] > 0) for m in self.degrees):
            raise ValueError("nonpositive radicand")
        r = sum(radicands[m] ** (1.0 / m) for m in self.degrees) - 1.0
        J = np.hstack([(radicands[m] ** (1.0 / m - 1.0) / m)[:, None] * features[m] for m in self.degrees])
        return r, J

    def project(self, coefficients, features):
        """Per-degree least-squares projection min sum_i [P_m(x_i; a) - P_m(x_i; a0)]^2 subject to Eq. (31)."""
        out = {}
        for m in self.degrees:
            R = np.linalg.qr(features[m] / np.sqrt(len(features[m])), mode="r")
            a = cp.Variable(len(coefficients[m]))
            cp.Problem(cp.Minimize(cp.sum_squares(R @ (a - coefficients[m]))),
                       self._constraints_single(m, a)).solve(solver=self.settings["solver"])
            out[m] = np.asarray(a.value)
        return out

    def _constraints_single(self, m, a):
        M, L, nz = self.systems[m]
        G = cp.Variable((nz, nz), symmetric=True)
        return [M @ cp.reshape(G, (nz * nz,), order="C") == L @ a, G >> self.settings["gram_margin"] * np.eye(nz)]

    def gauss_newton(self, coefficients, features, max_steps=None, log=None):
        """Damped Gauss--Newton steps of the MSE subject to Eq. (31); every iterate satisfies the constraint."""
        s = self.settings
        sizes = [len(self.term_map[m]) for m in self.degrees]
        parts = np.split(np.arange(sum(sizes)), np.cumsum(sizes)[:-1])
        r, J = self.residual(coefficients, features)
        mse = float(np.mean(r ** 2))
        kappa = s["initial_damping"]
        log = [] if log is None else log
        for step in range(max_steps or s["gauss_newton_steps"]):
            n = len(r)
            Q, R = np.linalg.qr(J / np.sqrt(n))
            rhs = Q.T @ (r / np.sqrt(n))
            scale = np.linalg.norm(R, axis=0)
            d = cp.Variable(sum(sizes))
            trial_vars = {m: coefficients[m] + d[idx] for m, idx in zip(self.degrees, parts)}
            constraints = self._constraints(trial_vars)
            mse_trial = np.inf
            while kappa <= 1.0e8:
                problem = cp.Problem(cp.Minimize(cp.sum_squares(R @ d + rhs) + kappa * cp.sum_squares(cp.multiply(scale, d))),
                                     constraints)
                problem.solve(solver=s["solver"])
                if d.value is not None:
                    trial = {m: coefficients[m] + np.asarray(d.value)[idx] for m, idx in zip(self.degrees, parts)}
                    try:
                        r_trial, J_trial = self.residual(trial, features)
                        mse_trial = float(np.mean(r_trial ** 2))
                    except ValueError:
                        mse_trial = np.inf
                    if mse_trial < mse:
                        break
                kappa *= s["damping_increase"]
            if not mse_trial < mse:
                log.append(dict(step=step, status="no decrease", damping=kappa, mse=mse))
                break
            decrease = (mse - mse_trial) / mse
            coefficients, r, J, mse = trial, r_trial, J_trial, mse_trial
            kappa = max(kappa * s["damping_decrease"], 1.0e-12)
            log.append(dict(step=step, mse=mse, damping=kappa, relative_decrease=decrease))
            print(f"  Gauss-Newton step {step:3d}  mse={mse:.6e}  damping={kappa:.1e}", flush=True)
            if decrease < s["relative_tolerance"]:
                break
        return coefficients, log

    def certificate(self, coefficients):
        """Recompute G_m from the final coefficients (maximizing its minimum eigenvalue), project it onto the
        linear identity, and report the minimum eigenvalue and the residual of the identity for every degree."""
        out = {}
        for m in self.degrees:
            M, L, nz = self.systems[m]
            G = cp.Variable((nz, nz), symmetric=True)
            t = cp.Variable()
            b = L @ coefficients[m]
            cp.Problem(cp.Maximize(t), [M @ cp.reshape(G, (nz * nz,), order="C") == b, G >> t * np.eye(nz), t <= 1.0]
                       ).solve(solver=self.settings["solver"])
            g = np.asarray(G.value); g = 0.5 * (g + g.T)
            iu = np.triu_indices(nz)
            S = np.zeros((nz * nz, len(iu[0])))
            for col, (a, c) in enumerate(zip(*iu)):
                S[a * nz + c, col] = 1
                S[c * nz + a, col] = 1
            correction = np.linalg.lstsq(M @ S, b - M @ g.reshape(-1), rcond=None)[0]
            gp = np.zeros((nz, nz)); gp[iu] = g[iu] + correction; gp = gp + gp.T - np.diag(np.diag(gp))
            out[str(m)] = dict(gram_size=nz, min_eigenvalue=float(np.linalg.eigvalsh(gp).min()),
                               identity_residual_max_abs=float(np.max(np.abs(M @ gp.reshape(-1) - b))),
                               coefficient_scale_max_abs=float(np.max(np.abs(b))))
        out["verified"] = bool(all(out[str(m)]["min_eigenvalue"] > 0 and out[str(m)]["identity_residual_max_abs"]
                                   < 1.0e-12 * max(1.0, out[str(m)]["coefficient_scale_max_abs"]) for m in self.degrees))
        return out
