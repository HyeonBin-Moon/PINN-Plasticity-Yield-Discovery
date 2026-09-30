"""Fit an identified neural yield function with the polynomial surrogate of Section 2.4.

The surrogate is a sum of signed roots of homogeneous polynomials of degrees 2, 4, 6 and 8
(94 coefficients). It is fitted to 100,000 stress samples on the level set of the neural yield
function, with 10,000 further samples for validation, using Adam for 20,000 epochs (learning rate
1e-4, reduced to 1e-5 after 1,000 epochs, gradient norm clipped at 10).

Example
-------
python code/fit_polynomial.py --yield-model Yld2000
python code/fit_polynomial.py --yield-model VM --checkpoint results/noise/VM/noise_0.00e+00/VM_noise0.00e+00_model.pth

Outputs: polynomial checkpoint, coefficient table (CSV), fit metrics (JSON), loss and yield contour
figures under results/polynomial/<model>/. With --export-fortran the fitted function and its first and
second derivatives are also written as a standalone Fortran subroutine (and inserted into an Abaqus
UMAT template when --umat-template is given).
"""

import argparse
import csv
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, for utils

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sympy as sp
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn

from utils.ai_model import (
    DEFAULT_POLYNOMIAL_DEGREES,
    MixedHomogeneousPolynomial,
    SymmetricICNN,
    build_monomial_features,
    enumerate_homogeneous_terms,
    signed_root,
)
from utils.bench_models import (
    hardening_law,
    yield_Hill1948,
    yield_VM,
    yield_Yld2000_2d,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "noise"
    / "Yld2000"
    / "noise_0.00e+00"
    / "Yld2000_noise0.00e+00_model.pth"
)
DEFAULT_UMAT_TEMPLATE = None
DEFAULT_OUTPUT_DIR = ROOT / "results" / "polynomial" / "Yld2000"
DEGREES = DEFAULT_POLYNOMIAL_DEGREES
FITTING_ROOT_EPSILON = 1.0e-8
OUTPUT_FILENAMES = {
    "checkpoint": "Yld2000_polynomial_model.pth",
    "coefficients": "Yld2000_polynomial_coefficients.csv",
    "metrics": "Yld2000_polynomial_fit_metrics.json",
    "loss_plot": "Yld2000_polynomial_fit_loss.png",
    "yield_plot": "Yld2000_polynomial_yield_surface.png",
    "fortran_subroutine": "Yield_Function_Yld2000_polynomial.for",
    "full_umat": "UMAT_Yld2000_polynomial_plane_stress.for",
}
FIELD_OUTPUT_FILENAMES = {
    "reaction_force_plot": "Yld2000_polynomial_fem_reaction_force.png",
    "field_validation_plot": "Yld2000_polynomial_fem_field_validation.png",
}
SHEAR_FRACTIONS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.55)

YIELD_MODEL_SPECS = {
    "VM": {
        "label": "von Mises",
        "yield_function": yield_VM,
        "checkpoint": (
            ROOT
            / "results"
            / "noise"
            / "VM"
            / "noise_0.00e+00"
            / "VM_noise0.00e+00_model.pth"
        ),
        "output_dir": (
            ROOT / "results" / "polynomial" / "VM"
        ),
        "umat_template": None,
        "mesh_inp": (
            ROOT / "data" / "Inp_VM_x_tension.inp"
        ),
        "ground_truth_fem_npz": (
            ROOT
            / "data"
            / "Data_Inp_VM_x_tension.npz"
        ),
    },
    "Hill": {
        "label": "Hill 1948",
        "yield_function": yield_Hill1948,
        "checkpoint": (
            ROOT
            / "results"
            / "noise"
            / "Hill"
            / "noise_0.00e+00"
            / "Hill_noise0.00e+00_model.pth"
        ),
        "output_dir": (
            ROOT / "results" / "polynomial" / "Hill"
        ),
        "umat_template": None,
        "mesh_inp": (
            ROOT / "data" / "Inp_Hill_x_tension.inp"
        ),
        "ground_truth_fem_npz": (
            ROOT
            / "data"
            / "Data_Inp_Hill_x_tension.npz"
        ),
    },
    "Yld2000": {
        "label": "Yld2000-2d",
        "yield_function": yield_Yld2000_2d,
        "checkpoint": DEFAULT_CHECKPOINT,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "umat_template": DEFAULT_UMAT_TEMPLATE,
        "mesh_inp": (
            ROOT / "data" / "Inp_Yld2000_x_tension.inp"
        ),
        "ground_truth_fem_npz": (
            ROOT
            / "data"
            / "Data_Inp_Yld2000_x_tension.npz"
        ),
    },
}


def fibonacci_sphere(samples, offset=0):
    """Generate deterministic, approximately uniform directions on a sphere."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    local_indices = np.arange(samples, dtype=np.float64)
    phase_indices = local_indices + offset
    y = 1.0 - 2.0 * (local_indices + 0.5) / samples
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * phase_indices
    return np.column_stack((np.cos(theta) * radius, y, np.sin(theta) * radius))


class SymbolicBundle(NamedTuple):
    symbols: tuple
    value: object
    gradient: tuple
    hessian: tuple
    radicands: tuple


def build_symbolic_bundle(model):
    """Build a signed-root value, gradient, and Hessian without delta terms."""
    s11, s22, s12 = sp.symbols("s11 s22 s12", real=True)
    symbols = (s11, s22, s12)
    value = sp.Integer(0)
    gradient = [sp.Integer(0) for _ in range(3)]
    hessian = [[sp.Integer(0) for _ in range(3)] for _ in range(3)]
    radicands = []

    for degree in model.degrees:
        coefficients = (
            model.coefficients[str(degree)]
            .detach()
            .cpu()
            .to(torch.float64)
            .numpy()
        )
        polynomial = sp.Integer(0)
        for coefficient, powers in zip(
            coefficients,
            model.term_map[degree],
        ):
            monomial = sp.Float(float(coefficient), 17)
            for symbol, power in zip(symbols, powers):
                monomial *= symbol ** power
            polynomial += monomial

        radicands.append(polynomial)
        grad_f = [sp.diff(polynomial, symbol) for symbol in symbols]
        hessian_f = [
            [sp.diff(grad_f[i], symbols[j]) for j in range(3)]
            for i in range(3)
        ]
        exponent = sp.Rational(1, degree)
        value += polynomial * sp.Abs(polynomial) ** (exponent - 1)
        first_factor = exponent * sp.Abs(polynomial) ** (exponent - 1)
        second_factor = (
            exponent
            * (exponent - 1)
            * polynomial
            * sp.Abs(polynomial) ** (exponent - 3)
        )
        for i in range(3):
            gradient[i] += first_factor * grad_f[i]
            for j in range(3):
                hessian[i][j] += (
                    second_factor * grad_f[i] * grad_f[j]
                    + first_factor * hessian_f[i][j]
                )

    return SymbolicBundle(
        symbols=symbols,
        value=value,
        gradient=tuple(gradient),
        hessian=tuple(tuple(row) for row in hessian),
        radicands=tuple(radicands),
    )


def _maximum_scaled_error(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    numerator = float(np.max(np.abs(actual - expected)))
    denominator = max(float(np.max(np.abs(expected))), 1.0e-12)
    return numerator / denominator


def validate_symbolic_parity(
    model,
    bundle,
    stress,
    singularity_tolerance=1.0e-10,
):
    """Compare symbolic value/derivatives against float64 PyTorch autograd."""
    stress = stress.detach().to(dtype=torch.float64, device="cpu")
    model = model.to(dtype=torch.float64, device="cpu")
    with torch.no_grad():
        radicands = model.radicands_from_features(
            build_monomial_features(stress, model.term_map)
        )
        minimum = torch.stack(
            [torch.abs(radicands[degree]) for degree in model.degrees],
            dim=1,
        ).min(dim=1).values
    valid_mask = minimum > singularity_tolerance
    valid_stress = stress[valid_mask]
    if valid_stress.shape[0] == 0:
        raise ValueError("No non-singular points available for symbolic parity")

    torch_values = []
    torch_gradients = []
    torch_hessians = []
    for point in valid_stress:
        point = point.detach().clone().requires_grad_(True)

        def scalar_value(candidate):
            return model(candidate.unsqueeze(0))[0]

        value = scalar_value(point)
        gradient = torch.autograd.grad(
            value,
            point,
            create_graph=True,
        )[0]
        hessian = torch.autograd.functional.hessian(scalar_value, point)
        torch_values.append(value.detach())
        torch_gradients.append(gradient.detach())
        torch_hessians.append(hessian.detach())

    torch_values_np = torch.stack(torch_values).numpy()
    torch_gradients_np = torch.stack(torch_gradients).numpy()
    torch_hessians_np = torch.stack(torch_hessians).numpy()

    coordinates = [valid_stress[:, i].numpy() for i in range(3)]
    value_fn = sp.lambdify(bundle.symbols, bundle.value, modules="numpy")
    gradient_fn = [
        sp.lambdify(bundle.symbols, expression, modules="numpy")
        for expression in bundle.gradient
    ]
    hessian_fn = [
        [
            sp.lambdify(bundle.symbols, expression, modules="numpy")
            for expression in row
        ]
        for row in bundle.hessian
    ]
    symbolic_values = np.asarray(value_fn(*coordinates), dtype=np.float64)
    symbolic_values = np.broadcast_to(
        symbolic_values,
        (valid_stress.shape[0],),
    )
    symbolic_gradients = np.column_stack(
        [
            np.broadcast_to(
                np.asarray(function(*coordinates), dtype=np.float64),
                (valid_stress.shape[0],),
            )
            for function in gradient_fn
        ]
    )
    symbolic_hessians = np.empty(
        (valid_stress.shape[0], 3, 3),
        dtype=np.float64,
    )
    for i in range(3):
        for j in range(3):
            symbolic_hessians[:, i, j] = np.broadcast_to(
                np.asarray(
                    hessian_fn[i][j](*coordinates),
                    dtype=np.float64,
                ),
                (valid_stress.shape[0],),
            )

    return {
        "valid_points": int(valid_stress.shape[0]),
        "excluded_points": int(stress.shape[0] - valid_stress.shape[0]),
        "value_max_relative_error": _maximum_scaled_error(
            symbolic_values,
            torch_values_np,
        ),
        "gradient_max_relative_error": _maximum_scaled_error(
            symbolic_gradients,
            torch_gradients_np,
        ),
        "hessian_max_relative_error": _maximum_scaled_error(
            symbolic_hessians,
            torch_hessians_np,
        ),
    }


def _fortran_assignment(expression, target):
    return sp.fcode(
        expression,
        assign_to=target,
        source_format="fixed",
        standard=95,
    )


def polynomial_banner(model_label):
    """Return the authoritative model-specific export banner."""
    return f"Polynomial fit of discovered {model_label} ICNN"


def generate_fortran_subroutine(
    bundle,
    model_label="Yld2000-2d",
):
    """Generate a fixed-form Abaqus yield subroutine from symbolic expressions."""
    lines = [
        "C",
        "C     ==================================================================",
        f"C     {polynomial_banner(model_label)}",
        "C     ==================================================================",
        "      SUBROUTINE Yield_Function(S, S_eq, dS_eq, ddS_eq)",
        "      INCLUDE 'ABA_PARAM.INC'",
        "",
        "      DOUBLE PRECISION, INTENT(IN)  :: S(3)",
        "      DOUBLE PRECISION, INTENT(OUT) :: S_eq",
        "      DOUBLE PRECISION, INTENT(OUT) :: dS_eq(3)",
        "      DOUBLE PRECISION, INTENT(OUT) :: ddS_eq(3,3)",
        "      DOUBLE PRECISION :: s11, s22, s12",
        "",
        "C     Finite elastic initialization at the non-differentiable origin",
        "      IF (MAXVAL(ABS(S)) .LE. 1.0D-12) THEN",
        "          S_eq = 0.0D0",
        "          dS_eq = 0.0D0",
        "          ddS_eq = 0.0D0",
        "          RETURN",
        "      END IF",
        "",
        "C     Mandel stress to tensorial plane-stress components",
        "      s11 = S(1)",
        "      s22 = S(2)",
        "      s12 = S(3) / DSQRT(2.0D0)",
        "",
        "C     Equivalent stress",
    ]
    lines.extend(_fortran_assignment(bundle.value, "S_eq").splitlines())
    lines.extend(["", "C     First derivative in tensorial notation"])
    for index, expression in enumerate(bundle.gradient, start=1):
        lines.extend(
            _fortran_assignment(
                expression,
                f"dS_eq({index})",
            ).splitlines()
        )
    lines.extend(["", "C     Second derivative in tensorial notation"])
    for i, row in enumerate(bundle.hessian, start=1):
        for j, expression in enumerate(row, start=1):
            lines.extend(
                _fortran_assignment(
                    expression,
                    f"ddS_eq({i},{j})",
                ).splitlines()
            )
    lines.extend(
        [
            "",
            "C     Tensorial derivatives to Mandel notation",
            "      dS_eq(3) = dS_eq(3) / DSQRT(2.0D0)",
            "      ddS_eq(1,3) = ddS_eq(1,3) / DSQRT(2.0D0)",
            "      ddS_eq(2,3) = ddS_eq(2,3) / DSQRT(2.0D0)",
            "      ddS_eq(3,1) = ddS_eq(3,1) / DSQRT(2.0D0)",
            "      ddS_eq(3,2) = ddS_eq(3,2) / DSQRT(2.0D0)",
            "      ddS_eq(3,3) = ddS_eq(3,3) / 2.0D0",
            "",
            "      RETURN",
            "      END",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_fortran_subroutine(
    source,
    model_label="Yld2000-2d",
):
    """Statically validate the generated Fortran subroutine contract."""
    upper = source.upper()
    if upper.count("SUBROUTINE YIELD_FUNCTION") != 1:
        raise ValueError("Fortran must contain exactly one Yield_Function")
    guard_start = upper.find("IF (MAXVAL(ABS(S)) .LE.")
    value_start = upper.find("C     EQUIVALENT STRESS")
    if guard_start < 0 or guard_start > value_start:
        raise ValueError(
            "Fortran must short-circuit zero stress before root evaluation"
        )
    guard_end = upper.find("END IF", guard_start)
    guard_region = upper[guard_start:guard_end]
    for line in (
        "S_EQ = 0.0D0",
        "DS_EQ = 0.0D0",
        "DDS_EQ = 0.0D0",
        "RETURN",
    ):
        if line not in guard_region:
            raise ValueError(f"Fortran zero-stress guard missing: {line}")
    required = ["S_EQ"]
    required.extend(f"DS_EQ({i})" for i in range(1, 4))
    required.extend(
        f"DDS_EQ({i},{j})"
        for i in range(1, 4)
        for j in range(1, 4)
    )
    for target in required:
        if re.search(rf"\b{re.escape(target)}\s*=", upper) is None:
            raise ValueError(f"Fortran assignment missing: {target}")
    mandel_lines = [
        "S12 = S(3) / DSQRT(2.0D0)",
        "DS_EQ(3) = DS_EQ(3) / DSQRT(2.0D0)",
        "DDS_EQ(3,3) = DDS_EQ(3,3) / 2.0D0",
    ]
    for line in mandel_lines:
        if line not in upper:
            raise ValueError(f"Fortran Mandel conversion missing: {line}")
    if polynomial_banner(model_label).upper() not in upper:
        raise ValueError("Fortran polynomial export banner is missing")


def replace_yield_subroutine(
    template_source,
    replacement_source,
    model_label="Yld2000-2d",
):
    """Replace exactly one complete Yield_Function subroutine in a UMAT."""
    lines = template_source.splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(
            r"^\s*SUBROUTINE\s+Yield_Function\b",
            line,
            flags=re.IGNORECASE,
        )
    ]
    if len(starts) != 1:
        raise ValueError(
            "UMAT template must contain exactly one Yield_Function subroutine"
        )
    start = starts[0]
    end = None
    for index in range(start + 1, len(lines)):
        if lines[index].strip().upper() == "END":
            end = index
            break
    if end is None:
        raise ValueError("Yield_Function END statement was not found")

    for index in range(max(0, start - 8), start):
        if "MANDEL NOTATION" in lines[index].upper():
            newline = "\r\n" if lines[index].endswith("\r\n") else "\n"
            lines[index] = (
                f"C     {polynomial_banner(model_label)}"
                + newline
            )

    newline = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"
    replacement = replacement_source.rstrip("\r\n") + newline
    return "".join(lines[:start]) + replacement + "".join(lines[end + 1 :])


def validate_complete_umat(
    template_source,
    generated_source,
    model_label="Yld2000-2d",
):
    """Validate replacement scope and the required UMAT subroutine structure."""
    del template_source
    upper = generated_source.upper()
    required_subroutines = [
        "SUBROUTINE UMAT",
        "SUBROUTINE RETURN_MAPPING",
        "SUBROUTINE ISO_HARDENING",
        "SUBROUTINE MAT_INVERSION",
    ]
    for name in required_subroutines:
        if name not in upper:
            raise ValueError(f"Generated UMAT is missing {name}")
    if upper.count("SUBROUTINE YIELD_FUNCTION") != 1:
        raise ValueError(
            "Generated UMAT must contain exactly one Yield_Function"
        )
    if polynomial_banner(model_label).upper() not in upper:
        raise ValueError("Generated UMAT polynomial banner is missing")

    yield_start = upper.index("SUBROUTINE YIELD_FUNCTION")
    next_subroutine = upper.find("SUBROUTINE ", yield_start + 1)
    yield_region = (
        upper[yield_start:]
        if next_subroutine < 0
        else upper[yield_start:next_subroutine]
    )
    for obsolete in ("ALPHA =", "C1(", "C2("):
        if obsolete in yield_region:
            raise ValueError(
                f"Analytical Yld2000 term remains in Yield_Function: {obsolete}"
            )


def model_output_filenames(
    model_tag,
    include_field_validation=False,
):
    """Return model-prefixed artifact filenames."""
    if model_tag not in YIELD_MODEL_SPECS:
        raise ValueError(f"Unknown yield model: {model_tag}")
    filenames = {
        "checkpoint": f"{model_tag}_polynomial_model.pth",
        "coefficients": (
            f"{model_tag}_polynomial_coefficients.csv"
        ),
        "metrics": f"{model_tag}_polynomial_fit_metrics.json",
        "loss_plot": f"{model_tag}_polynomial_fit_loss.png",
        "yield_plot": (
            f"{model_tag}_polynomial_yield_surface.png"
        ),
        "fortran_subroutine": (
            f"Yield_Function_{model_tag}_polynomial.for"
        ),
        "full_umat": (
            f"UMAT_{model_tag}_polynomial_plane_stress.for"
        ),
    }
    if include_field_validation:
        filenames.update(
            {
                "reaction_force_plot": (
                    f"{model_tag}_polynomial_fem_reaction_force.png"
                ),
                "field_validation_plot": (
                    f"{model_tag}_polynomial_fem_field_validation.png"
                ),
            }
        )
    return filenames


def planned_output_paths(
    output_dir,
    include_field_validation=False,
    model_tag="Yld2000",
):
    """Return every artifact path that a run is allowed to create."""
    output_dir = Path(output_dir)
    filenames = model_output_filenames(
        model_tag,
        include_field_validation=include_field_validation,
    )
    return {
        key: output_dir / filename
        for key, filename in filenames.items()
    }


def ensure_outputs_writable(paths, force=False):
    """Refuse accidental replacement of any planned artifact."""
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing and not force:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Planned output files already exist. Re-run with --force to "
            f"overwrite them:\n{formatted}"
        )


class FitConfig(NamedTuple):
    samples: int
    validation_samples: int
    epochs: int
    learning_rate: float
    learning_rate_after_drop: float
    learning_rate_drop_epoch: int
    gradient_clip: float
    seed: int
    smoke_test: bool
    plot_resolution: int


class FitResult(NamedTuple):
    model: MixedHomogeneousPolynomial
    loss_history: list
    elapsed_seconds: float


class FemResult(NamedTuple):
    displacement: np.ndarray
    strain: np.ndarray
    plastic_strain: np.ndarray
    equivalent_plastic_strain: np.ndarray
    stress: np.ndarray
    equivalent_stress: np.ndarray
    bottom_reaction: np.ndarray
    top_reaction: np.ndarray
    right_reaction: np.ndarray
    left_reaction: np.ndarray


def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def generate_icnn_surface_points(
    discovered_model,
    samples,
    sigma_y,
    device,
    offset=0,
):
    directions = torch.as_tensor(
        fibonacci_sphere(samples, offset=offset),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        equivalent = discovered_model(directions).reshape(-1)
    if not torch.isfinite(equivalent).all():
        raise ValueError("ICNN produced non-finite values on sphere samples")
    if torch.any(equivalent <= 1.0e-12):
        minimum = float(equivalent.min().detach().cpu())
        raise ValueError(
            f"ICNN must be positive on nonzero directions; minimum={minimum}"
        )
    return directions / equivalent.unsqueeze(1) * float(sigma_y)


def fit_polynomial(surface_points, sigma_y, config):
    """Fit the 94 polynomial coefficients with precomputed monomials."""
    set_deterministic_seed(config.seed)
    model = MixedHomogeneousPolynomial().to(
        device=surface_points.device,
        dtype=surface_points.dtype,
    )
    features = build_monomial_features(surface_points, model.term_map)
    target = torch.full(
        (surface_points.shape[0],),
        float(sigma_y),
        dtype=surface_points.dtype,
        device=surface_points.device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )
    loss_history = []
    started_at = time.time()
    progress_every = max(1, config.epochs // 20)

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.evaluate_from_features(
            features,
            stabilization_epsilon=FITTING_ROOT_EPSILON,
        )
        loss = torch.mean((prediction - target) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Polynomial fitting became non-finite at epoch {epoch}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.gradient_clip,
        )
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))

        if epoch == config.learning_rate_drop_epoch:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = config.learning_rate_after_drop

        if (
            epoch == 1
            or epoch == config.epochs
            or epoch % progress_every == 0
        ):
            elapsed = time.time() - started_at
            print(
                f"[{epoch}/{config.epochs}] "
                f"loss={loss_history[-1]:.6e} elapsed={elapsed:.1f}s",
                flush=True,
            )

    return FitResult(
        model=model,
        loss_history=loss_history,
        elapsed_seconds=time.time() - started_at,
    )


def evaluate_fit(model, surface_points, sigma_y):
    with torch.no_grad():
        prediction = model(surface_points).reshape(-1)
    target = torch.full_like(prediction, float(sigma_y))
    relative = torch.abs(prediction - target) / abs(float(sigma_y))
    finite = bool(
        torch.isfinite(prediction).all()
        and torch.isfinite(relative).all()
        and all(
            torch.isfinite(parameter).all()
            for parameter in model.parameters()
        )
    )
    return {
        "finite": finite,
        "relative_rmse": float(
            torch.sqrt(torch.mean(((prediction - target) / target) ** 2))
            .detach()
            .cpu()
        ),
        "mean_absolute_relative_error": float(
            torch.mean(relative).detach().cpu()
        ),
        "maximum_absolute_relative_error": float(
            torch.max(relative).detach().cpu()
        ),
    }


def validate_homogeneity(model, stress, scales=(0.5, 2.0, 3.0)):
    # The model is fitted in float32, but this identity is a structural
    # validation of the exported equation.  Evaluate it in float64 so the
    # gate does not measure avoidable signed-root round-off.
    model = model.double()
    parameter = next(model.parameters())
    stress = stress.to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    with torch.no_grad():
        base = model(stress).reshape(-1)
        denominator = torch.clamp(torch.abs(base), min=1.0e-12)
        errors = []
        for scale in scales:
            scaled = model(float(scale) * stress).reshape(-1)
            errors.append(
                torch.max(
                    torch.abs(scaled - float(scale) * base)
                    / (float(scale) * denominator)
                )
            )
    maximum = float(torch.stack(errors).max().detach().cpu())
    return {
        "scales": [float(scale) for scale in scales],
        "maximum_relative_error": maximum,
        "passed": maximum <= 1.0e-5,
    }


def save_coefficients_csv(model, path):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["degree", "i", "j", "k", "coefficient"],
        )
        writer.writeheader()
        for degree in model.degrees:
            values = (
                model.coefficients[str(degree)]
                .detach()
                .cpu()
                .to(torch.float64)
                .numpy()
            )
            for powers, coefficient in zip(model.term_map[degree], values):
                writer.writerow(
                    {
                        "degree": degree,
                        "i": powers[0],
                        "j": powers[1],
                        "k": powers[2],
                        "coefficient": f"{float(coefficient):.17e}",
                    }
                )


def _configure_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False


def plot_fitting_loss(loss_history, path):
    _configure_plot_style()
    epochs = np.arange(1, len(loss_history) + 1)
    figure, axis = plt.subplots(figsize=(7.5, 5.0), dpi=300)
    axis.semilogy(
        epochs,
        np.asarray(loss_history, dtype=float),
        color="tab:blue",
        linewidth=2.0,
        label="MSE loss",
    )
    axis.set_title("Fitting Loss", fontsize=20)
    axis.set_xlabel("Epoch", fontsize=20)
    axis.set_ylabel("Loss", fontsize=20)
    axis.set_xlim(1, max(1, len(loss_history)))
    axis.tick_params(axis="both", labelsize=20)
    axis.grid(True, which="major", linestyle="--", alpha=0.35)
    axis.grid(False, which="minor")
    axis.legend(fontsize=16, frameon=False, loc="upper right")
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close(figure)


def _module_device_dtype(module):
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


def _evaluate_yield_function(yield_function, stress):
    if isinstance(yield_function, nn.Module):
        device, dtype = _module_device_dtype(yield_function)
        stress = stress.to(device=device, dtype=dtype)
    with torch.no_grad():
        result = yield_function(stress).reshape(-1)
    return result.detach().cpu()


def _plot_yield_family(
    axis,
    yield_function,
    sigma_y,
    color,
    linestyle,
    linewidth,
    resolution,
):
    stress_axis = torch.linspace(
        -1.5 * sigma_y,
        1.5 * sigma_y,
        resolution,
    )
    s11, s22 = torch.meshgrid(
        stress_axis,
        stress_axis,
        indexing="ij",
    )
    for fraction in SHEAR_FRACTIONS:
        stress = torch.stack(
            [
                s11.reshape(-1),
                s22.reshape(-1),
                torch.full_like(s11.reshape(-1), fraction * sigma_y),
            ],
            dim=1,
        )
        equivalent = _evaluate_yield_function(
            yield_function,
            stress,
        ).numpy()
        axis.contour(
            s11.numpy() / sigma_y,
            s22.numpy() / sigma_y,
            equivalent.reshape(resolution, resolution) / sigma_y,
            levels=[1.0],
            colors=[color],
            linestyles=linestyle,
            linewidths=linewidth,
            alpha=0.95,
        )


def plot_yield_comparison(
    ground_truth_function,
    ground_truth_label,
    discovered_model,
    polynomial_model,
    sigma_y,
    path,
    resolution=350,
):
    _configure_plot_style()
    figure, axis = plt.subplots(figsize=(10.2, 5.0), dpi=300)
    _plot_yield_family(
        axis,
        ground_truth_function,
        sigma_y,
        "tab:blue",
        "--",
        2.4,
        resolution,
    )
    _plot_yield_family(
        axis,
        discovered_model,
        sigma_y,
        "tab:red",
        "--",
        2.0,
        resolution,
    )
    _plot_yield_family(
        axis,
        polynomial_model,
        sigma_y,
        "tab:green",
        ":",
        2.4,
        resolution,
    )
    axis.set_title("Polynomial Surrogate", fontsize=20)
    axis.set_xlabel(r"$\sigma_{xx}/\sigma_Y$", fontsize=20)
    axis.set_ylabel(r"$\sigma_{yy}/\sigma_Y$", fontsize=20)
    axis.set_xlim(-1.5, 1.5)
    axis.set_ylim(-1.5, 1.5)
    axis.set_aspect("equal", "box")
    axis.set_xticks(np.arange(-1.5, 1.51, 1.0))
    axis.set_yticks(np.arange(-1.5, 1.51, 1.0))
    axis.tick_params(axis="both", labelsize=20)
    axis.grid(True, linestyle="--", alpha=0.6)
    axis.legend(
        handles=[
            mlines.Line2D(
                [],
                [],
                color="tab:blue",
                linewidth=2.4,
                linestyle="--",
                label=ground_truth_label,
            ),
            mlines.Line2D(
                [],
                [],
                color="tab:red",
                linewidth=2.0,
                linestyle="--",
                label="ICNN",
            ),
            mlines.Line2D(
                [],
                [],
                color="tab:green",
                linewidth=2.4,
                linestyle=":",
                label="Polynomial",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, -0.02),
        fontsize=16,
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close(figure)


def element_to_node_average(faces, element_values, node_count):
    """Average one scalar value per triangle onto its referenced nodes."""
    faces = np.asarray(faces, dtype=np.int64)
    element_values = np.asarray(element_values, dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"faces must have shape (n_elements, 3), received {faces.shape}"
        )
    if element_values.shape != (faces.shape[0],):
        raise ValueError(
            "element_values must have one scalar per face; received "
            f"{element_values.shape} for {faces.shape[0]} faces"
        )
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    if faces.size and (faces.min() < 0 or faces.max() >= node_count):
        raise ValueError("faces contain node indices outside node_count")

    sums = np.zeros(node_count, dtype=np.float64)
    counts = np.zeros(node_count, dtype=np.int64)
    flat_faces = faces.reshape(-1)
    np.add.at(sums, flat_faces, np.repeat(element_values, 3))
    np.add.at(counts, flat_faces, 1)
    missing = np.flatnonzero(counts == 0)
    if missing.size:
        raise ValueError(
            "Mesh contains unreferenced nodes: "
            + ", ".join(map(str, missing[:10]))
        )
    return sums / counts


def validate_fem_result_shapes(ground_truth, polynomial):
    """Validate matching FEM result schemas before numerical comparison."""
    if not isinstance(ground_truth, FemResult):
        raise TypeError("ground_truth must be a FemResult")
    if not isinstance(polynomial, FemResult):
        raise TypeError("polynomial must be a FemResult")

    for field in FemResult._fields:
        true_value = np.asarray(getattr(ground_truth, field))
        poly_value = np.asarray(getattr(polynomial, field))
        if true_value.shape != poly_value.shape:
            raise ValueError(
                f"{field} shape mismatch: ground truth "
                f"{true_value.shape}, polynomial {poly_value.shape}"
            )
        if not np.isfinite(true_value).all():
            raise ValueError(f"{field} ground truth contains non-finite values")
        if not np.isfinite(poly_value).all():
            raise ValueError(f"{field} polynomial contains non-finite values")

    expected_ranks = {
        "displacement": (3, 2),
        "strain": (3, 3),
        "plastic_strain": (3, 3),
        "stress": (3, 3),
        "equivalent_plastic_strain": (2, None),
        "equivalent_stress": (2, None),
        "bottom_reaction": (2, 2),
        "top_reaction": (2, 2),
        "right_reaction": (2, 2),
        "left_reaction": (2, 2),
    }
    frame_count = None
    element_count = None
    for field, (rank, trailing_size) in expected_ranks.items():
        value = np.asarray(getattr(ground_truth, field))
        if value.ndim != rank:
            raise ValueError(
                f"{field} must have rank {rank}, received {value.shape}"
            )
        if trailing_size is not None and value.shape[-1] != trailing_size:
            raise ValueError(
                f"{field} final dimension must be {trailing_size}, "
                f"received {value.shape}"
            )
        if frame_count is None:
            frame_count = value.shape[0]
        elif value.shape[0] != frame_count:
            raise ValueError(
                f"{field} frame count {value.shape[0]} does not match "
                f"{frame_count}"
            )
        if field in {
            "strain",
            "plastic_strain",
            "stress",
            "equivalent_plastic_strain",
            "equivalent_stress",
        }:
            if element_count is None:
                element_count = value.shape[1]
            elif value.shape[1] != element_count:
                raise ValueError(
                    f"{field} element count {value.shape[1]} does not "
                    f"match {element_count}"
                )
    if frame_count is None or frame_count == 0:
        raise ValueError("FEM results must contain at least one frame")


def plot_reaction_force_validation(ground_truth, polynomial, path):
    """Plot right-boundary RFx/RFy histories for the two FEM runs."""
    validate_fem_result_shapes(ground_truth, polynomial)
    _configure_plot_style()
    increments = np.arange(ground_truth.right_reaction.shape[0])
    true_reaction = np.asarray(ground_truth.right_reaction)
    poly_reaction = np.asarray(polynomial.right_reaction)

    figure, axis = plt.subplots(figsize=(6, 4), dpi=300)
    colors = ("tab:blue", "tab:green")
    for component, direction in enumerate(("X", "Y")):
        axis.plot(
            increments,
            true_reaction[:, component] / 1.0e3,
            linewidth=2.6,
            color=colors[component],
            label=f"{direction}-dir Ground truth",
        )
        axis.plot(
            increments,
            poly_reaction[:, component] / 1.0e3,
            linewidth=2.6,
            linestyle="--",
            color=("tab:orange", "tab:red")[component],
            label=f"{direction}-dir Polynomial",
        )
    axis.set_xlabel("Increment", fontsize=14)
    axis.set_ylabel(r"Reaction Force $\mathrm{[kN]}$", fontsize=14)
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axis.legend(fontsize=10, frameon=False)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def _mesh_boundary_edges(faces):
    counts = {}
    for triangle in np.asarray(faces, dtype=np.int64):
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            counts[edge] = counts.get(edge, 0) + 1
    return [edge for edge, count in counts.items() if count == 1]


def _field_comparison_data(result, faces, node_count):
    element_fields = (
        result.stress[-1, :, 0],
        result.strain[-1, :, 0],
        result.plastic_strain[-1, :, 0],
        result.equivalent_stress[-1],
        result.equivalent_plastic_strain[-1],
    )
    return [
        element_to_node_average(faces, values, node_count)
        for values in element_fields
    ]


def plot_field_validation(
    vertices,
    faces,
    ground_truth,
    polynomial,
    path,
):
    """Plot final-frame ground truth, polynomial, and absolute errors."""
    validate_fem_result_shapes(ground_truth, polynomial)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError(
            f"vertices must have shape (n_nodes, 2), received {vertices.shape}"
        )
    if ground_truth.displacement.shape[1] != vertices.shape[0]:
        raise ValueError(
            "displacement node count does not match mesh vertices"
        )
    if ground_truth.stress.shape[1] != faces.shape[0]:
        raise ValueError("FEM element count does not match mesh faces")

    true_fields = _field_comparison_data(
        ground_truth,
        faces,
        vertices.shape[0],
    )
    poly_fields = _field_comparison_data(
        polynomial,
        faces,
        vertices.shape[0],
    )
    error_fields = [
        np.abs(true_value - poly_value)
        for true_value, poly_value in zip(true_fields, poly_fields)
    ]
    common_limits = [
        (
            min(true_value.min(), poly_value.min()),
            max(true_value.max(), poly_value.max()),
        )
        for true_value, poly_value in zip(true_fields, poly_fields)
    ]
    error_limits = [
        (0.0, max(float(error.max()), np.finfo(float).eps))
        for error in error_fields
    ]
    boundary_edges = _mesh_boundary_edges(faces)

    _configure_plot_style()
    figure, axes = plt.subplots(3, 5, figsize=(16, 10), dpi=300)
    for row in range(3):
        for column in range(5):
            axis = axes[row, column]
            if row == 0:
                data = true_fields[column]
                limits = common_limits[column]
            elif row == 1:
                data = poly_fields[column]
                limits = common_limits[column]
            else:
                data = error_fields[column]
                limits = error_limits[column]
            vmin, vmax = limits
            if np.isclose(vmin, vmax):
                padding = max(abs(float(vmin)), 1.0) * 1.0e-12
                vmin -= padding
                vmax += padding
            image = axis.tripcolor(
                vertices[:, 0],
                vertices[:, 1],
                faces,
                data,
                shading="gouraud",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                edgecolors="none",
            )
            for first, second in boundary_edges:
                points = vertices[[first, second]]
                axis.plot(
                    points[:, 0],
                    points[:, 1],
                    color="black",
                    linewidth=0.5,
                    zorder=10,
                )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            divider = make_axes_locatable(axis)
            colorbar_axis = divider.append_axes(
                "right",
                size="4%",
                pad=0.05,
            )
            colorbar = figure.colorbar(image, cax=colorbar_axis)
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            formatter.set_scientific(True)
            colorbar.formatter = formatter
            colorbar.ax.yaxis.set_offset_position("left")
            colorbar.update_ticks()
            colorbar_axis.tick_params(labelsize=16)
            colorbar.ax.yaxis.offsetText.set_fontsize(16)
    figure.subplots_adjust(hspace=0.01, wspace=0.15)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def _relative_l2_error(reference, prediction):
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = max(
        float(np.linalg.norm(reference.reshape(-1))),
        np.finfo(float).eps,
    )
    return float(
        np.linalg.norm((prediction - reference).reshape(-1))
        / denominator
    )


def calculate_fem_error_metrics(ground_truth, polynomial):
    validate_fem_result_shapes(ground_truth, polynomial)
    final_fields = {
        "Sxx": (
            ground_truth.stress[-1, :, 0],
            polynomial.stress[-1, :, 0],
        ),
        "Exx": (
            ground_truth.strain[-1, :, 0],
            polynomial.strain[-1, :, 0],
        ),
        "PExx": (
            ground_truth.plastic_strain[-1, :, 0],
            polynomial.plastic_strain[-1, :, 0],
        ),
        "S_eq": (
            ground_truth.equivalent_stress[-1],
            polynomial.equivalent_stress[-1],
        ),
        "PE_eq": (
            ground_truth.equivalent_plastic_strain[-1],
            polynomial.equivalent_plastic_strain[-1],
        ),
    }
    field_errors = {}
    for name, (reference, prediction) in final_fields.items():
        difference = np.asarray(prediction) - np.asarray(reference)
        field_errors[name] = {
            "mean_absolute_error": float(np.mean(np.abs(difference))),
            "maximum_absolute_error": float(np.max(np.abs(difference))),
            "relative_l2_error": _relative_l2_error(
                reference,
                prediction,
            ),
        }

    reaction_errors = {}
    for component, name in enumerate(("RFx", "RFy")):
        reference = ground_truth.right_reaction[:, component]
        prediction = polynomial.right_reaction[:, component]
        difference = prediction - reference
        reaction_errors[name] = {
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "maximum_absolute_error": float(np.max(np.abs(difference))),
            "relative_l2_error": _relative_l2_error(
                reference,
                prediction,
            ),
        }
    return {
        "field_errors": field_errors,
        "reaction_force_errors": reaction_errors,
    }


def _require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_fem_validation_inputs(
    mesh_path,
    ground_truth_path,
    polynomial_path,
):
    """Load the mesh and both Abaqus NPZ files through project utilities."""
    from utils.load_dataset import get_data_from_inp, get_data_from_npz

    mesh_path = _require_file(mesh_path, "FEM mesh")
    ground_truth_path = _require_file(
        ground_truth_path,
        "Ground-truth FEM NPZ",
    )
    polynomial_path = _require_file(
        polynomial_path,
        "Polynomial FEM NPZ",
    )
    mesh_data = get_data_from_inp(str(mesh_path))
    vertices, faces = mesh_data[0], mesh_data[1]
    b_matrices = mesh_data[4]
    ground_truth = FemResult(
        *get_data_from_npz(
            str(ground_truth_path),
            faces,
            b_matrices,
        )
    )
    polynomial = FemResult(
        *get_data_from_npz(
            str(polynomial_path),
            faces,
            b_matrices,
        )
    )
    validate_fem_result_shapes(ground_truth, polynomial)
    return vertices, faces, ground_truth, polynomial


def run_field_validation(
    mesh_path,
    ground_truth_path,
    polynomial_path,
    reaction_plot_path,
    field_plot_path,
):
    vertices, faces, ground_truth, polynomial = (
        load_fem_validation_inputs(
            mesh_path,
            ground_truth_path,
            polynomial_path,
        )
    )
    plot_reaction_force_validation(
        ground_truth,
        polynomial,
        reaction_plot_path,
    )
    plot_field_validation(
        vertices,
        faces,
        ground_truth,
        polynomial,
        field_plot_path,
    )
    metrics = calculate_fem_error_metrics(ground_truth, polynomial)
    return {
        "status": "complete",
        "mesh": str(Path(mesh_path).resolve()),
        "ground_truth_npz": str(Path(ground_truth_path).resolve()),
        "polynomial_npz": str(Path(polynomial_path).resolve()),
        "reaction_force_plot": str(Path(reaction_plot_path).resolve()),
        "field_validation_plot": str(Path(field_plot_path).resolve()),
        **metrics,
    }


def read_umat_template(path):
    raw = Path(path).read_bytes()
    for encoding in ("utf-8", "cp949"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "unknown",
        raw,
        0,
        1,
        f"Unable to decode UMAT template: {path}",
    )


def _config_as_dict(config):
    return {
        field: getattr(config, field)
        for field in config._fields
    }


def save_core_artifacts(
    paths,
    fit_result,
    metrics,
    config,
    checkpoint_path,
    model_tag,
    model_label,
    ground_truth_function,
    discovered_model,
    sigma_y,
):
    save_coefficients_csv(fit_result.model, paths["coefficients"])
    plot_fitting_loss(fit_result.loss_history, paths["loss_plot"])
    plot_yield_comparison(
        ground_truth_function,
        model_label,
        discovered_model,
        fit_result.model,
        sigma_y,
        paths["yield_plot"],
        resolution=config.plot_resolution,
    )
    checkpoint = {
        "model_state_dict": fit_result.model.state_dict(),
        "degrees": list(fit_result.model.degrees),
        "loss_history": {"loss": fit_result.loss_history},
        "source_icnn_checkpoint": str(Path(checkpoint_path).resolve()),
        "yield_model": model_tag,
        "yield_model_label": model_label,
        "fit_config": _config_as_dict(config),
        "validation_metrics": metrics,
    }
    torch.save(checkpoint, paths["checkpoint"])
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fit an identified neural yield function with the polynomial "
            "surrogate of Section 2.4 and optionally export Fortran."
        )
    )
    parser.add_argument(
        "--yield-model",
        choices=tuple(YIELD_MODEL_SPECS),
        default="Yld2000",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--export-fortran",
        action="store_true",
        help="write the fitted function and its derivatives as a Fortran subroutine",
    )
    parser.add_argument(
        "--umat-template",
        type=Path,
        help="Abaqus UMAT containing one Yield_Function subroutine to be replaced (with --export-fortran)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--validation-samples", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--learning-rate-after-drop",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--learning-rate-drop-epoch",
        type=int,
        default=1_000,
    )
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-relative-rmse", type=float, default=0.02)
    parser.add_argument("--max-relative-error", type=float, default=0.10)
    parser.add_argument(
        "--singularity-tolerance",
        type=float,
        default=1.0e-10,
    )
    parser.add_argument("--allow-failed-fit", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--mesh-inp",
        type=Path,
    )
    parser.add_argument(
        "--ground-truth-fem-npz",
        type=Path,
    )
    parser.add_argument("--polynomial-fem-npz", type=Path)
    return parser.parse_args(argv)


def make_fit_config(args):
    if args.smoke_test:
        return FitConfig(
            samples=256,
            validation_samples=128,
            epochs=2,
            learning_rate=args.learning_rate,
            learning_rate_after_drop=args.learning_rate_after_drop,
            learning_rate_drop_epoch=args.learning_rate_drop_epoch,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
            smoke_test=True,
            plot_resolution=80,
        )
    return FitConfig(
        samples=args.samples,
        validation_samples=args.validation_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        learning_rate_after_drop=args.learning_rate_after_drop,
        learning_rate_drop_epoch=args.learning_rate_drop_epoch,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        smoke_test=False,
        plot_resolution=350,
    )


def _validate_positive_integer(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be positive, received {value}")


def resolve_model_runtime_paths(args):
    """Fill unspecified CLI paths from the selected model specification."""
    spec = YIELD_MODEL_SPECS[args.yield_model]
    for argument_name, spec_name in (
        ("checkpoint", "checkpoint"),
        ("umat_template", "umat_template"),
        ("output_dir", "output_dir"),
        ("mesh_inp", "mesh_inp"),
        ("ground_truth_fem_npz", "ground_truth_fem_npz"),
    ):
        if getattr(args, argument_name) is None:
            setattr(args, argument_name, spec[spec_name])
    return spec


def main(argv=None):
    args = parse_args(argv)
    model_spec = resolve_model_runtime_paths(args)
    config = make_fit_config(args)
    _validate_positive_integer("samples", config.samples)
    _validate_positive_integer(
        "validation_samples",
        config.validation_samples,
    )
    _validate_positive_integer("epochs", config.epochs)
    include_field = args.polynomial_fem_npz is not None
    paths = planned_output_paths(
        args.output_dir,
        include_field_validation=include_field,
        model_tag=args.yield_model,
    )
    ensure_outputs_writable(paths.values(), force=args.force)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    set_deterministic_seed(config.seed)
    print(f"device={device}", flush=True)
    discovered_model = load_discovered_model(args.checkpoint, device)
    sigma_y = float(hardening_law(0.0))
    train_surface = generate_icnn_surface_points(
        discovered_model,
        config.samples,
        sigma_y,
        device,
        offset=0,
    )
    validation_surface = generate_icnn_surface_points(
        discovered_model,
        config.validation_samples,
        sigma_y,
        device,
        offset=config.samples,
    )
    fit_result = fit_polynomial(train_surface, sigma_y, config)
    fit_metrics = evaluate_fit(
        fit_result.model,
        validation_surface,
        sigma_y,
    )
    homogeneity = validate_homogeneity(
        fit_result.model,
        validation_surface[: min(64, config.validation_samples)],
    )

    symbolic_bundle = build_symbolic_bundle(fit_result.model)
    symbolic_parity = validate_symbolic_parity(
        fit_result.model,
        symbolic_bundle,
        validation_surface[: min(64, config.validation_samples)],
        singularity_tolerance=args.singularity_tolerance,
    )
    symbolic_passed = (
        symbolic_parity["valid_points"] >= 4
        and symbolic_parity["value_max_relative_error"] <= 1.0e-9
        and symbolic_parity["gradient_max_relative_error"] <= 1.0e-6
        and symbolic_parity["hessian_max_relative_error"] <= 1.0e-5
    )
    if not homogeneity["passed"]:
        symbolic_passed = False

    fit_quality_passed = (
        fit_metrics["finite"]
        and fit_metrics["relative_rmse"] <= args.max_relative_rmse
        and fit_metrics["maximum_absolute_relative_error"]
        <= args.max_relative_error
    )
    metrics = {
        "yield_model": args.yield_model,
        "yield_model_label": model_spec["label"],
        "smoke_test": config.smoke_test,
        "sigma_y": sigma_y,
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "fit": fit_metrics,
        "homogeneity": homogeneity,
        "symbolic_parity": symbolic_parity,
        "fit_quality_passed": fit_quality_passed,
        "symbolic_validation_passed": symbolic_passed,
        "training_elapsed_seconds": fit_result.elapsed_seconds,
        "field_validation": {
            "status": "pending" if include_field else "not_requested"
        },
        "fortran_export": {"status": "pending"},
    }

    save_core_artifacts(
        paths,
        fit_result,
        metrics,
        config,
        args.checkpoint,
        args.yield_model,
        model_spec["label"],
        model_spec["yield_function"],
        discovered_model,
        sigma_y,
    )

    if not symbolic_passed:
        raise RuntimeError(
            "Symbolic or homogeneity validation failed; Fortran was not written"
        )
    if (
        not fit_quality_passed
        and not config.smoke_test
        and not args.allow_failed_fit
    ):
        raise RuntimeError(
            "Held-out polynomial fit validation failed; use "
            "--allow-failed-fit only after reviewing metrics"
        )

    if args.export_fortran:
        standalone = generate_fortran_subroutine(
            symbolic_bundle,
            model_label=model_spec["label"],
        )
        validate_fortran_subroutine(
            standalone,
            model_label=model_spec["label"],
        )
        paths["fortran_subroutine"].write_text(
            standalone,
            encoding="ascii",
        )
        metrics["fortran_export"] = {
            "status": "complete",
            "standalone": str(paths["fortran_subroutine"].resolve()),
        }
        if args.umat_template is not None:
            template_source, template_encoding = read_umat_template(
                args.umat_template
            )
            complete_umat = replace_yield_subroutine(
                template_source,
                standalone,
                model_label=model_spec["label"],
            )
            validate_complete_umat(
                template_source,
                complete_umat,
                model_label=model_spec["label"],
            )
            paths["full_umat"].write_text(
                complete_umat,
                encoding=template_encoding,
            )
            metrics["fortran_export"].update({
                "complete_umat": str(paths["full_umat"].resolve()),
                "template_encoding": template_encoding,
            })
    else:
        metrics["fortran_export"] = {"status": "not_requested"}
    if include_field:
        print("Generating FEM validation figures", flush=True)
        metrics["field_validation"] = run_field_validation(
            args.mesh_inp,
            args.ground_truth_fem_npz,
            args.polynomial_fem_npz,
            paths["reaction_force_plot"],
            paths["field_validation_plot"],
        )
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
    checkpoint["validation_metrics"] = metrics
    torch.save(checkpoint, paths["checkpoint"])
    print(f"Completed: {args.output_dir.resolve()}", flush=True)
    return 0


def validate_checkpoint_state(state_dict):
    """Require the refreshed 32-wide, four-layer SymmetricICNN architecture."""
    expected_model = SymmetricICNN(hidden_dim=32, num_layers=4)
    expected_state = expected_model.state_dict()
    actual_keys = set(state_dict)
    expected_keys = set(expected_state)
    shape_mismatches = {
        key: (tuple(state_dict[key].shape), tuple(expected_state[key].shape))
        for key in actual_keys & expected_keys
        if tuple(state_dict[key].shape) != tuple(expected_state[key].shape)
    }
    if actual_keys != expected_keys or shape_mismatches:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "Checkpoint must match SymmetricICNN(hidden_dim=32, "
            f"num_layers=4); missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}"
        )


def load_discovered_model(checkpoint_path, device):
    """Load the authoritative discovered ICNN after architecture validation."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint has no model_state_dict: {checkpoint_path}"
        )
    state_dict = checkpoint["model_state_dict"]
    validate_checkpoint_state(state_dict)
    model = SymmetricICNN(hidden_dim=32, num_layers=4).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


if __name__ == "__main__":
    raise SystemExit(main())
