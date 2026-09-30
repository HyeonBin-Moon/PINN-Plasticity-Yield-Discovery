from utils.common_library import *

class SimpleICNN(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32,
                 output_dim: int = 1, num_layers: int = 3):
        super().__init__()

        self.activation = nn.ReLU()
        self.gamma = 0.1
        self.gz = lambda x: torch.sqrt(x**2 + self.gamma**2) - self.gamma
        self.gx = lambda x: x

        self.Wx = nn.ParameterList()
        self.Wz = nn.ParameterList()

        self.Wx.append(nn.Parameter(torch.empty(hidden_dim, input_dim)))

        for _ in range(num_layers - 1):
            self.Wx.append(nn.Parameter(torch.empty(hidden_dim, input_dim)))
            self.Wz.append(nn.Parameter(torch.empty(hidden_dim, hidden_dim)))

        self.Wx_out = nn.Parameter(torch.empty(output_dim, input_dim))
        self.Wz_out = nn.Parameter(torch.empty(output_dim, hidden_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        for w in list(self.Wx) + list(self.Wz) + [self.Wx_out, self.Wz_out]:
            nn.init.xavier_uniform_(w)

    def forward(self, x):
        z = self.activation(F.linear(x, self.Wx[0]))

        for i in range(1, len(self.Wx)):
            Wx = F.linear(x, self.gx(self.Wx[i]))
            Wz = F.linear(z, self.gz(self.Wz[i-1]))
            z = self.activation(Wz + Wx)

        Wx_out = F.linear(x, self.gx(self.Wx_out))
        Wz_out = F.linear(z, self.gz(self.Wz_out))

        return self.activation(Wz_out + Wx_out)


class SymmetricICNN(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 3):
        super().__init__()
        self.icnn = SimpleICNN(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers
        )

    def forward(self, x):
        s11, s22, s12 = x[:, 0], x[:, 1], x[:, 2]

        combos = torch.stack([
            torch.stack([+s11, +s22, +s12], dim=1),
            torch.stack([+s11, +s22, -s12], dim=1),
            torch.stack([-s11, -s22, +s12], dim=1),
            torch.stack([-s11, -s22, -s12], dim=1),
        ], dim=1)

        flat = combos.view(-1, 3)
        out = self.icnn(flat)

        return out.view(-1, 4, 1).mean(dim=1)


DEFAULT_POLYNOMIAL_DEGREES = (2, 4, 6, 8)


def enumerate_homogeneous_terms(
    degrees=DEFAULT_POLYNOMIAL_DEGREES,
):
    """Return exponent triples for every requested homogeneous degree."""
    return {
        degree: [
            (i, j, degree - i - j)
            for i in range(degree + 1)
            for j in range(degree + 1 - i)
        ]
        for degree in degrees
    }


def build_monomial_features(stress, term_map):
    """Precompute homogeneous monomials for stress shaped ``(N, 3)``."""
    if stress.ndim != 2 or stress.shape[1] != 3:
        raise ValueError("stress must have shape (N, 3)")
    features = {}
    for degree, terms in term_map.items():
        powers = torch.tensor(
            terms,
            dtype=stress.dtype,
            device=stress.device,
        )
        features[degree] = torch.prod(
            stress.unsqueeze(1) ** powers.unsqueeze(0),
            dim=2,
        )
    return features


def signed_root(value, degree, stabilization_epsilon=0.0):
    """Real signed root, optionally stabilized during optimization."""
    magnitude = torch.abs(value)
    if stabilization_epsilon:
        magnitude = magnitude + float(stabilization_epsilon)
    return torch.sign(value) * magnitude.pow(1.0 / degree)


class MixedHomogeneousPolynomial(nn.Module):
    """Sum of signed roots of even-degree homogeneous polynomials."""

    def __init__(self, degrees=DEFAULT_POLYNOMIAL_DEGREES):
        super().__init__()
        self.degrees = tuple(degrees)
        self.term_map = enumerate_homogeneous_terms(self.degrees)
        self.coefficients = nn.ParameterDict(
            {
                str(degree): nn.Parameter(
                    torch.randn(len(self.term_map[degree]))
                    * 1.0e-3
                )
                for degree in self.degrees
            }
        )

    def radicands_from_features(self, features):
        return {
            degree: (
                features[degree]
                @ self.coefficients[str(degree)]
            )
            for degree in self.degrees
        }

    def evaluate_from_features(
        self,
        features,
        stabilization_epsilon=0.0,
    ):
        radicands = self.radicands_from_features(features)
        result = torch.zeros_like(next(iter(radicands.values())))
        for degree in self.degrees:
            result = result + signed_root(
                radicands[degree],
                degree,
                stabilization_epsilon=stabilization_epsilon,
            )
        return result

    def forward(self, stress):
        features = build_monomial_features(
            stress,
            self.term_map,
        )
        return self.evaluate_from_features(features)


def load_polynomial_model_checkpoint(path, device="cpu"):
    """Reconstruct a frozen polynomial yield model from a checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Polynomial checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    if (
        not isinstance(checkpoint, dict)
        or "model_state_dict" not in checkpoint
        or "degrees" not in checkpoint
    ):
        raise ValueError(
            "Polynomial checkpoint requires model_state_dict and "
            f"degrees: {checkpoint_path}"
        )
    model = MixedHomogeneousPolynomial(
        degrees=tuple(checkpoint["degrees"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def compute_autograd_gradients(output_tensor, input_tensor):
    """
    Computes the Jacobian of output_tensor w.r.t input_tensor efficiently while ensuring identical results.
    Fully vectorized implementation while keeping gradients intact for training.
    """
    input_tensor.requires_grad_(True)

    batch_size, output_dim = output_tensor.shape  # (n, m)
    input_dim = input_tensor.shape[1]  # (n, d)

    # Flatten output tensor for batch-wise processing
    output_flat = output_tensor.view(batch_size, output_dim)  # (n, m)

    # Create grad_outputs as a list of one-hot vectors to compute gradients separately
    grad_outputs = torch.eye(output_dim, dtype=output_tensor.dtype, device=output_tensor.device)  # (m, m)

    # Compute the Jacobian by iterating over output dimensions (keeping gradients intact)
    jacobian_list = []
    for i in range(output_dim):
        grad = torch.autograd.grad(
            outputs=output_flat, 
            inputs=input_tensor, 
            grad_outputs=grad_outputs[i].expand(batch_size, output_dim),  # (n, m)
            create_graph=True, 
            retain_graph=True
        )[0]  # Shape: (n, d)

        jacobian_list.append(grad.unsqueeze(2))  # Shape: (n, d, 1)

    # Concatenate results to form full Jacobian matrix (n, d, m)
    jacobian = torch.cat(jacobian_list, dim=2)

    # Ensure correct reshape to (n, m, d)
    gradients = jacobian.permute(0, 2, 1).contiguous()  # Reshape from (n, d, m) to (n, m, d)

    return gradients
