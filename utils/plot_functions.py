"""Plotting helpers: iso-shear yield contours and loss histories."""

from utils.common_library import *


def plot_yield_surface_2D(yield_fun, ax, sigma_y, s12_fractions=(0.0, 0.2, 0.4, 0.5, 0.55),
                          s_range_mult=1.5, n_pts=300, contour_kwargs=None, label_kwargs=None,
                          normalize=False):
    """Draw the contours sigma_eq = sigma_y in the (sigma_xx, sigma_yy) plane for several shear levels.

    Parameters
    ----------
    yield_fun : callable mapping a (N, 3) stress tensor to (N,) or (N, 1) equivalent stresses
    ax : matplotlib axes
    sigma_y : reference stress used for the contour level and for the shear levels
    s12_fractions : shear stress levels sigma_xy / sigma_y, one contour per level
    normalize : plot the axes as sigma / sigma_y when True
    """
    contour_kwargs = contour_kwargs or dict(linestyles='--', linewidths=1.5)
    s12_vals = np.array(s12_fractions) * sigma_y

    s_range = s_range_mult * sigma_y
    s11 = torch.linspace(-s_range, s_range, n_pts, device=device)
    s22 = torch.linspace(-s_range, s_range, n_pts, device=device)
    S11, S22 = torch.meshgrid(s11, s22, indexing='ij')
    s11_flat, s22_flat = S11.reshape(-1), S22.reshape(-1)
    scale = sigma_y if normalize else 1.0

    for frac, s12 in zip(s12_fractions, s12_vals):
        stress = torch.stack([s11_flat, s22_flat, torch.full_like(s11_flat, s12)], dim=1)
        with torch.no_grad():
            S_eq = yield_fun(stress).detach().cpu().numpy().reshape(n_pts, n_pts)
        cs = ax.contour(S11.cpu().numpy() / scale, S22.cpu().numpy() / scale, S_eq,
                        levels=[sigma_y], **contour_kwargs)
        if label_kwargs is not None:
            ax.clabel(cs, fmt={sigma_y: f"s12={frac:.2f}"}, **label_kwargs)

    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_loss_history(loss_history, save_path=None):
    """Plot the per-case loss histories on a log axis and optionally save the figure."""
    min_length = min(len(values) for values in loss_history.values())

    plt.figure(figsize=(8, 6), dpi=200)
    for key, values in loss_history.items():
        values = np.array(values[:min_length])
        plt.plot(range(len(values)), values, label=key.replace('_', ' '), alpha=0.7)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Force equilibrium loss")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.close()
