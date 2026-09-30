"""Spatiotemporally correlated Gaussian displacement noise (Appendix B of the paper)."""

from utils.common_library import *


def generate_spatiotemporal_correlated_noise(U_clean, coord, noise_mag=0.1e-3, sigma_time=5.0, sigma_space=5.0):
    """Return a noise field with the shape of ``U_clean`` (n_frames, n_nodes, 2).

    1. independent standard normal samples,
    2. Gaussian smoothing along the frame axis (``sigma_time`` frames),
    3. Gaussian smoothing over the nodes with a kernel of width ``sigma_space`` (same unit as ``coord``),
    4. normalisation to zero mean and unit standard deviation per frame and component,
    5. scaling by ``noise_mag`` (same unit as the displacements).
    """
    from scipy.ndimage import gaussian_filter
    from scipy.spatial.distance import cdist

    T, m, d = U_clean.shape
    noise = np.random.randn(T, m, d)
    noise = gaussian_filter(noise, sigma=[sigma_time, 0, 0])

    D = cdist(coord, coord)
    W = np.exp(-D**2 / (2 * sigma_space**2))
    W /= W.sum(axis=1, keepdims=True)
    for t in range(T):
        for i in range(d):
            noise[t, :, i] = W @ noise[t, :, i]

    for t in range(T):
        for i in range(d):
            mean_ti = noise[t, :, i].mean()
            std_ti = noise[t, :, i].std()
            if std_ti > 0:
                noise[t, :, i] = (noise[t, :, i] - mean_ti) / std_ti

    return noise * noise_mag
