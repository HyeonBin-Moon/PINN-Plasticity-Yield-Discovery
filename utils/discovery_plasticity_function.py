"""Physics-informed identification of the neural yield function (Sections 2.2 and 2.3 of the paper).

The convex neural network ``SymmetricICNN`` is embedded in an explicit forward Euler stress update with
a sigmoid plasticity indicator. Strain increments computed from the FE displacement fields drive the
update, the resulting stress increments are assembled into internal nodal force increments, and the
network is trained by minimising the force equilibrium residual: the summed internal force increments
on each displacement-controlled boundary must match the measured reaction force increments, and the
internal force increments at all other nodes must vanish.
"""

from utils.common_library import *
from utils.bench_models import hardening_law
from utils.load_dataset import get_data_from_inp, get_data_from_npz
from utils.plot_functions import plot_yield_surface_2D, plot_loss_history
from utils.ai_model import SymmetricICNN, compute_autograd_gradients
from utils.spatiotemporal_noise import generate_spatiotemporal_correlated_noise
from utils.beta_scheduler import BetaScheduler


# Elastic constants shared by the three benchmarks (MPa)
E_MODULUS = 69e3
POISSON_RATIO = 0.3


def load_case_data(npz_file, fem_path):
    """Load the mesh (.inp) and the displacement/reaction force history (.npz) of one loading case."""
    inp_file = npz_file[5:-4] + ".inp"
    v, f, intp_tps, element_area, B_matrices, bot_index, top_index, right_index, left_index = \
        get_data_from_inp(os.path.join(fem_path, inp_file))
    U_sol, E_sol, _, _, _, _, bot_RF_sum, top_RF_sum, right_RF_sum, left_RF_sum = \
        get_data_from_npz(os.path.join(fem_path, npz_file), f, B_matrices)
    mesh = dict(v=v, f=f, intp_tps=intp_tps, element_area=element_area, B_matrices=B_matrices,
                bot_index=bot_index, top_index=top_index, right_index=right_index, left_index=left_index)
    field = dict(U_sol=U_sol, E_sol=E_sol, bot_RF_sum=bot_RF_sum, top_RF_sum=top_RF_sum,
                 right_RF_sum=right_RF_sum, left_RF_sum=left_RF_sum)
    return mesh, field


def compute_strain_from_disp(U_sol, f_np, B_matrices_np):
    """Element strains [eps_xx, eps_yy, gamma_xy] from nodal displacements."""
    U_temp = np.concatenate((U_sol[:, f_np, :][:, :, :, 0], U_sol[:, f_np, :][:, :, :, 1]), axis=2)
    return np.einsum('nij,tnj->tni', B_matrices_np, U_temp)


def add_noise_and_strain(field, v, f_np, B_matrices_np, noise_mag, noise_seed):
    """Add correlated displacement noise (in place) and recompute the strains."""
    set_seed(noise_seed)
    noise = generate_spatiotemporal_correlated_noise(U_clean=field['U_sol'].copy(), coord=v, noise_mag=noise_mag)
    field['U_sol'] = field['U_sol'] + noise
    field['E_sol'] = compute_strain_from_disp(field['U_sol'], f_np, B_matrices_np)
    return noise


def compute_increments(field):
    """Increments between consecutive sampled frames of every field."""
    keys = ['U_sol', 'E_sol', 'bot_RF_sum', 'top_RF_sum', 'right_RF_sum', 'left_RF_sum']
    return {'d' + k: field[k][1:] - field[k][:-1] for k in keys}


def increments_to_torch(dinc):
    return {k: np_to_torch(v).requires_grad_(False) for k, v in dinc.items()}


def build_L_e_transform(bc_nodes_x, bc_nodes_y, interior_x, interior_y, n_v, device):
    """Operator L_c of Eq. (25): two rows sum the nodal forces of the two displacement-controlled
    boundaries of a direction, the remaining rows select the free nodes of that direction."""
    def _build(bc_nodes, interior_nodes):
        n_int = len(interior_nodes)
        L = torch.zeros((2 + n_int, n_v), dtype=torch.float32, device=device)
        L[0, bc_nodes[0]] = 1.0
        L[1, bc_nodes[1]] = 1.0
        L[torch.arange(n_int, device=device) + 2, interior_nodes] = 1.0
        return L

    Lx = _build(bc_nodes_x, interior_x)
    Ly = _build(bc_nodes_y, interior_y)
    m, n = Lx.shape
    p, q = Ly.shape
    Lxy = torch.zeros((m + p, n + q), dtype=torch.float32, device=device)
    Lxy[:m, :n] = Lx
    Lxy[m:, n:] = Ly
    return Lxy


def plot_yield_comparison(yield_fun_ref, S_eq_net, sigma_y, ref_label, save_filepath=None):
    """Ground truth versus identified iso-shear yield contours, normalised by sigma_y."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    levels = (0.0, 0.2, 0.3, 0.4, 0.5, 0.55)
    plot_yield_surface_2D(yield_fun_ref, ax, sigma_y, s12_fractions=levels, normalize=True,
                          contour_kwargs=dict(colors='tab:blue', linestyles='-', linewidths=1.6))
    plot_yield_surface_2D(S_eq_net, ax, sigma_y, s12_fractions=levels, normalize=True,
                          contour_kwargs=dict(colors='tab:red', linestyles='--', linewidths=1.6, alpha=0.95))
    ax.set_xlabel(r"$\sigma_{xx}/\sigma_Y$")
    ax.set_ylabel(r"$\sigma_{yy}/\sigma_Y$")
    ax.set_title(f"{ref_label}: contours for $\\sigma_{{xy}}/\\sigma_Y$ = {', '.join(str(l) for l in levels)}")
    ax.legend(handles=[Line2D([0], [0], color='tab:blue', lw=1.6, ls='-', label='Ground truth'),
                       Line2D([0], [0], color='tab:red', lw=1.6, ls='--', label='NN')],
              loc='upper left', frameon=False)
    plt.tight_layout()
    if save_filepath:
        plt.savefig(save_filepath, dpi=200)
    plt.close(fig)


def Macro_PINN_train_function(config: dict):
    """Identify the neural yield function for one benchmark and one noise level.

    config keys
    -----------
    fem_dataset_path : folder with the Data_Inp_*.npz and Inp_*.inp files
    save_path        : output folder for the checkpoint and figures
    yield_model_npz  : dict model name -> list of the four npz files (UTx, UTy, SS, BT)
    yield_model_fun  : dict model name -> ground truth yield function (for the progress plots)
    yield_model      : "VM" | "Hill" | "Yld2000"
    hidden_layer, num_layers, learning_rate, num_epochs
    noise_mag        : displacement noise standard deviation [mm]
    use_cases        : list of four booleans, loss weights 1 or 0 for UTx, UTy, SS, BT
    noise_seed, model_seed : seeds for the noise realisation and for the network initialisation
    beta_scheduler_mode    : "alpha_decay" reproduces the sharpness schedule of Algorithm 2
    second_order_every     : G_c, the derivative of the flow direction is retained at every k-th increment
    plot_every_epoch       : progress printing and plotting interval
    plot_save_every_epoch  : interval for saving the evolving yield surface (0 disables)

    Returns
    -------
    S_eq_net, loss_history (dict of the four case losses), beta_history (dict with the beta values)
    """
    fem_dataset_path = config["fem_dataset_path"]
    save_path = config["save_path"]
    yield_model_npz = config["yield_model_npz"]
    yield_model_fun = config["yield_model_fun"]
    yield_model = config["yield_model"]
    hidden_layer = config["hidden_layer"]
    num_layers = config["num_layers"]
    learning_rate = config["learning_rate"]
    noise_mag = config["noise_mag"]
    num_epochs = config["num_epochs"]
    plot_every_epoch = config.get("plot_every_epoch", 100)
    plot_save_every_epoch = config.get("plot_save_every_epoch", 0)
    use_cases = config.get("use_cases", [True, True, True, True])
    noise_seed = config["noise_seed"]
    model_seed = config["model_seed"]
    beta_scheduler_mode = config.get("beta_scheduler_mode", "alpha_decay")
    second_order_every = config.get("second_order_every", 4)

    assert os.path.isdir(fem_dataset_path), f"Missing directory: {fem_dataset_path}"
    os.makedirs(save_path, exist_ok=True)

    # ---------------------------------------------------------------- data
    cases = []
    for npz_file in yield_model_npz[yield_model]:
        mesh, field = load_case_data(npz_file, fem_dataset_path)
        cases.append({'mesh': mesh, 'field': field})

    # the four loading cases share the same mesh
    v = cases[0]['mesh']['v']
    f_np = cases[0]['mesh']['f']
    element_area_np = cases[0]['mesh']['element_area']
    B_matrices_np = cases[0]['mesh']['B_matrices']
    bot_index_np = cases[0]['mesh']['bot_index']
    top_index_np = cases[0]['mesh']['top_index']
    right_index_np = cases[0]['mesh']['right_index']
    left_index_np = cases[0]['mesh']['left_index']

    for c in cases:
        add_noise_and_strain(c['field'], v, f_np, B_matrices_np, noise_mag, noise_seed)
    print(f"[Noise] standard deviation = {noise_mag:.3e} mm")

    # ---------------------------------------------------------------- precomputation
    C_el = torch.zeros((3, 3), dtype=torch.float32, device=device, requires_grad=False)
    C_el[0, 0] = 1.0
    C_el[1, 1] = 1.0
    C_el[0, 1] = POISSON_RATIO
    C_el[1, 0] = POISSON_RATIO
    C_el[2, 2] = (1.0 - POISSON_RATIO) / 2.0
    C_el *= E_MODULUS / (1.0 - POISSON_RATIO**2)

    f = np_to_torch(f_np).to(dtype=torch.long, device=device)
    v = np_to_torch(v)
    element_area = np_to_torch(element_area_np)
    B_matrices = np_to_torch(B_matrices_np)

    top_index = torch.tensor(top_index_np, dtype=torch.long, device=device)
    bot_index = torch.tensor(bot_index_np, dtype=torch.long, device=device)
    left_index = torch.tensor(left_index_np, dtype=torch.long, device=device)
    right_index = torch.tensor(right_index_np, dtype=torch.long, device=device)

    n_v = len(v)
    n_f = len(f)

    # element-to-global assembly operator
    L_e = torch.zeros((n_f, 2 * n_v, 6), dtype=torch.float32, device=device)
    row_idx = torch.arange(n_f, device=device).unsqueeze(1).repeat(1, 3)
    col_idx_1 = f
    col_idx_2 = f + n_v
    feat_idx_1 = torch.tensor([0, 1, 2], device=device).repeat(n_f, 1)
    feat_idx_2 = torch.tensor([3, 4, 5], device=device).repeat(n_f, 1)
    L_e[row_idx, col_idx_1, feat_idx_1] = 1.0
    L_e[row_idx, col_idx_2, feat_idx_2] = 1.0

    total_index = torch.arange(len(v), dtype=torch.long, device=device)

    def case_operator(bc_x, bc_y):
        """Assembly operator of one loading case for the displacement-controlled boundaries bc_x (x direction)
        and bc_y (y direction); all other nodes of a direction are free nodes."""
        bc_nodes_x = torch.unique(torch.cat(bc_x))
        bc_nodes_y = torch.unique(torch.cat(bc_y))
        interior_x = total_index[~torch.isin(total_index, bc_nodes_x)]
        interior_y = total_index[~torch.isin(total_index, bc_nodes_y)]
        return torch.einsum("eni,mn->emi", L_e,
                            build_L_e_transform(bc_x, bc_y, interior_x, interior_y, n_v, device))

    L_e_case1 = case_operator([left_index, right_index], [left_index, right_index])   # UTx
    L_e_case2 = case_operator([top_index, bot_index], [top_index, bot_index])         # UTy
    L_e_case3 = case_operator([top_index, bot_index], [top_index, bot_index])         # SS
    L_e_case4 = case_operator([left_index, right_index], [top_index, bot_index])      # BT
    del L_e, total_index, row_idx, col_idx_1, col_idx_2, feat_idx_1, feat_idx_2

    # ---------------------------------------------------------------- increments
    dinc = [increments_to_torch(compute_increments(c['field'])) for c in cases]
    del cases

    dE_sol_case1 = dinc[0]['dE_sol']
    dE_sol_case2 = dinc[1]['dE_sol']
    dE_sol_case3 = dinc[2]['dE_sol']
    dE_sol_case4 = dinc[3]['dE_sol']

    n_t = dE_sol_case1.shape[0]

    def external_force(L_e_case, rf_x, rf_y):
        """Measured reaction force increments in the layout of L_e_case: the two boundary sums of each
        direction followed by zeros for the free nodes."""
        half = L_e_case.shape[1] // 2
        zeros_pad = torch.zeros((n_t, half - 2), device=device)
        dRFx = torch.cat([rf_x[0][:, :1], rf_x[1][:, :1], zeros_pad], dim=1)
        dRFy = torch.cat([rf_y[0][:, 1:2], rf_y[1][:, 1:2], zeros_pad], dim=1)
        return torch.cat([dRFx, dRFy], dim=1)

    ext_dRF_case1 = external_force(L_e_case1, (dinc[0]['dleft_RF_sum'], dinc[0]['dright_RF_sum']),
                                   (dinc[0]['dleft_RF_sum'], dinc[0]['dright_RF_sum']))
    ext_dRF_case2 = external_force(L_e_case2, (dinc[1]['dtop_RF_sum'], dinc[1]['dbot_RF_sum']),
                                   (dinc[1]['dtop_RF_sum'], dinc[1]['dbot_RF_sum']))
    ext_dRF_case3 = external_force(L_e_case3, (dinc[2]['dtop_RF_sum'], dinc[2]['dbot_RF_sum']),
                                   (dinc[2]['dtop_RF_sum'], dinc[2]['dbot_RF_sum']))
    ext_dRF_case4 = external_force(L_e_case4, (dinc[3]['dleft_RF_sum'], dinc[3]['dright_RF_sum']),
                                   (dinc[3]['dtop_RF_sum'], dinc[3]['dbot_RF_sum']))
    del dinc

    for t in [L_e_case1, L_e_case2, L_e_case3, L_e_case4,
              ext_dRF_case1, ext_dRF_case2, ext_dRF_case3, ext_dRF_case4, element_area]:
        t.requires_grad_(False)

    # ---------------------------------------------------------------- stress update and loss
    def solve_plastic_explicit(yield_fun, dE, C_el, beta):
        """Explicit forward Euler stress update with the sigmoid plasticity indicator (Eqs. (13)-(22)).

        Returns the stress increments (T, n_f, 3) and equivalent plastic strain increments (T, n_f, 1).
        The derivative of the flow direction with respect to the network parameters is retained only at
        every ``second_order_every``-th increment and at the last one; elsewhere it is detached.
        """
        T, n_f = dE.shape[:2]
        device = dE.device

        accum_S = torch.zeros(n_f, 3, device=device)
        accum_PE = torch.zeros(n_f, 1, device=device)
        dS_list, dPE_list = [], []

        for t in range(T):
            dE_t = dE[t]

            if t == 0:
                delta_S = dE_t @ C_el.T
                delta_PE = torch.zeros_like(accum_PE)
            else:
                S_leaf = accum_S.detach().clone().requires_grad_()
                PE_leaf = accum_PE.detach().clone().requires_grad_()

                S_trial = S_leaf + dE_t @ C_el.T
                indicator = torch.sigmoid(beta * (yield_fun(S_trial) - hardening_law(PE_leaf)))

                S_eq = yield_fun(S_leaf)
                R = hardening_law(PE_leaf)

                use_2nd = (t % second_order_every == 0) or (t == T - 1)
                grad_S = torch.autograd.grad(S_eq.sum(), S_leaf, create_graph=use_2nd)[0]
                grad_R = torch.autograd.grad(R.sum(), PE_leaf, create_graph=use_2nd)[0]
                if not use_2nd:
                    grad_S = grad_S.detach()
                    grad_R = grad_R.detach()

                CgS = grad_S @ C_el
                num = (CgS * dE_t).sum(dim=1, keepdim=True)
                den = grad_R + (CgS * grad_S).sum(dim=1, keepdim=True)
                delta_PE = (num / den * indicator).clamp(min=0.0)
                delta_S = (dE_t - delta_PE * grad_S) @ C_el.T

            # increments that carry a graph stay on the device, the others are parked on the CPU
            if t > 0 and use_2nd:
                dS_list.append(delta_S)
                dPE_list.append(delta_PE)
            else:
                dS_list.append(delta_S.cpu())
                dPE_list.append(delta_PE.cpu())
            accum_S = (accum_S + delta_S).detach()
            accum_PE = (accum_PE + delta_PE).detach()

        dS = torch.stack([x.to(device) for x in dS_list], 0)
        dPE_eq = torch.stack([x.to(device) for x in dPE_list], 0)
        return dS, dPE_eq

    def residual_loss(dS_pred, L_e_case, ext_dRF):
        """Force equilibrium residual of Eq. (26): assembled internal force increments minus the
        measured increments, summed over all rows and sampled increments."""
        BdS_pred = torch.einsum("ein, tei, e -> ten", B_matrices, dS_pred, element_area)
        int_dRF_pred = torch.einsum("eni,tei->tn", L_e_case, BdS_pred)
        return torch.sum((int_dRF_pred - ext_dRF) ** 2)

    def closure(beta):
        dS1, _ = solve_plastic_explicit(S_eq_net, dE_sol_case1, C_el, beta)
        dS2, _ = solve_plastic_explicit(S_eq_net, dE_sol_case2, C_el, beta)
        dS3, _ = solve_plastic_explicit(S_eq_net, dE_sol_case3, C_el, beta)
        dS4, _ = solve_plastic_explicit(S_eq_net, dE_sol_case4, C_el, beta)
        return (residual_loss(dS1, L_e_case1, ext_dRF_case1),
                residual_loss(dS2, L_e_case2, ext_dRF_case2),
                residual_loss(dS3, L_e_case3, ext_dRF_case3),
                residual_loss(dS4, L_e_case4, ext_dRF_case4))

    # ---------------------------------------------------------------- training
    set_seed(model_seed)
    S_eq_net = SymmetricICNN(hidden_dim=hidden_layer, num_layers=num_layers).to(device)

    loss_history = {"loss_case1": [], "loss_case2": [], "loss_case3": [], "loss_case4": []}
    beta_history = {"beta": []}
    case_weights = [1.0 if u else 0.0 for u in use_cases]

    start_time = time.time()
    beta_scheduler = BetaScheduler(mode=beta_scheduler_mode)
    optimizer = torch.optim.Adam(S_eq_net.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        beta = beta_scheduler.get_beta(epoch)

        optimizer.zero_grad()
        loss_case1, loss_case2, loss_case3, loss_case4 = closure(beta)
        loss = (loss_case1 * case_weights[0] + loss_case2 * case_weights[1] +
                loss_case3 * case_weights[2] + loss_case4 * case_weights[3])
        loss.backward()
        optimizer.step()

        beta_scheduler.update(loss.item(), epoch)

        loss_history["loss_case1"].append(loss_case1.item())
        loss_history["loss_case2"].append(loss_case2.item())
        loss_history["loss_case3"].append(loss_case3.item())
        loss_history["loss_case4"].append(loss_case4.item())
        beta_history["beta"].append(beta)

        if plot_every_epoch and epoch % plot_every_epoch == 0:
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"[{epoch}/{num_epochs}] loss = {loss:.4e} | beta = {beta:.2f} MPa^-1 | elapsed = {elapsed_str}")

        if plot_save_every_epoch and epoch % plot_save_every_epoch == 0:
            evolution_dir = os.path.join(save_path, "evolution_history")
            os.makedirs(evolution_dir, exist_ok=True)
            plot_yield_comparison(yield_model_fun[yield_model], S_eq_net, hardening_law(0.0), ref_label=yield_model,
                                  save_filepath=os.path.join(evolution_dir, f"yield_surface_epoch_{epoch:05d}.png"))

    return S_eq_net, loss_history, beta_history
