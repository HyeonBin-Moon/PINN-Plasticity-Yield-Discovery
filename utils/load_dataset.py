"""Readers for the Abaqus mesh (.inp) and the extracted full-field data (.npz)."""

from utils.common_library import *


def _block(inp_data, start_keyword, end_keyword):
    pattern = re.compile(f'{start_keyword}(.*?){end_keyword}', re.DOTALL)
    return pattern.findall(inp_data)[0].split("\n")[1:-1]


def _nset(inp_data, name):
    lines = _block(inp_data, rf'\*Nset, nset={name}, instance=Part-1-1',
                   rf'\*Elset, elset={name}, instance=Part-1-1')
    return np.array([int(s.strip()) for line in lines for s in line.split(',') if s.strip()])


def get_data_from_inp(inp_file):
    """Parse nodes, CPS3 elements and the four boundary node sets from an Abaqus input file.

    Returns
    -------
    v : (n_nodes, 2) nodal coordinates [mm]
    f : (n_elements, 3) zero-based connectivity
    intp_tps : (n_elements, 2) element centroids
    element_area : (n_elements,)
    B_matrices : (n_elements, 3, 6) constant strain-displacement matrices, ordered [u_x(3 nodes), u_y(3 nodes)]
    bot_index, top_index, right_index, left_index : zero-based node indices of the four edges
    """
    with open(inp_file, 'r') as file:
        inp_data = file.read()

    lines = _block(inp_data, r'\*Node', r'\*Element, type=CPS3')
    nodes = np.array([list(map(float, i.split(','))) for i in lines])

    lines = _block(inp_data, r'\*Element, type=CPS3', r'\*Nset, nset=Set-1, generate')
    elements = np.array([list(map(int, i.split(','))) for i in lines])

    v = nodes[:, 1:].copy()
    f = elements[:, 1:].astype(int) - 1

    bot_index = _nset(inp_data, 'Bot') - 1
    top_index = _nset(inp_data, 'Top') - 1
    right_index = _nset(inp_data, 'Right') - 1
    left_index = _nset(inp_data, 'Left') - 1

    # Constant strain triangle B matrices
    x, y = v[f][:, :, 0], v[f][:, :, 1]
    a = 0.5 * np.abs(x[:, 0] * (y[:, 1] - y[:, 2]) + x[:, 1] * (y[:, 2] - y[:, 0]) + x[:, 2] * (y[:, 0] - y[:, 1]))
    d = 1.0 / (2 * a)
    dx = np.stack([(y[:, 1] - y[:, 2]), (y[:, 2] - y[:, 0]), (y[:, 0] - y[:, 1])], axis=1) * d[:, None]
    dy = np.stack([(x[:, 2] - x[:, 1]), (x[:, 0] - x[:, 2]), (x[:, 1] - x[:, 0])], axis=1) * d[:, None]
    B = np.zeros((len(f), 3, 6))
    B[:, 0, :3], B[:, 1, 3:], B[:, 2, :3], B[:, 2, 3:] = dx, dy, dy, dx

    intp_tps = v[f].mean(1)
    return v, f, intp_tps, a, B, bot_index, top_index, right_index, left_index


def get_data_from_npz(npz_file, f, B_matrices):
    """Load the FE displacement and reaction force history of one loading case.

    The training data files contain only ``displacements`` and ``rf_bot/top/right/left``
    (first column: node label, then the x and y components). Files that also contain the
    stress and state-variable fields written by the FE data extractor are accepted; those
    fields are returned for validation purposes and are None otherwise.

    Returns
    -------
    U_sol : (n_frames, n_nodes, 2) nodal displacements [mm]
    E_sol : (n_frames, n_elements, 3) strains [eps_xx, eps_yy, gamma_xy] from the B matrices
    PE_sol, PE_eq_sol, S_sol, S_eq_sol : validation fields or None
    bot_RF_sum, top_RF_sum, right_RF_sum, left_RF_sum : (n_frames, 2) total reaction forces [N]
    """
    data = np.load(npz_file)

    disp = data['displacements'][:, :, 1:]
    bot_rf = data['rf_bot'][:, :, 1:]
    top_rf = data['rf_top'][:, :, 1:]
    right_rf = data['rf_right'][:, :, 1:]
    left_rf = data['rf_left'][:, :, 1:]

    U_temp = np.concatenate((disp[:, f, :][:, :, :, 0], disp[:, f, :][:, :, :, 1]), axis=2)
    E_sol = np.einsum('nij,tnj->tni', B_matrices, U_temp)
    U_sol = disp.copy()

    PE_sol = PE_eq_sol = S_sol = S_eq_sol = None
    if 'stress' in data.files:
        stress = data['stress'][:, :, 1:]
        S_sol = np.stack((stress[:, :, 0], stress[:, :, 1], stress[:, :, 3]), axis=2)
        PE_sol = np.stack((data['sdv1'], data['sdv2'], data['sdv3'] * np.sqrt(2)), axis=2)
        PE_eq_sol = data['sdv4'].copy()
        S_eq_sol = data['sdv5'].copy()

    return (U_sol, E_sol, PE_sol, PE_eq_sol, S_sol, S_eq_sol,
            bot_rf.sum(axis=1), top_rf.sum(axis=1), right_rf.sum(axis=1), left_rf.sum(axis=1))
