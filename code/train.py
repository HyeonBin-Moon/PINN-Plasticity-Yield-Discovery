"""Identify the neural yield function from the FE displacement and reaction force data.

Runs of the paper
-----------------
noise    : one identification per benchmark and displacement noise level (0, 0.1, 0.2, 0.3 um), model seed 1
ensemble : ten identifications per benchmark from the noise-free data with model seeds 1 to 10

Examples
--------
python code/train.py --dry-run
python code/train.py --yield-model VM --study noise --noise 0.0
python code/train.py --yield-model all --study noise ensemble
python code/train.py --yield-model Yld2000 --study noise --noise 0.3 --force

Outputs (per run): <tag>_model.pth (state dict, loss and beta histories), <tag>_yield_surface.png,
<tag>_loss_history.png under results/<study>/<model>/<condition>/.
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

head_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, head_path)
os.chdir(head_path)

from utils.common_library import *                                             # noqa: E402
from utils.bench_models import yield_VM, yield_Hill1948, yield_Yld2000_2d, hardening_law   # noqa: E402
from utils.plot_functions import plot_loss_history                              # noqa: E402
from utils.discovery_plasticity_function import Macro_PINN_train_function, plot_yield_comparison  # noqa: E402

DATA_PATH = os.path.join(head_path, "data")
RESULTS_ROOT = os.path.join(head_path, "results")

YIELD_MODEL_NPZ = {
    "VM": ["Data_Inp_VM_x_tension.npz", "Data_Inp_VM_y_tension.npz", "Data_Inp_VM_shear.npz", "Data_Inp_VM_biaxial.npz"],
    "Hill": ["Data_Inp_Hill_x_tension.npz", "Data_Inp_Hill_y_tension.npz", "Data_Inp_Hill_shear.npz", "Data_Inp_Hill_biaxial.npz"],
    "Yld2000": ["Data_Inp_Yld2000_x_tension.npz", "Data_Inp_Yld2000_y_tension.npz", "Data_Inp_Yld2000_shear.npz", "Data_Inp_Yld2000_biaxial.npz"],
}
YIELD_MODEL_FUN = {"VM": yield_VM, "Hill": yield_Hill1948, "Yld2000": yield_Yld2000_2d}
YIELD_MODEL_LIST = ["VM", "Hill", "Yld2000"]

NOISE_LEVELS_UM = [0.0, 0.1, 0.2, 0.3]     # displacement noise standard deviation in micrometres
ENSEMBLE_SEEDS = list(range(1, 11))

# Settings of Table 1 of the paper
BASE_CONFIG = {
    "fem_dataset_path": DATA_PATH,
    "yield_model_npz": YIELD_MODEL_NPZ,
    "yield_model_fun": YIELD_MODEL_FUN,
    "hidden_layer": 32,
    "num_layers": 4,
    "learning_rate": 1e-3,
    "num_epochs": 2000,
    "use_cases": [True, True, True, True],
    "noise_seed": 42,
    "beta_scheduler_mode": "alpha_decay",   # beta = 4 MPa^-1, transition width multiplied by 0.9 every 200 epochs
    "second_order_every": 4,                # G_c = {4, 8, ..., 100}
    "plot_every_epoch": 100,
    "plot_save_every_epoch": 0,
}


def build_run_specs(yield_models, studies, noise_levels_um, seeds):
    specs = []
    for yield_model in yield_models:
        if "noise" in studies:
            for noise_um in noise_levels_um:
                noise_mag = noise_um * 1e-3
                tag = f"{yield_model}_noise{noise_mag:.2e}"
                specs.append(dict(study="noise", yield_model=yield_model, tag=tag,
                                  save_path=os.path.join(RESULTS_ROOT, "noise", yield_model, f"noise_{noise_mag:.2e}"),
                                  config={**BASE_CONFIG, "yield_model": yield_model, "noise_mag": noise_mag, "model_seed": 1}))
        if "ensemble" in studies:
            for seed in seeds:
                tag = f"{yield_model}_seed{seed:02d}"
                specs.append(dict(study="ensemble", yield_model=yield_model, tag=tag,
                                  save_path=os.path.join(RESULTS_ROOT, "ensemble", yield_model, f"seed_{seed:02d}"),
                                  config={**BASE_CONFIG, "yield_model": yield_model, "noise_mag": 0.0, "model_seed": seed}))
    return specs


def run_one(spec, force=False):
    save_path, tag, yield_model = spec["save_path"], spec["tag"], spec["yield_model"]
    model_path = os.path.join(save_path, f"{tag}_model.pth")
    if os.path.isfile(model_path) and not force:
        print(f"SKIP {tag} (checkpoint exists, use --force to retrain)")
        return

    print("=" * 70)
    print(f"RUN {tag}  ->  {os.path.relpath(save_path, head_path)}")
    print("=" * 70)
    start = time.time()
    S_eq_net, loss_history, beta_history = Macro_PINN_train_function({**spec["config"], "save_path": save_path})

    torch.save({"model_state_dict": S_eq_net.state_dict(),
                "loss_history": loss_history,
                "beta_history": beta_history,
                "config": {k: v for k, v in spec["config"].items() if k not in ("yield_model_npz", "yield_model_fun")}},
               model_path)
    plot_yield_comparison(YIELD_MODEL_FUN[yield_model], S_eq_net, hardening_law(0.0), ref_label=yield_model,
                          save_filepath=os.path.join(save_path, f"{tag}_yield_surface.png"))
    plot_loss_history(loss_history, save_path=os.path.join(save_path, f"{tag}_loss_history.png"))
    print(f"DONE {tag} in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

    del S_eq_net, loss_history, beta_history
    plt.close("all")
    clear_memory()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Neural yield function identification.")
    parser.add_argument("--yield-model", choices=YIELD_MODEL_LIST + ["all"], default="all")
    parser.add_argument("--study", nargs="+", choices=["noise", "ensemble"], default=["noise"])
    parser.add_argument("--noise", type=float, nargs="+", default=NOISE_LEVELS_UM,
                        help="noise standard deviations in micrometres for the noise study")
    parser.add_argument("--seeds", type=int, nargs="+", default=ENSEMBLE_SEEDS,
                        help="model seeds for the ensemble study")
    parser.add_argument("--epochs", type=int, default=BASE_CONFIG["num_epochs"])
    parser.add_argument("--force", action="store_true", help="retrain and overwrite existing checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="list the selected runs without training")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    BASE_CONFIG["num_epochs"] = args.epochs
    yield_models = YIELD_MODEL_LIST if args.yield_model == "all" else [args.yield_model]
    specs = build_run_specs(yield_models, set(args.study), args.noise, args.seeds)

    print(f"device: {device}")
    print(f"planned runs: {len(specs)}")
    if args.dry_run:
        for i, spec in enumerate(specs, 1):
            print(f"{i:02d}. {spec['study']:8s} {spec['tag']:28s} -> {os.path.relpath(spec['save_path'], head_path)}")
        return 0

    for spec in specs:
        run_one(spec, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
