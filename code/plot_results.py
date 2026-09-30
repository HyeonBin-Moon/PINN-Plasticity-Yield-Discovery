"""Plot an identified neural yield function against its ground truth.

Example
-------
python code/plot_results.py --checkpoint results/noise/VM/noise_0.00e+00/VM_noise0.00e+00_model.pth --yield-model VM

Produces <checkpoint stem>_contours.png (iso-shear yield contours, ground truth solid and NN dashed,
for sigma_xy / sigma_Y = 0, 0.2, 0.3, 0.4, 0.5, 0.55) and <checkpoint stem>_loss.png (loss history).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

head_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, head_path)
os.chdir(head_path)

from utils.common_library import *                                             # noqa: E402
from utils.bench_models import yield_VM, yield_Hill1948, yield_Yld2000_2d, hardening_law   # noqa: E402
from utils.ai_model import SymmetricICNN                                         # noqa: E402
from utils.plot_functions import plot_loss_history                              # noqa: E402
from utils.discovery_plasticity_function import plot_yield_comparison           # noqa: E402

YIELD_MODEL_FUN = {"VM": yield_VM, "Hill": yield_Hill1948, "Yld2000": yield_Yld2000_2d}
LABELS = {"VM": "von Mises", "Hill": "Hill 1948", "Yld2000": "Yld2000-2d"}


def load_model(checkpoint_path, hidden_dim=32, num_layers=4):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    model = SymmetricICNN(hidden_dim=cfg.get("hidden_layer", hidden_dim),
                          num_layers=cfg.get("num_layers", num_layers)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot an identified yield function.")
    parser.add_argument("--checkpoint", required=True, help="path to a <tag>_model.pth file written by train.py")
    parser.add_argument("--yield-model", choices=list(YIELD_MODEL_FUN), required=True)
    parser.add_argument("--output-dir", default=None, help="defaults to the checkpoint folder")
    args = parser.parse_args(argv)

    model, checkpoint = load_model(args.checkpoint)
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0].replace("_model", "")
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)

    contour_path = os.path.join(out_dir, f"{stem}_contours.png")
    plot_yield_comparison(YIELD_MODEL_FUN[args.yield_model], model, hardening_law(0.0),
                          ref_label=LABELS[args.yield_model], save_filepath=contour_path)
    print("saved", contour_path)

    if "loss_history" in checkpoint:
        loss_path = os.path.join(out_dir, f"{stem}_loss.png")
        plot_loss_history(checkpoint["loss_history"], save_path=loss_path)
        print("saved", loss_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
