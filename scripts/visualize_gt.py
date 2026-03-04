import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    p = argparse.ArgumentParser("Visualize occupancy GT (.npz).")
    p.add_argument("--occ_dir", type=str, required=True,
                   help="Directory with *.npz occupancy files")
    p.add_argument("--index", type=int, default=0,
                   help="Which file index to visualize (sorted by name)")
    p.add_argument("--save_path", type=str, default=None,
                   help="If set, save figure instead of showing it")
    return p.parse_args()

def main():
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.occ_dir, "*.npz")))
    if not files:
        raise RuntimeError(f"No .npz files found in {args.occ_dir}")

    idx = max(0, min(args.index, len(files) - 1))
    path = files[idx]
    print(f"Loading {path}")

    data = np.load(path, allow_pickle=True)
    occ = data["occupancy"]  # [H,W] or [D,H,W]

    # If 3D, collapse to BEV
    if occ.ndim == 3:
        occ_bev = occ.max(axis=0)
    else:
        occ_bev = occ

    plt.figure(figsize=(6, 6))
    plt.imshow(occ_bev, origin="lower", cmap="viridis")
    plt.colorbar(label="occupied")
    plt.title(os.path.basename(path))
    plt.axis("equal")

    if args.save_path:
        plt.savefig(args.save_path, bbox_inches="tight")
        print(f"Saved to {args.save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()