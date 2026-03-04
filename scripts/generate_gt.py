import argparse
import os

from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from src.data.occupancy_gt import OccupancyGridConfig, OccupancyGTGenerator

def parse_args():
    parser = argparse.ArgumentParser("Generate occupancy GT from nuScenes.")
    parser.add_argument("--nusc_root", type=str, required=True,
                        help="Path to nuScenes root (contains samples/, sweeps/, maps/, v1.0-*)")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        help="nuScenes version string, e.g. v1.0-mini")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for .npz occupancy files")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to process (from start of mini split)")
    parser.add_argument("--x_lim", type=float, nargs=2, default=[-30.0, 30.0])
    parser.add_argument("--y_lim", type=float, nargs=2, default=[-30.0, 30.0])
    parser.add_argument("--z_lim", type=float, nargs=2, default=[-3.0, 5.0])
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--nsweeps", type=int, default=5)
    parser.add_argument("--bev", action="store_true",
                        help="If set, generate BEV [H,W] grids; otherwise full 3D [D,H,W].")
    return parser.parse_args()

def main():
    args = parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=True)

    grid_cfg = OccupancyGridConfig(
        x_lim=tuple(args.x_lim),
        y_lim=tuple(args.y_lim),
        z_lim=tuple(args.z_lim),
        resolution=args.resolution,
        bev=args.bev or True,  # default to BEV for now
    )

    generator = OccupancyGTGenerator(
        nusc=nusc,
        grid_cfg=grid_cfg,
        nsweeps=args.nsweeps,
    )

    tokens = [s["token"] for s in nusc.sample][: args.num_samples]

    os.makedirs(args.out_dir, exist_ok=True)

    for token in tqdm(tokens, desc="Generating occupancy GT"):
        generator.save_sample(token, args.out_dir)

if __name__ == "__main__":
    main()
