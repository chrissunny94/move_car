import os
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

@dataclass
class OccupancyGridConfig:
    x_lim: Tuple[float, float] = (-30.0, 30.0)   # meters, forward/back
    y_lim: Tuple[float, float] = (-30.0, 30.0)   # meters, left/right
    z_lim: Tuple[float, float] = (-3.0, 5.0)     # meters, up/down
    resolution: float = 0.2                      # meters per cell
    bev: bool = True                             # collapse Z if True

class OccupancyGTGenerator:
    """
    Generate occupancy grids from nuScenes LIDAR_TOP in the ego-vehicle frame.
    """

    def __init__(self,
                 nusc: NuScenes,
                 grid_cfg: OccupancyGridConfig,
                 nsweeps: int = 5,
                 min_dist: float = 1.0) -> None:
        self.nusc = nusc
        self.grid_cfg = grid_cfg
        self.nsweeps = nsweeps
        self.min_dist = min_dist

        self.xmin, self.xmax = grid_cfg.x_lim
        self.ymin, self.ymax = grid_cfg.y_lim
        self.zmin, self.zmax = grid_cfg.z_lim
        self.res = grid_cfg.resolution

        self.nx = int(np.round((self.xmax - self.xmin) / self.res))
        self.ny = int(np.round((self.ymax - self.ymin) / self.res))
        self.nz = int(np.round((self.zmax - self.zmin) / self.res))

    # ----- point extraction -----

    def _get_points_ego(self, sample_token: str) -> np.ndarray:
        """
        Aggregate multiple LIDAR_TOP sweeps and return points in the ego frame.
        Returns: np.ndarray of shape (N, 3) with (x, y, z) in meters.
        """
        sample = self.nusc.get("sample", sample_token)
        pc, _ = LidarPointCloud.from_file_multisweep(
            self.nusc,
            sample_rec=sample,
            chan="LIDAR_TOP",
            ref_chan="LIDAR_TOP",
            nsweeps=self.nsweeps,
            min_distance=self.min_dist,
        )
        pts = pc.points[:3, :].T  # (N, 3)
        return pts

    # ----- occupancy computation -----

    def generate_for_sample(self, sample_token: str) -> np.ndarray:
        """
        Compute occupancy grid for one sample.
        Returns:
          - BEV: [H, W] uint8 (0/1)
          - 3D:  [D, H, W] uint8 (0/1)
        """
        pts = self._get_points_ego(sample_token)

        # filter to region of interest
        mask = (
            (pts[:, 0] >= self.xmin) & (pts[:, 0] < self.xmax) &
            (pts[:, 1] >= self.ymin) & (pts[:, 1] < self.ymax) &
            (pts[:, 2] >= self.zmin) & (pts[:, 2] < self.zmax)
        )
        pts = pts[mask]

        if pts.shape[0] == 0:
            if self.grid_cfg.bev:
                return np.zeros((self.ny, self.nx), dtype=np.uint8)
            else:
                return np.zeros((self.nz, self.ny, self.nx), dtype=np.uint8)

        # discretize to grid indices
        ix = ((pts[:, 0] - self.xmin) / self.res).astype(np.int32)
        iy = ((pts[:, 1] - self.ymin) / self.res).astype(np.int32)
        iz = ((pts[:, 2] - self.zmin) / self.res).astype(np.int32)

        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        iz = np.clip(iz, 0, self.nz - 1)

        if self.grid_cfg.bev:
            grid = np.zeros((self.ny, self.nx), dtype=np.uint8)
            # y → row, x → col
            grid[iy, ix] = 1
        else:
            grid = np.zeros((self.nz, self.ny, self.nx), dtype=np.uint8)
            grid[iz, iy, ix] = 1

        return grid

    # ----- disk I/O -----

    def save_sample(self, sample_token: str, out_dir: str) -> str:
        """
        Generate and save occupancy grid as a compressed .npz.
        """
        os.makedirs(out_dir, exist_ok=True)
        occ = self.generate_for_sample(sample_token)
        out_path = os.path.join(out_dir, f"{sample_token}.npz")
        np.savez_compressed(
            out_path,
            occupancy=occ.astype(np.uint8),
            sample_token=sample_token,
            grid_cfg=asdict(self.grid_cfg),
        )
        return out_path
