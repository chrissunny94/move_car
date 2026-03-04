import os
from typing import Any, Dict

from flask import Flask, jsonify, request
import numpy as np
import torch

from nuscenes.nuscenes import NuScenes

from src.data.occupancy_gt import OccupancyGridConfig, OccupancyGTGenerator
from src.models.occnet import SimpleOccNet

def create_app() -> Flask:
    app = Flask(__name__)

    # --------- Global, long-lived objects ---------
    nusc_root = os.environ.get("NUSCENES_ROOT", "/workspace/DATA/nuscenes")
    nusc_version = os.environ.get("NUSCENES_VERSION", "v1.0-mini")

    nusc = NuScenes(version=nusc_version, dataroot=nusc_root, verbose=False)

    grid_cfg = OccupancyGridConfig(
        x_lim=(-30.0, 30.0),
        y_lim=(-30.0, 30.0),
        z_lim=(-3.0, 5.0),
        resolution=0.2,
        bev=True,
    )
    gt_generator = OccupancyGTGenerator(
        nusc=nusc,
        grid_cfg=grid_cfg,
        nsweeps=5,
        min_dist=1.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleOccNet(in_channels=1, hidden_channels=32).to(device)
    model.eval()  # inference mode

    # --------- Helpers ---------

    def run_model_on_occ(occ: np.ndarray) -> Dict[str, Any]:
        """
        occ: [H, W] or [D, H, W] uint8/float
        Returns basic inference summary.
        """
        if occ.ndim == 3:
            occ = occ.max(axis=0)  # collapse Z for now

        x = occ.astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

        with torch.no_grad():
            y_pred = model(x)  # [1,H,W]
        y_pred = y_pred.squeeze(0).cpu().numpy()  # [H,W]

        # simple threshold for visualization / debugging
        occupied_pred = np.argwhere(y_pred > 0.5)

        return {
            "pred_shape": list(y_pred.shape),
            "num_pred_occupied": int(occupied_pred.shape[0]),
            # cap for response size
            "pred_occupied_indices_sample": occupied_pred[:1000].tolist(),
        }

    # --------- Routes ---------

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/gt/<sample_token>", methods=["GET"])
    def gt(sample_token: str):
        """
        Generate occupancy GT for a nuScenes sample token.
        """
        occ = gt_generator.generate_for_sample(sample_token)  # np.uint8
        occupied = np.argwhere(occ > 0)

        return jsonify({
            "sample_token": sample_token,
            "grid_cfg": {
                "x_lim": grid_cfg.x_lim,
                "y_lim": grid_cfg.y_lim,
                "z_lim": grid_cfg.z_lim,
                "resolution": grid_cfg.resolution,
                "bev": grid_cfg.bev,
            },
            "occ_shape": list(occ.shape),
            "num_occupied": int(occupied.shape[0]),
            "occupied_indices_sample": occupied[:1000].tolist(),
        })

    @app.route("/inference", methods=["POST"])
    def inference():
        """
        Run the placeholder model on occupancy for a given sample.
        Body JSON:
          {
            "sample_token": "<nuScenes sample token>"
          }
        """
        body = request.get_json(force=True)
        sample_token = body["sample_token"]

        # 1) generate GT occupancy
        occ = gt_generator.generate_for_sample(sample_token)

        # 2) run model
        result = run_model_on_occ(occ)

        return jsonify({
            "sample_token": sample_token,
            "occ_shape": list(occ.shape),
            **result,
        })

    return app

app = create_app()

if __name__ == "__main__":
    # optional: read host/port from env
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", "5004"))
    app.run(host=host, port=port, debug=True)