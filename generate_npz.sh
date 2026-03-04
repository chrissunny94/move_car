PYTHONPATH=. python3 scripts/generate_gt.py \
  --nusc_root DATA/nuscenes \
  --version v1.0-mini \
  --out_dir DATA/occ_gt_mini \
  --num_samples 10 \
  --bev
