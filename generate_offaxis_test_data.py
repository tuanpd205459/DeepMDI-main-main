"""Sinh test set off-axis vào autodl-fs/simu_test theo README."""

import argparse
import json
from pathlib import Path

import numpy as np

from generate_offaxis_training_data import generate_sample, prepare_folders


def main():
    parser = argparse.ArgumentParser(description="Sinh hologram off-axis cho test PNNet/UMNet.")
    parser.add_argument("--output", type=Path, default=Path("autodl-fs/simu_test"))
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true", help="Cho phep ghi de file test trung ten.")
    args = parser.parse_args()
    if args.num_samples <= 0 or args.size <= 0:
        raise ValueError("num-samples va size phai lon hon 0")
    folders = prepare_folders(args.output)
    existing = [folders[k] / f"{i:05d}.png" for i in range(args.num_samples)
                for k in ("frame1", "frame2", "frame1_n", "frame2_n")]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Da co file test ({existing[0]}). Dung --overwrite de thay the.")
    coords = np.linspace(-0.5, 0.5, args.size, dtype=np.float32)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    radius, angle = 2 * np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)
    pupil = (radius <= 1).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    for index in range(args.num_samples):
        generate_sample(index, x, y, radius, angle, pupil, folders, rng)
        if (index + 1) % 100 == 0 or index + 1 == args.num_samples:
            print(f"Generated {index + 1}/{args.num_samples} hologram pairs.")
    (args.output / "offaxis_test_config.json").write_text(json.dumps({
        "description": "Test set off-axis, delta(x,y) nghieng tuyen tinh",
        "num_samples": args.num_samples, "size": args.size, "seed": args.seed,
        "test_pipeline": "predict_PNNet.py -> predict_UMNet.py -> unwrap.m"}, indent=2), encoding="utf-8")
    print(f"Done. Test dataset: {args.output.resolve()}")


if __name__ == "__main__":
    main()
