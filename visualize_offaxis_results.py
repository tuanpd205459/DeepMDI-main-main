"""Hiển thị input, kết quả trung gian PNNet và kết quả cuối UMNet/delta."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_png(path):
    if not path.exists():
        return None
    image = np.asarray(Image.open(path))
    if image.ndim == 3:
        image = image[..., 0]
    maximum = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1
    return image.astype(np.float32) / maximum


def read_npy(path):
    return np.load(path) if path.exists() else None


def draw(ax, image, title, cmap="gray", vmin=None, vmax=None):
    ax.set_title(title)
    ax.axis("off")
    if image is None:
        ax.text(0.5, 0.5, "Chua co file", ha="center", va="center", transform=ax.transAxes)
        return
    plot = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)


def main():
    parser = argparse.ArgumentParser(description="Xem pipeline off-axis theo tung mau.")
    parser.add_argument("--data-root", type=Path, default=Path("autodl-fs/simu_test"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--pnnet-folder", default="PNNet")
    parser.add_argument("--umnet-name", default="UMNet")
    parser.add_argument("--save", type=Path, help="Luu figure PNG thay vi chi hien cua so.")
    args = parser.parse_args()
    stem, root = f"{args.index:05d}", args.data_root
    umnet = root / f"test_{args.umnet_name}"
    true_delta = None
    metadata = root / "metadata" / f"{stem}.npz"
    if metadata.exists():
        with np.load(metadata) as item:
            true_delta = item["delta_radians"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    draw(axes[0, 0], read_png(root / "frame1" / f"{stem}.png"), "Input hologram 1")
    draw(axes[0, 1], read_png(root / "frame2" / f"{stem}.png"), "Input hologram 2")
    draw(axes[0, 2], read_npy(root / "phi" / f"{stem}.npy"), "Pha vat the that", "turbo")
    draw(axes[1, 0], read_png(root / args.pnnet_folder / "frame1_n" / f"{stem}.png"), "PNNet: frame 1")
    draw(axes[1, 1], read_png(root / args.pnnet_folder / "frame2_n" / f"{stem}.png"), "PNNet: frame 2")
    draw(axes[1, 2], read_npy(root / "phi_wrapped" / f"{stem}.npy"), "Wrapped phase that", "twilight", -np.pi, np.pi)
    draw(axes[2, 0], read_png(umnet / "wrapped" / f"{stem}.png"), "UMNet: wrapped phase", "twilight")
    draw(axes[2, 1], true_delta, "Delta nghieng that", "turbo")
    draw(axes[2, 2], read_npy(umnet / "delta" / f"{stem}.npy"), "UMNet: delta du doan", "turbo")
    fig.suptitle(f"Off-axis pipeline - sample {stem}", fontsize=16)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=180)
        print(f"Saved visualization: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
