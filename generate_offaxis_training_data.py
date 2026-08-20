"""Sinh dữ liệu train: hai hologram off-axis theo Eq. (S1)-(S4) của supplement.

H1 = A + B*cos(phi_obj + carrier_1)
H2 = A + B*cos(phi_obj + carrier_1 + delta(x,y))
delta(x,y) = 2*pi*(dfx*x + dfy*y) + delta0
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image


def zernike_orders(count=37):
    """Tạo danh sách (n, m) cho các mode Zernike thực."""
    orders, n = [], 0
    while len(orders) < count:
        for m in range(-n, n + 1, 2):
            orders.append((n, m))
            if len(orders) == count:
                break
        n += 1
    return orders


ORDERS = zernike_orders()


def zernike(n, m, radius, angle):
    """Đa thức Zernike thực trên đĩa đơn vị."""
    am = abs(m)
    radial = np.zeros_like(radius)
    if (n - am) % 2:
        return radial
    for k in range((n - am) // 2 + 1):
        c = ((-1) ** k * math.factorial(n - k) /
             (math.factorial(k) * math.factorial((n + am) // 2 - k) *
              math.factorial((n - am) // 2 - k)))
        radial += c * radius ** (n - 2 * k)
    return radial * (np.cos(m * angle) if m >= 0 else np.sin(am * angle))


def make_phase(radius, angle, pupil, rng):
    """Tạo pha vật thể theo Table S1."""
    coefficients = np.zeros(37, dtype=np.float32)
    coefficients[1:3] = rng.uniform(-15, 15, 2)
    coefficients[3] = rng.uniform(-10, 10)
    coefficients[4:16] = rng.uniform(-3, 3, 12)
    coefficients[16:] = rng.uniform(-0.5, 0.5, 21)
    phase = sum(c * zernike(n, m, radius, angle) for c, (n, m) in zip(coefficients, ORDERS))
    return (phase * pupil).astype(np.float32), coefficients


def carrier(rng, lower, upper):
    """Carrier off-axis, đơn vị chu kỳ trên toàn bề rộng ảnh."""
    magnitude, direction = rng.uniform(lower, upper), rng.uniform(0, 2 * np.pi)
    return magnitude * np.cos(direction), magnitude * np.sin(direction)


def save_png(path, image):
    """Lưu ảnh [0,1] 16-bit; dùng Pillow nếu chưa cài OpenCV."""
    image = (np.clip(image, 0, 1) * 65535).astype(np.uint16)
    if cv2 is None:
        Image.fromarray(image).save(path)
    elif not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Khong the ghi anh: {path}")


def prepare_folders(output):
    """Tạo cấu trúc dữ liệu chuẩn cho train/test."""
    folders = {name: output / name for name in
               ("frame1", "frame2", "frame1_n", "frame2_n", "phi", "phi_wrapped", "metadata")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def generate_sample(index, x, y, radius, angle, pupil, folders, rng):
    """Sinh một cặp hologram off-axis, nhãn PNNet và metadata delta thật."""
    phi_object, coefficients = make_phase(radius, angle, pupil, rng)
    i0, w = rng.uniform(0.5, 0.95), rng.uniform(0.2, 1.0)
    x0, y0, beam_width = *rng.uniform(-0.3, 0.3, 2), rng.uniform(0.8, 1.2)
    iref = i0 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * beam_width ** 2))
    background, modulation = (1 + w) * iref, 2 * np.sqrt(w) * iref

    fx, fy = carrier(rng, 8, 16)
    dfx, dfy = carrier(rng, 1, 5)
    delta0 = rng.uniform(0.1, 3.0)
    phase1 = phi_object + 2 * np.pi * (fx * x + fy * y)
    delta = 2 * np.pi * (dfx * x + dfy * y) + delta0
    phase2 = phase1 + delta

    clean1 = background + modulation * np.cos(phase1) * pupil
    clean2 = background + modulation * np.cos(phase2) * pupil
    scale = max(float(clean1.max()), float(clean2.max()), np.finfo(np.float32).eps)
    raw1, raw2 = clean1 / scale, clean2 / scale
    snr_db = rng.uniform(15, 40)
    noise_std = math.sqrt(float(np.mean(raw1 ** 2)) / 10 ** (snr_db / 10))
    raw1 += rng.normal(0, noise_std, raw1.shape).astype(np.float32)
    raw2 += rng.normal(0, noise_std, raw2.shape).astype(np.float32)

    name = f"{index:05d}.png"
    save_png(folders["frame1"] / name, raw1)
    save_png(folders["frame2"] / name, raw2)
    save_png(folders["frame1_n"] / name, 0.5 + 0.5 * np.cos(phase1) * pupil)
    save_png(folders["frame2_n"] / name, 0.5 + 0.5 * np.cos(phase2) * pupil)
    np.save(folders["phi"] / f"{index:05d}.npy", phi_object)
    np.save(folders["phi_wrapped"] / f"{index:05d}.npy", np.angle(np.exp(1j * phase1)).astype(np.float32))
    np.savez_compressed(folders["metadata"] / f"{index:05d}.npz",
                        zernike_coefficients=coefficients, delta_radians=delta.astype(np.float32),
                        carrier_1_cycles=np.array([fx, fy], dtype=np.float32),
                        delta_cycles=np.array([dfx, dfy], dtype=np.float32),
                        delta_offset_rad=np.float32(delta0), snr_db=np.float32(snr_db))


def main():
    parser = argparse.ArgumentParser(description="Sinh hologram off-axis cho PNNet/UMNet.")
    parser.add_argument("--output", type=Path, default=Path("autodl-fs/simu_train"))
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.size <= 0:
        raise ValueError("num-samples va size phai lon hon 0")
    folders = prepare_folders(args.output)
    coords = np.linspace(-0.5, 0.5, args.size, dtype=np.float32)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    radius, angle = 2 * np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)
    pupil = (radius <= 1).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    for index in range(args.num_samples):
        generate_sample(index, x, y, radius, angle, pupil, folders, rng)
        if (index + 1) % 100 == 0 or index + 1 == args.num_samples:
            print(f"Generated {index + 1}/{args.num_samples} hologram pairs.")
    (args.output / "dataset_config.json").write_text(json.dumps({
        "description": "Hai hologram off-axis, delta(x,y) nghieng tuyen tinh",
        "num_samples": args.num_samples, "size": args.size, "seed": args.seed,
        "source_equations": "Supplement Eq. (S1)-(S4)", "snr_db": [15, 40]}, indent=2), encoding="utf-8")
    print(f"Done. Dataset: {args.output.resolve()}")


if __name__ == "__main__":
    main()
