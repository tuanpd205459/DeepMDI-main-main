import os
import math
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler, autocast

from utils.dataset_UMNet import ISBI_Loader
from Physics_loss import physics_driven_loss, fit_linear_delta_plane
from NAFNet_arch import NAFNet_arch  # Mạng nền của UMNet

# -------------------- Tái lập kết quả --------------------
seed_value = 3407
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = False   # Tắt để benchmark hoạt động
torch.backends.cudnn.benchmark = True        # Tự chọn kernel CUDA nhanh nhất

# -------------------- Siêu tham số --------------------
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- Mask hình tròn --------------------
def circle_mask(size1=256, size2=256, r=256 / 2 * 0.99, x_offset=0, y_offset=0, batchsize=batch_size):
    """
    Tạo mask hình tròn cho loss vật lý và phần hiển thị.
    """
    x0 = (size1 - 1) / 2 + x_offset
    y0 = (size2 - 1) / 2 + y_offset
    y, x = np.ogrid[:size1, :size2]
    y = y[::-1]  # Đảo chiều trục y.

    mask = torch.tensor(((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2, dtype=torch.bool).unsqueeze(0)
    mask = mask.repeat(batchsize, 1, 1)
    return mask.to(device)

circle3 = circle_mask(batchsize=batch_size)

# -------------------- Hàm huấn luyện --------------------
def train_net(net, device, train_data_path, csv_name, criterion, epochs=301, batch_size=batch_size, lr=1.5e-3, step=1):
    """
    Huấn luyện UMNet bằng loss dựa trên mô hình vật lý.
    Có thể điều chỉnh lr để loss hội tụ ổn định hơn.
    """

    # Nạp dữ liệu.
    if isinstance(train_data_path, list) and len(train_data_path) == 2:
        simul_dataset = ISBI_Loader(train_data_path[0])
        real_dataset = ISBI_Loader(train_data_path[1])
        all_dataset = torch.utils.data.dataset.ConcatDataset([simul_dataset, real_dataset])
    else:
        all_dataset = ISBI_Loader(train_data_path)

    train_size = int(len(all_dataset) * 1)
    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_size, len(all_dataset) - train_size])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    val_best_loss = float('inf')
    scaler = GradScaler()  # AMP scaler

    # Tạo file CSV để lưu lịch sử loss.
    if not os.path.exists(csv_name):
        csvdata = pd.DataFrame(columns=['epoch', 'loss_val'])
        csvdata.to_csv(csv_name, index=False)

    for epoch in range(epochs):
        net.train()
        epoch_losses = []

        print(f"Epoch {epoch} - Time: {datetime.datetime.now()}")

        for image in train_loader:
            image = image.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = net(image)
                loss = criterion(pred, image, circle3)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch} - Loss: {avg_loss:.6f}")

        # Lưu checkpoint có loss tốt nhất.
        if avg_loss < val_best_loss:
            val_best_loss = avg_loss
            torch.save(net.state_dict(), 'model/UMNet.pth')

        # Ghi loss của epoch vào CSV.
        csvdata = pd.DataFrame([[epoch, avg_loss]])
        csvdata.to_csv(csv_name, mode='a', header=False, index=False)

        # -------------------- Hiển thị kết quả --------------------
        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                for image in train_loader:
                    image = image.to(device, dtype=torch.float32)
                    pred = net(image)

                    # Tách các ảnh dùng để hiển thị.
                    input_image1 = image[0, 0].cpu().numpy()
                    input_image2 = image[0, 1].cpu().numpy()
                    predicted_phi = pred[0, 0].cpu().numpy()
                    predicted_delta, _ = fit_linear_delta_plane(pred[:, 1])

                    # Tái tạo hai ảnh theo mô hình vật lý với delta nghiêng tuyến tính.
                    phi_vis = pred[0, 0] * 2 * math.pi + math.pi
                    delta_vis = predicted_delta[0] * 2 * math.pi
                    AB = torch.tensor(0.5).to(device)
                    AB_expanded = AB.expand_as(phi_vis)
                    model1 = AB_expanded + AB_expanded * torch.cos(phi_vis)
                    model2 = AB_expanded + AB_expanded * torch.cos(phi_vis + delta_vis)

                    # Chuyển sang NumPy để vẽ.
                    model1_np = model1.cpu().numpy()
                    model2_np = model2.cpu().numpy()

                    # Vẽ kết quả.
                    plt.figure(figsize=(12, 5))
                    titles = ["Input Image1", "Input Image2", "Predicted Phi",
                              "Model1", "Model2"]
                    images = [input_image1, input_image2, predicted_phi,
                              model1_np, model2_np]

                    for i, (title, img) in enumerate(zip(titles, images), 1):
                        plt.subplot(2, 3, i)
                        im = plt.imshow(img)
                        plt.title(title)
                        plt.axis('off')
                        plt.colorbar(im)

                    plt.suptitle(f"Epoch {epoch}")
                    plt.show()
                    break  # Chỉ hiển thị batch đầu tiên.

# -------------------- Chương trình chính --------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    net = NAFNet_arch(
        in_channel=2, out_channel=2,
        width=32,                       # Tăng từ 16 → 32
        middle_blk_num=2,              # Tăng từ 1 → 2
        enc_blk_nums=[2, 2, 2, 2],    # Tăng từ [1,1,1,1]
        dec_blk_nums=[2, 2, 2, 2]
    )
    net.to(device)

    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Model parameters: {total_params:.2f}M")

    # UMNet học từ ảnh train đã được PNNet chuẩn hoá.
    train_data_path = 'autodl-fs/simu_train/PNNet'
    train_net(net, device, train_data_path, criterion=physics_driven_loss(),
              csv_name="excel/UMNet.csv",
              lr=1.5e-3)
