import glob
import os
import re
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from NAFNet_arch import NAFNet_arch

# -------------------- Cấu hình thiết bị --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- Cấu hình model --------------------
model_name = 'PNNet'
width = 16
enc_blks = [1, 1, 1, 1]
middle_blk_num = 1
dec_blks = [1, 1, 1, 1]

net = NAFNet_arch(
    in_channel=2, out_channel=2,
    width=width, middle_blk_num=middle_blk_num,
    enc_blk_nums=enc_blks, dec_blk_nums=dec_blks
).to(device)

# Nạp trọng số đã huấn luyện.
net.load_state_dict(torch.load(f'model/{model_name}.pth', map_location=device, weights_only=True))
net.eval()  # Chuyển model sang chế độ đánh giá.

criterion = F.l1_loss  # Hàm loss.
loss_list = []

# -------------------- Đường dẫn dữ liệu --------------------
tests_path1 = glob.glob('autodl-fs/simu_test/frame1/*.png')
tests_path1_sorted = sorted(
    tests_path1, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group())
)

# -------------------- Vòng lặp suy luận --------------------
for test_path in tests_path1_sorted:
    test_path2 = test_path.replace('frame1', 'frame2')

    # Tạo đường dẫn lưu theo tên model.
    save_res_path1 = test_path.replace('frame1', f'{model_name}/frame1_n')
    save_res_path2 = test_path.replace('frame1', f'{model_name}/frame2_n')

    lab_path1 = test_path.replace('frame1', 'frame1_n')
    lab_path2 = test_path.replace('frame1', 'frame2_n')

    # -------------------- Nạp ảnh --------------------
    img1 = cv2.imread(test_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(test_path2, cv2.IMREAD_UNCHANGED)
    lab1 = cv2.imread(lab_path1, cv2.IMREAD_UNCHANGED)
    lab2 = cv2.imread(lab_path2, cv2.IMREAD_UNCHANGED)

    # Chuẩn hoá về [0, 1].
    img1 = img1 / 65535.0 if img1.max() > 255 else img1 / 255.0
    img2 = img2 / 65535.0 if img2.max() > 255 else img2 / 255.0
    lab1 = lab1 / 65535.0 if lab1.max() > 255 else lab1 / 255.0
    lab2 = lab2 / 65535.0 if lab2.max() > 255 else lab2 / 255.0

    # Thêm chiều batch và chiều kênh.
    img1 = img1.reshape(1, 1, img1.shape[0], img1.shape[1])
    img2 = img2.reshape(1, 1, img2.shape[0], img2.shape[1])
    lab1 = lab1.reshape(1, 1, lab1.shape[0], lab1.shape[1])
    lab2 = lab2.reshape(1, 1, lab2.shape[0], lab2.shape[1])

    # Ghép hai ảnh theo chiều kênh.
    img_tensor = torch.from_numpy(np.concatenate((img1, img2), axis=1)).to(device, dtype=torch.float32)
    lab_tensor = torch.from_numpy(np.concatenate((lab1, lab2), axis=1)).to(device, dtype=torch.float32)

    # -------------------- Chạy suy luận --------------------
    pred = net(img_tensor)
    loss = criterion(pred, lab_tensor).detach().cpu().numpy()
    loss_list.append(np.sqrt(loss))
    print(f"Loss: {loss_list[-1]}")

    # Tách hai kênh kết quả và lưu ảnh 16-bit.
    pred0 = pred[:, 0, :, :].unsqueeze(1).detach().cpu().numpy()[0, 0]
    pred1 = pred[:, 1, :, :].unsqueeze(1).detach().cpu().numpy()[0, 0]

    pred0_16bit = np.clip(pred0 * 65535, 0, 65535).astype(np.uint16)
    pred1_16bit = np.clip(pred1 * 65535, 0, 65535).astype(np.uint16)

    # Tạo thư mục lưu nếu chưa tồn tại.
    os.makedirs(os.path.dirname(save_res_path1), exist_ok=True)
    os.makedirs(os.path.dirname(save_res_path2), exist_ok=True)

    cv2.imwrite(save_res_path1, pred0_16bit)
    cv2.imwrite(save_res_path2, pred1_16bit)

# -------------------- Vẽ loss --------------------
plt.figure(figsize=(10, 4))
plt.plot(range(len(loss_list)), loss_list, 'o-', label='Loss')
plt.xlabel('Sample Index')
plt.ylabel('Loss')
plt.title(f'{model_name} - Normalization Loss per Sample')
plt.legend()
plt.show()


# In loss trung bình.
avg_loss = sum(loss_list) / len(loss_list)
print(f"Average loss: {avg_loss}")
