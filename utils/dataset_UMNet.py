import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
import matplotlib.pyplot as plt


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        """
        Dataset cho UMNet huấn luyện theo mô hình vật lý.
        Chỉ nạp ảnh đầu vào, không cần nhãn.
        """
        self.data_path = data_path
        self.imgs_path1 = glob.glob(os.path.join(data_path, 'frame1_n/*.png'))
        self.imgs_path2 = glob.glob(os.path.join(data_path, 'frame2_n/*.png'))

    def augment(self, image, flipCode):
        """
        Tăng cường dữ liệu đơn giản bằng phép lật ảnh.
        flipCode: 1 ngang, 0 dọc, -1 cả hai.
        """
        return cv2.flip(image, flipCode)

    def __getitem__(self, index):
        """
        Nạp và tiền xử lý ảnh đầu vào.
        """
        image_path1 = self.imgs_path1[index]
        image_path2 = self.imgs_path2[index]

        # Đọc ảnh.
        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

        if image1 is None or image2 is None:
            raise ValueError(f"Failed to load images at {image_path1} or {image_path2}")

        # Thêm chiều kênh.
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1])
        image2 = image2.reshape(1, image2.shape[0], image2.shape[1])

        # Chuẩn hoá về [0, 1].
        image1 = image1 / 65535.0 if image1.max() > 255 else image1 / 255.0
        image2 = image2 / 65535.0 if image2.max() > 255 else image2 / 255.0

        # Ghép hai ảnh thành đầu vào của mạng.
        image = np.concatenate((image1, image2), axis=0)

        return image

    def __len__(self):
        """
        Trả về tổng số mẫu.
        """
        return len(self.imgs_path1)


if __name__ == "__main__":
    # Ví dụ sử dụng.
    dataset = ISBI_Loader("../data/simu_1_2/")
    print("Number of samples:", len(dataset))

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=15, shuffle=True)

    for image in train_loader:
        print("Batch shape:", image.shape)
