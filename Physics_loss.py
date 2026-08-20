import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fit_linear_delta_plane(delta):
    """Chiếu mỗi map delta lên mặt phẳng nghiêng a*x + b*y + c."""
    batch_size, height, width = delta.shape
    y = torch.linspace(-1.0, 1.0, height, device=delta.device, dtype=delta.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=delta.device, dtype=delta.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    design = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1),
                          torch.ones(height * width, device=delta.device, dtype=delta.dtype)), dim=1)
    coefficients = delta.reshape(batch_size, -1) @ torch.linalg.pinv(design).T
    plane = (coefficients[:, 0, None, None] * grid_x
             + coefficients[:, 1, None, None] * grid_y
             + coefficients[:, 2, None, None])
    return plane, coefficients


class RangeLoss(nn.Module):
    def __init__(self, lower_bound=0.0, upper_bound=1.0, penalty_weight=10.0, tolerance=0.02):
        """
        Loss giới hạn miền giá trị, có biên dung sai.
        Chỉ phạt các giá trị ngoài miền trên kênh đầu tiên của tensor đầu vào.

        Args:
            lower_bound (float): Giá trị nhỏ nhất cho phép của kênh đầu tiên.
            upper_bound (float): Giá trị lớn nhất cho phép của kênh đầu tiên.
            penalty_weight (float): Hệ số nhân của phần phạt.
            tolerance (float): Biên sai lệch được chấp nhận mà không bị phạt.
        """
        super(RangeLoss, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.penalty_weight = penalty_weight
        self.tolerance = tolerance

    def forward(self, x):
        # Chỉ giới hạn miền giá trị cho kênh pha (kênh 0).
        channel_1 = x[:, 0, :, :]

        # Tính cận hiệu dụng sau khi cộng biên dung sai.
        lower = self.lower_bound - self.tolerance
        upper = self.upper_bound + self.tolerance

        # Phạt giá trị vượt cận dưới hoặc cận trên.
        penalty_lower = torch.clamp(lower - channel_1, min=0) ** 2
        penalty_upper = torch.clamp(channel_1 - upper, min=0) ** 2

        penalty = penalty_lower + penalty_upper
        return self.penalty_weight * penalty.sum()


class physics_driven_loss(nn.Module):
    def __init__(self, flag=1):
        super().__init__()
        self.flag = flag
        # Giới hạn pha dự đoán trong khoảng xấp xỉ [-0.5, 0.5].
        self.range_loss_fn = RangeLoss(lower_bound=-0.5, upper_bound=0.5, penalty_weight=1.0, tolerance=0.02)

    def forward(self, pred, image, circle3):
        """
        Loss dựa trên mô hình vật lý cho bài toán khôi phục pha giao thoa.
        Kết hợp độ khớp dữ liệu, tính nhất quán vật lý và giới hạn miền giá trị.

        Args:
            pred (Tensor): Dự đoán mạng, kích thước [B, 2, H, W].
                           Kênh 0 là pha phi; kênh 1 dùng để tạo pha nghiêng delta.
            image (Tensor): Hai ảnh giao thoa đầu vào, kích thước [B, 2, H, W].
            circle3 (Tensor): Mask nhị phân xác định vùng hợp lệ.
        """
        # Tách hai ảnh giao thoa đầu vào.
        frame1 = image[:, 0, :, :]
        frame2 = image[:, 1, :, :]

        # Tách pha phi và delta do mạng dự đoán.
        phi = pred[:, 0, :, :]
        delta = pred[:, 1, :, :]

        # Delta là pha nghiêng tuyến tính theo không gian: a*x + b*y + c.
        delta_plane, _ = fit_linear_delta_plane(delta)
        delta_radians = delta_plane * (2 * math.pi)

        # Thành phần biên độ/nền hằng số.
        AB = torch.tensor(0.5).to(device)
        AB_expanded = AB.expand_as(phi)

        # Đổi phi về miền [0, 2π].
        phi = phi * 2 * math.pi + math.pi

        # Ràng buộc mô hình giao thoa vật lý.
        # Ảnh 1: I1 = A + B*cos(phi)
        # Ảnh 2: I2 = A + B*cos(phi + delta(x, y))
        physic1 = AB_expanded + AB_expanded * torch.cos(phi)
        physic2 = AB_expanded + AB_expanded * torch.cos(phi + delta_radians)

        # So sánh ảnh tái tạo với ảnh gốc trong vùng hợp lệ.
        pdloss1 = F.l1_loss(physic1 * circle3, frame1 * circle3)
        pdloss2 = F.l1_loss(physic2 * circle3, frame2 * circle3)

        # Loss cuối: loss vật lý cộng với phần giới hạn miền giá trị.
        loss = pdloss1 + pdloss2 + self.range_loss_fn(pred) * 0.01
        return loss
