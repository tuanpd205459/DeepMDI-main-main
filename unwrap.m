function out = unwrap(wphs, tol, iter)
% UNWRAP  Gỡ cuộn pha bền vững bằng phương pháp bình phương tối thiểu có trọng số.
%
%   Phần cài đặt này dựa trên thuật toán được mô tả trong:
%   Guo, Y., Chen, X., & Zhang, T. 
%   "Robust phase unwrapping algorithm based on least squares,"
%   Optics and Lasers in Engineering, 63, 25–29 (2014).
%
%   ĐẦU VÀO:
%       wphs.data : Map pha wrapped (radian)
%       wphs.mask : Mask nhị phân của pixel hợp lệ (1 = trong vùng đồng tử)
%       tol       : Ngưỡng hội tụ (ví dụ 1e-6)
%       iter      : Số vòng lặp tối đa (ví dụ 100)
%
%   ĐẦU RA:
%       out.data  : Map pha đã gỡ cuộn
%       out.mask  : Mask pixel hợp lệ
%       out.type  : 'phase'
%
%   Ví dụ:
%       out = unwrap(wphs, 1e-6, 100);

% ------------------ Xử lý tham số ------------------
switch nargin
    case 1
        tol = 1e-3;
        iter = 10;
    case 2
        iter = 10;
end

% ------------------ Trích xuất vùng hợp lệ ------------------
[m, n] = find(wphs.mask);
minr = min(m); maxr = max(m);
minc = min(n); maxc = max(n);

data = wphs.data(minr:maxr, minc:maxc);
mask = wphs.mask(minr:maxr, minc:maxc);
[rows, cols] = size(data);

% ------------------ Tính gradient pha wrapped ------------------
% Gradient ngang (hướng x)
dx1 = [diff(data,1,2), zeros(rows,1)];
dx1 = mod(dx1 + pi, 2*pi) - pi;
dx2 = circshift(dx1, [0 1]);

% Gradient dọc (hướng y)
dy1 = [diff(data); zeros(1,cols)];
dy1 = mod(dy1 + pi, 2*pi) - pi;
dy2 = circshift(dy1, [1 0]);

% ------------------ Tính trọng số ------------------
% Trọng số bảo đảm gradient chỉ tính trong vùng mask hợp lệ.
wx1 = double([mask(:,1:end-1) & mask(:,2:end), mask(:,end)]);
wx2 = double([mask(:,1), mask(:,1:end-1) & mask(:,2:end)]);
wy1 = double([mask(1:end-1,:) & mask(2:end,:); mask(end,:)]);
wy2 = double([mask(1,:); mask(1:end-1,:) & mask(2:end,:)]);

% ------------------ Tính phần dư (Laplacian) ------------------
r = (wx1.*dx1 - wx2.*dx2) + (wy1.*dy1 - wy2.*dy2);

% Tạo toán tử Laplacian rời rạc trong miền tần số.
t = 2*( repmat(cos((0:rows-1)'*(pi/rows)), 1, cols) + ...
        repmat(cos((0:cols-1) *(pi/cols)), rows, 1) - 2 );
t(1,1) = 1;

data(:) = 0;
crit = tol * norm(r, 'fro');   % Điều kiện hội tụ.

% ------------------ Bộ giải gradient liên hợp lặp ------------------
for k = 1:iter
    % Giải phương trình Laplacian bằng DCT (bộ giải Poisson nhanh).
    z = idct2(dct2(r) ./ t);
    b1 = sum(sum(r .* z));
    if k == 1
        p = z;
    else
        p = z + (b1 / b0) * p;
    end
    b0 = b1;

    % Tính gradient của p.
    dx1 = [diff(p,1,2), zeros(rows,1)];
    dx2 = circshift(dx1,[0 1]);
    dy1 = [diff(p); zeros(1,cols)];
    dy2 = circshift(dy1,[1 0]);

    % Tính phần dư cho vòng lặp hiện tại.
    rp = (wx1.*dx1 - wx2.*dx2) + (wy1.*dy1 - wy2.*dy2);
    a = sum(sum(p .* rp));
    r = r - (b1/a) * rp;
    data = data + (b1/a) * p;

    % Kiểm tra hội tụ.
    if norm(r, 'fro') < crit
        break;
    end
end

% ------------------ Ghép kết quả đầu ra ------------------
out = wphs;
out.data(minr:maxr, minc:maxc) = data .* mask;
out.mask(minr:maxr, minc:maxc) = mask;
out.type = 'phase';

end
