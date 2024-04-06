import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import create_kernel


class L1LOSS(nn.Module):
    def __init__(self):
        super(L1LOSS, self).__init__()

    def forward(self, x, y):
        loss = torch.mean(torch.abs(x - y))
        return loss


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, x, y):
        mse = ((x - y) ** 2).mean(axis=(1, 2, 3)) + 1e-12
        MAX_PIXEL = 1
        batch_psnr = 10 * torch.log10((MAX_PIXEL ** 2) / mse)
        psnr_mean = batch_psnr.mean()

        return psnr_mean


class SSIMLoss(nn.Module):
    def __init__(self, window_size, device):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.device = device

    def forward(self, x, y):
        kernel = create_kernel(self.window_size, 3).to(device=self.device)
        kernel.requires_grad = False
        mean_sr = F.conv2d(x, kernel, padding=self.window_size // 2, groups=3)
        mean_hr = F.conv2d(y, kernel, padding=self.window_size // 2, groups=3)

        mean_sr_square = mean_sr.pow(2)
        mean_hr_square = mean_hr.pow(2)

        var_sr = F.conv2d(x * x, kernel, padding=self.window_size // 2, groups=3) - mean_sr_square
        var_hr = F.conv2d(y * y, kernel, padding=self.window_size // 2, groups=3) - mean_hr_square
        covar = F.conv2d(x * y, kernel, padding=self.window_size // 2, groups=3) - mean_sr * mean_hr

        C1 = 1e-4
        C2 = 9e-4

        ssim_pixwise = ((2 * mean_sr * mean_hr + C1) * (2 * covar + C2)) / \
                     ((mean_sr_square + mean_hr_square + C1) * (var_sr + var_hr + C2))

        ssim_mean = torch.mean(ssim_pixwise)
        return -torch.log10(ssim_mean), ssim_mean

