import numpy as np
import torch


def psnr(x, y):
    if type(x) == torch.Tensor:
        x = x.numpy().transpose((0, 2, 3, 1))[:, :, :, (2, 1, 0)]
    if type(y) == torch.Tensor:
        y = y.numpy().transpose((0, 2, 3, 1))[:, :, :, (2, 1, 0)]

    mse = np.mean((x - y) ** 2, axis=(1, 2, 3)) + 1e-12

    MAX_PIXEL = 1
    batch_psnr = 20 * np.log10(MAX_PIXEL / np.sqrt(mse))
    avg_psnr = np.mean(batch_psnr).item()

    return avg_psnr


def ssim(x, y):
    if type(x) == torch.Tensor:
        x = x.numpy().transpose((0, 2, 3, 1))[:, :, :, (2, 1, 0)]
    if type(y) == torch.Tensor:
        y = y.numpy().transpose((0, 2, 3, 1))[:, :, :, (2, 1, 0)]

