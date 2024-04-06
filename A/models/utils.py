import torch
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_kernel(window_size, channel):
    gauss = gaussian(window_size, 1.5).unsqueeze(1)
    kernel_2d = torch.mm(gauss, torch.transpose(gauss, 1, 0)).unsqueeze(0).unsqueeze(0)

    kernel_expanded = kernel_2d.expand(channel, 1, window_size, window_size)
    kernel = Variable(kernel_expanded).contiguous()

    return kernel

