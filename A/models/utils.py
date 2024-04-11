import torch
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    """
    The generation of a tensor following gaussian distribution
    :param window_size: The size of the tensor
    :param sigma: The standard deviation
    :return:
    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_kernel(window_size, channel):
    """
    The creation of a gaussian kernel
    :param window_size: The size of the kernel
    :param channel: The channel number of the kernel
    :return: The generated gaussian filter
    """
    gauss = gaussian(window_size, 1.5).unsqueeze(1)
    kernel_2d = torch.mm(gauss, torch.transpose(gauss, 1, 0)).unsqueeze(0).unsqueeze(0)

    kernel_expanded = kernel_2d.expand(channel, 1, window_size, window_size)
    kernel = Variable(kernel_expanded).contiguous()

    return kernel
