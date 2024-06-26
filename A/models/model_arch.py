import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SRModel(nn.Module):
    """
    The class of the Super-Resolution model
    """
    def __init__(self, cin, cmid, cup, cout, n_block):
        """
        The initialization method
        :param cin: The number of input channels
        :param cmid: The number of channels of the latent features
        :param cup: The number of channels of the features in the reconstruction module
        :param cout: The number of output channels
        :param n_block: The number of blocks, deciding the depth of the model
        """
        super(SRModel, self).__init__()

        self.conv_embed = nn.Conv2d(
            cin, cmid, kernel_size=3, padding=1, stride=1, bias=True
        )

        scblock_list = []
        for i in range(n_block):
            scblock_list.append(SMU(cmid) if i % 2 == 0 else SCBlock(cmid))
        self.sc_blocks = nn.Sequential(*scblock_list)
        self.conv_after_sc = nn.Conv2d(
            cmid, cmid, kernel_size=3, padding=1, stride=1, bias=True
        )
        self.sr_block = SRBlock(cmid, cup)
        self.conv_out = nn.Conv2d(
            cup, cout, kernel_size=3, padding=1, stride=1, bias=True
        )

    def forward(self, x):
        """
        The computing process
        :param x: The input image batch
        :return: The generated SR images
        """
        x_embed = self.conv_embed(x)
        x_mid = self.conv_after_sc(self.sc_blocks(x_embed))
        x_up = self.sr_block(x_mid)
        x_out = self.conv_out(x_up) + F.interpolate(
            x, scale_factor=4, mode="bilinear", align_corners=False
        )

        return x_out


class SCBlock(nn.Module):
    """
    The class of the self-calibrated block. The original architecture is from
    https://github.com/zhaohengyuan1/PAN, which is re-implemented in this project.
    """
    def __init__(self, cin):
        """
        The initialization method
        :param cin: The number of channels of the input features
        """
        super(SCBlock, self).__init__()
        cmid = cin // 2
        self.conv_upper = nn.Conv2d(
            cin, cmid, kernel_size=1, padding=0, stride=1, bias=False
        )
        self.conv_lower = nn.Conv2d(
            cin, cmid, kernel_size=1, padding=0, stride=1, bias=False
        )

        self.att_conv = nn.Conv2d(cmid, cmid, kernel_size=1)
        self.conv_upper1 = nn.Conv2d(cmid, cmid, kernel_size=3, padding=1, bias=False)
        self.conv_upper2 = nn.Conv2d(cmid, cmid, kernel_size=3, padding=1, bias=False)

        self.conv_lower1 = nn.Conv2d(
            cmid, cmid, kernel_size=3, padding=1, stride=1, bias=False, dilation=1
        )

        self.conv_out = nn.Conv2d(cmid * 2, cmid * 2, kernel_size=1, bias=False)

    def forward(self, x):
        """
        The computing process
        :param x: The input batch of features
        :return: The features after processing
        """

        n, c, h, w = x.shape
        y = F.layer_norm(x, [c, h, w])
        y_upper = F.leaky_relu(self.conv_upper(y), negative_slope=0.2, inplace=True)
        y_lower = F.leaky_relu(self.conv_lower(y), negative_slope=0.2, inplace=True)

        y_att = F.sigmoid(self.att_conv(y_upper))
        y_upper = self.conv_upper1(y_upper)

        y_upper = torch.mul(y_upper, y_att)
        y_upper = F.leaky_relu(
            self.conv_upper2(y_upper), negative_slope=0.2, inplace=True
        )

        y_lower = F.leaky_relu(
            self.conv_lower1(y_lower), negative_slope=0.2, inplace=True
        )

        y_full = torch.cat((y_lower, y_upper), dim=1)

        y_out = self.conv_out(y_full) + y

        out = y_out + x
        return out


class PA(nn.Module):
    """
    The pixel wise attention from https://github.com/zhaohengyuan1/PAN,
    which is re-implemented in this project.
    """
    def __init__(self, cin):
        """
        The initialization method
        :param cin: The number of channels of input features
        """
        super(PA, self).__init__()
        self.conv_att = nn.Conv2d(cin, cin, kernel_size=1)

    def forward(self, x):
        """
        The computing process
        :param x: The batch of input features
        :return: The features after processing
        """
        x_att = F.sigmoid(self.conv_att(x))
        out = F.leaky_relu(torch.mul(x, x_att), negative_slope=0.2, inplace=True)
        return out


class SMU(nn.Module):
    """
    The class of the Separable Modulation Unit (https://dl.acm.org/doi/abs/10.1145/3581783.3612353).
    There is no official code. It is re-implemented from scratch in this project.
    """
    def __init__(self, cin):
        """
        The initialization method
        :param cin: The number of channels of the input features
        """
        super(SMU, self).__init__()

        self.trans_in = nn.Conv2d(cin, cin, 1)
        self.dconv_k = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
        self.dconv_v = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
        self.dconv_q1 = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
        self.dconv_q2 = nn.Conv2d(cin, 1, 1, 1)
        self.trans_out = nn.Conv2d(cin, cin, 1)

    def forward(self, x):
        """
        The computing process
        :param x: The batch of input features
        :return: The features after processing
        """
        n, c, h, w = x.shape
        y = F.layer_norm(x, [c, h, w])
        y = F.leaky_relu(self.trans_in(y), negative_slope=0.2, inplace=True)
        k = F.leaky_relu(self.dconv_k(y), negative_slope=0.2, inplace=True)
        v = F.leaky_relu(self.dconv_v(y), negative_slope=0.2, inplace=True)
        q = F.leaky_relu(
            self.dconv_q2(self.dconv_q1((y))), negative_slope=0.2, inplace=True
        )

        q = F.softmax(rearrange(q, "n 1 h w -> n 1 (h w)"), dim=2)
        k = rearrange(k, "n c h w -> n c (h w)")
        v = rearrange(v, "n c h w -> n c (h w)")

        modulator = torch.sum(q * k, dim=2, keepdim=True)
        v = rearrange(modulator * v, "n c (h w) -> n c h w", h=h, w=w)
        out = F.leaky_relu(self.trans_out(v), negative_slope=0.2, inplace=True)
        out = x + out
        return out


class SRBlock(nn.Module):
    """
    The class of the reconstruction module
    """
    def __init__(self, cin, cup):
        """
        The initialization method
        :param cin: The number channels of the input features
        :param cup: The number of channel of the features in the reconstruction block
        """
        super(SRBlock, self).__init__()
        self.up_conv1 = nn.Conv2d(
            cin, cup, kernel_size=3, padding=1, stride=1, bias=True
        )
        self.pa1 = PA(cup)
        self.up_conv2 = nn.Conv2d(
            cup, cup, kernel_size=3, padding=1, stride=1, bias=True
        )

        self.up_conv3 = nn.Conv2d(
            cup, cup, kernel_size=3, padding=1, stride=1, bias=True
        )
        self.pa2 = PA(cup)
        self.up_conv4 = nn.Conv2d(
            cup, cup, kernel_size=3, padding=1, stride=1, bias=True
        )

    def forward(self, x):
        """
        The computing process
        :param x: The batch of input features
        :return: The output SR images
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up_conv1(x)
        x = self.pa1(x)
        x = F.leaky_relu(self.up_conv2(x), negative_slope=0.2, inplace=True)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up_conv3(x)
        x = self.pa2(x)
        x = F.leaky_relu(self.up_conv4(x), negative_slope=0.2, inplace=True)
        return x


def param_count(model):
    """
    The function counting the parameter number of a model
    :param model: The model
    :return: The parameter count of the model
    """
    num = 0
    for n, v in model.named_parameters():
        num += v.nelement()
    return num



