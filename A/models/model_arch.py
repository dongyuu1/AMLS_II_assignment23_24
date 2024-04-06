import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from einops import rearrange

class SRModel(nn.Module):
    def __init__(self, cin, cmid, cup, cout, n_block):
        super(SRModel, self).__init__()

        self.conv_embed = nn.Conv2d(cin, cmid, kernel_size=3, padding=1, stride=1, bias=True)

        scblock_list = []
        for i in range(n_block):
            scblock_list.append(SMU(cmid) if i % 2 == 0 else SCBlock(cmid))
        self.sc_blocks = nn.Sequential(*scblock_list)
        self.conv_after_sc = nn.Conv2d(cmid, cmid, kernel_size=3, padding=1, stride=1, bias=True)
        self.sr_block = SRBlock(cmid, cup)
        self.conv_out = nn.Conv2d(cup, cout, kernel_size=3, padding=1, stride=1, bias=True)


    def forward(self, x):
        x_embed = self.conv_embed(x)
        x_mid = self.conv_after_sc(self.sc_blocks(x_embed))
        x_up = self.sr_block(x_mid)
        x_out = self.conv_out(x_up) + F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)

        return x_out


class SCBlock(nn.Module):
    def __init__(self, cin):
        super(SCBlock, self).__init__()
        cmid = cin // 2
        self.conv_upper = nn.Conv2d(cin, cmid, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv_lower = nn.Conv2d(cin, cmid, kernel_size=1, padding=0, stride=1, bias=False)

        self.att_conv = nn.Conv2d(cmid, cmid, kernel_size=1)
        self.conv_upper1 = nn.Conv2d(cmid, cmid, kernel_size=3, padding=1, bias=False)
        self.conv_upper2 = nn.Conv2d(cmid, cmid, kernel_size=3, padding=1, bias=False)

        self.conv_lower1 = nn.Conv2d(cmid, cmid, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)

        self.conv_out = nn.Conv2d(cmid * 2, cmid * 2, kernel_size=1, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        y = F.layer_norm(x, [c, h, w])
        y_upper = F.leaky_relu(self.conv_upper(y), negative_slope=0.2, inplace=True)
        y_lower = F.leaky_relu(self.conv_lower(y), negative_slope=0.2, inplace=True)

        y_att = F.sigmoid(self.att_conv(y_upper))
        y_upper = self.conv_upper1(y_upper)

        y_upper = torch.mul(y_upper, y_att)
        y_upper = F.leaky_relu(self.conv_upper2(y_upper), negative_slope=0.2, inplace=True)

        y_lower = F.leaky_relu(self.conv_lower1(y_lower), negative_slope=0.2, inplace=True)

        y_full = torch.cat((y_lower, y_upper), dim=1)

        y_out = self.conv_out(y_full) + y

        out = y_out + x
        return out


class PA(nn.Module):
    def __init__(self, cin):
        super(PA, self).__init__()
        self.conv_att = nn.Conv2d(cin, cin, kernel_size=1)

    def forward(self, x):
        x_att = F.sigmoid(self.conv_att(x))
        out = F.leaky_relu(torch.mul(x, x_att), negative_slope=0.2, inplace=True)
        return out


class SMU(nn.Module):
    def __init__(self, cin):
        super(SMU, self).__init__()

        self.trans_in = nn.Conv2d(cin, cin, 1)
        self.dconv_k = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
        self.dconv_v = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
        self.dconv_q1 = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
        self.dconv_q2 = nn.Conv2d(cin, 1, 1, 1)
        self.trans_out = nn.Conv2d(cin, cin, 1)


    def forward(self, x):
        n, c, h, w = x.shape
        y = F.layer_norm(x, [c, h, w])
        y = F.leaky_relu(self.trans_in(y), negative_slope=0.2, inplace=True)
        k = F.leaky_relu(self.dconv_k(y), negative_slope=0.2, inplace=True)
        v = F.leaky_relu(self.dconv_v(y), negative_slope=0.2, inplace=True)
        q = F.leaky_relu(self.dconv_q2(self.dconv_q1((y))), negative_slope=0.2, inplace=True)


        q = F.softmax(rearrange(q, "n 1 h w -> n 1 (h w)"), dim=2)
        k = rearrange(k, "n c h w -> n c (h w)")
        v = rearrange(v, "n c h w -> n c (h w)")

        modulator = torch.sum(q * k, dim=2, keepdim=True)
        v = rearrange(modulator * v, "n c (h w) -> n c h w", h=h, w=w)
        out = F.leaky_relu(self.trans_out(v), negative_slope=0.2, inplace=True)
        out = x + out
        return out

"""
class LSN(nn.Module):
    def __init__(self, cin):
        super(LSN, self).__init__()
        self.trans_in == nn.Conv2d(cin, cin, 1)
        self.dconv = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
        self.trans_out = nn.Conv2d(cin, cin, 1)
        self.gamma = Variable()
"""

class SRBlock(nn.Module):
    def __init__(self, cin, cup):
        super(SRBlock, self).__init__()
        self.up_conv1 = nn.Conv2d(cin, cup, kernel_size=3, padding=1, stride=1, bias=True)
        self.pa1 = PA(cup)
        self.up_conv2 = nn.Conv2d(cup, cup, kernel_size=3, padding=1, stride=1, bias=True)

        self.up_conv3 = nn.Conv2d(cup, cup, kernel_size=3, padding=1, stride=1, bias=True)
        self.pa2 = PA(cup)
        self.up_conv4 = nn.Conv2d(cup, cup, kernel_size=3, padding=1, stride=1, bias=True)


    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up_conv1(x)
        x = self.pa1(x)
        x = F.leaky_relu(self.up_conv2(x), negative_slope=0.2, inplace=True)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up_conv3(x)
        x = self.pa2(x)
        x = F.leaky_relu(self.up_conv4(x), negative_slope=0.2, inplace=True)
        return x


def param_count(model):
    num = 0
    for n, v in model.named_parameters():
        num += v.nelement()
    print(num)


if __name__ == "__main__":
    x = torch.randn((32, 40, 64, 64))
    block = SRModel(3, 40, 24, 3, 16)
    print(param_count(block))
    x = block(x)

