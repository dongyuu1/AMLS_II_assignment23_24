import torch
from einops import rearrange

a = torch.randn((2, 2, 2, 2))
print(a)
n, c, h, w  = a.shape
a = rearrange(a, 'n c h w -> (n c) 1 h w')
a = rearrange(a, '(n c) 1 h w -> n c h w', c=c, n=n)
print(a)
Py