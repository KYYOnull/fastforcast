

import torch
from torch import nn
from layers.Invertible import RevIN

# 傅里叶变换
class FNetBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, dropout=0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm_1 = nn.LayerNorm(input_dim)
        self.norm_2 = nn.LayerNorm(input_dim)

    def fourier_transform(self, x):
        # return torch.fft.fft2(x, dim=(1, 2)).real # 二维傅里叶  x 的第一个和第二个维度上执行变换
        return torch.fft.fft(x, dim=-1).real # x 的最后一个维度上执行变换

    def forward(self, x): # 一层傅里叶
        resid = x
        x = self.fourier_transform(x)
        x = self.norm_1(x + resid) # add then norm
        resid = x
        x = self.mlp(x)
        out = self.norm_2(x + resid)
        return out


class Model(nn.Module):
    def __init__(self, configs): # the real model
        super().__init__()

        self.encoder = nn.ModuleList([
            FNetBlock(configs.enc_in, configs.d_model) for _ in range(configs.e_layers) # 2
        ]) # encoder input size=7  d_model=512
        
        self.projection = nn.Linear(configs.seq_len, configs.pred_len) # 126 4
        self.norm = nn.LayerNorm(configs.enc_in) if configs.norm else None
        self.rev = RevIN(configs.enc_in) if configs.rev else None # 可逆标准化层

    def forward(self, x):
        x = self.rev(x, 'norm') if self.rev else x # 标准化
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x) if self.norm else x
        x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        x = self.rev(x, 'denorm') if self.rev else x

        return x
