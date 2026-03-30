import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=3)[0].max(dim=2)[0]
        avg_pool = torch.mean(x, dim=(2, 3))
        
        channel_weights = self.sigmoid(
            self.mlp(max_pool) + self.mlp(avg_pool)
        ).unsqueeze(2).unsqueeze(3)
        
        return x * channel_weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        spatial_weights = self.sigmoid(
            self.conv(torch.cat([max_pool, avg_pool], dim=1))
        )
        return x * spatial_weights
    
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class DoubleConvNoPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=True):
        super().__init__()
        self.use_pool = use_pool
        if use_pool:
            self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvNoPool(in_channels, out_channels)
    def forward(self, x):
        if self.use_pool:
            x = self.pool(x)
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels_SIV=2, predict_days=7):
        super().__init__()
        self.predict_days = predict_days
        base_channels = 32
        self.inc = DoubleConvNoPool(in_channels, base_channels)
        
        # encoder
        self.down1 = DoubleConvNoPool(base_channels, base_channels*2)      # 32
        self.down2 = Down(base_channels*2, base_channels*4, use_pool=True)  # 64
        self.down3 = Down(base_channels*4, base_channels*8, use_pool=True)  # 128
        self.down4 = Down(base_channels*8, base_channels*16, use_pool=True)  # 256
        
        # decoder
        self.up1 = Up(base_channels*16, base_channels*8)  # 128
        self.up2 = Up(base_channels*8, base_channels*4)   # 64
        self.up3 = Up(base_channels*4, base_channels*2)   # 32
        self.outc = nn.Conv2d(base_channels*2, out_channels_SIV * predict_days, kernel_size=1)

    def forward(self, x):
        x_shared = self.inc(x)
        x_SIV1 = self.down1(x_shared)
        x_SIV2 = self.down2(x_SIV1)
        x_SIV3 = self.down3(x_SIV2) 
        x_SIV4 = self.down4(x_SIV3)
        x_SIV5 = self.up1(x_SIV4, x_SIV3)
        x_SIV6 = self.up2(x_SIV5, x_SIV2)
        x_SIV = self.up3(x_SIV6, x_SIV1)
        
        SIV = torch.sigmoid(self.outc(x_SIV))
        SIV = SIV.chunk(self.predict_days, dim=1)
        return SIV

if __name__ == "__main__":
    model = Unet(in_channels = 14, predict_days = 15)
    dummy_input = torch.randn(8, 14, 256, 256)
    SIV_pred = model(dummy_input)
    print(f"Output Shape: {SIV_pred[0].shape}")
