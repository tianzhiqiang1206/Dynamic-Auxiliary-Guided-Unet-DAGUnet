import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from optical_flow import WrapedProcessing
import matplotlib.pyplot as plt

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

class WAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.A_in_SIV = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.A_in_SIC = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.A_in_FLOW = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5)
        self.A_out_SIV = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.A_out_SIC = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.A_out_FLOW = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.cbam = CBAM(channels)
        self.cbam_sic = CBAM(channels)
        self.cbam_siv = CBAM(channels)
        self.cbam_flow = CBAM(channels)

    def forward(self, x_SIV, x_SIC, x_FLOW):
        assert x_SIV.size(1) == x_SIC.size(1)
        weight_flow = 0.0

        x_shared = (1-weight_flow) * self.A_in_SIV * x_SIV + self.A_in_SIC * x_SIC + weight_flow * self.A_in_FLOW * x_FLOW
        x_shared = self.cbam(x_shared)

        x_SIV_out = self.A_out_SIV * x_shared + self.cbam_siv(x_SIV)
        x_SIC_out = self.A_out_SIC * x_shared + self.cbam_sic(x_SIC)
        x_FLOW_out = self.A_out_FLOW * x_shared + self.cbam_flow(x_FLOW)
        return x_SIV_out, x_SIC_out, x_FLOW_out

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

class HISUnet(nn.Module):
    def __init__(self, in_channels, out_channels_SIV=2, out_channels_SIC=1, out_channels_FLOW=2, predict_days=15):
        super().__init__()
        self.predict_days = predict_days
        init_channels = 32
        base_channels = 32
        self.inc = DoubleConvNoPool(in_channels, init_channels)
        self.inc_flow = DoubleConvNoPool(in_channels=self.predict_days*2, out_channels=init_channels)
        self.wrapped_processing = WrapedProcessing(pretrained = None)

        self.down1_SIV = DoubleConvNoPool(init_channels, base_channels*2)
        self.down2_SIV = Down(base_channels*2, base_channels*4, use_pool=True)
        self.down3_SIV = Down(base_channels*4, base_channels*8, use_pool=True)
        self.down4_SIV = Down(base_channels*8, base_channels*16, use_pool=True)

        self.up1_SIV = Up(base_channels*16, base_channels*8)
        self.up2_SIV = Up(base_channels*8, base_channels*4)
        self.up3_SIV = Up(base_channels*4, base_channels*2)
        self.outc_SIV = nn.Conv2d(base_channels*2, out_channels_SIV * predict_days, kernel_size=1)

        self.down1_SIC = DoubleConvNoPool(init_channels, base_channels*2)
        self.down2_SIC = Down(base_channels*2, base_channels*4, use_pool=True)
        self.down3_SIC = Down(base_channels*4, base_channels*8, use_pool=True)
        self.down4_SIC = Down(base_channels*8, base_channels*16, use_pool=True)

        self.up1_SIC = Up(base_channels*16, base_channels*8)
        self.up2_SIC = Up(base_channels*8, base_channels*4)
        self.up3_SIC = Up(base_channels*4, base_channels*2)
        self.outc_SIC = nn.Conv2d(base_channels*2, out_channels_SIC * predict_days, kernel_size=1)

        self.down1_FLOW = DoubleConvNoPool(init_channels, base_channels*2)
        self.down2_FLOW = Down(base_channels*2, base_channels*4, use_pool=True)
        self.down3_FLOW = Down(base_channels*4, base_channels*8, use_pool=True)
        self.down4_FLOW = Down(base_channels*8, base_channels*16, use_pool=True)

        self.up1_FLOW = Up(base_channels*16, base_channels*8)
        self.up2_FLOW = Up(base_channels*8, base_channels*4)
        self.up3_FLOW = Up(base_channels*4, base_channels*2)
        self.outc_FLOW = nn.Conv2d(base_channels*2, out_channels_FLOW * predict_days, kernel_size=1)

        self.wam1 = WAM(base_channels*2)
        self.wam2 = WAM(base_channels*4)
        self.wam3 = WAM(base_channels*8)
        self.wam4 = WAM(base_channels*16)
        self.wam5 = WAM(base_channels*8)
        self.wam6 = WAM(base_channels*4)

    def forward(self, x):
        u_channels = [0, 1, 2, 3, 4, 5, 6]
        v_channels = [7, 8, 9, 10, 11, 12, 13]
        u_data = x[:, u_channels, :, :]
        v_data = x[:, v_channels, :, :]

        u_wrapped = self.wrapped_processing(u_data)
        v_wrapped = self.wrapped_processing(v_data)
        uv_flow = torch.stack([u_wrapped, v_wrapped], dim=2)
        uv_flow = uv_flow.view(x.shape[0], self.predict_days * 2, x.shape[2], x.shape[3])

        x_shared = self.inc(x)
        uv_flow = self.inc_flow(uv_flow)

        x_SIV1 = self.down1_SIV(x_shared)
        x_SIC1 = self.down1_SIC(x_shared)
        x_FLOW1 = self.down1_FLOW(uv_flow)
        x_SIV1, x_SIC1, x_FLOW1 = self.wam1(x_SIV1, x_SIC1, x_FLOW1)

        x_SIV2 = self.down2_SIV(x_SIV1)
        x_SIC2 = self.down2_SIC(x_SIC1)
        x_FLOW2 = self.down2_FLOW(x_FLOW1)
        x_SIV2, x_SIC2, x_FLOW2 = self.wam2(x_SIV2, x_SIC2, x_FLOW2)

        x_SIV3 = self.down3_SIV(x_SIV2)
        x_SIC3 = self.down3_SIC(x_SIC2)
        x_FLOW3 = self.down3_FLOW(x_FLOW2)
        x_SIV3, x_SIC3, x_FLOW3 = self.wam3(x_SIV3, x_SIC3, x_FLOW3)

        x_SIV4 = self.down4_SIV(x_SIV3)
        x_SIC4 = self.down4_SIC(x_SIC3)
        x_FLOW4 = self.down4_FLOW(x_FLOW3)
        x_SIV4, x_SIC4, x_FLOW4 = self.wam4(x_SIV4, x_SIC4, x_FLOW4)

        x_SIV5 = self.up1_SIV(x_SIV4, x_SIV3)
        x_SIC5 = self.up1_SIC(x_SIC4, x_SIC3)
        x_FLOW5 = self.up1_FLOW(x_FLOW4, x_FLOW3)
        x_SIV5, x_SIC5, x_FLOW5 = self.wam5(x_SIV5, x_SIC5, x_FLOW5)

        x_SIV6 = self.up2_SIV(x_SIV5, x_SIV2)
        x_SIC6 = self.up2_SIC(x_SIC5, x_SIC2)
        x_FLOW6 = self.up2_FLOW(x_FLOW5, x_FLOW2)
        x_SIV6, x_SIC6, x_FLOW6 = self.wam6(x_SIV6, x_SIC6, x_FLOW6)

        x_SIV = self.up3_SIV(x_SIV6, x_SIV1)
        x_SIC = self.up3_SIC(x_SIC6, x_SIC1)

        SIV = torch.sigmoid(self.outc_SIV(x_SIV))
        SIC = torch.sigmoid(self.outc_SIC(x_SIC))

        SIV = SIV.chunk(self.predict_days, dim=1)
        SIC = SIC.chunk(self.predict_days, dim=1)

        return SIV, SIC

if __name__ == "__main__":
    model = HISUnet(in_channels = 21, predict_days = 7)
    dummy_input = torch.randn(8, 21, 256, 256)
    SIV_pred, SIC_pred = model(dummy_input)
    print(f"SIV Output Shape: {SIV_pred[0].shape}")
    print(f"SIC Output Shape: {SIC_pred[0].shape}")