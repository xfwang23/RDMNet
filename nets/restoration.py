import torch
from torch import nn


class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels * s_factor, 1, stride=1, padding=0))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0))

    def forward(self, x):
        x = self.up(x)
        return x


class Focus(nn.Module):
    def __init__(self, in_channels, n_feats, ksize=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 4, n_feats, ksize, stride, padding=ksize // 2)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, n_feats, reduction, relu=True):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, kernel_size=1),
            nn.ReLU(inplace=True) if relu else nn.SiLU(True),
            nn.Conv2d(n_feats // reduction, n_feats, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.ca(self.avg_pool(x))
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feats, reduction, relu=True):
        super().__init__()
        self.conv_1 = nn.Conv2d(n_feats, n_feats // reduction, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True) if relu else nn.SiLU(inplace=True)
        self.conv_2 = nn.Conv2d(n_feats // reduction, n_feats, kernel_size=3, padding=1)
        self.ca = ChannelAttention(n_feats, reduction, relu=relu)

    def forward(self, x):
        x = self.conv_2(self.act(self.conv_1(x)))
        x = self.ca(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, n_feats=32, reduction=8, relu=True):
        super().__init__()
        self.num_blocks = [1, 1, 1, 1, 1]
        # 640x640
        self.emb_in = Focus(in_channels, n_feats, ksize=3, stride=1)
        # 320x320
        self.e1 = nn.Sequential(*[CAB(n_feats, reduction, relu=relu) for _ in range(self.num_blocks[0])])
        self.down2 = DownSample(n_feats, s_factor=2)
        # 160x160
        self.e2 = nn.Sequential(*[CAB(n_feats * 2, reduction, relu=relu) for _ in range(self.num_blocks[1])])
        self.down3 = DownSample(n_feats * 2, s_factor=2)
        # 80x80
        self.e3 = nn.Sequential(*[CAB(n_feats * 4, reduction, relu=relu) for _ in range(self.num_blocks[2])])
        self.down4 = DownSample(n_feats * 4, s_factor=2)
        # 40x40
        self.e4 = nn.Sequential(*[CAB(n_feats * 8, reduction, relu=relu) for _ in range(self.num_blocks[3])])
        self.down5 = DownSample(n_feats * 8, s_factor=2)
        # 20x20
        self.e5 = nn.Sequential(*[CAB(n_feats * 16, reduction, relu=relu) for _ in range(self.num_blocks[4])])

        self.att1 = nn.Sequential(*[nn.Conv2d(n_feats, n_feats, kernel_size=1), nn.Sigmoid()])
        self.att2 = nn.Sequential(*[nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=1), nn.Sigmoid()])
        self.att3 = nn.Sequential(*[nn.Conv2d(n_feats * 4, n_feats * 4, kernel_size=1), nn.Sigmoid()])
        self.att4 = nn.Sequential(*[nn.Conv2d(n_feats * 8, n_feats * 8, kernel_size=1), nn.Sigmoid()])
        self.att5 = nn.Sequential(*[nn.Conv2d(n_feats * 16, n_feats * 16, kernel_size=1), nn.Sigmoid()])

    def forward(self, x, inter):
        # 640x640x32
        emb = self.emb_in(x)
        # 320x320x64
        e1 = self.e1(self.att1(emb + inter[0]) * emb)
        down2 = self.down2(e1)
        # 160x160x128
        e2 = self.e2(self.att2(down2 + inter[1]) * down2)
        down3 = self.down3(e2)
        # 80x80x256
        e3 = self.e3(self.att3(down3 + inter[2]) * down3)
        down4 = self.down4(e3)
        # 40x40x512
        e4 = self.e4(self.att4(down4 + inter[3]) * down4)
        down5 = self.down5(e4)
        # 20x20x1024
        e5 = self.e5(self.att5(down5 + inter[4]) * down5)

        return e1, e2, e3, e4, e5


class Decoder(nn.Module):
    def __init__(self, out_channel=3, n_feats=32, reduction=8, relu=True):
        super().__init__()
        self.num_blocks = [1, 1]
        # 80x80x256
        self.up3 = UpSample(n_feats * 4, s_factor=2)
        # 160x160x128
        self.d2 = nn.Sequential(*[CAB(n_feats * 2, reduction, relu=relu) for _ in range(self.num_blocks[0])])
        self.up2 = UpSample(n_feats * 2, s_factor=2)
        # 320x320x64
        self.d1 = nn.Sequential(*[CAB(n_feats * 1, reduction, relu=relu) for _ in range(self.num_blocks[1])])
        self.up1 = UpSample(n_feats * 1, s_factor=2)
        # 640x640x32
        self.emb_out = nn.Sequential(
            CAB(n_feats // 2, reduction, relu=relu),
            nn.Conv2d(n_feats // 2, out_channel, kernel_size=3, padding=1),
        )

    def forward(self, x, e, img):
        # x 为多尺度特征融合后的最后一个尺度的特征
        up3 = self.up3(x)
        d2 = self.d2(up3 + e[1])
        up2 = self.up2(d2)
        d1 = self.d1(up2 + e[0])
        up1 = self.up1(d1)
        out = self.emb_out(up1)
        return out + img


