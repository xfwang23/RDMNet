from torch import nn
from nets.moco import MoCo


class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.SiLU(True),
            nn.Conv2d(out_feat, out_feat, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat)
        )

    def forward(self, x):
        return nn.SiLU(True)(self.backbone(x) + self.shortcut(x))


class ResEncoder(nn.Module):
    def __init__(self, in_channel=3, n_feat=32, reduction=8):
        super(ResEncoder, self).__init__()
        self.emb_in = nn.Conv2d(in_channel, n_feat, kernel_size=3, padding=1, bias=False)
        self.E1 = ResBlock(in_feat=n_feat, out_feat=n_feat, stride=2)  # 320x320x32
        self.E2 = ResBlock(in_feat=n_feat, out_feat=n_feat * 2, stride=2)  # 160x160x64
        self.E3 = ResBlock(in_feat=n_feat * 2, out_feat=n_feat * 4, stride=2)  # 80x80
        self.E4 = ResBlock(in_feat=n_feat * 4, out_feat=n_feat * 8, stride=2)  # 40x40
        self.E5 = ResBlock(in_feat=n_feat * 8, out_feat=n_feat * 16, stride=2)  # 20x20

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_feat * 16, n_feat * 8),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feat * 8, 256),
        )

    def forward(self, x):
        emb = self.emb_in(x)
        l1 = self.E1(emb)
        l2 = self.E2(l1)
        l3 = self.E3(l2)
        l4 = self.E4(l3)
        l5 = self.E5(l4)
        out = self.mlp(l5)

        return out, (l1, l2, l3, l4, l5)


class UDE(nn.Module):
    def __init__(self):
        super(UDE, self).__init__()
        self.moco = MoCo(base_encoder=ResEncoder, dim=256, K=12 * 256)

    def forward(self, x_query, x_key=None):
        if self.training:
            # degradation-aware represenetion learning
            logits, labels, inter = self.moco(x_query, x_key)

            return logits, labels, inter
        else:
            # degradation-aware represenetion learning
            inter = self.moco(x_query)
            return inter
