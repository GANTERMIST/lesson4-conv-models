"""Кастомные слои: Attention, активации, pooling, Residual варианты."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -------- 3.1: Кастомные слои --------

class ChannelAttention(nn.Module):
    """Канальный Attention (упрощённый SE-block)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 1), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """Пространственный Attention для CNN."""

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        w = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * w


class Swish(nn.Module):
    """Кастомная функция активации Swish."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Кастомная функция активации Mish."""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class StochasticDepthConv(nn.Module):
    """Кастомный сверточный слой co stochastic depth."""

    def __init__(self, in_ch, out_ch, kernel_size=3, drop_prob=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.drop_prob = drop_prob

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        if self.training and torch.rand(1).item() < self.drop_prob:
            return torch.zeros_like(out)
        return out


class AdaptiveStridedPool(nn.Module):
    """Кастомный pooling: среднее + максимальное с весами."""

    def __init__(self, output_size=1):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size)
        self.max = nn.AdaptiveMaxPool2d(output_size)
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        w = torch.sigmoid(self.weight)
        return w * self.avg(x) + (1 - w) * self.max(x)


# -------- 3.2: Варианты Residual блоков --------

class BasicResBlock(nn.Module):
    """Базовый Residual блок (identity shortcut)."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class BottleneckResBlock(nn.Module):
    """Боттлнецк Residual блок (1x1 -> 3x3 -> 1x1)."""

    def __init__(self, channels, bottleneck_ratio=4):
        super().__init__()
        mid = max(channels // bottleneck_ratio, 1)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class WideResBlock(nn.Module):
    """Широкий Residual блок (wide channels)."""

    def __init__(self, channels, width_mult=2):
        super().__init__()
        wide = channels * width_mult
        self.conv1 = nn.Conv2d(channels, wide, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(wide)
        self.conv2 = nn.Conv2d(wide, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# -------- Модели с кастомными слоями --------

class AttentionCNN(nn.Module):
    """CNN с Channel + Spatial Attention."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            AdaptiveStridedPool(1), nn.Flatten(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.ca(x)
        x = self.sa(x)
        x = self.layer2(x)
        return self.classifier(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResBlockVariantNet(nn.Module):
    """CNN с выбираемым типом Residual блока."""

    def __init__(self, block_type='basic', num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        blocks = {'basic': BasicResBlock, 'bottleneck': BottleneckResBlock, 'wide': WideResBlock}
        BlkCls = blocks[block_type]
        self.layer1 = BlkCls(64)
        self.pool = nn.MaxPool2d(2)
        self.layer2 = BlkCls(64)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(self.layer1(x))
        x = self.layer2(x)
        return self.classifier(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
