"""Сверточные модели: простая CNN, CNN c Residual, модели для CIFAR."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Базовый Residual блок."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class SimpleCNNMNIST(nn.Module):
    """Простая CNN для MNIST (2-3 conv слоя)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNetMNIST(nn.Module):
    """CNN с Residual блоками для MNIST."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.layer1 = ResidualBlock(32)
        self.pool1 = nn.MaxPool2d(2)
        self.layer2 = ResidualBlock(32)
        self.pool2 = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        return self.classifier(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleCNNCIFAR(nn.Module):
    """CNN с Residual блоками для CIFAR-10."""

    def __init__(self, num_classes=10, use_dropout=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.res1 = ResidualBlock(64)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.res2 = ResidualBlock(128)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4) if use_dropout else nn.Identity(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.res1(x)
        x = self.block2(x)
        x = self.res2(x)
        return self.classifier(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------- Модели для анализа архитектур (Задание 2) ----------


class CNNKernelStudy(nn.Module):
    """Сеть с выбираемым размером ядра свертки."""

    def __init__(self, kernel_size=3, num_classes=10):
        super().__init__()
        pad = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNMixedKernel(nn.Module):
    """Комбинация 1x1 + 3x3 ядер."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.conv2(self.conv1(x)))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_depth_cnn(n_conv_layers: int, num_classes=10) -> nn.Module:
    """Строит CNN с заданной глубиной (2, 4 или 6+ слоёв)."""

    class DepthCNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_ch = 3
            out_ch = 32
            for i in range(n_conv_layers):
                layers += [
                    nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                ]
                if i % 2 == 1:
                    layers.append(nn.MaxPool2d(2))
                in_ch = out_ch
                if i < n_conv_layers - 1:
                    out_ch = min(out_ch * 2, 256)
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, num_classes)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

        def count_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return DepthCNN()
