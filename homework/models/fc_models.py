"""Полносвязные модели для сравнения с CNN."""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class FCModelMNIST(nn.Module):
    """Полносвязная сеть для MNIST (3-4 слоя)."""

    def __init__(self, hidden_sizes=(512, 256, 128), dropout=0.3, num_classes=10):
        super().__init__()
        layers = []
        in_size = 28 * 28
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        self.net = nn.Sequential(*layers)
        logger.info("FCModelMNIST: %d params", self.count_params())

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FCModelCIFAR(nn.Module):
    """Глубокая полносвязная сеть для CIFAR-10."""

    def __init__(self, hidden_sizes=(1024, 512, 256, 128), dropout=0.4, num_classes=10):
        super().__init__()
        layers = []
        in_size = 3 * 32 * 32
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        self.net = nn.Sequential(*layers)
        logger.info("FCModelCIFAR: %d params", self.count_params())

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
