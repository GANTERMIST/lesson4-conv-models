"""
Задание 3: Кастомные слои и эксперименты.

3.1  Кастомные слои (15 баллов)
     - ChannelAttention (SE-block), SpatialAttention
     - активации Swish / Mish vs ReLU
     - кастомный pooling (avg + max с обучаемым весом)
     - StochasticDepthConv против стандартного conv

3.2  Эксперименты с Residual блоками (15 баллов)
     - BasicResBlock vs BottleneckResBlock vs WideResBlock
     - параметры, точность, стабильность
"""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('hw3')

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.custom_layers import (
    ChannelAttention, SpatialAttention, Swish, Mish,
    StochasticDepthConv, AdaptiveStridedPool,
    BasicResBlock, BottleneckResBlock, WideResBlock,
    AttentionCNN, ResBlockVariantNet,
)
from utils.comparison_utils import get_cifar10_loaders, run_comparison, print_comparison_table
from utils.visualization_utils import plot_learning_curves, plot_comparison_bar
from utils.training_utils import get_device, set_seed


def test_custom_layers_forward():
    """Тестируем forward-проход всех кастомных слоёв."""
    logger.info('=== Unit tests for custom layers ===')
    x = torch.randn(4, 64, 16, 16)

    # ChannelAttention
    ca = ChannelAttention(64)
    out = ca(x)
    assert out.shape == x.shape, f'ChannelAttention shape mismatch: {out.shape}'
    logger.info('ChannelAttention: OK  params=%d', sum(p.numel() for p in ca.parameters()))

    # SpatialAttention
    sa = SpatialAttention()
    out = sa(x)
    assert out.shape == x.shape
    logger.info('SpatialAttention: OK  params=%d', sum(p.numel() for p in sa.parameters()))

    # Swish vs ReLU
    swish = Swish()
    relu = nn.ReLU()
    inp = torch.linspace(-3, 3, 100)
    sw_out = swish(inp)
    re_out = relu(inp)
    logger.info('Swish range: [%.3f, %.3f]', sw_out.min().item(), sw_out.max().item())
    logger.info('ReLU range:  [%.3f, %.3f]', re_out.min().item(), re_out.max().item())

    # Mish
    mish = Mish()
    mi_out = mish(inp)
    logger.info('Mish range:  [%.3f, %.3f]', mi_out.min().item(), mi_out.max().item())

    # StochasticDepthConv
    sdc = StochasticDepthConv(64, 64)
    out = sdc(x)
    assert out.shape == x.shape
    logger.info('StochasticDepthConv: OK')

    # AdaptiveStridedPool
    pool = AdaptiveStridedPool(output_size=1)
    out = pool(x)
    assert out.shape == (4, 64, 1, 1)
    logger.info('AdaptiveStridedPool: OK  learned_weight=%.4f', torch.sigmoid(pool.weight).item())

    # Residual blocks
    for BlkCls, name in [(BasicResBlock, 'Basic'), (BottleneckResBlock, 'Bottleneck'), (WideResBlock, 'Wide')]:
        blk = BlkCls(64)
        out = blk(x)
        assert out.shape == x.shape
        logger.info('%sResBlock: OK  params=%d', name, blk.count_params())

    logger.info('All unit tests passed!')


def task3_1_custom_layers_comparison(num_epochs: int = 10):
    """
    3.1 Сравнение кастомных слоёв на CIFAR-10:
    базовая CNN vs CNN+Attention vs CNN+Swish
    """
    logger.info('=== Task 3.1: Custom Layers Comparison ===')
    device = get_device()
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, augment=True)

    # Базовая CNN (без attention) - используем ResBlockVariantNet c basic block
    class BaselineCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes),
            )
        def forward(self, x): return self.net(x)
        def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

    class SwishCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), Swish(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), Swish(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes),
            )
        def forward(self, x): return self.net(x)
        def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

    models = {
        'Baseline-CNN': BaselineCNN(),
        'AttentionCNN': AttentionCNN(),
        'Swish-CNN': SwishCNN(),
    }

    results = run_comparison(
        models_dict=models,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        device=device,
        save_dir='results/custom_layers',
    )

    print_comparison_table(results)
    plot_learning_curves(results, 'plots/custom_layers_curves.png', title='Custom Layers Comparison')
    plot_comparison_bar(results, 'best_test_acc', 'plots/custom_layers_acc_bar.png',
                        title='Custom Layers: Test Accuracy')
    return results


def task3_2_residual_variants(num_epochs: int = 10):
    """
    3.2 Сравнение вариантов Residual блоков.
    """
    logger.info('=== Task 3.2: Residual Block Variants ===')
    device = get_device()
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, augment=True)

    models = {
        'ResNet-Basic': ResBlockVariantNet(block_type='basic'),
        'ResNet-Bottleneck': ResBlockVariantNet(block_type='bottleneck'),
        'ResNet-Wide': ResBlockVariantNet(block_type='wide'),
    }

    # Параметры блоков
    for name, blk_cls in [('Basic', BasicResBlock), ('Bottleneck', BottleneckResBlock), ('Wide', WideResBlock)]:
        blk = blk_cls(64)
        logger.info('[%s] block params = %d', name, blk.count_params())

    results = run_comparison(
        models_dict=models,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        device=device,
        save_dir='results/residual_variants',
    )

    print_comparison_table(results)
    plot_learning_curves(results, 'plots/residual_variants_curves.png', title='Residual Block Variants')
    plot_comparison_bar(results, 'best_test_acc', 'plots/residual_variants_acc.png',
                        title='Residual Variants: Test Accuracy')
    plot_comparison_bar(results, 'params', 'plots/residual_variants_params.png',
                        title='Residual Variants: # Parameters')
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Task 3: Custom Layers and Residual Experiments')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--skip-unit-tests', action='store_true')
    parser.add_argument('--skip-custom', action='store_true')
    parser.add_argument('--skip-residual', action='store_true')
    args = parser.parse_args()

    if not args.skip_unit_tests:
        test_custom_layers_forward()
    if not args.skip_custom:
        task3_1_custom_layers_comparison(args.epochs)
    if not args.skip_residual:
        task3_2_residual_variants(args.epochs)

    logger.info('Task 3 complete. Plots saved to ./plots/, results to ./results/')
