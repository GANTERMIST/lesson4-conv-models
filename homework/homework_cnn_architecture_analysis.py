"""
Задание 2: Анализ архитектур CNN.

2.1  Влияние размера ядра свертки (15 баллов)
     - 3x3, 5x5, 7x7, 1x1+3x3
     - одинаковое (приблизительно) количество параметров
     - визуализация активаций первого слоя

2.2  Влияние глубины CNN (15 баллов)
     - 2, 4, 6 conv-слоёв + CNN c Residual-связями
     - vanishing/exploding gradients
     - feature maps
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
logger = logging.getLogger('hw2')

from models.cnn_models import CNNKernelStudy, CNNMixedKernel, build_depth_cnn, SimpleCNNCIFAR
from utils.comparison_utils import get_cifar10_loaders, run_comparison, print_comparison_table
from utils.visualization_utils import (
    plot_learning_curves, plot_comparison_bar,
    plot_feature_maps, plot_gradient_flow,
)
from utils.training_utils import get_device, set_seed, run_experiment, compute_gradient_norms
import torch


def task2_1_kernel_size(num_epochs: int = 10):
    """
    2.1 Исследуем влияние размера ядра.
    Анализ: рецептивное поле, точность, время.
    """
    logger.info('=== Task 2.1: Kernel Size Analysis ===')
    device = get_device()
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, augment=True)

    # Рецептивные поля
    receptive_fields = {3: 'RF=5 (2x3)', 5: 'RF=9 (2x5)', 7: 'RF=13 (2x7)'}
    for ks, rf in receptive_fields.items():
        params = CNNKernelStudy(kernel_size=ks).count_params()
        logger.info('kernel=%dx%d  params=%d  %s', ks, ks, params, rf)
    logger.info('Mixed(1x1+3x3)  params=%d  RF~5', CNNMixedKernel().count_params())

    models = {
        'CNN-3x3': CNNKernelStudy(kernel_size=3),
        'CNN-5x5': CNNKernelStudy(kernel_size=5),
        'CNN-7x7': CNNKernelStudy(kernel_size=7),
        'CNN-Mixed(1x1+3x3)': CNNMixedKernel(),
    }

    results = run_comparison(
        models_dict=models,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        device=device,
        save_dir='results/architecture_analysis/kernel',
    )

    print_comparison_table(results)
    plot_learning_curves(results, 'plots/kernel_learning_curves.png', title='Kernel Size Comparison')
    plot_comparison_bar(results, 'best_test_acc', 'plots/kernel_acc_bar.png', title='Kernel: Test Accuracy')
    plot_comparison_bar(results, 'infer_ms', 'plots/kernel_infer_bar.png', title='Kernel: Inference Time (ms)')

    # Визуализация активаций первого слоя
    sample_img = next(iter(test_loader))[0][0]  # одно изображение
    for name, model in models.items():
        model = model.to(device)
        plot_feature_maps(
            model, sample_img.to(device), 'features.0',
            f'plots/feature_maps_{name.replace("/","_")}.png',
        )

    return results


def task2_2_depth(num_epochs: int = 10):
    """
    2.2 Исследуем влияние глубины CNN.
    Анализ: vanishing/exploding gradients, feature maps.
    """
    logger.info('=== Task 2.2: Depth Analysis ===')
    device = get_device()
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, augment=True)

    models = {
        'CNN-Shallow(2)': build_depth_cnn(n_conv_layers=2),
        'CNN-Medium(4)': build_depth_cnn(n_conv_layers=4),
        'CNN-Deep(6)': build_depth_cnn(n_conv_layers=6),
        'CNN-ResNet': SimpleCNNCIFAR(use_dropout=True),
    }

    results = run_comparison(
        models_dict=models,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        device=device,
        save_dir='results/architecture_analysis/depth',
    )

    print_comparison_table(results)
    plot_learning_curves(results, 'plots/depth_learning_curves.png', title='CNN Depth Comparison')
    plot_comparison_bar(results, 'best_test_acc', 'plots/depth_acc_bar.png', title='Depth: Test Accuracy')

    # Gradient flow для каждой модели
    criterion = torch.nn.CrossEntropyLoss()
    x_batch, y_batch = next(iter(train_loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    for name, model in models.items():
        model = model.to(device)
        model.train()
        model.zero_grad()
        criterion(model(x_batch), y_batch).backward()
        gnorms = compute_gradient_norms(model)
        plot_gradient_flow(gnorms, f'plots/depth_gradient_{name}.png',
                           title=f'Gradient Flow: {name}')
        logger.info('[%s] grad norms: min=%.2e  max=%.2e',
                    name, min(gnorms.values()), max(gnorms.values()))

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Task 2: CNN Architecture Analysis')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--skip-kernel', action='store_true')
    parser.add_argument('--skip-depth', action='store_true')
    args = parser.parse_args()

    if not args.skip_kernel:
        task2_1_kernel_size(args.epochs)
    if not args.skip_depth:
        task2_2_depth(args.epochs)

    logger.info('Task 2 complete. Plots saved to ./plots/, results to ./results/')
