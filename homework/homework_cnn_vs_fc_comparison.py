"""
Задание 1: Сравнение CNN и полносвязных сетей на MNIST и CIFAR-10.

1.1  FC vs SimpleCNN vs ResNet на MNIST   (20 баллов)
1.2  FC vs ResNet vs ResNet+Dropout на CIFAR-10 (20 баллов)

Для каждой модели:
 - одинаковые гиперпараметры
 - точность train/test
 - время обучения и инференса
 - кривые обучения, confusion matrix, gradient flow
 - количество параметров
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
logger = logging.getLogger('hw1')

from models.fc_models import FCModelMNIST, FCModelCIFAR
from models.cnn_models import SimpleCNNMNIST, ResNetMNIST, SimpleCNNCIFAR
from utils.comparison_utils import (
    get_mnist_loaders, get_cifar10_loaders,
    run_comparison, print_comparison_table,
    CIFAR10_CLASSES, MNIST_CLASSES,
)
from utils.visualization_utils import (
    plot_learning_curves, plot_comparison_bar,
    plot_confusion_matrix, plot_gradient_flow,
)
from utils.training_utils import get_device, set_seed, run_experiment, compute_gradient_norms
import torch


def task1_1_mnist(num_epochs: int = 10):
    """
    1.1 Сравнение на MNIST:
    FC (3-4 слоя) | Простая CNN (2-3 conv) | CNN c Residual Block
    """
    logger.info('=== Task 1.1: MNIST Comparison ===')
    device = get_device()
    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    models = {
        'FC-MNIST': FCModelMNIST(hidden_sizes=(512, 256, 128)),
        'SimpleCNN-MNIST': SimpleCNNMNIST(),
        'ResNet-MNIST': ResNetMNIST(),
    }

    results = run_comparison(
        models_dict=models,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        device=device,
        save_dir='results/mnist_comparison',
    )

    print_comparison_table(results)

    # Визуализация
    plot_learning_curves(results, 'plots/mnist_learning_curves.png', title='MNIST: FC vs CNN')
    plot_comparison_bar(results, 'best_test_acc', 'plots/mnist_acc_bar.png', title='MNIST Test Accuracy')
    plot_comparison_bar(results, 'params', 'plots/mnist_params_bar.png', title='MNIST # Parameters')
    plot_comparison_bar(results, 'train_time_s', 'plots/mnist_time_bar.png', title='MNIST Training Time (s)')

    # Confusion matrix для лучшей модели
    best = max(results, key=lambda r: r['best_test_acc'])
    logger.info('Best MNIST model: %s (acc=%.4f)', best['name'], best['best_test_acc'])

    return results


def task1_2_cifar(num_epochs: int = 15):
    """
    1.2 Сравнение на CIFAR-10:
    FC (глубокая) | CNN + Residual | CNN + Residual + Dropout
    """
    logger.info('=== Task 1.2: CIFAR-10 Comparison ===')
    device = get_device()
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, augment=True)

    models = {
        'FC-CIFAR': FCModelCIFAR(hidden_sizes=(1024, 512, 256, 128)),
        'CNN-ResNet-CIFAR': SimpleCNNCIFAR(use_dropout=False),
        'CNN-ResNet-Dropout-CIFAR': SimpleCNNCIFAR(use_dropout=True),
    }

    results = run_comparison(
        models_dict=models,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        device=device,
        save_dir='results/cifar_comparison',
    )

    print_comparison_table(results)

    plot_learning_curves(results, 'plots/cifar_learning_curves.png', title='CIFAR-10: FC vs CNN')
    plot_comparison_bar(results, 'best_test_acc', 'plots/cifar_acc_bar.png', title='CIFAR-10 Test Accuracy')
    plot_comparison_bar(results, 'train_time_s', 'plots/cifar_time_bar.png', title='CIFAR-10 Training Time (s)')

    # Confusion matrix для лучшей модели
    best_name = max(results, key=lambda r: r['best_test_acc'])['name']
    best_model = models[best_name].to(device)
    plot_confusion_matrix(
        best_model, test_loader, CIFAR10_CLASSES, device,
        f'plots/cifar_confusion_{best_name}.png',
        title=f'CIFAR-10 Confusion Matrix: {best_name}',
    )

    # Gradient flow: последняя эпоха train
    criterion = torch.nn.CrossEntropyLoss()
    best_model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    best_model.zero_grad()
    criterion(best_model(x), y).backward()
    grad_norms = compute_gradient_norms(best_model)
    plot_gradient_flow(grad_norms, f'plots/cifar_gradient_flow_{best_name}.png',
                       title=f'Gradient Flow: {best_name}')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Task 1: CNN vs FC comparison')
    parser.add_argument('--epochs-mnist', type=int, default=10)
    parser.add_argument('--epochs-cifar', type=int, default=15)
    parser.add_argument('--skip-mnist', action='store_true')
    parser.add_argument('--skip-cifar', action='store_true')
    args = parser.parse_args()

    if not args.skip_mnist:
        task1_1_mnist(args.epochs_mnist)
    if not args.skip_cifar:
        task1_2_cifar(args.epochs_cifar)

    logger.info('Task 1 complete. Plots saved to ./plots/, results to ./results/')
