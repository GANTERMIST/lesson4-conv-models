"""Утилиты для сравнения моделей: запуск экспериментов и форматирование таблиц."""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import torch
import torchvision
import torchvision.transforms as T

from utils.training_utils import run_experiment, get_device, set_seed

logger = logging.getLogger(__name__)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
MNIST_CLASSES = [str(i) for i in range(10)]


def get_mnist_loaders(batch_size: int = 128, data_dir: str = './data'):
    """Загружает MNIST с нормализацией."""
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def get_cifar10_loaders(batch_size: int = 128, data_dir: str = './data', augment: bool = True):
    """Загружает CIFAR-10 с аугментацией для train."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    if augment:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        train_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def run_comparison(
    models_dict: Dict,
    train_loader,
    test_loader,
    num_epochs: int = 10,
    device=None,
    save_dir: str = 'results',
) -> List[Dict]:
    """Запускает сравнение нескольких моделей.

    Args:
        models_dict: {name: model_instance}
    Returns:
        Список словарей с результатами.
    """
    if device is None:
        device = get_device()
    results = []
    for name, model in models_dict.items():
        set_seed(42)
        logger.info('Running experiment: %s', name)
        result = run_experiment(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            device=device,
            name=name,
        )
        results.append(result)
    # Сохраняем JSON (history отдельно)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    summary = [{k: v for k, v in r.items() if k != 'history'} for r in results]
    out_path = os.path.join(save_dir, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info('Saved summary to %s', out_path)
    return results


def print_comparison_table(results: List[Dict]) -> None:
    """Выводит таблицу сравнения в консоль."""
    header = f"{'Model':<25} {'Params':>10} {'Best Acc':>10} {'Train Acc':>10} {'Time(s)':>10} {'Infer(ms)':>12}"
    print('\n' + '=' * len(header))
    print(header)
    print('=' * len(header))
    for r in results:
        print(
            f"{r['name']:<25} {r['params']:>10,} {r['best_test_acc']:>10.4f} "
            f"{r['final_train_acc']:>10.4f} {r['train_time_s']:>10.1f} {r['infer_ms']:>12.4f}"
        )
    print('=' * len(header))
