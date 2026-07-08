"""Функции визуализации: кривые обучения, confusion matrix, фечер-мапы, градиенты."""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_learning_curves(
    results: List[Dict],
    save_path: str,
    title: str = 'Learning Curves',
) -> None:
    """Центральная визуализация: loss + accuracy для всех моделей."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14)
    for res in results:
        h = res['history']
        epochs = range(1, len(h['train_loss']) + 1)
        axes[0].plot(epochs, h['train_loss'], label=f"{res['name']} train")
        axes[0].plot(epochs, h['test_loss'], '--', label=f"{res['name']} test")
        axes[1].plot(epochs, h['train_acc'], label=f"{res['name']} train")
        axes[1].plot(epochs, h['test_acc'], '--', label=f"{res['name']} test")
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(fontsize=7)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info('Saved learning curves to %s', save_path)


def plot_comparison_bar(
    results: List[Dict],
    metric: str,
    save_path: str,
    title: str = '',
) -> None:
    """Сравнительная барная диаграмма по выбранной метрике."""
    names = [r['name'] for r in results]
    values = [r[metric] for r in results]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    bars = ax.bar(names, values, color='steelblue', edgecolor='black')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(title or f'Comparison: {metric}')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info('Saved bar chart to %s', save_path)


def plot_confusion_matrix(
    model: nn.Module,
    loader,
    class_names: List[str],
    device: torch.device,
    save_path: str,
    title: str = 'Confusion Matrix',
) -> None:
    """Строит и сохраняет confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=7,
                    color='white' if cm_norm[i, j] > 0.5 else 'black')
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info('Saved confusion matrix to %s', save_path)


def plot_feature_maps(
    model: nn.Module,
    image: torch.Tensor,
    layer_name: str,
    save_path: str,
    n_maps: int = 16,
) -> None:
    """Визуализирует feature maps после заданного слоя."""
    activations = {}

    def hook_fn(module, inp, out):
        activations['feat'] = out.detach().cpu()

    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    if handle is None:
        logger.warning('Layer %s not found', layer_name)
        return

    model.eval()
    with torch.no_grad():
        model(image.unsqueeze(0) if image.dim() == 3 else image)
    handle.remove()

    feat = activations.get('feat')
    if feat is None:
        return
    feat = feat[0]  # first sample
    n = min(n_maps, feat.shape[0])
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).ravel()
    for i in range(n):
        axes[i].imshow(feat[i].numpy(), cmap='viridis')
        axes[i].set_title(f'ch {i}', fontsize=8)
        axes[i].axis('off')
    for i in range(n, len(axes)):
        axes[i].axis('off')
    plt.suptitle(f'Feature maps: {layer_name}', fontsize=12)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=100)
    plt.close()
    logger.info('Saved feature maps to %s', save_path)


def plot_gradient_flow(
    grad_norms: Dict[str, float],
    save_path: str,
    title: str = 'Gradient Flow',
) -> None:
    """Визуализирует нормы градиентов для каждого слоя."""
    layers = list(grad_norms.keys())
    norms = list(grad_norms.values())
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.4), 5))
    ax.plot(norms, marker='o', markersize=4)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=6)
    ax.set_ylabel('Gradient Norm')
    ax.set_title(title)
    ax.set_yscale('log')
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=100)
    plt.close()
    logger.info('Saved gradient flow to %s', save_path)
