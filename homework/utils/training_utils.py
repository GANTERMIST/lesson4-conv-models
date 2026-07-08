"""Утилиты для обучения и оценки моделей."""
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    """Одна эпоха обучения. Возвращает (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    """Оценка модели. Возвращает (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def measure_inference_time(
    model: nn.Module,
    input_shape: Tuple,
    device: torch.device,
    n_runs: int = 100,
) -> float:
    """Измеряет среднее время инференса (ms)."""
    model.eval()
    dummy = torch.randn(input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):  # warm-up
            model(dummy)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    return elapsed


def run_experiment(
    model: nn.Module,
    train_loader,
    test_loader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    name: str = 'model',
) -> Dict:
    """Полный цикл обучения + логирование результатов."""
    if device is None:
        device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("[%s] params=%d, device=%s", name, n_params, device)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    train_start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)
        logger.info(
            "[%s] epoch %02d/%02d  train=%.4f/%.4f  test=%.4f/%.4f",
            name, epoch, num_epochs, tr_loss, tr_acc, te_loss, te_acc,
        )

    train_time = time.perf_counter() - train_start
    # Инференс для одного батча
    sample_batch = next(iter(test_loader))[0][:1]
    infer_ms = measure_inference_time(model, tuple(sample_batch.shape), device)

    result = {
        'name': name,
        'params': n_params,
        'train_time_s': round(train_time, 2),
        'infer_ms': round(infer_ms, 4),
        'best_test_acc': max(history['test_acc']),
        'final_train_acc': history['train_acc'][-1],
        'history': history,
    }
    logger.info("[%s] DONE  best_test=%.4f  time=%.1fs", name, result['best_test_acc'], train_time)
    return result


def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """Возвращает нормы градиентов для каждого параметра (gradient flow analysis)."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms
