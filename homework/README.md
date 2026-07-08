# Homework: Lesson 4 — Convolutional Neural Networks

This homework explores CNNs, architectural analysis, and custom layers using PyTorch on MNIST and CIFAR-10 datasets.

## Project Structure

```
homework/
├── models/
│   ├── fc_models.py          # Fully-connected baseline models
│   ├── cnn_models.py         # CNN architectures (depth, width, kernel studies)
│   └── custom_layers.py      # Custom layers: ChannelAttention, SpatialAttention,
│                             #   Swish, Mish, StochasticDepthConv, residual blocks
├── utils/
│   ├── training_utils.py     # Training loop, evaluation, seed, device utils
│   ├── visualization_utils.py# Plotting learning curves, confusion matrix, feature maps
│   └── comparison_utils.py   # Data loaders, run_comparison, print_comparison_table
├── homework_cnn_vs_fc_comparison.py     # Task 1: CNN vs FC comparison
├── homework_cnn_architecture_analysis.py # Task 2: CNN architecture analysis
├── homework_custom_layers_experiments.py # Task 3: Custom layers & residual blocks
└── README.md
```

## Tasks

### Task 1: CNN vs FC Comparison (`homework_cnn_vs_fc_comparison.py`)
- Compare FC and CNN models on MNIST and CIFAR-10
- Analyze accuracy, parameters, and inference time
- Visualize learning curves and bar charts

### Task 2: CNN Architecture Analysis (`homework_cnn_architecture_analysis.py`)
- Study effect of **depth** (2 to 6 conv layers)
- Study effect of **width** (16 to 256 channels)
- Study effect of **kernel size** (3x3, 5x5, 7x7, mixed)
- Study **regularization** (Dropout, BatchNorm, Weight Decay)

### Task 3: Custom Layers & Residual Experiments (`homework_custom_layers_experiments.py`)
- Unit tests for custom layers (forward pass shapes, gradients)
- Compare residual block variants: Basic, Bottleneck, Wide, Attention
- Custom activations: Swish, Mish
- StochasticDepthConv, AdaptiveStridedPool

## Running

```bash
# Task 1
python homework/homework_cnn_vs_fc_comparison.py --epochs 10

# Task 2
python homework/homework_cnn_architecture_analysis.py --epochs 10

# Task 3
python homework/homework_custom_layers_experiments.py --epochs 10

# Skip unit tests
python homework/homework_custom_layers_experiments.py --skip-unit-tests
```

## Requirements

- Python 3.8+
- PyTorch >= 1.12
- torchvision
- matplotlib
- numpy
- tabulate

```bash
pip install torch torchvision matplotlib numpy tabulate
```

## Results

Results and plots are saved to:
- `results/cnn_vs_fc/` — Task 1 outputs
- `results/architecture_analysis/` — Task 2 outputs  
- `results/residual_variants/` — Task 3 outputs
- `plots/` — All generated figures

## Notes

- All experiments use `set_seed(42)` for reproducibility
- GPU is used automatically if available (`get_device()`)
- Default: 10 epochs; increase for better convergence
