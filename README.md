<p align="center">
  <img src="https://raw.githubusercontent.com/SattamAltwaim/PILOT/assets/header.png" alt="PILOT" width="100%">
</p>

<h1 align="center">PILOT: Policy-Informed Learned Optimization for Adaptive Deep Network Training</h1>

<p align="center">
  <a href="https://pypi.org/project/pilot-optimizer/"><img src="https://img.shields.io/pypi/v/pilot-optimizer?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://arxiv.org/abs/2605.24570"><img src="https://img.shields.io/badge/arXiv-2605.24570-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://github.com/SattamAltwaim/PILOT/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
</p>

<p align="center">
  <a href="https://sattamaltwaim.github.io/PILOT/">Project Page</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2605.24570">Paper</a> &nbsp;|&nbsp;
  <a href="https://pypi.org/project/pilot-optimizer/">PyPI</a>
</p>

---



**PILOT** is an online adaptive optimizer that adjusts its update behavior during training. Instead of applying a fixed update rule from the first step to the last, PILOT reads a gradient-direction agreement signal and reshapes the update through a lightweight learned policy — no offline search, no meta-training, no second-order estimation.

## Installation

```bash
pip install pilot-optimizer
```

<details>
<summary>Or install from source</summary>

```bash
git clone https://github.com/SattamAltwaim/PILOT.git
cd PILOT
pip install -e .
```
</details>

## Usage

```python
from pilot import PILOT

optimizer = PILOT(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    gamma=0.95,        # smoothing for agreement signal
    eta_phi=0.01,      # policy learning rate
    degree=2           # polynomial degree
)

for batch in dataloader:
    loss = criterion(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Key Results

### CNN Architecture

| Dataset | Optimizer | Accuracy (%) ↑ | Val Loss ↓ | Loss Var. ↓ |
|---|---|---|---|---|
| FashionMNIST | Adam | 93.28 | 0.1957 | **0.0033** |
| FashionMNIST | AdamW | 93.22 | 0.1944 | 0.0034 |
| FashionMNIST | Lion | 92.91 | 0.2091 | 0.0041 |
| FashionMNIST | AdaBelief | 93.66 | 0.1822 | 0.0046 |
| FashionMNIST | **PILOT (Ours)** | **94.13** | **0.1719** | 0.0045 |
| CIFAR-10 | Adam | 79.91 | 0.5794 | 0.0103 |
| CIFAR-10 | Lion | 80.87 | 0.5487 | 0.0105 |
| CIFAR-10 | **PILOT (Ours)** | **81.94** | **0.5302** | **0.0073** |

### ResNet-18 Architecture

| Dataset | Optimizer | Accuracy (%) ↑ | Val Loss ↓ | Loss Var. ↓ |
|---|---|---|---|---|
| FashionMNIST | AdaBelief | 95.33 | 0.1711 | 0.0056 |
| FashionMNIST | **PILOT (Ours)** | **95.71** | 0.2690 | 0.0030 |
| CIFAR-10 | Adam | 93.18 | **0.2140** | 0.0073 |
| CIFAR-10 | AdamW | 92.90 | 0.2514 | 0.0066 |
| CIFAR-10 | **PILOT (Ours)** | **93.42** | 0.2496 | **0.0001** |

Full results with all baselines in the [paper](https://arxiv.org/abs/2605.24570).

### Training Curves

<p align="center">
  <img src="https://raw.githubusercontent.com/SattamAltwaim/PILOT/assets/fig1_loss_accuracy.png" alt="Training loss and validation accuracy" width="100%">
</p>

<p align="center"><em>Training loss (left, middle) and validation accuracy (right) across FashionMNIST (top) and CIFAR-10 (bottom) over 30 epochs.</em></p>

---

## Loss Landscape

<p align="center">
  <img src="https://raw.githubusercontent.com/SattamAltwaim/PILOT/assets/fig2_landscape.png" alt="Loss landscape trajectories — CIFAR-10 / SmallCNN" width="65%">
</p>

<p align="center"><em>PILOT follows a distinct trajectory through the loss surface and converges to a lower-loss region compared to Adam, AdamW, Lion, and Sophia.</em></p>

---

## Method

PILOT monitors gradient-direction agreement (cosine similarity between successive gradients, smoothed into a running signal) and feeds it through a learned polynomial to produce three control knobs:

- **Momentum reliance** — trust the accumulated trend vs. react to the current gradient
- **Variance normalization** — how aggressively to apply adaptive scaling
- **Sign compression** — use full gradient magnitudes vs. compress toward ±1

Only **3(d+1)** learnable coefficients (9 for degree 2). Initialized so PILOT starts as Adam and learns to deviate. The policy is updated each step via analytic meta-gradients at negligible cost. See the [paper](https://arxiv.org/abs/2605.24570) for the full derivation.

---

## Hyperparameters

| Parameter | Description | Default / Range |
|-----------|-------------|-----------------|
| `lr` | Learning rate | `1e-3` |
| `betas` | Moment coefficients | `(0.9, 0.999)` |
| `weight_decay` | Decoupled weight decay | `0.01` |
| `gamma` | Agreement signal smoothing | `0.85`–`0.99` |
| `eta_phi` | Policy learning rate | `5e-4`–`5e-2` |
| `degree` | Polynomial degree | `1`–`4` |

<details>
<summary>Tuned configurations from the paper</summary>

| Dataset | Architecture | γ | η_φ | Degree |
|---------|-------------|------|--------|--------|
| CIFAR-10 | SmallCNN | 0.882 | 0.00312 | 1 |
| CIFAR-10 | ResNet-18 | 0.950 | 0.00500 | 2 |
| FashionMNIST | SmallCNN | 0.950 | 0.01000 | 2 |
| FashionMNIST | ResNet-18 | 0.957 | 0.00273 | 3 |

Selected via Bayesian optimization (TPE + ASHA early stopping, 30–40 trials).
</details>

---

## Experiment Setup

30 epochs · cross-entropy loss · cosine annealing LR · batch size 128 · AMP · 3-epoch linear warmup for ResNet-18. Benchmarked on NVIDIA V100 GPUs. See the [paper](https://arxiv.org/abs/2605.24570) for full details.

---

## Citation

```bibtex
@misc{altuuaim2026pilotpolicyinformedlearnedoptimization,
      title={PILOT: Policy-Informed Learned Optimization for Adaptive Deep Network Training}, 
      author={Sattam Altuuaim and Lama Ayash and Muhammad Mubashar and Naeemullah Khan},
      year={2026},
      eprint={2605.24570},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.24570}, 
}
```

---

## License

[MIT](LICENSE)
