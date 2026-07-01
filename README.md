# Quorum Sensing Optimisation (QSO)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official Python implementation of **Quorum Sensing Optimisation (QSO)**, a novel bio-inspired metaheuristic algorithm for single-objective continuous optimisation.

> Medi, I. (2025). Quorum Sensing Optimisation: A Novel Bio-Inspired Metaheuristic Algorithm for Global Optimisation. *Swarm and Evolutionary Computation*. [Under Review]

---

## Algorithm Overview

QSO derives its core exploration-exploitation switching logic directly from the biochemical cycle of bacterial quorum sensing:

- **Autoinducer (AI) signal accumulation** — population fitness quality is aggregated into a scalar concentration signal C ∈ [0, 1]
- **Adaptive threshold detection** — a declining threshold θ(t) triggers the switch from exploration (Lévy flight) to exploitation (colony centroid + best attraction) when C ≥ θ(t)
- **Signal decay anti-stagnation reset** — when the population stagnates for τ consecutive iterations, the AI signal is decayed by factor e^(−λ), forcing re-exploration

This mechanism is distinct from prior quorum-sensing-inspired algorithms (QBSO, QBHO) which use collective signalling only as a peripheral enhancement to pre-existing frameworks.

---

## Key Results

Evaluated on 23 classical benchmark functions and the CEC 2017 suite (30D and 50D) against 15 state-of-the-art algorithms:

| Suite | QSO Rank | MFR | Friedman p-value |
|---|---|---|---|
| Classical 23 | 9/16 | 8.70 | 6.78×10⁻¹⁹ |
| CEC 2017 30D | 3/16 | 5.07 | 7.98×10⁻³⁶ |
| CEC 2017 50D | **1/16** | **3.50** | 9.51×10⁻⁴⁸ |

QSO ranks **1st of 16** at 50D with a mean Friedman rank of 3.50, improving from 3rd at 30D — the largest positive scalability trend among all compared algorithms.

---

## Repository Structure

```
QSO/
├── algorithms/
│   ├── qso.py          # Core QSO implementation
│   └── registry.py     # Algorithm name registry
├── supplementary/
│   └── Engineering_success_rates.csv   # Full 15-algorithm engineering success rates
├── README.md
└── LICENSE
```

Experimental notebooks (sensitivity analysis, benchmarking, engineering problems, ablation study) are available from the corresponding author on reasonable request, or will be linked here following journal publication.

---

## Installation

No external dependencies beyond NumPy and SciPy:

```bash
pip install numpy scipy
```

---

## Quick Start

```python
import numpy as np
from algorithms.qso import qso

# Define your objective function (minimisation)
def sphere(x):
    return np.sum(x ** 2)

# Run QSO
best_fitness, best_position, convergence, *_ = qso(
    func=sphere,
    lb=-100.0,
    ub=100.0,
    dim=30,
    pop_size=30,
    max_iter=500,
    theta_min=0.3,
    theta_max=0.7,
    lambda_=0.05,
    tau=10,
    alpha=1.25,
    seed=42
)

print(f"Best fitness: {best_fitness:.6e}")
print(f"Best position: {best_position[:5]}...")  # first 5 dimensions
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 30 | Population size |
| `max_iter` | 500 | Maximum iterations |
| `theta_min` | 0.3 | Minimum adaptive threshold |
| `theta_max` | 0.7 | Maximum adaptive threshold |
| `lambda_` | 0.05 | AI signal decay rate |
| `tau` | 10 | Stagnation window (iterations before reset) |
| `alpha` | 1.25 | Lévy flight index |

Sensitivity analysis confirms robustness across wide parameter ranges — see paper Section 5.1.

---

## Returns

`qso()` returns a tuple of 7 values:

```python
best_fitness, best_position, convergence, diversity, quorum_history, phase_history, theta_history = qso(...)
```

| Return | Description |
|---|---|
| `best_fitness` | Best objective value found |
| `best_position` | Corresponding solution vector |
| `convergence` | Best fitness at each iteration |
| `diversity` | Mean std of population at each iteration |
| `quorum_history` | AI concentration C at each iteration |
| `phase_history` | Phase (1=exploit, 0=explore) at each iteration |
| `theta_history` | Adaptive threshold θ(t) at each iteration |

---

## Citation

If you use QSO in your research, please cite:

```bibtex
@article{medi2025qso,
  title   = {Quorum Sensing Optimisation: A Novel Bio-Inspired Metaheuristic Algorithm for Global Optimisation},
  author  = {Medi, Imran},
  journal = {Swarm and Evolutionary Computation},
  year    = {2025},
  note    = {Under Review}
}
```

This entry will be updated with volume, issue, and DOI upon acceptance.

---

## Experimental Data

Full experimental results will be permanently archived on Zenodo upon journal acceptance. The Zenodo DOI will be linked here and in the paper's Data Availability statement.

Supplementary data available in this repository:
- `supplementary/Engineering_success_rates.csv` — success rates for all 15 algorithms across 4 engineering problems

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Imran Medi**
School of Computing, Asia Pacific University of Technology and Innovation
Kuala Lumpur, Malaysia
