# Quorum Sensing Optimisation (QSO)

Reference implementation and complete experimental data for:

> Medi, I. (2026). Quorum Sensing Optimisation: Mechanism, Evaluation and Limits of a
> Population-Signal Switching Metaheuristic. *Applied Intelligence* (submitted to journal).

QSO applies one of two operator sets at each iteration according to whether an aggregate
population-fitness signal exceeds a declining threshold. The transition is discrete and
reversible, occurring 13–179 times within a 500-iteration run.

## Install

```bash
pip install -r requirements.txt   # opfunu is pinned to 1.0.1
```

The CEC 2017 experiments use the **official competition shift vectors and rotation
matrices**, loaded by `opfunu`. Results obtained with locally generated transformations are
not comparable and are not used anywhere in the paper.

## Quick start

```python
from src.qso import qso
from opfunu.cec_based import cec2017

f = cec2017.F12017(ndim=30)
best_fitness, best_position, convergence, diversity, quorum, phase, theta = qso(
    f.evaluate, lb=-100, ub=100, dim=30,
    pop_size=30, max_iter=500, seed=42)
```

`quorum`, `phase` and `theta` are the switching instrumentation used in Fig. 3.

## Reproducing the paper

| Manuscript item | Notebook / script |
|---|---|
| Tables 3–5, Figs 1–2 | `notebooks/04_cec2017_official.ipynb` |
| Table 6, Fig. 3 | `notebooks/06_diagnostics.ipynb` |
| Table 7 | `notebooks/05_ablation_official.ipynb` |
| Tables 8–11, Fig. 4 | `notebooks/06_diagnostics.ipynb` |
| Tables 12–14 | `notebooks/07_engineering_rerun.ipynb` |
| Section 5.6 verification | `notebooks/08_feasibility_check.ipynb` |
| Appendix A, Fig. A1 | `notebooks/03_sensitivity_analysis.ipynb` |

Figures 1, 3 and 4 can be regenerated from the released data without re-running any
experiment:

```bash
python scripts/make_figures.py --data data/ --out figures/
```

## Data

All per-run results are in `data/`. Every number in the paper is derived from these files;
nothing is reported that is not reproducible from them. This includes the 1,920 engineering
solution vectors in `all_runs.json`, which allow the constraint-feasibility verification of
Section 5.6 to be checked independently.

## A note on the constraint-handling results

The engineering experiments use a static squared penalty (10⁶ × Σ max(0, gᵢ)²) applied
identically to all sixteen algorithms. Section 5.6 shows this does not exclude infeasible
solutions: on two of the four problems, fewer than a fifth of all runs across all algorithms
are feasible at a 10⁻⁶ tolerance, and reported values that appear to improve on published
optima correspond to constraint-violating solutions. `08_feasibility_check.ipynb` reproduces
this check on any of the stored solutions.

## Licence

[MIT / CC-BY-4.0]

## Citation

```bibtex
@article{medi2026qso,
  title  = {Quorum Sensing Optimisation: Mechanism, Evaluation and Limits of a
            Population-Signal Switching Metaheuristic},
  author = {Medi, Imran},
  year   = {2026},
  note   = {Under review}
}
```
