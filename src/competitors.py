# %% [markdown]
# # 02_Competitors
# Competitor algorithm implementations — all 14 algorithms

# %%
import numpy as np
import os
import json
from google.colab import drive

drive.mount('/content/drive')

BASE = '/content/drive/MyDrive/QSO_Research'

print('='*50)
print('NOTEBOOK 02 — COMPETITOR IMPLEMENTATIONS')
print('='*50)
print('\nAlgorithms to implement:')
algos = {
    'Classical':      ['PSO', 'GA', 'DE'],
    'Modern':         ['GWO', 'WOA', 'SCA', 'HHO', 'MPA'],
    'Biology-Based':  ['BFO', 'QBHO', 'QBSO'],
    'Recent':         ['DBO', 'POA', 'EVO'],
}
for group, names in algos.items():
    print(f'  {group}: {", ".join(names)}')
print('\n✅ Ready to implement')

# %%
def initialise_population(pop_size, dim, lb, ub, seed=42):
    """Standardised population initialisation for all algorithms."""
    np.random.seed(seed)
    lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb)
    ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub)
    X  = np.random.uniform(lb, ub, (pop_size, dim))
    return X, lb, ub


def evaluate_population(func, X):
    """Evaluate fitness for all agents."""
    return np.array([func(X[i]) for i in range(len(X))])


def bound_check(X, lb, ub):
    """Reflect positions back into bounds."""
    return np.clip(X, lb, ub)


def get_best(fitness, X):
    """Return best fitness and position."""
    idx = np.argmin(fitness)
    return fitness[idx], X[idx].copy()


# Standard return format for ALL algorithms:
# (best_fitness, best_position, convergence_curve)
# This uniform interface is critical for fair comparison

print('✅ Shared utilities defined')

# %%
def pso(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        w=0.7, c1=1.5, c2=1.5,
        seed=42):
    """
    Particle Swarm Optimisation (Kennedy & Eberhart, 1995)

    Parameters:
    -----------
    w  : float — inertia weight (default 0.7)
    c1 : float — cognitive coefficient (default 1.5)
    c2 : float — social coefficient (default 1.5)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)

    # Velocities
    V       = np.zeros((pop_size, dim))
    v_max   = 0.2 * (ub - lb)

    # Personal and global bests
    pbest_X = X.copy()
    fitness = evaluate_population(func, X)
    pbest_f = fitness.copy()

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    for t in range(max_iter):
        r1 = np.random.rand(pop_size, dim)
        r2 = np.random.rand(pop_size, dim)

        # Velocity update
        V = (w * V
             + c1 * r1 * (pbest_X - X)
             + c2 * r2 * (gbest_X  - X))
        V = np.clip(V, -v_max, v_max)

        # Position update
        X = X + V
        X = bound_check(X, lb, ub)

        # Fitness evaluation
        fitness = evaluate_population(func, X)

        # Update personal bests
        improved = fitness < pbest_f
        pbest_f[improved] = fitness[improved]
        pbest_X[improved] = X[improved]

        # Update global best
        curr_f, curr_X = get_best(fitness, X)
        if curr_f < gbest_f:
            gbest_f = curr_f
            gbest_X = curr_X.copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


# --- Test ---
print('Testing PSO...')
from numpy import sum as npsum
sphere = lambda x: npsum(x**2)
f, _, c = pso(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'PSO did not improve'
print('✅ PSO operational')

# %%
def ga(func, lb, ub, dim,
       pop_size=30, max_iter=500,
       cr=0.9, mr=0.01,
       seed=42):
    """
    Genetic Algorithm (Holland, 1992)

    Parameters:
    -----------
    cr : float — crossover rate (default 0.9)
    mr : float — mutation rate (default 0.01)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    for t in range(max_iter):
        new_X = np.zeros_like(X)

        for i in range(pop_size):
            # ── Tournament selection ──────────────────────────
            t1, t2 = np.random.randint(0, pop_size, 2)
            parent1 = X[t1] if fitness[t1] < fitness[t2] else X[t2]

            t3, t4 = np.random.randint(0, pop_size, 2)
            parent2 = X[t3] if fitness[t3] < fitness[t4] else X[t4]

            # ── Single-point crossover ────────────────────────
            if np.random.rand() < cr:
                point   = np.random.randint(1, dim)
                child   = np.concatenate([
                              parent1[:point],
                              parent2[point:]])
            else:
                child = parent1.copy()

            # ── Gaussian mutation ─────────────────────────────
            mask          = np.random.rand(dim) < mr
            child[mask]  += np.random.normal(
                                0, 0.1*(ub[mask]-lb[mask]))
            child         = np.clip(child, lb, ub)
            new_X[i]      = child

        X       = new_X
        fitness = evaluate_population(func, X)

        # Elite preservation — keep best from previous generation
        worst_idx = np.argmax(fitness)
        if fitness[worst_idx] > gbest_f:
            X[worst_idx]       = gbest_X.copy()
            fitness[worst_idx] = gbest_f

        curr_f, curr_X = get_best(fitness, X)
        if curr_f < gbest_f:
            gbest_f = curr_f
            gbest_X = curr_X.copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


# --- Test ---
print('Testing GA...')
f, _, c = ga(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'GA did not improve'
print('✅ GA operational')

# %%
def de(func, lb, ub, dim,
       pop_size=30, max_iter=500,
       F=0.5, cr=0.9,
       seed=42):
    """
    Differential Evolution (Storn & Price, 1997)
    DE/rand/1/bin variant.

    Parameters:
    -----------
    F  : float — scaling factor (default 0.5)
             Lower values (0.4-0.6) work better on
             continuous unimodal problems. Original
             paper recommends F in [0.4, 1.0].
    cr : float — crossover rate (default 0.9)

    Note on parameters:
    -------------------
    F=0.5 chosen based on parameter sensitivity analysis
    showing F=0.8 causes over-exploration on 30D continuous
    problems with pop_size=30 at 500 iterations.
    This is consistent with Storn & Price (1997) who note
    F in [0.4, 0.6] works well for most continuous problems.
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    for t in range(max_iter):
        for i in range(pop_size):
            # ── Mutation — DE/rand/1 ──────────────────────────
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c_idx = np.random.choice(idxs, 3, replace=False)

            mutant = X[a] + F * (X[b] - X[c_idx])
            mutant = np.clip(mutant, lb, ub)

            # ── Binomial crossover ────────────────────────────
            cross_mask = np.random.rand(dim) < cr
            # Guarantee at least one dimension crosses over
            cross_mask[np.random.randint(dim)] = True
            trial = np.where(cross_mask, mutant, X[i])

            # ── Greedy selection ──────────────────────────────
            trial_f = func(trial)
            if trial_f < fitness[i]:
                X[i]       = trial
                fitness[i] = trial_f
                if trial_f < gbest_f:
                    gbest_f = trial_f
                    gbest_X = trial.copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


# --- Test ---
print('Testing DE (fixed)...')
de_results = []
for seed in [42, 43, 44, 45, 46]:
    f, _, c = de(sphere, -100, 100, dim=30, seed=seed)
    de_results.append(f)
    improvement = (c[0] - f) / c[0] * 100
    print(f'  Seed {seed}: {f:.4e} '
          f'(improvement: {improvement:.1f}%)')

print(f'\n  Mean: {np.mean(de_results):.4e}')
print(f'  Std:  {np.std(de_results):.4e}')

# Convergence check
print('\n  Convergence check (seed 42):')
f, _, c = de(sphere, -100, 100, dim=30, seed=42)
checkpoints = [0, 50, 100, 200, 300, 400, 499]
for cp in checkpoints:
    print(f'    Iter {cp:3d}: {c[cp]:.4e}')

assert f < c[0], 'DE did not improve'
print('\n✅ DE (fixed) operational')

# %%
def gwo(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        seed=42):
    """
    Grey Wolf Optimiser (Mirjalili et al., 2014)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    # Alpha, beta, delta wolves
    sorted_idx = np.argsort(fitness)
    alpha_f, alpha_X = fitness[sorted_idx[0]], X[sorted_idx[0]].copy()
    beta_f,  beta_X  = fitness[sorted_idx[1]], X[sorted_idx[1]].copy()
    delta_f, delta_X = fitness[sorted_idx[2]], X[sorted_idx[2]].copy()

    convergence = [alpha_f]

    for t in range(max_iter):
        # Linearly decreasing a from 2 to 0
        a = 2 - 2 * (t / max_iter)

        for i in range(pop_size):
            # Update position based on alpha, beta, delta
            X1 = _gwo_update(X[i], alpha_X, a)
            X2 = _gwo_update(X[i], beta_X,  a)
            X3 = _gwo_update(X[i], delta_X, a)
            X[i] = np.clip((X1 + X2 + X3) / 3, lb, ub)

        fitness = evaluate_population(func, X)

        # Update hierarchy
        sorted_idx = np.argsort(fitness)
        if fitness[sorted_idx[0]] < alpha_f:
            alpha_f = fitness[sorted_idx[0]]
            alpha_X = X[sorted_idx[0]].copy()
        if fitness[sorted_idx[1]] < beta_f:
            beta_f  = fitness[sorted_idx[1]]
            beta_X  = X[sorted_idx[1]].copy()
        if fitness[sorted_idx[2]] < delta_f:
            delta_f = fitness[sorted_idx[2]]
            delta_X = X[sorted_idx[2]].copy()

        convergence.append(alpha_f)

    return alpha_f, alpha_X, convergence


def _gwo_update(x, leader, a):
    """Helper — update position toward a leader wolf."""
    r1, r2 = np.random.rand(len(x)), np.random.rand(len(x))
    A = 2 * a * r1 - a
    C = 2 * r2
    D = np.abs(C * leader - x)
    return leader - A * D


# --- Test ---
print('Testing GWO...')
f, _, c = gwo(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'GWO did not improve'
print('✅ GWO operational')

# %%
def woa(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        seed=42):
    """
    Whale Optimisation Algorithm (Mirjalili & Lewis, 2016)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    for t in range(max_iter):
        a  = 2 - 2 * (t / max_iter)  # Decreases from 2 to 0
        a2 = -1 - (t / max_iter)     # Decreases from -1 to -2

        for i in range(pop_size):
            r  = np.random.rand()
            A  = 2 * a * np.random.rand(dim) - a
            C  = 2 * np.random.rand(dim)
            b  = 1.0   # Spiral shape constant
            l  = (a2 - 1) * np.random.rand() + 1
            p  = np.random.rand()

            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    # Shrinking encircling
                    D       = np.abs(C * gbest_X - X[i])
                    X[i]    = gbest_X - A * D
                else:
                    # Random search
                    rand_X  = X[np.random.randint(pop_size)]
                    D       = np.abs(C * rand_X - X[i])
                    X[i]    = rand_X - A * D
            else:
                # Spiral bubble-net attack
                D       = np.abs(gbest_X - X[i])
                X[i]    = (D * np.exp(b * l)
                           * np.cos(2 * np.pi * l)
                           + gbest_X)

            X[i] = np.clip(X[i], lb, ub)

        fitness = evaluate_population(func, X)

        curr_f, curr_X = get_best(fitness, X)
        if curr_f < gbest_f:
            gbest_f = curr_f
            gbest_X = curr_X.copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


# --- Test ---
print('Testing WOA...')
f, _, c = woa(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'WOA did not improve'
print('✅ WOA operational')

# %%
def sca(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        seed=42):
    """
    Sine Cosine Algorithm (Mirjalili, 2016)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    for t in range(max_iter):
        # Decreasing r1 from 2 to 0
        r1 = 2 - 2 * (t / max_iter)

        for i in range(pop_size):
            r2 = 2 * np.pi * np.random.rand(dim)
            r3 = np.random.rand(dim)
            r4 = np.random.rand()

            if r4 < 0.5:
                X[i] = (X[i]
                        + r1 * np.sin(r2)
                        * np.abs(r3 * gbest_X - X[i]))
            else:
                X[i] = (X[i]
                        + r1 * np.cos(r2)
                        * np.abs(r3 * gbest_X - X[i]))

            X[i] = np.clip(X[i], lb, ub)

        fitness = evaluate_population(func, X)

        curr_f, curr_X = get_best(fitness, X)
        if curr_f < gbest_f:
            gbest_f = curr_f
            gbest_X = curr_X.copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


# --- Test ---
print('Testing SCA...')
f, _, c = sca(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'SCA did not improve'
print('✅ SCA operational')

# %%
def hho(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        seed=42):
    """
    Harris Hawks Optimisation (Heidari et al., 2019)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    for t in range(max_iter):
        E0 = 2 * np.random.rand() - 1   # Initial energy
        E  = 2 * E0 * (1 - t / max_iter) # Escaping energy

        for i in range(pop_size):
            r = np.random.rand()

            if np.abs(E) >= 1:
                # ── Exploration ───────────────────────────────
                if r >= 0.5:
                    rand_X   = X[np.random.randint(pop_size)]
                    X[i]     = (rand_X
                                - np.random.rand()
                                * np.abs(rand_X
                                - 2 * np.random.rand() * X[i]))
                else:
                    X[i]     = ((gbest_X - np.mean(X, axis=0))
                                - np.random.rand()
                                * (lb + np.random.rand() * (ub - lb)))
            else:
                # ── Exploitation ──────────────────────────────
                J        = 2 * (1 - np.random.rand())
                delta_X  = gbest_X - X[i]

                if r >= 0.5 and np.abs(E) >= 0.5:
                    # Soft besiege
                    X[i] = delta_X - E * np.abs(J * gbest_X - X[i])

                elif r >= 0.5 and np.abs(E) < 0.5:
                    # Hard besiege
                    X[i] = gbest_X - E * np.abs(delta_X)

                elif r < 0.5 and np.abs(E) >= 0.5:
                    # Soft besiege with progressive rapid dives
                    Y = gbest_X - E * np.abs(J * gbest_X - X[i])
                    Z = Y + np.random.rand(dim) * _levy_hho(dim)
                    X[i] = (Y if func(Y) < func(Z) else Z)

                else:
                    # Hard besiege with progressive rapid dives
                    Y = gbest_X - E * np.abs(J * gbest_X - np.mean(X, axis=0))
                    Z = Y + np.random.rand(dim) * _levy_hho(dim)
                    X[i] = (Y if func(Y) < func(Z) else Z)

            X[i] = np.clip(X[i], lb, ub)

        fitness = evaluate_population(func, X)

        curr_f, curr_X = get_best(fitness, X)
        if curr_f < gbest_f:
            gbest_f = curr_f
            gbest_X = curr_X.copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


def _levy_hho(dim, beta=1.5):
    """Lévy flight helper for HHO."""
    from scipy.special import gamma
    sigma = (gamma(1+beta) * np.sin(np.pi*beta/2) /
             (gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v)**(1/beta))


# --- Test ---
print('Testing HHO...')
f, _, c = hho(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'HHO did not improve'
print('✅ HHO operational')

# %%
def mpa(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        seed=42):
    """
    Marine Predators Algorithm (Faramarzi et al., 2020)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)

    # Elite matrix — top predator
    Elite   = np.tile(gbest_X, (pop_size, 1))
    convergence = [gbest_f]
    P       = 0.5
    FADs    = 0.2

    for t in range(max_iter):
        CF = (1 - t/max_iter) ** (2*t/max_iter)

        RL = 0.05 * _levy_mpa(pop_size, dim)
        RB = np.random.randn(pop_size, dim)

        for i in range(pop_size):
            r  = np.random.rand()
            R  = np.random.rand(dim)

            if t < max_iter / 3:
                # Phase 1 — High velocity ratio (prey moves faster)
                stepsize   = RB[i] * (Elite[i] - RB[i] * X[i])
                X[i]      += P * stepsize

            elif t < 2 * max_iter / 3:
                if i < pop_size // 2:
                    # Phase 2a — Unit velocity ratio (Lévy)
                    stepsize = RL[i] * (Elite[i] - RL[i] * X[i])
                    X[i]    += P * stepsize
                else:
                    # Phase 2b — Unit velocity ratio (Brownian)
                    stepsize = RB[i] * (RB[i] * Elite[i] - X[i])
                    X[i]    += P * CF * stepsize
            else:
                # Phase 3 — Low velocity ratio (predator moves faster)
                stepsize   = RL[i] * (RL[i] * Elite[i] - X[i])
                X[i]      += P * CF * stepsize

            # FADs effect
            if np.random.rand() < FADs:
                U    = np.random.rand(dim) < FADs
                X[i]+= CF * (lb + np.random.rand(dim)*(ub-lb)) * U

            X[i] = np.clip(X[i], lb, ub)

        fitness = evaluate_population(func, X)

        curr_f, curr_X = get_best(fitness, X)
        if curr_f < gbest_f:
            gbest_f = curr_f
            gbest_X = curr_X.copy()

        # Update elite matrix
        Elite = np.tile(gbest_X, (pop_size, 1))
        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


def _levy_mpa(n, d, beta=1.5):
    """Lévy flight helper for MPA."""
    from scipy.special import gamma
    sigma = (gamma(1+beta) * np.sin(np.pi*beta/2) /
             (gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, (n, d))
    v = np.random.normal(0, 1, (n, d))
    return u / (np.abs(v)**(1/beta))


# --- Test ---
print('Testing MPA...')
f, _, c = mpa(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'MPA did not improve'
print('✅ MPA operational')

# %%
def bfo(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        n_swim=4, n_tumble=4,
        seed=42):
    """
    Bacterial Foraging Optimisation (Passino, 2002)

    Parameters:
    -----------
    n_swim   : int — swim steps per chemotaxis (default 4)
    n_tumble : int — tumble steps (default 4)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    step_size = 0.1 * (ub - lb)
    iters_per_cycle = max(1, max_iter // (n_tumble * n_swim + 1))

    for t in range(max_iter):
        for i in range(pop_size):
            # ── Tumble — random direction ─────────────────────
            delta = np.random.randn(dim)
            delta /= (np.linalg.norm(delta) + 1e-10)

            # ── Swim — move in tumble direction ───────────────
            for s in range(n_swim):
                X_new    = X[i] + step_size * delta
                X_new    = np.clip(X_new, lb, ub)
                f_new    = func(X_new)

                if f_new < fitness[i]:
                    X[i]       = X_new
                    fitness[i] = f_new
                    if f_new < gbest_f:
                        gbest_f = f_new
                        gbest_X = X_new.copy()
                else:
                    break

        # ── Reproduction — top half survives ─────────────────
        if t % iters_per_cycle == 0:
            sorted_idx   = np.argsort(fitness)
            X            = np.vstack([
                               X[sorted_idx[:pop_size//2]],
                               X[sorted_idx[:pop_size//2]]
                           ])
            fitness      = np.concatenate([
                               fitness[sorted_idx[:pop_size//2]],
                               fitness[sorted_idx[:pop_size//2]]
                           ])

        # Decrease step size over time
        step_size *= 0.99

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


# --- Test ---
print('Testing BFO...')
f, _, c = bfo(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'BFO did not improve'
print('✅ BFO operational')

# %%
def qbso(func, lb, ub, dim,
         pop_size=30, max_iter=500,
         qs_threshold=0.5,
         seed=42):
    """
    Quorum Sensing Bacterial Swarm Optimisation (QBSO)
    Based on: Li et al. (2019)

    QS used as enhancement to bacterial swarm —
    NOT as standalone framework (key distinction from QSO)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    step_size = 0.1 * (ub - lb)

    for t in range(max_iter):
        # ── Compute quorum signal ─────────────────────────────
        f_worst = np.max(fitness)
        f_best  = np.min(fitness)
        epsilon = 1e-10

        if f_worst - f_best < epsilon:
            qs_signal = 0.5
        else:
            qs_signal = np.mean(
                (f_worst - fitness) / (f_worst - f_best + epsilon))

        for i in range(pop_size):
            delta = np.random.randn(dim)
            delta /= (np.linalg.norm(delta) + 1e-10)

            if qs_signal >= qs_threshold:
                # QS triggered — move toward global best
                direction = gbest_X - X[i]
                norm      = np.linalg.norm(direction) + 1e-10
                X[i]     += step_size * (direction/norm)
            else:
                # QS not triggered — random walk
                X[i]     += step_size * delta

            X[i] = np.clip(X[i], lb, ub)

        fitness = evaluate_population(func, X)

        curr_f, curr_X = get_best(fitness, X)
        if curr_f < gbest_f:
            gbest_f = curr_f
            gbest_X = curr_X.copy()

        step_size *= 0.995
        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


def qbho(func, lb, ub, dim,
         pop_size=30, max_iter=500,
         qs_threshold=0.5,
         seed=42):
    """
    Quorum Sensing Bacterial Horde Optimisation (QBHO)
    Based on: Alzaqebah et al. (2023)

    QS used to identify optimal bacterial positions —
    NOT as standalone framework (key distinction from QSO)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    for t in range(max_iter):
        # ── Quorum detection ──────────────────────────────────
        f_worst   = np.max(fitness)
        f_best    = np.min(fitness)
        epsilon   = 1e-10

        qs_signal = np.mean(
            (f_worst - fitness) / (f_worst - f_best + epsilon + 1e-10))

        # ── Worst position used as reference (per QBHO paper) ─
        worst_idx = np.argmax(fitness)

        for i in range(pop_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            if qs_signal >= qs_threshold:
                # Quorum active — avoid worst, move to best
                X[i] = (X[i]
                        + r1 * (gbest_X - X[i])
                        - r2 * (X[worst_idx] - X[i]))
            else:
                # Quorum inactive — standard foraging
                rand_X = X[np.random.randint(pop_size)]
                X[i]   = X[i] + r1 * (rand_X - X[i])

            X[i] = np.clip(X[i], lb, ub)

        fitness = evaluate_population(func, X)

        curr_f, curr_X = get_best(fitness, X)
        if curr_f < gbest_f:
            gbest_f = curr_f
            gbest_X = curr_X.copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


# --- Tests ---
print('Testing QBSO...')
f, _, c = qbso(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'QBSO did not improve'
print('✅ QBSO operational')

print('Testing QBHO...')
f, _, c = qbho(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'QBHO did not improve'
print('✅ QBHO operational')

# %%
def dbo(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        seed=42):
    """
    Dung Beetle Optimisation (Xue & Shen, 2022)

    Four beetle roles:
    - Ball-rollers  : navigate using celestial cues (exploration)
    - Dancers       : reorient when lost (escape local optima)
    - Foragers      : search near best site (exploitation)
    - Brood-stealers: compete for best positions (intensification)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    # Population split into 4 roles
    n_rollers  = pop_size // 4
    n_dancers  = pop_size // 4
    n_foragers = pop_size // 4
    n_thieves  = pop_size - n_rollers - n_dancers - n_foragers

    # Role index boundaries
    r_end = n_rollers
    d_end = n_rollers + n_dancers
    f_end = n_rollers + n_dancers + n_foragers

    for t in range(max_iter):
        R  = 1 - t / max_iter       # Decreasing radius
        CF = (1 - t/max_iter) ** 2  # Convergence factor

        # ── Ball-rolling beetles (exploration) ────────────────────
        for i in range(r_end):
            if np.random.rand() > 0.9:
                # Dancing reorientation
                X[i] = X[i] + np.tan(
                    np.random.rand(dim)) * np.abs(X[i] - gbest_X)
            else:
                # Navigate toward best with decreasing radius
                r1   = np.random.rand(dim)
                X[i] = X[i] + R * r1 * (gbest_X - X[i])
            X[i] = np.clip(X[i], lb, ub)

        # ── Dancing beetles (escape local optima) ─────────────────
        for i in range(r_end, d_end):
            r1   = np.random.rand(dim)
            X[i] = gbest_X + r1 * np.abs(X[i] - gbest_X) * CF
            X[i] = np.clip(X[i], lb, ub)

        # ── Foraging beetles (exploitation) — FIXED ───────────────
        for i in range(d_end, f_end):
            r1   = np.random.rand(dim)
            r2   = np.random.rand(dim)
            # Move toward global best with random perturbation
            X[i] = (X[i]
                    + r1 * (gbest_X - X[i])
                    + r2 * CF * np.random.randn(dim))
            X[i] = np.clip(X[i], lb, ub)

        # ── Brood-stealing beetles (intensification) ──────────────
        for i in range(f_end, pop_size):
            r1   = np.random.rand(dim)
            r2   = np.random.rand(dim)
            # Steal position near global best
            X[i] = (gbest_X
                    + r1 * CF * (X[i] - gbest_X)
                    + r2 * np.random.randn(dim) * R)
            X[i] = np.clip(X[i], lb, ub)

        # ── Evaluate & update best ────────────────────────────────
        fitness = evaluate_population(func, X)

        # Elite preservation
        worst_idx = np.argmax(fitness)
        if fitness[worst_idx] > gbest_f:
            X[worst_idx]       = gbest_X.copy()
            fitness[worst_idx] = gbest_f

        curr_f, curr_X = get_best(fitness, X)
        if curr_f < gbest_f:
            gbest_f = curr_f
            gbest_X = curr_X.copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


# --- Test ---
print('Testing DBO (fixed)...')
dbo_results = []
for seed in [42, 43, 44, 45, 46]:
    f, _, c = dbo(sphere, -100, 100, dim=30, seed=seed)
    dbo_results.append(f)
    improvement = (c[0] - f) / c[0] * 100
    print(f'  Seed {seed}: {f:.4e} '
          f'(improvement: {improvement:.1f}%)')

print(f'\n  Mean: {np.mean(dbo_results):.4e}')
print(f'  Std:  {np.std(dbo_results):.4e}')

# Convergence check
print('\n  Convergence check (seed 42):')
f, _, c = dbo(sphere, -100, 100, dim=30, seed=42)
checkpoints = [0, 50, 100, 200, 300, 400, 499]
for cp in checkpoints:
    print(f'    Iter {cp:3d}: {c[cp]:.4e}')

assert f < c[0], 'DBO did not improve'
assert np.mean(dbo_results) < 1e3, \
    f'DBO mean still too high: {np.mean(dbo_results):.4e}'
print('\n✅ DBO (fixed) operational')

# %%
def poa(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        seed=42):
    """
    Pelican Optimisation Algorithm (Trojovský & Dehghani, 2022)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    for t in range(max_iter):
        for i in range(pop_size):
            # ── Phase 1: Moving toward prey ───────────────────
            # Random prey selection
            prey_idx  = np.random.randint(pop_size)
            prey_X    = X[prey_idx]
            prey_f    = fitness[prey_idx]

            X1 = X[i] + np.random.rand(dim) * (
                prey_X - np.random.randint(1, 3) * X[i])
            X1 = np.clip(X1, lb, ub)
            f1 = func(X1)

            if f1 < fitness[i]:
                X[i]       = X1
                fitness[i] = f1

            # ── Phase 2: Winging on water surface ─────────────
            R    = 0.2 * (1 - t / max_iter)
            X2   = X[i] + R * (2 * np.random.rand(dim) - 1) * X[i]
            X2   = np.clip(X2, lb, ub)
            f2   = func(X2)

            if f2 < fitness[i]:
                X[i]       = X2
                fitness[i] = f2

            if fitness[i] < gbest_f:
                gbest_f = fitness[i]
                gbest_X = X[i].copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


def evo(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        seed=42):
    """
    Electric Eel Foraging Optimiser (EVO)
    Based on: Wang et al. (2024)
    """
    X, lb, ub = initialise_population(pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    gbest_f, gbest_X = get_best(fitness, X)
    convergence = [gbest_f]

    for t in range(max_iter):
        a = 2 * (1 - t / max_iter)  # Decreasing factor

        for i in range(pop_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            # ── Electric discharge hunting ────────────────────
            if np.random.rand() < 0.5:
                # Discharge toward best
                X[i] = (X[i]
                        + a * r1 * (gbest_X - X[i])
                        + (1-a) * r2 * (
                            X[np.random.randint(pop_size)] - X[i]))
            else:
                # Passive drift with random component
                beta   = np.random.randn(dim)
                X[i]   = (gbest_X
                          + beta * np.abs(gbest_X - X[i]) * (1 - a))

            X[i] = np.clip(X[i], lb, ub)

        fitness = evaluate_population(func, X)

        curr_f, curr_X = get_best(fitness, X)
        if curr_f < gbest_f:
            gbest_f = curr_f
            gbest_X = curr_X.copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


# --- Tests ---
print('Testing POA...')
f, _, c = poa(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'POA did not improve'
print('✅ POA operational')

print('Testing EVO...')
f, _, c = evo(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'EVO did not improve'
print('✅ EVO operational')

# %%


# %%
def _levy_gjo(n, d, beta=1.5):
    """Lévy flight helper for GJO."""
    from scipy.special import gamma
    sigma = (gamma(1+beta) * np.sin(np.pi*beta/2) /
             (gamma((1+beta)/2) * beta
              * 2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, (n, d))
    v = np.random.normal(0, 1, (n, d))
    return u / (np.abs(v)**(1/beta))


def gjo(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        seed=42):
    """
    Golden Jackal Optimizer (Chopra & Ansari, 2022)
    Published: Expert Systems with Applications, 198, 116924

    Models male and female jackal hunting behaviour:
    - Male jackal: tracks prey (global best)
    - Female jackal: supports male (second best)
    - Prey escape energy decreases over iterations
    """
    X, lb, ub = initialise_population(
                    pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    # Male and female jackal (best two solutions)
    sorted_idx = np.argsort(fitness)
    male_pos   = X[sorted_idx[0]].copy()
    male_f     = fitness[sorted_idx[0]]
    female_pos = X[sorted_idx[1]].copy()
    female_f   = fitness[sorted_idx[1]]

    gbest_f     = male_f
    gbest_X     = male_pos.copy()
    convergence = [gbest_f]

    for t in range(max_iter):
        E1 = 1.5 * (1 - t / max_iter)
        RL = 0.05 * _levy_gjo(pop_size, dim)

        for i in range(pop_size):
            E0 = 2 * np.random.rand() - 1
            E  = E1 * E0

            # Update toward male jackal
            D_male   = np.abs(RL[i] * male_pos - X[i])
            X1       = male_pos - E * D_male

            # Update toward female jackal
            D_female = np.abs(RL[i] * female_pos - X[i])
            X2       = female_pos - E * D_female

            # Average of both updates
            X[i] = np.clip((X1 + X2) / 2, lb, ub)

        fitness = evaluate_population(func, X)

        # Update male and female jackals
        sorted_idx = np.argsort(fitness)

        if fitness[sorted_idx[0]] < male_f:
            male_f   = fitness[sorted_idx[0]]
            male_pos = X[sorted_idx[0]].copy()

        if fitness[sorted_idx[1]] < female_f:
            female_f   = fitness[sorted_idx[1]]
            female_pos = X[sorted_idx[1]].copy()

        if male_f < gbest_f:
            gbest_f = male_f
            gbest_X = male_pos.copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


# --- Test ---
sphere = lambda x: np.sum(x**2)

print('Testing GJO...')
f, _, c = gjo(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'GJO did not improve'
print('✅ GJO operational')

# %%
# Import QSO from Drive
import importlib.util, sys

def load_qso():
    """Load QSO from saved Drive file."""
    spec   = importlib.util.spec_from_file_location(
                 'qso', f'{BASE}/algorithms/qso.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.qso

qso_func = load_qso()

# ── Algorithm registry ────────────────────────────────────────────
ALGORITHM_REGISTRY = {
    'QSO':  lambda func, lb, ub, dim, seed:
                qso_func(func, lb, ub, dim,
                         pop_size=30, max_iter=500,
                         theta_min=0.3, theta_max=0.7,
                         lambda_=0.05, tau=10, alpha=1.25,
                         seed=seed),
    'PSO':  lambda func, lb, ub, dim, seed:
                pso(func, lb, ub, dim,
                    pop_size=30, max_iter=500,
                    w=0.7, c1=1.5, c2=1.5, seed=seed),
    'GA':   lambda func, lb, ub, dim, seed:
                ga(func, lb, ub, dim,
                   pop_size=30, max_iter=500,
                   cr=0.9, mr=0.01, seed=seed),
    'DE':   lambda func, lb, ub, dim, seed:
                de(func, lb, ub, dim,
                   pop_size=30, max_iter=500,
                   F=0.5, cr=0.9, seed=seed),
    'GWO':  lambda func, lb, ub, dim, seed:
                gwo(func, lb, ub, dim,
                    pop_size=30, max_iter=500, seed=seed),
    'WOA':  lambda func, lb, ub, dim, seed:
                woa(func, lb, ub, dim,
                    pop_size=30, max_iter=500, seed=seed),
    'SCA':  lambda func, lb, ub, dim, seed:
                sca(func, lb, ub, dim,
                    pop_size=30, max_iter=500, seed=seed),
    'HHO':  lambda func, lb, ub, dim, seed:
                hho(func, lb, ub, dim,
                    pop_size=30, max_iter=500, seed=seed),
    'MPA':  lambda func, lb, ub, dim, seed:
                mpa(func, lb, ub, dim,
                    pop_size=30, max_iter=500, seed=seed),
    'BFO':  lambda func, lb, ub, dim, seed:
                bfo(func, lb, ub, dim,
                    pop_size=30, max_iter=500, seed=seed),
    'QBSO': lambda func, lb, ub, dim, seed:
                qbso(func, lb, ub, dim,
                     pop_size=30, max_iter=500, seed=seed),
    'QBHO': lambda func, lb, ub, dim, seed:
                qbho(func, lb, ub, dim,
                     pop_size=30, max_iter=500, seed=seed),
    'DBO':  lambda func, lb, ub, dim, seed:
                dbo(func, lb, ub, dim,
                    pop_size=30, max_iter=500, seed=seed),
    'POA':  lambda func, lb, ub, dim, seed:
                poa(func, lb, ub, dim,
                    pop_size=30, max_iter=500, seed=seed),
    'EVO':  lambda func, lb, ub, dim, seed:
                evo(func, lb, ub, dim,
                    pop_size=30, max_iter=500, seed=seed),
    'GJO':  lambda func, lb, ub, dim, seed:
                gjo(func, lb, ub, dim,
                    pop_size=30, max_iter=500, seed=seed),
}


def run_algorithm(algo_name, func, lb, ub, dim, seed):
    """
    Unified runner — same interface for all 16 algorithms.
    Returns: (best_fitness, best_position, convergence_curve)
    """
    runner = ALGORITHM_REGISTRY[algo_name]
    result = runner(func, lb, ub, dim, seed)
    return result[0], result[1], result[2]


# ── Print summary ─────────────────────────────────────────────────
algos = {
    'Classical':      ['PSO', 'GA', 'DE'],
    'Modern':         ['GWO', 'WOA', 'SCA', 'HHO', 'MPA'],
    'Biology-Based':  ['BFO', 'QBSO', 'QBHO'],
    'Recent':         ['DBO', 'POA', 'EVO', 'GJO'],
}

print('✅ Algorithm registry built')
print(f'   {len(ALGORITHM_REGISTRY)} algorithms registered')
print()
for group, names in algos.items():
    print(f'  {group}: {", ".join(names)}')
print('  + QSO (your algorithm)')

# ── Quick verification ────────────────────────────────────────────
sphere = lambda x: np.sum(x**2)
print('\nVerifying GJO in registry...')
f, _, _ = run_algorithm('GJO', sphere, -100, 100,
                         dim=30, seed=42)
print(f'  GJO Sphere 30D: {f:.4e}')
print(f'✅ All {len(ALGORITHM_REGISTRY)} algorithms ready')

# %%
print('='*60)
print('QUICK COMPARISON — ALL 15 ALGORITHMS ON SPHERE 30D')
print('='*60)

sphere = lambda x: np.sum(x**2)

results_quick = {}
for algo in ALGORITHM_REGISTRY:
    try:
        best_f, _, conv = run_algorithm(
            algo, sphere, -100, 100, dim=30, seed=42)
        improved = conv[0] > conv[-1]
        results_quick[algo] = best_f
        status = '✅' if improved else '⚠️'
        print(f'  {status} {algo:<6}: {best_f:.4e}')
    except Exception as e:
        print(f'  ❌ {algo:<6}: ERROR — {e}')
        results_quick[algo] = None

print('\n' + '='*60)
print('RANKING (best to worst):')
print('='*60)
valid = {k: v for k, v in results_quick.items() if v is not None}
ranked = sorted(valid.items(), key=lambda x: x[1])
for rank, (algo, val) in enumerate(ranked, 1):
    marker = ' ← MY ALGORITHM' if algo == 'QSO' else ''
    print(f'  {rank:2d}. {algo:<6}: {val:.4e}{marker}')

# %%
# Save registry to Drive
import json

registry_code = f'''"""
QSO Research — Competitor Algorithm Registry
All 15 algorithms with unified interface
Generated from 02_Competitors.ipynb
"""

ALGORITHM_NAMES = {list(ALGORITHM_REGISTRY.keys())}

def get_algorithms():
    """Returns list of all algorithm names."""
    return ALGORITHM_NAMES
'''

with open(f'{BASE}/algorithms/registry.py', 'w') as f:
    f.write(registry_code)

print('✅ Registry saved to Drive')
print(f'\n📋 All 15 algorithms implemented and tested:')
for group, names in algos.items():
    print(f'  {group}: {", ".join(names)}')
print('  + QSO (your algorithm)')

# Also save algorithm names as JSON for easy loading in other notebooks
algo_list = list(ALGORITHM_REGISTRY.keys())
with open(f'{BASE}/algorithms/algo_list.json', 'w') as f:
    json.dump({'algorithms': algo_list}, f, indent=2)

print(f'\n✅ Algorithm list saved: {algo_list}')
print('\n🎯 Ready for Notebook 03 — Sensitivity Analysis')

# %%
##diagnostics

# %%
print('='*55)
print('DIAGNOSTIC — DE AND DBO INVESTIGATION')
print('='*55)

# Test DE with verbose output
print('\nDE Investigation:')
print('  Testing with different seeds...')
de_results = []
for seed in [42, 43, 44, 45, 46]:
    f, _, c = de(sphere, -100, 100, dim=30, seed=seed)
    de_results.append(f)
    print(f'  Seed {seed}: {f:.4e} '
          f'(initial: {c[0]:.4e}, improvement: {(c[0]-f)/c[0]*100:.1f}%)')

print(f'  DE mean: {np.mean(de_results):.4e}')
print(f'  DE std:  {np.std(de_results):.4e}')

# Check if DE is actually converging
print('\n  DE convergence check (seed 42):')
f, _, c = de(sphere, -100, 100, dim=30, seed=42)
checkpoints = [0, 50, 100, 200, 300, 400, 499]
for cp in checkpoints:
    print(f'    Iter {cp:3d}: {c[cp]:.4e}')

print('\nDBO Investigation:')
print('  Testing with different seeds...')
dbo_results = []
for seed in [42, 43, 44, 45, 46]:
    f, _, c = dbo(sphere, -100, 100, dim=30, seed=seed)
    dbo_results.append(f)
    print(f'  Seed {seed}: {f:.4e}')

print(f'  DBO mean: {np.mean(dbo_results):.4e}')

# Check DBO convergence
print('\n  DBO convergence check (seed 42):')
f, _, c = dbo(sphere, -100, 100, dim=30, seed=42)
for cp in checkpoints:
    print(f'    Iter {cp:3d}: {c[cp]:.4e}')

# %%
# Run this diagnostic first to understand the real issue
print('DE Deep Diagnostic')
print('='*50)

# Test 1 — Check if pop_size is the bottleneck
print('\nTest 1 — Population size effect:')
for ps in [30, 50, 100]:
    f, _, c = de(sphere, -100, 100, dim=30,
                 pop_size=ps, max_iter=500, seed=42)
    print(f'  pop_size={ps}: {f:.4e}')

# Test 2 — Check if iterations is the bottleneck
print('\nTest 2 — Iteration count effect:')
for mi in [500, 1000, 2000]:
    f, _, c = de(sphere, -100, 100, dim=30,
                 pop_size=30, max_iter=mi, seed=42)
    print(f'  max_iter={mi}: {f:.4e}')

# Test 3 — Check original F values
print('\nTest 3 — F value effect (original, no adaptive):')
for f_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
    f, _, c = de(sphere, -100, 100, dim=30,
                 pop_size=30, max_iter=500,
                 F=f_val, cr=0.9, seed=42)
    print(f'  F={f_val}: {f:.4e}')

# Test 4 — Check CR values
print('\nTest 4 — CR value effect:')
for cr_val in [0.5, 0.7, 0.9, 1.0]:
    f, _, c = de(sphere, -100, 100, dim=30,
                 pop_size=30, max_iter=500,
                 F=0.8, cr=cr_val, seed=42)
    print(f'  CR={cr_val}: {f:.4e}')

# Test 5 — Check 10D vs 30D
print('\nTest 5 — Dimensionality effect:')
for d in [10, 20, 30]:
    f, _, c = de(sphere, -100, 100, dim=d,
                 pop_size=30, max_iter=500, seed=42)
    print(f'  dim={d}: {f:.4e}')

# %%
print('='*55)
print('QSO vs TOP COMPETITORS — MULTIMODAL FUNCTIONS')
print('='*55)

def rastrigin(x):
    d = len(x)
    return 10*d + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def ackley(x):
    d = len(x)
    return (-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/d))
            - np.exp(np.sum(np.cos(2*np.pi*x))/d)
            + 20 + np.e)

def levy(x):
    d = len(x)
    w = 1 + (x - 1) / 4
    return (np.sin(np.pi*w[0])**2
            + np.sum((w[:-1]-1)**2
            * (1 + 10*np.sin(np.pi*w[:-1]+1)**2))
            + (w[-1]-1)**2 * (1 + np.sin(2*np.pi*w[-1])**2))

test_funcs = [
    ('Rastrigin', rastrigin, -5.12, 5.12),
    ('Ackley',    ackley,    -32,   32),
    ('Levy',      levy,      -10,   10),
]

# Test QSO vs top 5 competitors from Cell 16
top_competitors = ['QSO', 'POA', 'HHO', 'GWO', 'SCA', 'PSO']

for fname, func, lb, ub in test_funcs:
    print(f'\n{fname} (30D):')
    func_results = {}

    for algo in top_competitors:
        best_f, _, _ = run_algorithm(algo, func, lb, ub,
                                      dim=30, seed=42)
        func_results[algo] = best_f

    # Rank them
    ranked = sorted(func_results.items(), key=lambda x: x[1])
    for rank, (algo, val) in enumerate(ranked, 1):
        marker = ' ← QSO' if algo == 'QSO' else ''
        print(f'  {rank}. {algo:<6}: {val:.4e}{marker}')

# %%
print('='*55)
print('30-RUN AVERAGE — SPHERE 30D (proper comparison)')
print('='*55)

seeds = list(range(42, 72))
algo_subset = ['QSO', 'POA', 'HHO', 'GWO', 'SCA', 'WOA', 'PSO', 'MPA']

mean_results = {}
for algo in algo_subset:
    runs = []
    for seed in seeds:
        f, _, _ = run_algorithm(sphere, -100, 100,
                                 dim=30, seed=seed) \
                  if False else \
                  (lambda: run_algorithm(
                      algo, sphere, -100, 100,
                      dim=30, seed=seed))()
        runs.append(f)
    mean_results[algo] = {
        'mean': np.mean(runs),
        'std':  np.std(runs)
    }

print(f'\n{"Rank":<6} {"Algorithm":<8} {"Mean":<16} {"Std"}')
print('-'*45)
ranked = sorted(mean_results.items(),
                key=lambda x: x[1]['mean'])
for rank, (algo, stats) in enumerate(ranked, 1):
    marker = ' ← QSO' if algo == 'QSO' else ''
    print(f'{rank:<6} {algo:<8} '
          f'{stats["mean"]:<16.4e} '
          f'{stats["std"]:.4e}{marker}')

# %%
def gjo(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        seed=42):
    """
    Golden Jackal Optimizer (Chopra & Ansari, 2022)
    Published: Expert Systems with Applications, 198, 116924

    Models male and female jackal hunting behaviour:
    - Male jackal: tracks prey (global best)
    - Female jackal: supports male (second best)
    - Prey escape energy decreases over iterations
    """
    X, lb, ub = initialise_population(
                    pop_size, dim, lb, ub, seed)
    fitness    = evaluate_population(func, X)

    # Male and female jackal (best two solutions)
    sorted_idx  = np.argsort(fitness)
    male_pos    = X[sorted_idx[0]].copy()
    male_f      = fitness[sorted_idx[0]]
    female_pos  = X[sorted_idx[1]].copy()
    female_f    = fitness[sorted_idx[1]]

    gbest_f     = male_f
    gbest_X     = male_pos.copy()
    convergence = [gbest_f]

    for t in range(max_iter):
        # Prey escape energy
        E1  = 1.5 * (1 - t / max_iter)
        RL  = 0.05 * _levy_mpa(pop_size, dim)

        for i in range(pop_size):
            E0  = 2 * np.random.rand() - 1
            E   = E1 * E0

            r1  = np.random.rand()

            # ── Update toward male jackal ─────────────────
            D_male   = np.abs(
                RL[i] * male_pos - X[i])
            X1       = (male_pos - E * D_male)

            # ── Update toward female jackal ───────────────
            D_female = np.abs(
                RL[i] * female_pos - X[i])
            X2       = (female_pos - E * D_female)

            # ── Average of both updates ───────────────────
            X[i] = np.clip(
                (X1 + X2) / 2, lb, ub)

        fitness = evaluate_population(func, X)

        # Update male and female jackals
        sorted_idx = np.argsort(fitness)

        if fitness[sorted_idx[0]] < male_f:
            male_f   = fitness[sorted_idx[0]]
            male_pos = X[sorted_idx[0]].copy()

        if fitness[sorted_idx[1]] < female_f:
            female_f   = fitness[sorted_idx[1]]
            female_pos = X[sorted_idx[1]].copy()

        if male_f < gbest_f:
            gbest_f = male_f
            gbest_X = male_pos.copy()

        convergence.append(gbest_f)

    return gbest_f, gbest_X, convergence


# --- Test ---
print('Testing GJO...')
f, _, c = gjo(sphere, -100, 100, dim=30, seed=42)
print(f'  Sphere 30D: {f:.4e}')
assert f < c[0], 'GJO did not improve'
print('✅ GJO operational')

# %%
sphere = lambda x: np.sum(x**2)

print(f'Total algorithms: {len(ALGORITHM_REGISTRY)}')
assert 'GJO' in ALGORITHM_REGISTRY, 'GJO not in registry'

f, _, _ = run_algorithm('GJO', sphere, -100, 100,
                         dim=30, seed=42)
print(f'GJO Sphere 30D: {f:.4e}')
print('✅ GJO in registry and working')

# %%
# ── Run GJO on all benchmark suites ──────────────────────────────
import time

SEEDS     = list(range(42, 72))
GJO_ONLY  = ['GJO']

print('='*55)
print('RUNNING GJO ON ALL BENCHMARK SUITES')
print('='*55)
print(f'  Seeds: {len(SEEDS)}')
print(f'  Estimated time: ~15-25 minutes total\n')

# ── Classical 23 ──────────────────────────────────────────────────
print('1. Classical 23...')
start = time.time()
run_benchmark_suite(
    funcs_dict = CLASSICAL_FUNCS,
    algo_list  = GJO_ONLY,
    seeds      = SEEDS,
    save_path  = f'{BASE}/results/raw/classical',
    suite_name = 'GJO Classical 23'
)
print(f'   Done in {(time.time()-start)/60:.1f} mins\n')

# ── CEC 2017 30D ──────────────────────────────────────────────────
print('2. CEC 2017 30D...')
start = time.time()
run_benchmark_suite(
    funcs_dict = CEC2017_30D,
    algo_list  = GJO_ONLY,
    seeds      = SEEDS,
    save_path  = f'{BASE}/results/raw/cec2017_30d',
    suite_name = 'GJO CEC 2017 30D'
)
print(f'   Done in {(time.time()-start)/60:.1f} mins\n')

# ── CEC 2017 50D ──────────────────────────────────────────────────
print('3. CEC 2017 50D...')
start = time.time()
run_benchmark_suite(
    funcs_dict = CEC2017_50D,
    algo_list  = GJO_ONLY,
    seeds      = SEEDS,
    save_path  = f'{BASE}/results/raw/cec2017_50d',
    suite_name = 'GJO CEC 2017 50D'
)
print(f'   Done in {(time.time()-start)/60:.1f} mins\n')

print('='*55)
print('✅ GJO complete on all suites')
print('   Rerun Cell 11 to see updated rankings')
print('='*55)


