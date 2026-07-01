
import numpy as np

def levy_flight(n, d, alpha=1.25):
    from scipy.special import gamma
    sigma_u = (
        gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
        (gamma((1 + alpha) / 2) * alpha * 2**((alpha-1)/2))
    ) ** (1/alpha)
    u = np.random.normal(0, sigma_u, (n, d))
    v = np.random.normal(0, 1.0, (n, d))
    return u / (np.abs(v) ** (1/alpha))

def clip_to_bounds(x, lb, ub):
    repair_lb = lb + np.random.rand(*x.shape) * (ub - lb) * 0.1
    repair_ub = ub - np.random.rand(*x.shape) * (ub - lb) * 0.1
    x = np.where(x < lb, repair_lb, x)
    x = np.where(x > ub, repair_ub, x)
    return x

def compute_ai_concentration(fitness_values, f_best, f_worst):
    epsilon = 1e-10
    if (f_worst - f_best) < epsilon:
        return 0.5
    phi = (f_worst - fitness_values) / (f_worst - f_best + epsilon)
    return float(np.clip(np.mean(phi), 0.0, 1.0))

def adaptive_threshold(t, max_iter, theta_min=0.3, theta_max=0.7):
    return float(theta_max - (theta_max - theta_min)
                 * (t / max_iter))

def exploration_phase(X, lb, ub, alpha=1.25):
    n, d = X.shape
    r1 = np.random.rand(n, d)
    r2 = np.random.rand(n, d)
    X_rand = X[np.random.randint(0, n, size=n)]
    L = levy_flight(n, d, alpha)
    scale = (ub - lb) * 0.01
    L_scaled = np.clip(L * scale, -0.5*(ub-lb), 0.5*(ub-lb))
    X_new = X + r1*(X_rand - X) + r2*L_scaled
    return clip_to_bounds(X_new, lb, ub)

def exploitation_phase(X, X_best, lb, ub):
    n, d = X.shape
    r3 = np.random.rand(n, d)
    r4 = np.random.rand(n, d)
    X_colony = np.mean(X, axis=0)
    X_new = (X
             + r3 * (X_best   - X)
             + r4 * (X_colony - X))
    return clip_to_bounds(X_new, lb, ub)

def apply_ai_decay(C, lambda_=0.05):
    return float(np.clip(C * np.exp(-lambda_), 0.0, 1.0))

def qso(func, lb, ub, dim,
        pop_size=30, max_iter=500,
        theta_min=0.3, theta_max=0.7,
        lambda_=0.05, tau=10, alpha=1.25,
        seed=42, verbose=False):
    np.random.seed(seed)
    lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb)
    ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub)
    X = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([func(X[i]) for i in range(pop_size)])
    best_idx      = np.argmin(fitness)
    best_fitness  = fitness[best_idx]
    best_position = X[best_idx].copy()
    theta_t = adaptive_threshold(
                  0, max_iter, theta_min, theta_max)
    C = compute_ai_concentration(
            fitness, best_fitness, np.max(fitness))
    convergence    = [best_fitness]
    diversity      = [np.mean(np.std(X, axis=0))]
    quorum_history = [C]
    phase_history  = [1 if C >= theta_t else 0]
    theta_history  = [theta_t]
    no_improve_count = 0
    for t in range(max_iter):
        theta_t = adaptive_threshold(
                      t, max_iter, theta_min, theta_max)
        if C >= theta_t:
            X     = exploitation_phase(
                        X, best_position, lb, ub)
            phase = 1
        else:
            X     = exploration_phase(X, lb, ub, alpha)
            phase = 0
        fitness = np.array([func(X[i]) for i in range(pop_size)])
        worst_idx = np.argmax(fitness)
        if fitness[worst_idx] > best_fitness:
            X[worst_idx]       = best_position.copy()
            fitness[worst_idx] = best_fitness
        current_best_idx     = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]
        if current_best_fitness < best_fitness:
            best_fitness     = current_best_fitness
            best_position    = X[current_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        C = compute_ai_concentration(
                fitness, best_fitness, np.max(fitness))
        if no_improve_count >= tau:
            C = apply_ai_decay(C, lambda_)
            no_improve_count = 0
        convergence.append(best_fitness)
        diversity.append(np.mean(np.std(X, axis=0)))
        quorum_history.append(C)
        phase_history.append(phase)
        theta_history.append(theta_t)
        if verbose and (t+1) % 100 == 0:
            print(f"Iter {t+1}/{max_iter} | "
                  f"Best: {best_fitness:.6e} | "
                  f"C: {C:.3f} | theta: {theta_t:.3f}")
    return (best_fitness, best_position, convergence,
            diversity, quorum_history, phase_history,
            theta_history)
