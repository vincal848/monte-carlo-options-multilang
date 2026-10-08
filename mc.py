"""Monte Carlo pricer for European options under risk-neutral geometric Brownian motion.

S_T = S0 * exp((r - sigma^2/2) * T + sigma * W_T), simulated by taking n_steps
log-Euler steps of size dt = T / n_steps. For GBM with constant drift and
volatility this step is exact, not an approximation -- there is no discretization
error from the step count itself, only Monte Carlo sampling error from the finite
number of paths. The original scripts in legacy/ got the risk-neutral part wrong:
they simulated under the real-world drift mu and discounted at r, which is not a
risk-neutral measure and does not converge to Black-Scholes.

Pricing reuses the same simulated S_T across every strike, so put-call parity
holds exactly path by path (call_payoff - put_payoff = S_T - K for every path),
rather than as two independent estimates that merely happen to agree.
"""

import math

import numpy as np


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError("%s must be strictly positive, got %r" % (name, value))


def simulate_terminal(S0: float, r: float, sigma: float, T: float, n_steps: int,
                      n_paths: int, seed: int | None = None,
                      antithetic: bool = False) -> np.ndarray:
    """Terminal prices S_T, one per path, via n_steps exact log-Euler steps.

    With antithetic=True, n_paths must be even and the paths are returned
    interleaved as (plus_1, minus_1, plus_2, minus_2, ...): each pair shares the
    same draws Z negated, so price_european can average pair by pair.
    """
    _check_positive("S0", S0)
    _check_positive("sigma", sigma)
    _check_positive("T", T)
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer, got %r" % (n_steps,))
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer, got %r" % (n_paths,))
    if antithetic and n_paths % 2 != 0:
        raise ValueError("antithetic sampling needs an even n_paths, got %r" % (n_paths,))

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        log_plus = np.sum(drift + vol * Z, axis=1)
        log_minus = np.sum(drift - vol * Z, axis=1)
        ST = np.empty(n_paths)
        ST[0::2] = S0 * np.exp(log_plus)
        ST[1::2] = S0 * np.exp(log_minus)
    else:
        Z = rng.standard_normal((n_paths, n_steps))
        log_terminal = np.sum(drift + vol * Z, axis=1)
        ST = S0 * np.exp(log_terminal)
    return ST


def simulate_paths(S0: float, r: float, sigma: float, T: float, n_steps: int,
                   n_paths: int, seed: int | None = None) -> np.ndarray:
    """Full paths including S0 as step 0, shape (n_paths, n_steps + 1). For plots,
    not for pricing -- n_paths is normally small here.
    """
    _check_positive("S0", S0)
    _check_positive("sigma", sigma)
    _check_positive("T", T)
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer, got %r" % (n_steps,))
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer, got %r" % (n_paths,))

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    Z = rng.standard_normal((n_paths, n_steps))
    log_paths = np.cumsum(drift + vol * Z, axis=1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def price_european(S0: float, strikes: list[float], r: float, sigma: float, T: float,
                   n_steps: int, n_paths: int, seed: int | None = None,
                   antithetic: bool = False) -> list[dict[str, float]]:
    """Call and put price plus standard error for each strike, from one shared sample.

    Returns a list of dicts: {K, call, call_se, put, put_se}, one per strike, in
    the order given.
    """
    strikes = list(strikes)
    if not strikes:
        raise ValueError("strikes must be a non-empty list")
    for K in strikes:
        _check_positive("K", K)

    ST = simulate_terminal(S0, r, sigma, T, n_steps, n_paths, seed, antithetic)
    discount = math.exp(-r * T)

    rows = []
    for K in strikes:
        call_pay = np.maximum(ST - K, 0.0)
        put_pay = np.maximum(K - ST, 0.0)

        if antithetic:
            call_sample = (call_pay[0::2] + call_pay[1::2]) / 2.0
            put_sample = (put_pay[0::2] + put_pay[1::2]) / 2.0
        else:
            call_sample = call_pay
            put_sample = put_pay

        n_eff = call_sample.shape[0]
        rows.append({
            "K": float(K),
            "call": discount * call_sample.mean(),
            "call_se": discount * call_sample.std(ddof=1) / math.sqrt(n_eff),
            "put": discount * put_sample.mean(),
            "put_se": discount * put_sample.std(ddof=1) / math.sqrt(n_eff),
        })
    return rows
