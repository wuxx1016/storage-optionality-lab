import numpy as np

from .config import PriceConfig, StorageConfig


def month_index(step: int) -> int:
    return int(step % 12)


def seasonal_log_level(step: int, storage_cfg: StorageConfig, price_cfg: PriceConfig) -> float:
    angle = 2.0 * np.pi * (step / 12.0 + price_cfg.seasonal_phase)
    return price_cfg.mean_log_price + price_cfg.seasonal_amplitude * np.cos(angle)


def _winter_loading(months: np.ndarray, price_cfg: PriceConfig) -> np.ndarray:
    winter = np.isin(months % 12, list(price_cfg.winter_months)).astype(float)
    summer = np.isin(months % 12, list(price_cfg.summer_months)).astype(float)
    return winter - summer


def simulate_market_paths(
    n_paths: int,
    storage_cfg: StorageConfig,
    price_cfg: PriceConfig,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Simulate prompt, curve-spread, basis, and volatility proxy paths.

    The curve is not arbitrage-free. It is a practical toy state generator:
    prompt risk, seasonal spread risk, local basis risk, and volatility proxies
    are explicit so valuation features resemble a desk model.
    """
    rng = np.random.default_rng(seed)
    n_steps = storage_cfg.n_steps
    dt = storage_cfg.dt
    tenors = price_cfg.curve_tenors

    log_prompt = np.empty((n_paths, n_steps + 1), dtype=float)
    spread_factor = np.empty_like(log_prompt)
    basis = np.empty_like(log_prompt)
    vol_factor = np.empty_like(log_prompt)
    curve = np.empty((n_paths, n_steps + 1, tenors), dtype=float)

    log_prompt[:, 0] = np.log(price_cfg.s0)
    spread_factor[:, 0] = 0.0
    basis[:, 0] = 0.0
    vol_factor[:, 0] = 0.0

    corr = np.array(
        [
            [1.00, -0.25, 0.10, 0.15],
            [-0.25, 1.00, 0.05, 0.20],
            [0.10, 0.05, 1.00, 0.00],
            [0.15, 0.20, 0.00, 1.00],
        ]
    )
    chol = np.linalg.cholesky(corr)

    for t in range(n_steps):
        shocks = rng.standard_normal((n_paths, 4)) @ chol.T
        prompt_sigma = price_cfg.sigma_prompt * np.exp(0.30 * vol_factor[:, t])
        spread_sigma = price_cfg.sigma_spread * np.exp(0.35 * vol_factor[:, t])
        theta_t = seasonal_log_level(t, storage_cfg, price_cfg)

        log_prompt[:, t + 1] = (
            log_prompt[:, t]
            + price_cfg.kappa_prompt * (theta_t - log_prompt[:, t]) * dt
            + prompt_sigma * np.sqrt(dt) * shocks[:, 0]
        )
        spread_factor[:, t + 1] = (
            spread_factor[:, t]
            - price_cfg.kappa_spread * spread_factor[:, t] * dt
            + spread_sigma * np.sqrt(dt) * shocks[:, 1]
        )
        basis[:, t + 1] = (
            basis[:, t]
            - price_cfg.kappa_basis * basis[:, t] * dt
            + price_cfg.sigma_basis * np.sqrt(dt) * shocks[:, 2]
        )
        vol_factor[:, t + 1] = (
            0.92 * vol_factor[:, t]
            + price_cfg.vol_of_vol * np.sqrt(dt) * shocks[:, 3]
        )

    for t in range(n_steps + 1):
        delivery_steps = t + np.arange(tenors)
        seasonal = np.array([seasonal_log_level(int(m), storage_cfg, price_cfg) for m in delivery_steps])
        prompt_season = seasonal_log_level(t, storage_cfg, price_cfg)
        tenor_decay = np.exp(-np.arange(tenors) / 4.0)
        winter_load = _winter_loading(delivery_steps, price_cfg)
        log_curve = (
            log_prompt[:, [t]]
            + (seasonal - prompt_season)[None, :]
            + 0.18 * spread_factor[:, [t]] * winter_load[None, :]
            - 0.06 * spread_factor[:, [t]] * tenor_decay[None, :]
        )
        curve[:, t, :] = np.exp(log_curve)

    prompt = curve[:, :, 0]
    next_month = curve[:, :, 1]
    prompt_next = prompt - next_month

    winter_avg = np.zeros_like(prompt)
    summer_avg = np.zeros_like(prompt)
    for t in range(n_steps + 1):
        delivery_months = (t + np.arange(tenors)) % 12
        winter_mask = np.isin(delivery_months, list(price_cfg.winter_months))
        summer_mask = np.isin(delivery_months, list(price_cfg.summer_months))
        winter_avg[:, t] = curve[:, t, winter_mask].mean(axis=1)
        summer_avg[:, t] = curve[:, t, summer_mask].mean(axis=1)

    return {
        "curve": curve,
        "prompt": prompt,
        "next_month": next_month,
        "winter_avg": winter_avg,
        "summer_avg": summer_avg,
        "prompt_next_spread": prompt_next,
        "winter_summer_spread": winter_avg - summer_avg,
        "basis": basis,
        "local_prompt": np.maximum(0.01, prompt + basis),
        "prompt_vol": price_cfg.sigma_prompt * np.exp(0.30 * vol_factor),
        "spread_vol": price_cfg.sigma_spread * np.exp(0.35 * vol_factor),
        "spread_factor": spread_factor,
        "vol_factor": vol_factor,
    }


def market_feature_matrix(
    market: dict[str, np.ndarray],
    t: int,
    inventory: np.ndarray,
    storage_cfg: StorageConfig,
    price_cfg: PriceConfig,
    include_constant: bool = False,
) -> np.ndarray:
    inv = np.asarray(inventory, dtype=float) / storage_cfg.capacity
    n = inv.shape[0]
    prompt = market["prompt"][:, t] / price_cfg.s0
    next_month = market["next_month"][:, t] / price_cfg.s0
    pnext = market["prompt_next_spread"][:, t] / price_cfg.s0
    wsum = market["winter_summer_spread"][:, t] / price_cfg.s0
    basis = market["basis"][:, t] / price_cfg.s0
    pvol = market["prompt_vol"][:, t] / price_cfg.sigma_prompt - 1.0
    svol = market["spread_vol"][:, t] / price_cfg.sigma_spread - 1.0

    month = month_index(t)
    month_dummies = np.zeros((n, 11), dtype=float)
    if month < 11:
        month_dummies[:, month] = 1.0
    winter = float(month in price_cfg.winter_months)
    summer = float(month in price_cfg.summer_months)
    shoulder = 1.0 - max(winter, summer)
    season_dummies = np.column_stack(
        [
            np.full(n, winter),
            np.full(n, summer),
            np.full(n, shoulder),
        ]
    )

    core = np.column_stack(
        [
            np.full(n, t / storage_cfg.n_steps),
            inv,
            inv * inv,
            prompt,
            np.log(np.maximum(prompt, 1e-6)),
            next_month,
            pnext,
            wsum,
            basis,
            pvol,
            svol,
            inv * pnext,
            inv * wsum,
            inv * basis,
            (inv < 0.25).astype(float),
            (inv > 0.75).astype(float),
        ]
    )
    out = np.column_stack([core, month_dummies, season_dummies])
    if include_constant:
        out = np.column_stack([np.ones(n), out])
    return out


def deterministic_forward_curve(storage_cfg: StorageConfig, price_cfg: PriceConfig) -> np.ndarray:
    months = np.arange(price_cfg.curve_tenors)
    return np.exp([seasonal_log_level(int(m), storage_cfg, price_cfg) for m in months])


def simulate_log_ou_paths(
    n_paths: int,
    storage_cfg: StorageConfig,
    price_cfg: PriceConfig,
    seed: int = 7,
) -> np.ndarray:
    """Backward-compatible prompt-path helper."""
    return simulate_market_paths(n_paths, storage_cfg, price_cfg, seed=seed)["prompt"]
