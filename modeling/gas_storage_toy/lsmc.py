import numpy as np

from .config import PriceConfig, StorageConfig
from .env import feasible_quantity
from .price import market_feature_matrix, simulate_market_paths


def basis(market: dict[str, np.ndarray], inventory: np.ndarray, month: int, cfg: StorageConfig, pcfg: PriceConfig) -> np.ndarray:
    return market_feature_matrix(market, month, inventory, cfg, pcfg, include_constant=True)


def immediate_cashflow(direction: int, qty: np.ndarray, local_price: np.ndarray, cfg: StorageConfig) -> np.ndarray:
    if direction > 0:
        return -(local_price + cfg.inject_cost) * qty
    if direction < 0:
        return (local_price - cfg.withdraw_cost) * (-qty)
    return np.zeros_like(local_price)


def terminal_value(inventory: np.ndarray, local_price: np.ndarray, cfg: StorageConfig) -> np.ndarray:
    salvage = cfg.terminal_salvage_fraction * local_price * inventory
    penalty = cfg.terminal_penalty * local_price * np.abs(inventory - cfg.terminal_target)
    return salvage - penalty


def train_lsmc(
    n_paths: int = 10000,
    storage_cfg: StorageConfig | None = None,
    price_cfg: PriceConfig | None = None,
    seed: int = 42,
) -> dict:
    cfg = storage_cfg or StorageConfig()
    pcfg = price_cfg or PriceConfig()
    market = simulate_market_paths(n_paths, cfg, pcfg, seed=seed)
    disc = np.exp(-cfg.discount_rate * cfg.dt)
    value_models: list[np.ndarray | None] = [None] * cfg.n_steps

    rng = np.random.default_rng(seed + 100)
    inventory_design = rng.uniform(0.0, cfg.capacity, size=n_paths)

    for t in range(cfg.n_steps - 1, -1, -1):
        values_by_action = []
        for direction in (-1, 0, 1):
            qty = feasible_quantity(direction, inventory_design, cfg)
            next_inv = inventory_design + qty
            cf = immediate_cashflow(direction, qty, market["local_prompt"][:, t], cfg)
            if t + 1 < cfg.n_steps:
                x_next = basis(market, next_inv, t + 1, cfg, pcfg)
                cont = x_next @ value_models[t + 1]
            else:
                cont = terminal_value(next_inv, market["local_prompt"][:, t + 1], cfg)
            values_by_action.append(cf + disc * cont)

        y = np.max(np.column_stack(values_by_action), axis=1)
        x = basis(market, inventory_design, t, cfg, pcfg)
        coef, *_ = np.linalg.lstsq(x, y, rcond=1e-8)
        value_models[t] = coef
        inventory_design = rng.uniform(0.0, cfg.capacity, size=n_paths)

    one_path_market = simulate_market_paths(1, cfg, pcfg, seed=seed + 1)
    one_path_market["prompt"][:, 0] = pcfg.s0
    one_path_market["local_prompt"][:, 0] = pcfg.s0
    initial_value = float(
        (basis(one_path_market, np.array([cfg.initial_inventory]), 0, cfg, pcfg) @ value_models[0])[0]
    )
    return {"models": value_models, "initial_value": initial_value, "storage_cfg": cfg, "price_cfg": pcfg}


def _single_path_market_view(market: dict[str, np.ndarray], path_idx: int) -> dict[str, np.ndarray]:
    out = {}
    for key, value in market.items():
        if value.ndim == 3:
            out[key] = value[path_idx : path_idx + 1, :, :]
        else:
            out[key] = value[path_idx : path_idx + 1, :]
    return out


def lsmc_action(model: dict, t: int, inventory: float, market_one_path: dict[str, np.ndarray]) -> int:
    cfg: StorageConfig = model["storage_cfg"]
    pcfg: PriceConfig = model["price_cfg"]
    disc = np.exp(-cfg.discount_rate * cfg.dt)
    best_idx = 1
    best_value = -np.inf
    local_price = market_one_path["local_prompt"][:, t]
    for idx, direction in enumerate((-1, 0, 1)):
        qty = float(feasible_quantity(direction, inventory, cfg))
        next_inv = inventory + qty
        cf = immediate_cashflow(direction, np.array([qty]), local_price, cfg)[0]
        if t + 1 < cfg.n_steps:
            cont = basis(market_one_path, np.array([next_inv]), t + 1, cfg, pcfg) @ model["models"][t + 1]
            cont = float(cont[0])
        else:
            cont = terminal_value(np.array([next_inv]), market_one_path["local_prompt"][:, t + 1], cfg)[0]
        value = cf + disc * cont
        if value > best_value:
            best_value = value
            best_idx = idx
    return best_idx


def evaluate_lsmc_policy(model: dict, n_paths: int = 3000, seed: int = 99) -> dict:
    cfg: StorageConfig = model["storage_cfg"]
    pcfg: PriceConfig = model["price_cfg"]
    market = simulate_market_paths(n_paths, cfg, pcfg, seed=seed)
    values = np.zeros(n_paths)
    final_inv = np.zeros(n_paths)
    actions = np.zeros((n_paths, cfg.n_steps), dtype=int)
    for p in range(n_paths):
        inv = cfg.initial_inventory
        total = 0.0
        market_p = _single_path_market_view(market, p)
        for t in range(cfg.n_steps):
            action_idx = lsmc_action(model, t, inv, market_p)
            direction = (-1, 0, 1)[action_idx]
            qty = float(feasible_quantity(direction, inv, cfg))
            cf = immediate_cashflow(direction, np.array([qty]), market_p["local_prompt"][:, t], cfg)[0]
            inv += qty
            total += np.exp(-cfg.discount_rate * cfg.dt * (t + 1)) * cf
            actions[p, t] = direction
        total += np.exp(-cfg.discount_rate * cfg.dt * cfg.n_steps) * terminal_value(
            np.array([inv]), market_p["local_prompt"][:, -1], cfg
        )[0]
        values[p] = total
        final_inv[p] = inv
    return {
        "mean_value": float(values.mean()),
        "std_value": float(values.std(ddof=1)),
        "stderr": float(values.std(ddof=1) / np.sqrt(n_paths)),
        "mean_final_inventory": float(final_inv.mean()),
        "actions": actions,
    }
