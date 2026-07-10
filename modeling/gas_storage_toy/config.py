from dataclasses import dataclass


@dataclass(frozen=True)
class StorageConfig:
    n_steps: int = 24
    dt: float = 1.0 / 12.0
    discount_rate: float = 0.04
    capacity: float = 100.0
    initial_inventory: float = 50.0
    max_inject: float = 12.0
    max_withdraw: float = 14.0
    ratchet_floor: float = 0.35
    inject_cost: float = 0.08
    withdraw_cost: float = 0.05
    terminal_target: float = 50.0
    terminal_penalty: float = 1.0
    terminal_salvage_fraction: float = 0.95


@dataclass(frozen=True)
class PriceConfig:
    s0: float = 3.0
    mean_log_price: float = 1.05
    kappa_prompt: float = 1.35
    sigma_prompt: float = 0.42
    kappa_spread: float = 1.05
    sigma_spread: float = 0.28
    kappa_basis: float = 2.0
    sigma_basis: float = 0.10
    vol_of_vol: float = 0.45
    seasonal_amplitude: float = 0.18
    seasonal_phase: float = 0.0
    curve_tenors: int = 13
    winter_months: tuple[int, ...] = (10, 11, 0, 1, 2)
    summer_months: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
