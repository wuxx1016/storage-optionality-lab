import numpy as np

from .config import PriceConfig, StorageConfig
from .price import market_feature_matrix, simulate_market_paths


def inject_rate(inventory: np.ndarray | float, cfg: StorageConfig) -> np.ndarray | float:
    inv_norm = np.asarray(inventory) / cfg.capacity
    multiplier = cfg.ratchet_floor + (1.0 - cfg.ratchet_floor) * (1.0 - inv_norm)
    return cfg.max_inject * multiplier


def withdraw_rate(inventory: np.ndarray | float, cfg: StorageConfig) -> np.ndarray | float:
    inv_norm = np.asarray(inventory) / cfg.capacity
    multiplier = cfg.ratchet_floor + (1.0 - cfg.ratchet_floor) * inv_norm
    return cfg.max_withdraw * multiplier


def feasible_quantity(direction: int, inventory: np.ndarray | float, cfg: StorageConfig) -> np.ndarray | float:
    if direction > 0:
        return np.minimum(inject_rate(inventory, cfg), cfg.capacity - inventory)
    if direction < 0:
        return -np.minimum(withdraw_rate(inventory, cfg), inventory)
    return np.zeros_like(np.asarray(inventory), dtype=float)


class GasStorageEnv:
    """Episodic gas storage environment with curve features and ratchets."""

    ACTIONS = np.array([-1, 0, 1], dtype=int)

    def __init__(
        self,
        storage_cfg: StorageConfig | None = None,
        price_cfg: PriceConfig | None = None,
        seed: int = 11,
    ):
        self.storage_cfg = storage_cfg or StorageConfig()
        self.price_cfg = price_cfg or PriceConfig()
        self.rng = np.random.default_rng(seed)
        self.market = None
        self.t = 0
        self.inventory = self.storage_cfg.initial_inventory

    @property
    def obs_dim(self) -> int:
        dummy_market = simulate_market_paths(1, self.storage_cfg, self.price_cfg, seed=1)
        dummy_inventory = np.array([self.storage_cfg.initial_inventory])
        return market_feature_matrix(dummy_market, 0, dummy_inventory, self.storage_cfg, self.price_cfg).shape[1]

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    def reset(self) -> np.ndarray:
        seed = int(self.rng.integers(0, 2**31 - 1))
        self.market = simulate_market_paths(1, self.storage_cfg, self.price_cfg, seed=seed)
        self.t = 0
        self.inventory = self.storage_cfg.initial_inventory
        return self._obs()

    def _obs(self) -> np.ndarray:
        inv = np.array([self.inventory])
        return market_feature_matrix(self.market, self.t, inv, self.storage_cfg, self.price_cfg)[0].astype(np.float32)

    @property
    def prompt_price(self) -> float:
        return float(self.market["prompt"][0, self.t])

    @property
    def local_price(self) -> float:
        return float(self.market["local_prompt"][0, self.t])

    def feasible_action_indices(self) -> list[int]:
        feasible = []
        for idx, direction in enumerate(self.ACTIONS):
            qty = float(feasible_quantity(direction, self.inventory, self.storage_cfg))
            if -1e-9 <= self.inventory + qty <= self.storage_cfg.capacity + 1e-9:
                feasible.append(idx)
        return feasible

    def _quantity(self, direction: int) -> float:
        return float(feasible_quantity(direction, self.inventory, self.storage_cfg))

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict]:
        cfg = self.storage_cfg
        direction = int(self.ACTIONS[action_idx])
        qty = self._quantity(direction)
        local_price = self.local_price

        if direction > 0:
            cashflow = -(local_price + cfg.inject_cost) * qty
        elif direction < 0:
            cashflow = (local_price - cfg.withdraw_cost) * (-qty)
        else:
            cashflow = 0.0

        self.inventory += qty
        self.t += 1
        done = self.t >= cfg.n_steps

        if done:
            final_price = float(self.market["local_prompt"][0, self.t])
            salvage = cfg.terminal_salvage_fraction * final_price * self.inventory
            penalty = cfg.terminal_penalty * final_price * abs(self.inventory - cfg.terminal_target)
            cashflow += salvage - penalty

        reward = cashflow * np.exp(-cfg.discount_rate * cfg.dt * self.t)
        next_obs = self._obs() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        info = {
            "quantity": qty,
            "cashflow": cashflow,
            "prompt": self.prompt_price if not done else float(self.market["prompt"][0, -1]),
            "basis": float(self.market["basis"][0, min(self.t, cfg.n_steps)]),
        }
        return next_obs, float(reward), done, info


def evaluate_policy(policy_fn, n_episodes: int = 1000, seed: int = 123) -> dict:
    env = GasStorageEnv(seed=seed)
    rewards = []
    inventories = []
    for _ in range(n_episodes):
        obs = env.reset()
        total = 0.0
        done = False
        while not done:
            action_idx = policy_fn(obs, env)
            obs, reward, done, _ = env.step(action_idx)
            total += reward
        rewards.append(total)
        inventories.append(env.inventory)
    arr = np.asarray(rewards)
    return {
        "mean_value": float(arr.mean()),
        "std_value": float(arr.std(ddof=1)),
        "stderr": float(arr.std(ddof=1) / np.sqrt(n_episodes)),
        "mean_final_inventory": float(np.mean(inventories)),
    }
