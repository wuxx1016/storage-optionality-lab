from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gas_storage_toy.config import PriceConfig, StorageConfig
from gas_storage_toy.dqn import dqn_policy_fn, train_dqn
from gas_storage_toy.env import GasStorageEnv, evaluate_policy
from gas_storage_toy.lsmc import evaluate_lsmc_policy, lsmc_action, train_lsmc
from gas_storage_toy.price import deterministic_forward_curve, simulate_market_paths


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def heuristic_policy(obs, env: GasStorageEnv) -> int:
    prompt = env.prompt_price
    pnext = float(env.market["prompt_next_spread"][0, env.t])
    wsum = float(env.market["winter_summer_spread"][0, env.t])
    inv_norm = env.inventory / env.storage_cfg.capacity
    month = env.t % 12
    if inv_norm < 0.85 and (wsum > 0.35 or (pnext < -0.10 and month in {3, 4, 5, 6, 7, 8})):
        return 2
    if inv_norm > 0.15 and (prompt > 3.35 or (wsum < 0.05 and month in {10, 11, 0, 1, 2})):
        return 0
    return 1


def lsmc_policy_fn(model):
    def policy(obs, env: GasStorageEnv) -> int:
        return lsmc_action(model, env.t, env.inventory, env.market)

    return policy


def make_slice_market(storage_cfg, price_cfg, t=6, seed=777):
    market = simulate_market_paths(1, storage_cfg, price_cfg, seed=seed)
    for key, value in market.items():
        if value.ndim == 3:
            value[:, :, :] = value[:, [t], :]
        else:
            value[:, :] = value[:, [t]]
    return market


def set_prompt_slice(market, t, price, price_cfg):
    old_prompt = float(market["prompt"][0, t])
    scale = price / old_prompt
    market["curve"][0, t, :] *= scale
    market["prompt"][0, t] = price
    market["next_month"][0, t] *= scale
    market["winter_avg"][0, t] *= scale
    market["summer_avg"][0, t] *= scale
    market["prompt_next_spread"][0, t] = market["prompt"][0, t] - market["next_month"][0, t]
    market["winter_summer_spread"][0, t] = market["winter_avg"][0, t] - market["summer_avg"][0, t]
    market["local_prompt"][0, t] = max(0.01, price + market["basis"][0, t])


def set_winter_summer_slice(market, t, spread):
    mid = 0.5 * (market["winter_avg"][0, t] + market["summer_avg"][0, t])
    market["winter_avg"][0, t] = mid + 0.5 * spread
    market["summer_avg"][0, t] = mid - 0.5 * spread
    market["winter_summer_spread"][0, t] = spread


def policy_grid_lsmc(model, storage_cfg, price_cfg):
    t = 6
    prices = np.linspace(2.0, 4.5, 45)
    inventories = np.linspace(0.0, storage_cfg.capacity, 41)
    grid = np.zeros((len(inventories), len(prices)))
    for i, inv in enumerate(inventories):
        for j, price in enumerate(prices):
            market = make_slice_market(storage_cfg, price_cfg, t=t)
            set_prompt_slice(market, t, price, price_cfg)
            grid[i, j] = (-1, 0, 1)[lsmc_action(model, t, inv, market)]
    return prices, inventories, grid


def policy_grid_dqn(model, storage_cfg, price_cfg):
    t = 6
    env = GasStorageEnv(storage_cfg, price_cfg, seed=321)
    env.market = make_slice_market(storage_cfg, price_cfg, t=t)
    env.t = t
    prices = np.linspace(2.0, 4.5, 45)
    inventories = np.linspace(0.0, storage_cfg.capacity, 41)
    grid = np.zeros((len(inventories), len(prices)))
    policy = dqn_policy_fn(model)
    for i, inv in enumerate(inventories):
        for j, price in enumerate(prices):
            env.inventory = float(inv)
            set_prompt_slice(env.market, t, float(price), price_cfg)
            action = policy(env._obs(), env)
            grid[i, j] = env.ACTIONS[action]
    return prices, inventories, grid


def spread_grid_lsmc(model, storage_cfg, price_cfg):
    t = 6
    spreads = np.linspace(-0.50, 1.00, 45)
    inventories = np.linspace(0.0, storage_cfg.capacity, 41)
    grid = np.zeros((len(inventories), len(spreads)))
    for i, inv in enumerate(inventories):
        for j, spread in enumerate(spreads):
            market = make_slice_market(storage_cfg, price_cfg, t=t)
            set_winter_summer_slice(market, t, float(spread))
            grid[i, j] = (-1, 0, 1)[lsmc_action(model, t, inv, market)]
    return spreads, inventories, grid


def save_policy_plot(x, inventories, grid, xlabel, title, path):
    plt.figure(figsize=(7, 4.5))
    plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), inventories.min(), inventories.max()],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.colorbar(ticks=[-1, 0, 1], label="action (-1 withdraw, 0 hold, +1 inject)")
    plt.xlabel(xlabel)
    plt.ylabel("inventory")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_table(rows, path_csv, path_md):
    cols = ["method", "mean_value", "std_value", "stderr", "mean_final_inventory"]
    with open(path_csv, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")

    with open(path_md, "w", encoding="utf-8") as f:
        f.write("| method | mean_value | std_value | stderr | mean_final_inventory |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['mean_value']:.3f} | {row['std_value']:.3f} | "
                f"{row['stderr']:.3f} | {row['mean_final_inventory']:.3f} |\n"
            )


def main():
    RESULTS.mkdir(exist_ok=True)
    storage_cfg = StorageConfig()
    price_cfg = PriceConfig()

    lsmc_model = train_lsmc(n_paths=9000, storage_cfg=storage_cfg, price_cfg=price_cfg, seed=42)
    lsmc_eval = evaluate_lsmc_policy(lsmc_model, n_paths=2500, seed=100)

    dqn_model = train_dqn(episodes=1200, seed=22)
    dqn_eval = evaluate_policy(dqn_policy_fn(dqn_model), n_episodes=1000, seed=101)
    heuristic_eval = evaluate_policy(heuristic_policy, n_episodes=1000, seed=102)

    comparison = [
        {"method": "LSMC rich features", **{k: v for k, v in lsmc_eval.items() if k != "actions"}},
        {"method": "NumPy DQN rich state", **dqn_eval},
        {"method": "Curve heuristic", **heuristic_eval},
    ]
    write_table(comparison, RESULTS / "comparison_table.csv", RESULTS / "comparison_table.md")

    fwd = deterministic_forward_curve(storage_cfg, price_cfg)
    plt.figure(figsize=(7, 3.8))
    plt.plot(np.arange(len(fwd)), fwd, marker="o")
    plt.xlabel("delivery month ahead")
    plt.ylabel("forward price")
    plt.title("Toy seasonal forward mean")
    plt.tight_layout()
    plt.savefig(RESULTS / "toy_forward_curve.png", dpi=160)
    plt.close()

    sample_market = simulate_market_paths(300, storage_cfg, price_cfg, seed=222)
    plt.figure(figsize=(7, 3.8))
    plt.plot(sample_market["winter_summer_spread"].mean(axis=0), label="winter-summer")
    plt.plot(sample_market["prompt_next_spread"].mean(axis=0), label="prompt-next")
    plt.xlabel("month")
    plt.ylabel("spread")
    plt.title("Average simulated curve spreads")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "simulated_spreads.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 3.8))
    rewards = dqn_model["episode_rewards"]
    kernel = np.ones(40) / 40.0
    rolling = np.convolve(rewards, kernel, mode="valid")
    plt.plot(np.arange(len(rolling)) + 39, rolling, label="40-episode rolling mean")
    plt.xlabel("episode")
    plt.ylabel("discounted reward")
    plt.title("NumPy DQN training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "dqn_training_curve.png", dpi=160)
    plt.close()

    x, inventories, grid = policy_grid_lsmc(lsmc_model, storage_cfg, price_cfg)
    save_policy_plot(x, inventories, grid, "prompt price", "LSMC policy: prompt/inventory slice", RESULTS / "lsmc_policy_prompt_inventory.png")

    x, inventories, grid = policy_grid_dqn(dqn_model, storage_cfg, price_cfg)
    save_policy_plot(x, inventories, grid, "prompt price", "DQN policy: prompt/inventory slice", RESULTS / "dqn_policy_prompt_inventory.png")

    x, inventories, grid = spread_grid_lsmc(lsmc_model, storage_cfg, price_cfg)
    save_policy_plot(x, inventories, grid, "winter-summer spread", "LSMC policy: spread/inventory slice", RESULTS / "lsmc_policy_spread_inventory.png")

    for row in comparison:
        print(
            f"{row['method']:>22}  value={row['mean_value']:8.3f}  "
            f"stderr={row['stderr']:6.3f}  final_inv={row['mean_final_inventory']:6.2f}"
        )
    print(f"LSMC regression initial value estimate: {lsmc_model['initial_value']:.3f}")
    print(f"Results written to {RESULTS}")


if __name__ == "__main__":
    main()
