from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gas_storage_toy.config import PriceConfig, StorageConfig
from gas_storage_toy.env import feasible_quantity
from gas_storage_toy.lsmc import (
    basis,
    evaluate_lsmc_policy,
    immediate_cashflow,
    lsmc_action,
    terminal_value,
    train_lsmc,
)
from gas_storage_toy.price import deterministic_forward_curve, simulate_market_paths


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lab"
DATA = ROOT / "public" / "data"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)


def eval_lsmc_case(
    name: str,
    storage_cfg: StorageConfig,
    price_cfg: PriceConfig,
    train_paths: int = 1400,
    eval_paths: int = 360,
    seed: int = 500,
) -> dict:
    model = train_lsmc(
        n_paths=train_paths,
        storage_cfg=storage_cfg,
        price_cfg=price_cfg,
        seed=seed,
    )
    result = evaluate_lsmc_policy(model, n_paths=eval_paths, seed=seed + 17)
    return {
        "name": name,
        "value": result["mean_value"],
        "stderr": result["stderr"],
        "std": result["std_value"],
        "final_inventory": result["mean_final_inventory"],
        "model": model,
    }


def model_value(model: dict, market: dict[str, np.ndarray], t: int, inventory: float) -> float:
    cfg = model["storage_cfg"]
    pcfg = model["price_cfg"]
    x = basis(market, np.array([inventory]), t, cfg, pcfg)
    return float((x @ model["models"][t])[0])


def recompute_spreads(market: dict[str, np.ndarray], t: int, price_cfg: PriceConfig) -> None:
    curve = market["curve"][0, t, :]
    delivery_months = (t + np.arange(price_cfg.curve_tenors)) % 12
    winter_mask = np.isin(delivery_months, list(price_cfg.winter_months))
    summer_mask = np.isin(delivery_months, list(price_cfg.summer_months))
    market["prompt"][0, t] = curve[0]
    market["next_month"][0, t] = curve[1]
    market["winter_avg"][0, t] = curve[winter_mask].mean()
    market["summer_avg"][0, t] = curve[summer_mask].mean()
    market["prompt_next_spread"][0, t] = curve[0] - curve[1]
    market["winter_summer_spread"][0, t] = market["winter_avg"][0, t] - market["summer_avg"][0, t]
    market["local_prompt"][0, t] = max(0.01, market["prompt"][0, t] + market["basis"][0, t])


def set_spread_state(market: dict[str, np.ndarray], t: int, prompt_next: float, winter_summer: float) -> None:
    prompt = float(market["prompt"][0, t])
    market["next_month"][0, t] = prompt - prompt_next
    market["prompt_next_spread"][0, t] = prompt_next
    mid = 0.5 * (market["winter_avg"][0, t] + market["summer_avg"][0, t])
    market["winter_avg"][0, t] = mid + 0.5 * winter_summer
    market["summer_avg"][0, t] = mid - 0.5 * winter_summer
    market["winter_summer_spread"][0, t] = winter_summer


def plot_model_process(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.axis("off")
    steps = [
        ("Market state", "prompt, spreads,\nvol, basis, month"),
        ("Facility state", "inventory, ratchets,\navailability, costs"),
        ("Feasible actions", "inject, hold,\nwithdraw"),
        ("Bellman / LSMC", "cashflow + fitted\ncontinuation value"),
        ("Policy value", "out-of-sample\npolicy simulation"),
        ("Risk views", "deltas, hedge P&L,\nsensitivity surfaces"),
    ]
    xs = np.linspace(0.06, 0.94, len(steps))
    for i, ((title, body), x) in enumerate(zip(steps, xs)):
        rect = plt.Rectangle((x - 0.075, 0.42), 0.15, 0.24, facecolor="#f4f7f5", edgecolor="#9fb0aa", lw=1.2)
        ax.add_patch(rect)
        ax.text(x, 0.59, title, ha="center", va="center", fontsize=10, weight="bold", color="#17211e")
        ax.text(x, 0.49, body, ha="center", va="center", fontsize=8.5, color="#58645f")
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.085, 0.54), xytext=(x + 0.085, 0.54), arrowprops=dict(arrowstyle="->", color="#275d77", lw=1.5))
    ax.text(0.5, 0.2, "One environment, common random numbers, frozen policy evaluation", ha="center", fontsize=11, color="#116149")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_value_bars(cases: list[dict], path: Path) -> None:
    labels = [c["name"] for c in cases]
    values = np.array([c["value"] for c in cases])
    stderr = np.array([c["stderr"] for c in cases])
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    colors = ["#116149" if i == 0 else "#275d77" if values[i] >= values[0] else "#a2472f" for i in range(len(cases))]
    ax.barh(np.arange(len(cases)), values, xerr=1.96 * stderr, color=colors, alpha=0.9)
    ax.axvline(values[0], color="#17211e", lw=1.1, ls="--", label="base case")
    ax.set_yticks(np.arange(len(cases)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("policy value, simulated $/storage unit")
    ax.set_title("LSMC value sensitivity by modeling assumption")
    ax.grid(axis="x", color="#dbe1de", lw=0.8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_surface(surface: np.ndarray, spread_vols: np.ndarray, kappas: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    im = ax.imshow(
        surface,
        origin="lower",
        aspect="auto",
        extent=[spread_vols.min(), spread_vols.max(), kappas.min(), kappas.max()],
        cmap="viridis",
    )
    ax.set_xlabel("seasonal spread volatility")
    ax.set_ylabel("spread mean reversion")
    ax.set_title("Extrinsic value surface: spread volatility vs mean reversion")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("LSMC policy value")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_inventory_sensitivity(rows: list[dict], path: Path) -> None:
    inv = np.array([r["inventory"] for r in rows])
    val = np.array([r["value"] for r in rows])
    err = np.array([r["stderr"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(inv, val, color="#116149", marker="o", lw=2)
    ax.fill_between(inv, val - 1.96 * err, val + 1.96 * err, color="#116149", alpha=0.15, linewidth=0)
    ax.set_xlabel("initial inventory")
    ax.set_ylabel("policy value")
    ax.set_title("Value depends on starting inventory and reachable flexibility")
    ax.grid(color="#dbe1de", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def lsmc_action_value(model: dict, t: int, inventory: float, market: dict[str, np.ndarray], direction: int) -> float:
    cfg = model["storage_cfg"]
    pcfg = model["price_cfg"]
    disc = np.exp(-cfg.discount_rate * cfg.dt)
    qty = float(feasible_quantity(direction, inventory, cfg))
    next_inv = inventory + qty
    cf = immediate_cashflow(direction, np.array([qty]), market["local_prompt"][:, t], cfg)[0]
    if t + 1 < cfg.n_steps:
        cont = basis(market, np.array([next_inv]), t + 1, cfg, pcfg) @ model["models"][t + 1]
        cont_value = float(cont[0])
    else:
        cont_value = terminal_value(np.array([next_inv]), market["local_prompt"][:, t + 1], cfg)[0]
    return float(cf + disc * cont_value)


def plot_policy_spread_maps(model: dict, storage_cfg: StorageConfig, price_cfg: PriceConfig, path: Path) -> None:
    t = 6
    base_market = simulate_market_paths(1, storage_cfg, price_cfg, seed=722)
    pnext = np.linspace(-0.45, 0.45, 41)
    wsum = np.linspace(-0.20, 1.00, 41)
    inventories = [20.0, 50.0, 80.0]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, inv in zip(axes, inventories):
        grid = np.zeros((len(wsum), len(pnext)))
        center_market = {k: v.copy() for k, v in base_market.items()}
        set_spread_state(center_market, t, 0.0, 0.35)
        center_value = model_value(model, center_market, t, inv)
        for i, ws in enumerate(wsum):
            for j, pn in enumerate(pnext):
                market = {k: v.copy() for k, v in base_market.items()}
                set_spread_state(market, t, float(pn), float(ws))
                grid[i, j] = model_value(model, market, t, inv) - center_value
        vmax = max(1.0, float(np.nanpercentile(np.abs(grid), 95)))
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[pnext.min(), pnext.max(), wsum.min(), wsum.max()],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.contour(pnext, wsum, grid, levels=[0.0], colors="#17211e", linewidths=1.1)
        ax.set_title(f"inventory = {inv:.0f}")
        ax.set_xlabel("prompt - next")
        ax.grid(color="white", alpha=0.18)
    axes[0].set_ylabel("winter - summer")
    cbar = fig.colorbar(im, ax=axes, shrink=0.86, pad=0.04)
    cbar.set_label("fitted continuation value change")
    fig.suptitle("LSMC continuation value responds to inventory and curve spreads", y=1.02)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_ladder(model: dict, path: Path) -> None:
    cfg = model["storage_cfg"]
    pcfg = model["price_cfg"]
    t = 3
    inventory = cfg.initial_inventory
    market = simulate_market_paths(1, cfg, pcfg, seed=1331)
    base_value = model_value(model, market, t, inventory)
    deltas = []
    eps = 0.01
    for tenor in range(pcfg.curve_tenors):
        bumped = {k: v.copy() for k, v in market.items()}
        bumped["curve"][0, t, tenor] += eps
        recompute_spreads(bumped, t, pcfg)
        deltas.append((model_value(model, bumped, t, inventory) - base_value) / eps)
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.bar(np.arange(pcfg.curve_tenors), deltas, color="#275d77")
    ax.axhline(0.0, color="#17211e", lw=1)
    ax.set_xlabel("delivery month ahead")
    ax.set_ylabel("finite-difference value delta")
    ax.set_title("Illustrative hedge ladder from the fitted LSMC value map")
    ax.grid(axis="y", color="#dbe1de", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sample_paths(path: Path) -> None:
    cfg = StorageConfig()
    pcfg = PriceConfig()
    market = simulate_market_paths(120, cfg, pcfg, seed=900)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.4), sharex=True)
    items = [
        ("prompt", "prompt price"),
        ("prompt_next_spread", "prompt-next spread"),
        ("winter_summer_spread", "winter-summer spread"),
        ("basis", "local basis"),
    ]
    x = np.arange(cfg.n_steps + 1)
    for ax, (key, title) in zip(axes.ravel(), items):
        arr = market[key]
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9], axis=0)
        ax.plot(x, q50, color="#116149", lw=2)
        ax.fill_between(x, q10, q90, color="#116149", alpha=0.16, linewidth=0)
        ax.set_title(title)
        ax.grid(color="#dbe1de", lw=0.8)
    fig.suptitle("Simulated state variables used by the policy", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(base_case: dict, cases: list[dict], surface: np.ndarray, inv_rows: list[dict]) -> None:
    best = max(cases[1:], key=lambda x: x["value"])
    worst = min(cases[1:], key=lambda x: x["value"])
    payload = {
        "base_value": round(base_case["value"], 3),
        "base_stderr": round(base_case["stderr"], 3),
        "best_scenario": best["name"],
        "best_scenario_value": round(best["value"], 3),
        "worst_scenario": worst["name"],
        "worst_scenario_value": round(worst["value"], 3),
        "surface_min": round(float(np.min(surface)), 3),
        "surface_max": round(float(np.max(surface)), 3),
        "best_initial_inventory": max(inv_rows, key=lambda x: x["value"])["inventory"],
        "figures": [
            "/figures/lab/model-process.png",
            "/figures/lab/state-path-fan.png",
            "/figures/lab/value-sensitivity-bars.png",
            "/figures/lab/spread-vol-mean-reversion-surface.png",
            "/figures/lab/initial-inventory-sensitivity.png",
            "/figures/lab/policy-spread-maps.png",
            "/figures/lab/hedge-delta-ladder.png",
        ],
        "note": "Toy simulated values; not calibrated market valuation.",
    }
    (DATA / "model-summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    storage = StorageConfig()
    price = PriceConfig()

    plot_model_process(OUT / "model-process.png")
    plot_sample_paths(OUT / "state-path-fan.png")

    cases = [
        eval_lsmc_case("Base case", storage, price, seed=510),
        eval_lsmc_case("Higher prompt vol", storage, replace(price, sigma_prompt=price.sigma_prompt * 1.35), seed=511),
        eval_lsmc_case("Higher spread vol", storage, replace(price, sigma_spread=price.sigma_spread * 1.45), seed=512),
        eval_lsmc_case("Faster spread mean reversion", storage, replace(price, kappa_spread=price.kappa_spread * 1.70), seed=513),
        eval_lsmc_case("Higher basis risk", storage, replace(price, sigma_basis=price.sigma_basis * 2.00), seed=514),
        eval_lsmc_case("Stronger ratchets", replace(storage, max_inject=15.0, max_withdraw=18.0, ratchet_floor=0.50), price, seed=515),
        eval_lsmc_case("Tighter terminal target", replace(storage, terminal_penalty=2.50), price, seed=516),
    ]
    plot_value_bars(cases, OUT / "value-sensitivity-bars.png")

    spread_vols = np.array([0.18, 0.32, 0.52])
    kappas = np.array([0.45, 1.05, 1.80])
    surface = np.zeros((len(kappas), len(spread_vols)))
    for i, kappa in enumerate(kappas):
        for j, spread_vol in enumerate(spread_vols):
            case = eval_lsmc_case(
                "surface",
                storage,
                replace(price, sigma_spread=float(spread_vol), kappa_spread=float(kappa)),
                train_paths=900,
                eval_paths=240,
                seed=620 + i * 10 + j,
            )
            surface[i, j] = case["value"]
    plot_surface(surface, spread_vols, kappas, OUT / "spread-vol-mean-reversion-surface.png")

    inv_rows = []
    for inv in [5.0, 25.0, 50.0, 75.0, 95.0]:
        case = eval_lsmc_case(
            f"initial inventory {inv:.0f}",
            replace(storage, initial_inventory=inv),
            price,
            train_paths=950,
            eval_paths=260,
            seed=710 + int(inv),
        )
        inv_rows.append({"inventory": inv, "value": case["value"], "stderr": case["stderr"]})
    plot_inventory_sensitivity(inv_rows, OUT / "initial-inventory-sensitivity.png")

    base_model = cases[0]["model"]
    plot_policy_spread_maps(base_model, storage, price, OUT / "policy-spread-maps.png")
    plot_delta_ladder(base_model, OUT / "hedge-delta-ladder.png")
    write_summary(cases[0], cases, surface, inv_rows)

    fwd = deterministic_forward_curve(storage, price)
    np.savetxt(DATA / "deterministic-forward-curve.csv", fwd, delimiter=",")
    print(f"Wrote figures to {OUT}")
    print(f"Base value: {cases[0]['value']:.3f} +/- {1.96 * cases[0]['stderr']:.3f}")


if __name__ == "__main__":
    main()
