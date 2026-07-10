# Storage Optionality Modeling Lab

This folder contains the Python model used to generate the valuation and policy figures on the website.

It is intentionally CPU-friendly and simulated. The goal is to make the modeling assumptions visible:

- monthly prompt, prompt-next, winter-summer, volatility-proxy, and local-basis states;
- normalized inventory and inventory-dependent injection/withdrawal ratchets;
- LSMC continuation regression with month/season dummies and inventory-spread interactions;
- a dependency-light NumPy DQN baseline inherited from the toy package;
- sensitivity tests for volatility, spread mean reversion, basis risk, terminal penalties, ratchets, and initial inventory.

## Setup

```powershell
cd C:\Users\dwu30\Documents\Codex\2026-07-10\referenced-chatgpt-conversation-this-is-untrusted\outputs\gas-storage-math-site\modeling
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Generate Website Figures

```powershell
python run_sensitivity_lab.py
```

The script writes:

- `../public/figures/lab/model-process.png`
- `../public/figures/lab/state-path-fan.png`
- `../public/figures/lab/value-sensitivity-bars.png`
- `../public/figures/lab/spread-vol-mean-reversion-surface.png`
- `../public/figures/lab/initial-inventory-sensitivity.png`
- `../public/figures/lab/policy-spread-maps.png`
- `../public/figures/lab/hedge-delta-ladder.png`
- `../public/data/model-summary.json`

## Important Boundary

These outputs are research diagnostics from simulated data. They are not a calibrated market valuation. Production use would require market curve ingestion, factor calibration, option-implied volatility, daily nomination rules, contract ratchet tables, hedge instruments, transaction costs, and out-of-sample backtesting.
