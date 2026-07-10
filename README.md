# Storage Optionality Lab

A math-first website for the gas storage valuation research project. It documents
the recommended market model, physical constraints, dynamic program, LSMC basis,
RL formulation, extrinsic-value definition, hedge construction, and the planned
valuation and risk-analysis work.

## Run locally

```powershell
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:3000`.

## Build

```powershell
npm.cmd run build
```

The images in `public/figures` are simulated illustrations from the companion
toy model. They are not market calibrations or valuation outputs.

## Regenerate model diagnostics

```powershell
cd C:\Users\dwu30\Documents\Codex\2026-07-10\referenced-chatgpt-conversation-this-is-untrusted\outputs\gas-storage-math-site
python modeling\run_sensitivity_lab.py
```

The modeling code writes the sensitivity charts used by the website into
`public/figures/lab` and a small summary payload into `public/data/model-summary.json`.
