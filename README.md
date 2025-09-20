# gpcplot — GPC overlay plotter (v0.6.0)

High-signal, reproducible overlay plots for GPC/SEC data.  
Solid defaults, sensible axes, and a baked-in style for paper-ready figures.

> **This README describes v0.6.0.** For upcoming changes, see `CHANGELOG.md` and `TODO.md`.

---

## Features

- **One source of truth** for colors/linestyles; predictable **solid-first** defaults.
- **Okabe–Ito palette** by default; `tab10`, `Set2`, `Dark2`, `viridis`, `colorblind`, `mono` available.
- **Baked-in `.mplstyle`** so CLI and Jupyter look the same.
- **Clean log-x axis**: major ticks at powers of 10 (custom formatter), minor ticks 2–9.
- **Baselines**: `min`, `poly` (on logM), or **ASLS** (Eilers & Boelens).
- **Smoothing** in log10(M) or index domain (Gaussian; σ=0 is identity).
- **Normalization** by ∫W(log10M) (default) or linear M.
- **Moments** (Mn, Mw, Mz), **PDI**, **mode Mp**, and optional DP axis (`M/M0`).
- Robust input parser (tolerates commas/semicolons, duplicates, non-numeric lines).
- **PNG/PDF/SVG** export, with optional **transparent** background or forced white.

---

## Installation

### Option A: project-local virtual environment (recommended)

```powershell
# From the project root
python -m venv .venv
# Activate
#   Windows (PowerShell):
. .\.venv\Scripts\Activate.ps1
#   macOS/Linux:
# source .venv/bin/activate

python -m pip install -U pip
pip install -e .            # editable install
pip install pytest ipykernel

# Make a Jupyter kernel for this env (so notebooks use the same deps)
python -m ipykernel install --user --name gpcplot-venv --display-name "Python (gpcplot)"
