# TDPAD Streamlit Toy Simulator

This is a small interactive Python/Streamlit app for event-wise TDPAD toy simulations, unbinned likelihood scans, and binned Gaussian chi-squared scans.

## What it does

On launch, the sidebar lets you choose:

1. lifetime `tau` in ns,
2. magnetic field `B` in Tesla,
3. two detector angles in degrees,
4. the analysis time range in ns,
5. the `g` and `A2` grid ranges,
6. whether the true `g` and `A2` are drawn uniformly from those ranges or set manually,
7. the raw event count, grid resolution, and number of slider snapshots.

The app simulates raw exponential-decay events, keeps only events in the chosen time range, and computes cumulative event-wise likelihood snapshots on a `g x A2` grid.

For a selected cumulative event snapshot, the app shows two 2x2 analysis blocks:

### Event-wise likelihood block

- bottom left: event-wise posterior over `(g, A2)`,
- top left: marginalized posterior density in `g`, including 68% and 95% HPD regions,
- bottom right: marginalized posterior density in `A2`, including 68% and 95% HPD regions,
- top right: binned detector asymmetry `(det1 - det2)/(det1 + det2)` with Gaussian error bars, together with a smooth event-wise MAP-prediction curve.

### Binned Gaussian chi-squared block

A bin-count slider sits between the two blocks. It controls the top-right visualization in the event-wise block and the complete binned chi-squared analysis in the second block.

The binned analysis excludes every time bin where either detector has zero counts, computes

```text
chi2(g, A2) = sum_i [(r_i - r_pred_i(g, A2)) / sigma_i]^2
L(g, A2) ∝ exp(-0.5 * chi2(g, A2))
```

on the same `g x A2` grid, and displays:

- bottom left: chi-squared posterior over `(g, A2)`,
- top left: marginalized chi-squared posterior density in `g`, including 68% and 95% HPD regions,
- bottom right: marginalized chi-squared posterior density in `A2`, including 68% and 95% HPD regions,
- top right: binned detector asymmetry with Gaussian error bars, together with the best chi-squared MAP-prediction curve.

A slider selects the cumulative number of accepted events used in both analyses.

## Install and run

```bash
cd tdpad_streamlit_toy
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Physics convention

The app uses

```text
omega = g * (mu_N / hbar) * B
mu_N / hbar = 4.789412e7 rad s^-1 T^-1
```

All times are in ns, and the conversion to seconds is applied inside the Larmor phase.

The angular distribution is

```text
W(theta, t) = 1 + A2 * (0.25 + 0.75 * cos(2 theta - 2 omega t))
```

with unit detector efficiencies for the two hard-coded detectors.
