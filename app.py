#!/usr/bin/env python3
"""Interactive Streamlit app for a toy TDPAD event-wise analysis."""

from __future__ import annotations

import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from tdpad_core import (
    DetectorSetup,
    compute_chi2_posterior,
    compute_posterior_snapshots,
    log_spaced_event_counts,
    simulate_events,
)
from tdpad_plotting import figure_as_svg, make_chi2_frame_figure, make_frame_figure


st.set_page_config(page_title="TDPAD toy simulation", layout="wide")


def _format_float(x: float, digits: int = 5) -> str:
    return f"{x:.{digits}g}"


def _validate_analysis_inputs(
    *,
    lifetime_ns: float,
    b_field_t: float,
    phi1_deg: float,
    phi2_deg: float,
    t_min_ns: float,
    t_max_ns: float,
    g_min: float,
    g_max: float,
    a2_min: float,
    a2_max: float,
    true_parameter_mode: str,
    fixed_g: float,
    fixed_a2: float,
    raw_events: int,
    grid_points: int,
    n_frames: int,
) -> None:
    """Validate all user inputs once before simulation and analysis."""
    if not math.isfinite(lifetime_ns) or lifetime_ns <= 0.0:
        raise ValueError("Lifetime must be finite and positive")
    if not math.isfinite(b_field_t):
        raise ValueError("Magnetic field must be finite")
    if not math.isfinite(phi1_deg) or not math.isfinite(phi2_deg):
        raise ValueError("Detector angles must be finite")
    if not math.isfinite(t_min_ns) or not math.isfinite(t_max_ns):
        raise ValueError("Time-window limits must be finite")
    if t_max_ns <= t_min_ns:
        raise ValueError("t_max must be larger than t_min")
    if not math.isfinite(g_min) or not math.isfinite(g_max):
        raise ValueError("g-range limits must be finite")
    if g_max <= g_min:
        raise ValueError("g range must have min < max")
    if not math.isfinite(a2_min) or not math.isfinite(a2_max):
        raise ValueError("A₂-range limits must be finite")
    if a2_max <= a2_min:
        raise ValueError("A₂ range must have min < max")
    if a2_min <= -1.0 or a2_max >= 1.0:
        raise ValueError("A₂ range must satisfy -1 < A₂ < 1")
    if not math.isfinite(fixed_g):
        raise ValueError("Fixed true g must be finite")
    if not math.isfinite(fixed_a2) or abs(fixed_a2) >= 1.0:
        raise ValueError("Fixed true A₂ must satisfy -1 < A₂ < 1")
    if true_parameter_mode not in {"Draw uniformly from ranges", "Set fixed g and A₂"}:
        raise ValueError("Unknown true-parameter mode")
    if raw_events <= 0:
        raise ValueError("Raw event count must be positive")
    if grid_points <= 0:
        raise ValueError("Grid point count must be positive")
    if n_frames <= 0:
        raise ValueError("Snapshot count must be positive")



@st.cache_data(show_spinner=False)
def run_analysis_cached(
    lifetime_ns: float,
    b_field_t: float,
    phi1_deg: float,
    phi2_deg: float,
    t_min_ns: float,
    t_max_ns: float,
    g_min: float,
    g_max: float,
    a2_min: float,
    a2_max: float,
    true_parameter_mode: str,
    fixed_g: float,
    fixed_a2: float,
    seed: int,
    raw_events: int,
    grid_points: int,
    n_frames: int,
):
    _validate_analysis_inputs(
        lifetime_ns=lifetime_ns,
        b_field_t=b_field_t,
        phi1_deg=phi1_deg,
        phi2_deg=phi2_deg,
        t_min_ns=t_min_ns,
        t_max_ns=t_max_ns,
        g_min=g_min,
        g_max=g_max,
        a2_min=a2_min,
        a2_max=a2_max,
        true_parameter_mode=true_parameter_mode,
        fixed_g=fixed_g,
        fixed_a2=fixed_a2,
        raw_events=raw_events,
        grid_points=grid_points,
        n_frames=n_frames,
    )
    setup = DetectorSetup(phi1_deg=phi1_deg, phi2_deg=phi2_deg)

    use_fixed = true_parameter_mode == "Set fixed g and A₂"
    events = simulate_events(
        lifetime_ns=lifetime_ns,
        b_field_t=b_field_t,
        detector_setup=setup,
        t_min_ns=t_min_ns,
        t_max_ns=t_max_ns,
        g_range=(g_min, g_max),
        a2_range=(a2_min, a2_max),
        true_g=fixed_g if use_fixed else None,
        true_a2=fixed_a2 if use_fixed else None,
        raw_events=raw_events,
        seed=seed,
    )

    if events.accepted_events == 0:
        return events, None

    g_grid = np.linspace(g_min, g_max, grid_points)
    a2_grid = np.linspace(a2_min, a2_max, grid_points)
    event_counts = log_spaced_event_counts(events.accepted_events, n_frames)

    posterior = compute_posterior_snapshots(
        events=events,
        b_field_t=b_field_t,
        detector_setup=setup,
        g_grid=g_grid,
        a2_grid=a2_grid,
        event_counts=event_counts,
    )

    return events, posterior
@st.cache_data(show_spinner=False)
def compute_chi2_snapshot_cached(
    times_ns: np.ndarray,
    detectors: np.ndarray,
    b_field_t: float,
    phi1_deg: float,
    phi2_deg: float,
    g_grid: np.ndarray,
    a2_grid: np.ndarray,
    t_min_ns: float,
    t_max_ns: float,
    bins: int,
):
    """Cache the binned chi-squared posterior for a slider snapshot."""
    setup = DetectorSetup(phi1_deg=phi1_deg, phi2_deg=phi2_deg)
    return compute_chi2_posterior(
        times_ns=times_ns,
        detectors=detectors,
        b_field_t=b_field_t,
        detector_setup=setup,
        g_grid=g_grid,
        a2_grid=a2_grid,
        t_min_ns=t_min_ns,
        t_max_ns=t_max_ns,
        bins=bins,
    )
st.title("Interactive TDPAD toy simulation and analysis")
st.markdown(
    "This app simulates Time-Dependent Angular Distribution data for a given level of statistics. "
    "The data are then analyzed in a Bayesian framework using first a binning-*independent* likelihood and then a binning-*dependent* likelihood."
    "For both approaches, a corner plot is shown with the marginalized posteriors for the two parameters of interest, `g` and `A₂`, as well as the detector asymmetry as a function of time."
    "For details see the implementation notes at the bottom of the page."
)

with st.sidebar:
    st.header("Simulation inputs")
    lifetime_ns = st.number_input("Lifetime τ [ns]", min_value=0.001, value=1300.0, step=10.0)
    b_field_t = st.number_input("Magnetic field B [T]", value=0.15, step=0.1, format="%.6g")

    st.subheader("Two detector angles")
    phi1_deg = st.number_input("Detector 0 angle θ₀ [deg]", value=45.0, step=5.0, format="%.6g")
    phi2_deg = st.number_input("Detector 1 angle θ₁ [deg]", value=135.0, step=5.0, format="%.6g")

    st.subheader("Analysis time window")
    t_min_ns = st.number_input("t_min [ns]", value=300.0, step=10.0, format="%.6g")
    t_max_ns = st.number_input("t_max [ns]", value=3000.0, step=10.0, format="%.6g")

    st.subheader("Flat Prior and Draw ranges")
    g_min, g_max = st.slider("g range", -2.0, 2.0, (0.05, 1.05), step=0.01)
    a2_min, a2_max = st.slider("A₂ range", -0.99, 0.99, (0.0, 0.3), step=0.01)

    st.subheader("True parameter choice")
    true_parameter_mode = st.radio(
        "How should true g and A₂ be chosen?",
        ["Draw uniformly from ranges", "Set fixed g and A₂"],
        index=0,
    )
    fixed_g = st.number_input("Fixed true g", value=0.2, step=0.01, format="%.6g")
    fixed_a2 = st.number_input(
        "Fixed true A₂", min_value=-0.99, max_value=0.99, value=0.2, step=0.01, format="%.6g"
    )
    if true_parameter_mode == "Draw uniformly from ranges":
        st.caption("The fixed-value fields are ignored in draw mode.")
    else:
        st.caption("The likelihood grid still uses the ranges above.")

    st.subheader("Computation")
    seed = st.number_input("Random seed", min_value=0, value=12345, step=1)
    raw_events = st.number_input("Raw events to generate", min_value=100, max_value=200_000, value=10_000, step=1_000)
    grid_points = st.number_input("Grid points per axis", min_value=20, max_value=200, value=100, step=10)
    n_frames = st.number_input("Slider snapshots", min_value=10, max_value=200, value=100, step=10)
    show_local_gaussian = st.checkbox(
        "Fit local Gaussian through g MAP",
        value=False,
        help=(
            "Overlay a Gaussian on the binned g marginal that passes through its MAP "
            "and matches the posterior's local curvature."
        ),
    )

    run_button = st.button("Simulate / rescan", type="primary")

if true_parameter_mode == "Set fixed g and A₂":
    if not (g_min <= fixed_g <= g_max):
        st.warning("Fixed true g is outside the likelihood grid range, so the posterior cannot peak at the true g.")
    if not (a2_min <= fixed_a2 <= a2_max):
        st.warning("Fixed true A₂ is outside the likelihood grid range, so the posterior cannot peak at the true A₂.")

params = dict(
    lifetime_ns=lifetime_ns,
    b_field_t=b_field_t,
    phi1_deg=phi1_deg,
    phi2_deg=phi2_deg,
    t_min_ns=t_min_ns,
    t_max_ns=t_max_ns,
    g_min=g_min,
    g_max=g_max,
    a2_min=a2_min,
    a2_max=a2_max,
    true_parameter_mode=true_parameter_mode,
    fixed_g=fixed_g,
    fixed_a2=fixed_a2,
    seed=seed,
    raw_events=raw_events,
    grid_points=grid_points,
    n_frames=n_frames,
)

if run_button or "last_params" not in st.session_state:
    st.session_state.last_params = params
else:
    params = st.session_state.last_params

try:
    with st.spinner("Simulating events and scanning likelihood grid..."):
        start = time.perf_counter()
        events, posterior = run_analysis_cached(**params)
        elapsed = time.perf_counter() - start
except ValueError as exc:
    st.error(str(exc))
    st.stop()

setup = DetectorSetup(phi1_deg=params["phi1_deg"], phi2_deg=params["phi2_deg"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("True g", _format_float(events.true_g, 6))
col2.metric("True A₂", _format_float(events.true_a2, 6))
col3.metric("Accepted events", f"{events.accepted_events:,} / {events.raw_events_generated:,}")
larmor_period_ns = (
    2.0 * np.pi / abs(events.omega_rad_per_s) * 1e9
    if events.omega_rad_per_s != 0.0
    else np.inf
)
col4.metric("Period [ns]", _format_float(larmor_period_ns/2, 6))

st.caption(
    f"True parameters are `{events.true_parameter_mode}`. The displayed period is "
    f"T = π / |ω|. "
    f"Computation time for current cached run: {elapsed:.2f} s."
)

if posterior is None:
    st.warning(
        "No generated events survived the requested time window. Increase t_max, "
        "decrease t_min, increase raw events, or choose a longer lifetime."
    )
    st.stop()

frame_idx = st.slider(
    "Cumulative accepted events included in likelihood",
    min_value=0,
    max_value=len(posterior.event_counts) - 1,
    value=len(posterior.event_counts) - 1,
    format="snapshot %d",
)
selected_count = int(posterior.event_counts[frame_idx])
st.write(
    f"Showing snapshot **{frame_idx + 1} / {len(posterior.event_counts)}**, "
    f"using the first **{selected_count:,} accepted events**."
)

top_figure_slot = st.empty()

st.divider()
bins = st.slider(
    "Number of time bins",
    min_value=4,
    max_value=100,
    value=16,
    step=1,
    help=(
        "Used for the asymmetry visualization in the event-wise plot above, "
        "and for the full binned χ² analysis below. Bins with zero counts in "
        "either detector are excluded from the χ² sum."
    ),
)
st.caption(
    "The bin count slider is placed between the two analysis blocks. It controls "
    "the top-right asymmetry visualization in the event-wise block, and the full "
    "binned χ² likelihood block below."
)

fig = make_frame_figure(
    events=events,
    posterior=posterior,
    frame_idx=frame_idx,
    detector_setup=setup,
    b_field_t=params["b_field_t"],
    t_min_ns=params["t_min_ns"],
    t_max_ns=params["t_max_ns"],
    bins=bins,
)
fig_svg = figure_as_svg(fig)
top_figure_slot.pyplot(fig, clear_figure=True)
plt.close(fig)
st.download_button(
    "Download binning-independent figure as SVG",
    data=fig_svg,
    file_name=f"event_wise_snapshot_{frame_idx + 1}.svg",
    mime="image/svg+xml",
)

st.subheader("Binned Gaussian χ² analysis")
with st.spinner("Computing binned χ² posterior for the selected snapshot..."):
    chi2_result = compute_chi2_snapshot_cached(
        events.times_ns[:selected_count],
        events.detectors[:selected_count],
        params["b_field_t"],
        params["phi1_deg"],
        params["phi2_deg"],
        posterior.g_grid,
        posterior.a2_grid,
        params["t_min_ns"],
        params["t_max_ns"],
        bins,
    )

valid_chi2_bins = int(np.sum(chi2_result.valid_bins))
if valid_chi2_bins == 0:
    st.warning(
        "No bins have positive counts in both detectors for this snapshot/binning. "
        "The χ² posterior is therefore uninformative. Use more events or fewer bins."
    )
elif valid_chi2_bins < 3:
    st.info(
        f"Only {valid_chi2_bins} bins have positive counts in both detectors. "
        "The χ² result may be very weak; use more events or fewer bins for a stabler scan."
    )

fig_chi2 = make_chi2_frame_figure(
    events=events,
    chi2_result=chi2_result,
    n_events=selected_count,
    detector_setup=setup,
    b_field_t=params["b_field_t"],
    t_min_ns=params["t_min_ns"],
    t_max_ns=params["t_max_ns"],
    bins=bins,
    show_local_gaussian=show_local_gaussian,
)
fig_chi2_svg = figure_as_svg(fig_chi2)
st.pyplot(fig_chi2, clear_figure=True)
plt.close(fig_chi2)
st.download_button(
    "Download binning-dependent figure as SVG",
    data=fig_chi2_svg,
    file_name=f"binned_chi2_snapshot_{frame_idx + 1}.svg",
    mime="image/svg+xml",
)

with st.expander("Implementation notes"):
    notes_path = Path(__file__).with_name("implementation_notes.md")
    st.markdown(notes_path.read_text(encoding="utf-8"))
