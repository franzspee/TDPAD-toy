#!/usr/bin/env python3
"""Interactive Streamlit app for a toy TDPAD event-wise analysis."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from tdpad_core import (
    DetectorSetup,
    MU_N_OVER_HBAR_RAD_PER_S_T,
    binned_asymmetry,
    compute_posterior_snapshots,
    log_spaced_event_counts,
    map_asymmetry_prediction,
    simulate_events,
)


st.set_page_config(page_title="TDPAD toy simulation", layout="wide")


HPD_COLOR = "lightgreen"



def _format_float(x: float, digits: int = 5) -> str:
    return f"{x:.{digits}g}"



def _grid_spacing(grid: np.ndarray) -> float:
    grid = np.asarray(grid, dtype=float)
    if len(grid) < 2:
        return 1.0
    return float(np.mean(np.diff(grid)))



def _density_from_grid_probability(grid: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Convert a discrete grid probability vector into an approximate density."""
    density = np.asarray(probability, dtype=float).copy()
    dx = _grid_spacing(grid)
    norm = float(np.sum(density) * dx)
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros_like(density)
    return density / norm



def _credible_density_level_1d(
    density: np.ndarray,
    dx: float,
    credible_mass: float,
) -> float:
    """Density threshold for a 1D highest-posterior-density region."""
    density = np.asarray(density, dtype=float)
    if not 0.0 < credible_mass < 1.0:
        raise ValueError("credible_mass must be between 0 and 1")
    finite = np.isfinite(density)
    if not np.any(finite):
        return np.inf

    sorted_density = np.sort(density[finite])[::-1]
    cumulative_mass = np.cumsum(sorted_density * dx)
    idx = int(np.searchsorted(cumulative_mass, credible_mass, side="left"))
    idx = min(idx, len(sorted_density) - 1)
    return float(sorted_density[idx])



def _fill_masked_regions_under_curve(
    ax: plt.Axes,
    *,
    grid: np.ndarray,
    density: np.ndarray,
    mask: np.ndarray,
    color: str,
    alpha: float,
    label: str,
) -> None:
    """Fill contiguous masked regions under a 1D density curve."""
    grid = np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if len(grid) == 0 or not np.any(mask):
        return

    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])

    first = True
    for start, stop in zip(starts, stops):
        region = slice(start, stop)
        ax.fill_between(
            grid[region],
            0.0,
            density[region],
            color=color,
            alpha=alpha,
            label=label if first else None,
        )
        first = False



def _plot_1d_marginal_with_hpd(
    ax: plt.Axes,
    *,
    grid: np.ndarray,
    probability: np.ndarray,
    name: str,
    true_value: float,
    map_value: float,
) -> None:
    """Plot a marginalized posterior density with 68% and 95% HPD shading."""
    grid = np.asarray(grid, dtype=float)
    density = _density_from_grid_probability(grid, probability)
    dx = _grid_spacing(grid)

    if np.sum(density) <= 0.0 or not np.any(np.isfinite(density)):
        ax.text(0.5, 0.5, "invalid marginal", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"Marginal posterior in {name}")
        return

    level68 = _credible_density_level_1d(density, dx, credible_mass=0.68)
    level95 = _credible_density_level_1d(density, dx, credible_mass=0.95)
    mask68 = density >= level68
    mask95 = density >= level95

    _fill_masked_regions_under_curve(
        ax,
        grid=grid,
        density=density,
        mask=mask95,
        color=HPD_COLOR,
        alpha=0.35,
        label="95% HPD region",
    )
    _fill_masked_regions_under_curve(
        ax,
        grid=grid,
        density=density,
        mask=mask68,
        color=HPD_COLOR,
        alpha=0.75,
        label="68% HPD region",
    )

    ax.plot(grid, density, label=f"p({name})")
    map_idx = int(np.argmin(np.abs(grid - map_value)))
    ax.plot(
        map_value,
        density[map_idx],
        "rx",
        markersize=8,
        markeredgewidth=1.5,
        label="MAP",
    )
    ax.axvline(true_value, linestyle="--", linewidth=1.5, label=f"true {name}")
    ax.axvline(map_value, linestyle=":", linewidth=1.5, label=f"MAP {name}")
    ax.set_title(f"Marginal posterior in {name}")
    ax.set_xlabel(name)
    ax.set_ylabel("posterior density")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", fontsize=8)



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
    clip_negative_weights: bool,
):
    setup = DetectorSetup(phi1_deg=phi1_deg, phi2_deg=phi2_deg, eff1=1.0, eff2=1.0)

    use_fixed = true_parameter_mode == "Set fixed g and A₂"
    events = simulate_events(
        lifetime_ns=lifetime_ns,
        b_field_t=b_field_t,
        detector_setup=setup,
        t_min_ns=t_min_ns,
        t_max_ns=t_max_ns,
        g_range=(g_min, g_max),
        a2_range=(a2_min, a2_max),
        true_g=float(fixed_g) if use_fixed else None,
        true_a2=float(fixed_a2) if use_fixed else None,
        raw_events=raw_events,
        seed=seed,
        clip_negative_weights=clip_negative_weights,
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
        clip_negative_weights=clip_negative_weights,
    )

    return events, posterior



def make_frame_figure(
    *,
    events,
    posterior,
    frame_idx: int,
    detector_setup: DetectorSetup,
    b_field_t: float,
    t_min_ns: float,
    t_max_ns: float,
    clip_negative_weights: bool,
):
    n_events = int(posterior.event_counts[frame_idx])
    post = posterior.posteriors[frame_idx]
    g_grid = posterior.g_grid
    a2_grid = posterior.a2_grid
    g_marginal = post.sum(axis=1)
    a2_marginal = post.sum(axis=0)
    map_g = float(posterior.map_g[frame_idx])
    map_a2 = float(posterior.map_a2[frame_idx])

    times = events.times_ns[:n_events]
    dets = events.detectors[:n_events]
    centers, asym, asym_err, det0_counts, det1_counts, edges = binned_asymmetry(
        times, dets, t_min_ns=t_min_ns, t_max_ns=t_max_ns, bins=16
    )
    nonempty = np.isfinite(asym) & np.isfinite(asym_err)

    curve_times = np.linspace(t_min_ns, t_max_ns, 600)
    prediction_curve = map_asymmetry_prediction(
        curve_times,
        g=map_g,
        a2=map_a2,
        b_field_t=b_field_t,
        detector_setup=detector_setup,
        clip_negative_weights=clip_negative_weights,
    )

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.4], width_ratios=[1.15, 1.0])
    ax_g = fig.add_subplot(gs[0, 0])
    ax_asym = fig.add_subplot(gs[0, 1])
    ax_post = fig.add_subplot(gs[1, 0])
    ax_a2 = fig.add_subplot(gs[1, 1])

    _plot_1d_marginal_with_hpd(
        ax_g,
        grid=g_grid,
        probability=g_marginal,
        name="g",
        true_value=events.true_g,
        map_value=map_g,
    )

    ax_asym.axhline(0.0, linewidth=0.8)
    ax_asym.errorbar(
        centers[nonempty],
        asym[nonempty],
        yerr=asym_err[nonempty],
        marker="o",
        linestyle="",
        capsize=3,
        label="binned data ± Gaussian error",
    )
    ax_asym.plot(curve_times, prediction_curve, linewidth=1.8, label="MAP prediction")
    ax_asym.set_ylim(-1.05, 1.05)
    ax_asym.set_title("16-bin detector asymmetry")
    ax_asym.set_xlabel("time [ns]")
    ax_asym.set_ylabel("(det1 - det2) / (det1 + det2)")
    ax_asym.legend(loc="best", fontsize=8)

    im = ax_post.imshow(
        post.T,
        origin="lower",
        aspect="auto",
        extent=[g_grid[0], g_grid[-1], a2_grid[0], a2_grid[-1]],
    )
    ax_post.plot(events.true_g, events.true_a2, marker="x", markersize=9, label="true")
    ax_post.plot(map_g, map_a2, marker="+", markersize=10, label="MAP")
    ax_post.set_title(f"Posterior after {n_events} accepted events")
    ax_post.set_xlabel("g")
    ax_post.set_ylabel("A₂")
    ax_post.legend(loc="best", fontsize=8)
    fig.colorbar(im, ax=ax_post, label="posterior probability")

    _plot_1d_marginal_with_hpd(
        ax_a2,
        grid=a2_grid,
        probability=a2_marginal,
        name="A₂",
        true_value=events.true_a2,
        map_value=map_a2,
    )

    fig.suptitle(
        "TDPAD toy likelihood scan: "
        f"MAP g={map_g:.5g}, MAP A₂={map_a2:.5g}",
        fontsize=14,
    )
    return fig



st.title("Interactive TDPAD toy simulation and event-wise likelihood scan")
st.markdown(
    "This app either draws one random `(g, A₂)` pair from flat input ranges or uses "
    "fixed values, simulates 10,000 raw exponential-decay events, keeps only events "
    "in the analysis time window, and updates a flat-prior posterior over a `g × A₂` grid."
)

with st.sidebar:
    st.header("Simulation inputs")
    lifetime_ns = st.number_input("Lifetime τ [ns]", min_value=0.001, value=100.0, step=10.0)
    b_field_t = st.number_input("Magnetic field B [T]", value=0.5, step=0.1, format="%.6g")

    st.subheader("Two detector angles")
    phi1_deg = st.number_input("Detector 1 angle θ₁ [deg]", value=0.0, step=5.0, format="%.6g")
    phi2_deg = st.number_input("Detector 2 angle θ₂ [deg]", value=90.0, step=5.0, format="%.6g")

    st.subheader("Analysis time window")
    t_min_ns = st.number_input("t_min [ns]", value=0.0, step=10.0, format="%.6g")
    t_max_ns = st.number_input("t_max [ns]", value=500.0, step=10.0, format="%.6g")

    st.subheader("Analysis grid / draw ranges")
    g_min, g_max = st.slider("g range", -2.0, 2.0, (-0.5, 0.5), step=0.01)
    a2_min, a2_max = st.slider("A₂ range", -2.0, 2.0, (-0.4, 0.4), step=0.01)

    st.subheader("True parameter choice")
    true_parameter_mode = st.radio(
        "How should true g and A₂ be chosen?",
        ["Draw uniformly from ranges", "Set fixed g and A₂"],
        index=0,
    )
    fixed_g = st.number_input("Fixed true g", value=0.2, step=0.01, format="%.6g")
    fixed_a2 = st.number_input("Fixed true A₂", value=0.2, step=0.01, format="%.6g")
    if true_parameter_mode == "Draw uniformly from ranges":
        st.caption("The fixed-value fields are ignored in draw mode.")
    else:
        st.caption("The likelihood grid still uses the ranges above.")

    st.subheader("Computation")
    seed = st.number_input("Random seed", min_value=0, value=12345, step=1)
    raw_events = st.number_input("Raw events to generate", min_value=100, max_value=200_000, value=10_000, step=1_000)
    grid_points = st.number_input("Grid points per axis", min_value=20, max_value=200, value=100, step=10)
    n_frames = st.number_input("Slider snapshots", min_value=10, max_value=200, value=100, step=10)
    clip_negative_weights = st.checkbox("Clip negative W weights", value=True)

    run_button = st.button("Simulate / rescan", type="primary")

if t_max_ns <= t_min_ns:
    st.error("t_max must be larger than t_min.")
    st.stop()
if g_max <= g_min:
    st.error("g range must have min < max.")
    st.stop()
if a2_max <= a2_min:
    st.error("A₂ range must have min < max.")
    st.stop()

if true_parameter_mode == "Set fixed g and A₂":
    if not (g_min <= fixed_g <= g_max):
        st.warning("Fixed true g is outside the likelihood grid range, so the posterior cannot peak at the true g.")
    if not (a2_min <= fixed_a2 <= a2_max):
        st.warning("Fixed true A₂ is outside the likelihood grid range, so the posterior cannot peak at the true A₂.")

params = dict(
    lifetime_ns=float(lifetime_ns),
    b_field_t=float(b_field_t),
    phi1_deg=float(phi1_deg),
    phi2_deg=float(phi2_deg),
    t_min_ns=float(t_min_ns),
    t_max_ns=float(t_max_ns),
    g_min=float(g_min),
    g_max=float(g_max),
    a2_min=float(a2_min),
    a2_max=float(a2_max),
    true_parameter_mode=str(true_parameter_mode),
    fixed_g=float(fixed_g),
    fixed_a2=float(fixed_a2),
    seed=int(seed),
    raw_events=int(raw_events),
    grid_points=int(grid_points),
    n_frames=int(n_frames),
    clip_negative_weights=bool(clip_negative_weights),
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

setup = DetectorSetup(phi1_deg=params["phi1_deg"], phi2_deg=params["phi2_deg"], eff1=1.0, eff2=1.0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("True g", _format_float(events.true_g, 6))
col2.metric("True A₂", _format_float(events.true_a2, 6))
col3.metric("Accepted events", f"{events.accepted_events:,} / {events.raw_events_generated:,}")
col4.metric("ω [rad/s]", _format_float(events.omega_rad_per_s, 6))

st.caption(
    f"True parameters are `{events.true_parameter_mode}`. Likelihood convention: "
    f"ω = g μ_N B / ℏ, with μ_N/ℏ = "
    f"{MU_N_OVER_HBAR_RAD_PER_S_T:.6g} rad s⁻¹ T⁻¹. "
    f"Computation time for current cached run: {elapsed:.2f} s."
)

if posterior is None:
    st.warning(
        "No raw exponential events survived the requested time window. Increase t_max, "
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

fig = make_frame_figure(
    events=events,
    posterior=posterior,
    frame_idx=frame_idx,
    detector_setup=setup,
    b_field_t=params["b_field_t"],
    t_min_ns=params["t_min_ns"],
    t_max_ns=params["t_max_ns"],
    clip_negative_weights=params["clip_negative_weights"],
)
st.pyplot(fig, clear_figure=True)
plt.close(fig)

with st.expander("Implementation notes"):
    st.markdown(
        "- Event times are drawn from an untruncated exponential with lifetime τ; "
        "only events inside `[t_min, t_max]` are retained, so the accepted sample can "
        "contain fewer than the raw 10,000 generated events.\n"
        "- Detector efficiencies are hard-coded to 1 for both detectors.\n"
        "- True `g` and `A₂` can either be drawn uniformly from the displayed ranges or fixed manually.\n"
        "- The prior over the displayed `g × A₂` grid is flat.\n"
        "- The posterior snapshots are cumulative: each likelihood product contains all "
        "accepted events up to the selected snapshot.\n"
        "- The 1D marginalized posteriors are plotted as densities, start at zero, and mark "
        "68% / 95% highest-posterior-density regions.\n"
        "- The top-right points use Gaussian error propagation for "
        "`(det1 - det2) / (det1 + det2)`, and the MAP prediction is drawn as a smooth curve."
    )
