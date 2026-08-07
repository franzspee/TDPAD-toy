"""Matplotlib figure construction for the Streamlit TDPAD toy app."""

from __future__ import annotations

import io
import math

import matplotlib.pyplot as plt
import numpy as np

from tdpad_core import DetectorSetup, binned_asymmetry, map_asymmetry_prediction


HPD_COLOR = "lightgreen"


def figure_as_svg(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(buffer, format="svg", bbox_inches="tight")
    return buffer.getvalue()


def _grid_spacing(grid: np.ndarray) -> float:
    return float(np.mean(np.diff(grid)))


def _density_from_grid_probability(grid: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Convert a discrete grid probability vector into an approximate density."""
    dx = _grid_spacing(grid)
    return np.asarray(probability, dtype=float) / dx


def _local_curvature_gaussian_for_marginal(
    grid: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float, float, bool]:
    """Return a Gaussian passing through the marginal MAP with matching curvature."""
    density = _density_from_grid_probability(grid, probability)

    map_idx = int(np.argmax(density))
    mu = float(grid[map_idx])
    peak = float(density[map_idx])
    if map_idx == 0 or map_idx == len(grid) - 1:
        return mu, float("nan"), peak, False

    local_grid = grid[map_idx - 1 : map_idx + 2]
    local_density = density[map_idx - 1 : map_idx + 2]
    quadratic, _linear, _constant = np.polyfit(local_grid, local_density, deg=2)

    second_derivative = float(2.0 * quadratic)
    if second_derivative >= 0.0:
        return mu, float("nan"), peak, False

    sigma = math.sqrt(-peak / second_derivative)
    return mu, sigma, peak, True


def _gaussian_density_curve(
    grid: np.ndarray,
    *,
    mu: float,
    sigma: float,
    peak: float,
) -> np.ndarray:
    """Evaluate a MAP-normalized Gaussian density on a grid."""
    return peak * np.exp(-0.5 * ((grid - mu) / sigma) ** 2)


def _credible_density_level_1d(
    density: np.ndarray,
    dx: float,
    credible_mass: float,
) -> float:
    """Density threshold for a 1D highest-posterior-density region."""
    sorted_density = np.sort(density)[::-1]
    cumulative_mass = np.cumsum(sorted_density * dx)
    idx = int(np.searchsorted(cumulative_mass, credible_mass, side="left"))
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
    show_local_gaussian: bool = False,
) -> None:
    """Plot a marginalized posterior density with 68% and 95% HPD shading."""
    grid = np.asarray(grid, dtype=float)
    density = _density_from_grid_probability(grid, probability)
    dx = _grid_spacing(grid)

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
    if show_local_gaussian:
        gaussian_mu, gaussian_sigma, gaussian_peak, gaussian_ok = (
            _local_curvature_gaussian_for_marginal(grid, probability)
        )
        if gaussian_ok:
            ax.plot(
                grid,
                _gaussian_density_curve(
                    grid,
                    mu=gaussian_mu,
                    sigma=gaussian_sigma,
                    peak=gaussian_peak,
                ),
                linestyle="--",
                linewidth=1.8,
                label=f"local Gaussian σ={gaussian_sigma:.3g}",
            )
        else:
            ax.text(
                0.02,
                0.95,
                "local Gaussian unavailable",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
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


def make_frame_figure(
    *,
    events,
    posterior,
    frame_idx: int,
    detector_setup: DetectorSetup,
    b_field_t: float,
    t_min_ns: float,
    t_max_ns: float,
    bins: int,
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
    centers, asym, asym_err, det0_counts, det1_counts, edges, valid_bins = binned_asymmetry(
        times,
        dets,
        t_min_ns=t_min_ns,
        t_max_ns=t_max_ns,
        bins=bins,
    )
    nonempty = valid_bins

    curve_times = np.linspace(t_min_ns, t_max_ns, 600)
    prediction_curve = map_asymmetry_prediction(
        curve_times,
        g=map_g,
        a2=map_a2,
        b_field_t=b_field_t,
        detector_setup=detector_setup,
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
    ax_asym.set_title(f"R(t) visualization using {bins} bins")
    ax_asym.set_xlabel("time [ns]")
    ax_asym.set_ylabel("R(t)")
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
        "Binning-independent analysis:\n "
        f"MAP g={map_g:.5g}, MAP A₂={map_a2:.5g}",
        fontsize=14,
    )
    return fig


def make_chi2_frame_figure(
    *,
    events,
    chi2_result,
    n_events: int,
    detector_setup: DetectorSetup,
    b_field_t: float,
    t_min_ns: float,
    t_max_ns: float,
    bins: int,
    show_local_gaussian: bool = False,
):
    """Create the second 2x2 figure set for the binned chi-squared analysis."""
    post = chi2_result.posterior
    g_grid = chi2_result.g_grid
    a2_grid = chi2_result.a2_grid
    g_marginal = post.sum(axis=1)
    a2_marginal = post.sum(axis=0)
    map_g = float(chi2_result.map_g)
    map_a2 = float(chi2_result.map_a2)

    curve_times = np.linspace(t_min_ns, t_max_ns, 600)
    prediction_curve = map_asymmetry_prediction(
        curve_times,
        g=map_g,
        a2=map_a2,
        b_field_t=b_field_t,
        detector_setup=detector_setup,
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
        show_local_gaussian=show_local_gaussian,
    )

    valid = chi2_result.valid_bins
    ax_asym.axhline(0.0, linewidth=0.8)
    ax_asym.errorbar(
        chi2_result.bin_centers[valid],
        chi2_result.asymmetry[valid],
        yerr=chi2_result.asymmetry_error[valid],
        marker="o",
        linestyle="",
        capsize=3,
        label="binned data ± Gaussian error",
    )
    ax_asym.plot(curve_times, prediction_curve, linewidth=1.8, label="χ² MAP prediction")
    ax_asym.set_title(f"R(t) fit using {bins} bins")
    ax_asym.set_xlabel("time [ns]")
    ax_asym.set_ylabel("R(t)")
    ax_asym.legend(loc="best", fontsize=8)

    im = ax_post.imshow(
        post.T,
        origin="lower",
        aspect="auto",
        extent=[g_grid[0], g_grid[-1], a2_grid[0], a2_grid[-1]],
    )
    ax_post.plot(events.true_g, events.true_a2, marker="x", markersize=9, label="true")
    ax_post.plot(map_g, map_a2, marker="+", markersize=10, label="χ² MAP")
    ax_post.set_title(f"χ² posterior after {n_events} accepted events")
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

    valid_count = int(np.sum(chi2_result.valid_bins))
    fig.suptitle(
        "Binning-dependent analysis:\n "
        f"MAP g={map_g:.5g}, MAP A₂={map_a2:.5g}, "
        f"valid bins={valid_count}/{bins}",
        fontsize=14,
    )
    return fig
