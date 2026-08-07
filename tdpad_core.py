"""Core simulation and likelihood tools for the Streamlit TDPAD toy app."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# Same convention as tdpad_sim_eventwise.py
MU_N_OVER_HBAR_RAD_PER_S_T = 4.789412e7


@dataclass(frozen=True)
class DetectorSetup:
    """Two-detector geometry and efficiencies."""

    phi1_deg: float
    phi2_deg: float
    eff1: float = 1.0
    eff2: float = 1.0

    @property
    def phis_rad(self) -> np.ndarray:
        return np.radians([self.phi1_deg, self.phi2_deg]).astype(float)

    @property
    def efficiencies(self) -> np.ndarray:
        return np.array([self.eff1, self.eff2], dtype=float)


@dataclass
class SimulatedEvents:
    """Event-wise TDPAD data after the requested analysis time-window cut."""

    times_ns: np.ndarray
    detectors: np.ndarray
    true_g: float
    true_a2: float
    omega_rad_per_s: float
    raw_events_generated: int
    accepted_events: int
    true_parameter_mode: str = "drawn"


@dataclass
class PosteriorResult:
    """Precomputed posterior snapshots for the Streamlit slider."""

    event_counts: np.ndarray
    posteriors: np.ndarray  # shape: n_frames, n_g, n_a
    map_g: np.ndarray
    map_a2: np.ndarray
    g_grid: np.ndarray
    a2_grid: np.ndarray


@dataclass
class ChiSquaredPosteriorResult:
    """Binned chi-squared posterior for one cumulative event snapshot."""

    posterior: np.ndarray  # shape: n_g, n_a
    chi2: np.ndarray
    map_g: float
    map_a2: float
    g_grid: np.ndarray
    a2_grid: np.ndarray
    bin_centers: np.ndarray
    asymmetry: np.ndarray
    asymmetry_error: np.ndarray
    det0_counts: np.ndarray
    det1_counts: np.ndarray
    valid_bins: np.ndarray



def p2_from_delta(delta_rad: np.ndarray) -> np.ndarray:
    """Return P2(cos(delta)) = 0.25 + 0.75 cos(2 delta)."""
    return 0.25 + 0.75 * np.cos(2.0 * delta_rad)



def omega_from_g_b(g: np.ndarray | float, b_field_t: float) -> np.ndarray | float:
    """Larmor angular frequency in rad/s."""
    return np.asarray(g) * MU_N_OVER_HBAR_RAD_PER_S_T * b_field_t


def w_value(
    theta_rad: np.ndarray | float,
    times_ns: np.ndarray,
    g: float,
    a2: float,
    b_field_t: float,
) -> np.ndarray:
    """Evaluate W(theta, t) for one parameter pair."""
    omega = float(omega_from_g_b(g, b_field_t))
    phase = omega * np.asarray(times_ns, dtype=float) * 1e-9
    return 1.0 + a2 * p2_from_delta(theta_rad - phase)



def simulate_events(
    *,
    lifetime_ns: float,
    b_field_t: float,
    detector_setup: DetectorSetup,
    t_min_ns: float,
    t_max_ns: float,
    g_range: tuple[float, float],
    a2_range: tuple[float, float],
    true_g: float | None = None,
    true_a2: float | None = None,
    raw_events: int = 10_000,
    seed: int | None = None,
) -> SimulatedEvents:
    """
    Draw detector-time pairs from the joint TDPAD event distribution.

    Candidate times follow the untruncated exponential decay law. Rejection
    sampling adds the time-dependent total detector weight, after which the
    detector is drawn from its conditional probability at the accepted time.
    Exactly ``raw_events`` joint events are produced before the requested
    analysis time-window cut is applied.

    If true_g and true_a2 are supplied, those fixed values are used. Otherwise
    each missing value is drawn uniformly from its corresponding input range.
    """
    phis = detector_setup.phis_rad
    efficiencies = detector_setup.efficiencies
    if not np.all(np.isfinite(efficiencies)) or np.any(efficiencies < 0.0):
        raise ValueError("Detector efficiencies must be finite and non-negative")
    efficiency_sum = float(efficiencies.sum())
    if efficiency_sum == 0.0:
        raise ValueError("At least one detector efficiency must be positive")

    rng = np.random.default_rng(seed)
    parameter_mode = "fixed" if true_g is not None and true_a2 is not None else "drawn"
    used_g = float(true_g) if true_g is not None else float(rng.uniform(g_range[0], g_range[1]))
    used_a2 = float(true_a2) if true_a2 is not None else float(rng.uniform(a2_range[0], a2_range[1]))
    omega = float(omega_from_g_b(used_g, b_field_t))

    # S(t) = sum_i eps_i W_i(t) is a constant plus one sinusoid. Its maximum
    # supplies an exact rejection bound when omega is nonzero. For omega == 0,
    # S is constant and every candidate can be accepted.
    if omega == 0.0:
        total_weight_bound = float(
            np.sum(efficiencies * (1.0 + used_a2 * p2_from_delta(phis)))
        )
    else:
        constant = efficiency_sum * (1.0 + 0.25 * used_a2)
        phasor_magnitude = abs(np.sum(efficiencies * np.exp(2.0j * phis)))
        amplitude = 0.75 * abs(used_a2) * phasor_magnitude
        total_weight_bound = float(constant + amplitude)

    joint_times_ns = np.empty(int(raw_events), dtype=float)
    joint_detectors = np.empty(int(raw_events), dtype=int)
    events_filled = 0
    acceptance_estimate = 0.5
    max_batch_size = 1_000_000

    while events_filled < raw_events:
        remaining = int(raw_events) - events_filled
        candidate_count = math.ceil(1.1 * remaining / acceptance_estimate)
        candidate_count = min(max(candidate_count, 1_024), max_batch_size)

        candidate_times_ns = rng.exponential(scale=lifetime_ns, size=candidate_count)
        phase = omega * candidate_times_ns * 1e-9
        q = p2_from_delta(phis[None, :] - phase[:, None])
        weights = efficiencies[None, :] * (1.0 + used_a2 * q)
        totals = weights.sum(axis=1)

        accepted = rng.random(candidate_count) * total_weight_bound < totals
        accepted_count = int(np.count_nonzero(accepted))
        if accepted_count == 0:
            acceptance_estimate = max(acceptance_estimate * 0.5, 1e-6)
            continue
        acceptance_estimate = accepted_count / candidate_count

        accepted_indices = np.flatnonzero(accepted)[:remaining]
        count_to_store = len(accepted_indices)
        detector0_probability = (
            weights[accepted_indices, 0] / totals[accepted_indices]
        )
        detector_draws = (
            rng.random(count_to_store) > detector0_probability
        ).astype(int)

        output_slice = slice(events_filled, events_filled + count_to_store)
        joint_times_ns[output_slice] = candidate_times_ns[accepted_indices]
        joint_detectors[output_slice] = detector_draws
        events_filled += count_to_store

    window_mask = (joint_times_ns >= t_min_ns) & (joint_times_ns <= t_max_ns)
    times_ns = joint_times_ns[window_mask]
    detectors = joint_detectors[window_mask]

    return SimulatedEvents(
        times_ns=times_ns,
        detectors=detectors,
        true_g=used_g,
        true_a2=used_a2,
        omega_rad_per_s=omega,
        raw_events_generated=int(raw_events),
        accepted_events=int(len(times_ns)),
        true_parameter_mode=parameter_mode,
    )



def log_spaced_event_counts(n_events: int, n_points: int = 100) -> np.ndarray:
    """Return approximately log-equally spaced cumulative event counts."""
    if n_events <= 0:
        return np.array([], dtype=int)
    if n_events <= n_points:
        return np.arange(1, n_events + 1, dtype=int)

    counts = np.unique(np.rint(np.geomspace(1, n_events, n_points)).astype(int))
    counts[0] = 1
    counts[-1] = n_events

    if len(counts) < n_points:
        all_counts = np.arange(1, n_events + 1, dtype=int)
        missing = np.setdiff1d(all_counts, counts, assume_unique=True)
        needed = n_points - len(counts)
        extra_idx = np.linspace(0, len(missing) - 1, needed).round().astype(int)
        counts = np.unique(np.concatenate([counts, missing[extra_idx]]))

    return np.unique(counts).astype(int)



def _normalize_log_grid(log_grid: np.ndarray) -> np.ndarray:
    """Convert log posterior grid to a normalized probability grid."""
    finite = np.isfinite(log_grid)
    if not np.any(finite):
        return np.full_like(log_grid, 1.0 / log_grid.size, dtype=float)
    max_log = np.max(log_grid[finite])
    shifted = np.where(finite, log_grid - max_log, -np.inf)
    posterior = np.exp(shifted)
    return posterior / posterior.sum()



def compute_posterior_snapshots(
    *,
    events: SimulatedEvents,
    b_field_t: float,
    detector_setup: DetectorSetup,
    g_grid: np.ndarray,
    a2_grid: np.ndarray,
    event_counts: np.ndarray,
) -> PosteriorResult:
    """
    Cumulatively update the event-wise log-likelihood over a g x A2 grid.

    The per-event term is
        eps_det W(theta_det,t) / sum_j eps_j W(theta_j,t)
    with a flat prior over the grid.
    """
    times_ns = np.asarray(events.times_ns, dtype=float)
    dets = np.asarray(events.detectors, dtype=int)
    n_events = len(times_ns)
    if n_events == 0:
        raise ValueError("No accepted events available for analysis")

    g_grid = np.asarray(g_grid, dtype=float)
    a2_grid = np.asarray(a2_grid, dtype=float)
    event_counts_array = np.asarray(event_counts, dtype=int)

    phis = detector_setup.phis_rad
    eff = detector_setup.efficiencies
    omega_grid = omega_from_g_b(g_grid, b_field_t)  # shape: ng

    # q_det0 and q_det1 each have shape n_events x n_g.
    phase = times_ns[:, None] * 1e-9 * omega_grid[None, :]
    q_det0 = p2_from_delta(phis[0] - phase)
    q_det1 = p2_from_delta(phis[1] - phase)

    ng = len(g_grid)
    na = len(a2_grid)
    log_like = np.zeros((ng, na), dtype=float)
    posteriors = np.empty((len(event_counts_array), ng, na), dtype=float)
    map_g = np.empty(len(event_counts_array), dtype=float)
    map_a2 = np.empty(len(event_counts_array), dtype=float)

    next_snapshot = 0
    targets = set(int(x) for x in event_counts_array)
    a2_row = a2_grid[None, :]

    for i in range(n_events):
        q0 = q_det0[i, :, None]
        q1 = q_det1[i, :, None]

        w0 = 1.0 + q0 * a2_row
        w1 = 1.0 + q1 * a2_row

        denom = eff[0] * w0 + eff[1] * w1
        numer = eff[0] * w0 if dets[i] == 0 else eff[1] * w1
        log_like += np.log(numer) - np.log(denom)

        event_count = i + 1
        if event_count in targets:
            posterior = _normalize_log_grid(log_like)
            posteriors[next_snapshot] = posterior
            max_idx = np.unravel_index(np.argmax(posterior), posterior.shape)
            map_g[next_snapshot] = g_grid[max_idx[0]]
            map_a2[next_snapshot] = a2_grid[max_idx[1]]
            next_snapshot += 1

    return PosteriorResult(
        event_counts=event_counts_array,
        posteriors=posteriors,
        map_g=map_g,
        map_a2=map_a2,
        g_grid=g_grid,
        a2_grid=a2_grid,
    )



def binned_asymmetry(
    times_ns: np.ndarray,
    detectors: np.ndarray,
    t_min_ns: float,
    t_max_ns: float,
    bins: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute detector histograms, asymmetry, and Gaussian propagated errors.

    The plotted asymmetry follows the original app convention:
        (det1 - det2) / (det1 + det2)
    where det1 is detector index 0 and det2 is detector index 1.
    With Poisson Gaussian count errors sqrt(N), error propagation gives
        sigma_A = 2 sqrt(N1 N2) / (N1 + N2)^(3/2)
                = sqrt((1 - A^2) / (N1 + N2)).

    Bins are marked invalid unless both detectors have at least one count.
    """
    edges = np.linspace(t_min_ns, t_max_ns, bins + 1)
    det0_counts, _ = np.histogram(times_ns[detectors == 0], bins=edges)
    det1_counts, _ = np.histogram(times_ns[detectors == 1], bins=edges)
    denom = det0_counts + det1_counts
    asym = np.full(bins, np.nan, dtype=float)
    asym_err = np.full(bins, np.nan, dtype=float)
    valid = (det0_counts > 0) & (det1_counts > 0)
    asym[valid] = (det0_counts[valid] - det1_counts[valid]) / denom[valid]
    asym_err[valid] = 2.0 * np.sqrt(det0_counts[valid] * det1_counts[valid]) / (denom[valid] ** 1.5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, asym, asym_err, det0_counts, det1_counts, edges, valid



def map_asymmetry_prediction(
    times_ns: np.ndarray,
    *,
    g: float,
    a2: float,
    b_field_t: float,
    detector_setup: DetectorSetup,
) -> np.ndarray:
    """Prediction for (det1-det2)/(det1+det2) at arbitrary times."""
    phis = detector_setup.phis_rad
    eff = detector_setup.efficiencies
    phase = float(omega_from_g_b(g, b_field_t)) * np.asarray(times_ns, dtype=float) * 1e-9
    w0 = eff[0] * (1.0 + a2 * p2_from_delta(phis[0] - phase))
    w1 = eff[1] * (1.0 + a2 * p2_from_delta(phis[1] - phase))
    return (w0 - w1) / (w0 + w1)



def _prediction_grid_for_times(
    times_ns: np.ndarray,
    *,
    g_grid: np.ndarray,
    a2_grid: np.ndarray,
    b_field_t: float,
    detector_setup: DetectorSetup,
) -> np.ndarray:
    """Return model asymmetry r(t) on a time x g x A2 grid."""
    times_ns = np.asarray(times_ns, dtype=float)
    g_grid = np.asarray(g_grid, dtype=float)
    a2_grid = np.asarray(a2_grid, dtype=float)
    phis = detector_setup.phis_rad
    eff = detector_setup.efficiencies

    phase = times_ns[:, None] * 1e-9 * omega_from_g_b(g_grid, b_field_t)[None, :]
    q0 = p2_from_delta(phis[0] - phase)[:, :, None]
    q1 = p2_from_delta(phis[1] - phase)[:, :, None]
    a2 = a2_grid[None, None, :]

    w0 = eff[0] * (1.0 + q0 * a2)
    w1 = eff[1] * (1.0 + q1 * a2)
    return (w0 - w1) / (w0 + w1)


def compute_chi2_posterior(
    *,
    times_ns: np.ndarray,
    detectors: np.ndarray,
    b_field_t: float,
    detector_setup: DetectorSetup,
    g_grid: np.ndarray,
    a2_grid: np.ndarray,
    t_min_ns: float,
    t_max_ns: float,
    bins: int,
) -> ChiSquaredPosteriorResult:
    """
    Build a binned chi-squared posterior on the same g x A2 grid.

    The data are binned into detector asymmetries, bins with zero counts in
    either detector are excluded, and the grid likelihood is
        L(g,A2) ∝ exp[-chi2(g,A2)/2].
    """
    if bins <= 0:
        raise ValueError("Number of bins must be positive")

    centers, asym, asym_err, det0_counts, det1_counts, _edges, valid_bins = binned_asymmetry(
        times_ns,
        detectors,
        t_min_ns=t_min_ns,
        t_max_ns=t_max_ns,
        bins=bins,
    )

    chi2 = np.full((len(g_grid), len(a2_grid)), np.inf, dtype=float)
    if np.any(valid_bins):
        pred = _prediction_grid_for_times(
            centers[valid_bins],
            g_grid=g_grid,
            a2_grid=a2_grid,
            b_field_t=b_field_t,
            detector_setup=detector_setup,
        )
        residual = asym[valid_bins, None, None] - pred
        sigma = asym_err[valid_bins, None, None]
        term = (residual / sigma) ** 2
        chi2 = np.sum(term, axis=0)

    posterior = _normalize_log_grid(-0.5 * chi2)
    max_idx = np.unravel_index(np.argmax(posterior), posterior.shape)
    return ChiSquaredPosteriorResult(
        posterior=posterior,
        chi2=chi2,
        map_g=float(g_grid[max_idx[0]]),
        map_a2=float(a2_grid[max_idx[1]]),
        g_grid=np.asarray(g_grid, dtype=float),
        a2_grid=np.asarray(a2_grid, dtype=float),
        bin_centers=centers,
        asymmetry=asym,
        asymmetry_error=asym_err,
        det0_counts=det0_counts,
        det1_counts=det1_counts,
        valid_bins=valid_bins,
    )



#### legacy simulation that draws pure-exponential times independently


def simulate_events_old(
    *,
    lifetime_ns: float,
    b_field_t: float,
    detector_setup: DetectorSetup,
    t_min_ns: float,
    t_max_ns: float,
    g_range: tuple[float, float],
    a2_range: tuple[float, float],
    true_g: float | None = None,
    true_a2: float | None = None,
    raw_events: int = 10_000,
    seed: int | None = None,
) -> SimulatedEvents:
    """
    Legacy simulation that draws pure-exponential times independently.

    Draw raw event times from an untruncated exponential and then select the
    requested time range. Detector choices are drawn from angular weights.

    If true_g and true_a2 are supplied, those fixed values are used. Otherwise
    each missing value is drawn uniformly from its corresponding input range.
    """
    if lifetime_ns <= 0:
        raise ValueError("lifetime_ns must be positive")
    if t_max_ns <= t_min_ns:
        raise ValueError("t_max_ns must be larger than t_min_ns")
    if raw_events <= 0:
        raise ValueError("raw_events must be positive")
    if g_range[1] < g_range[0]:
        raise ValueError("g_range must be ordered as (min, max)")
    if a2_range[1] < a2_range[0]:
        raise ValueError("a2_range must be ordered as (min, max)")

    rng = np.random.default_rng(seed)
    parameter_mode = "fixed" if true_g is not None and true_a2 is not None else "drawn"
    used_g = float(true_g) if true_g is not None else float(rng.uniform(g_range[0], g_range[1]))
    used_a2 = float(true_a2) if true_a2 is not None else float(rng.uniform(a2_range[0], a2_range[1]))

    raw_times_ns = rng.exponential(scale=lifetime_ns, size=int(raw_events))
    mask = (raw_times_ns >= t_min_ns) & (raw_times_ns <= t_max_ns)
    times_ns = raw_times_ns[mask]

    if len(times_ns) == 0:
        return SimulatedEvents(
            times_ns=times_ns,
            detectors=np.array([], dtype=int),
            true_g=used_g,
            true_a2=used_a2,
            omega_rad_per_s=float(omega_from_g_b(used_g, b_field_t)),
            raw_events_generated=int(raw_events),
            accepted_events=0,
            true_parameter_mode=parameter_mode,
        )

    phis = detector_setup.phis_rad
    eff = detector_setup.efficiencies
    omega = float(omega_from_g_b(used_g, b_field_t))
    phase = omega * times_ns * 1e-9

    q = p2_from_delta(phis[None, :] - phase[:, None])
    weights = 1.0 + used_a2 * q
    weights *= eff[None, :]

    totals = weights.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError(
            "At least one event has zero total detector weight. Check the detector efficiencies."
        )

    probabilities = weights / totals[:, None]
    random_numbers = rng.random(size=len(times_ns))
    detectors = (random_numbers > probabilities[:, 0]).astype(int)

    return SimulatedEvents(
        times_ns=times_ns.astype(float),
        detectors=detectors.astype(int),
        true_g=used_g,
        true_a2=used_a2,
        omega_rad_per_s=omega,
        raw_events_generated=int(raw_events),
        accepted_events=int(len(times_ns)),
        true_parameter_mode=parameter_mode,
    )
