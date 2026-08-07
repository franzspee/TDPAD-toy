**Data generation**

- Single events are of the form `(i, t)` where `i` is the detector index (either 0 or 1) and `t` is the time in ns.
- The event generation uses the standard TDPAD angular distribution formula:

$$
\begin{aligned}
p(i,t) &\propto \epsilon(i)\,\exp(-\lambda t)\,
W\!\left(\theta(i),t\right) \\[0.4em]
W(\theta,t) &= 1 + A_2\left[
\frac{1}{4} + \frac{3}{4}
\cos\!\left(2\theta - 2 g \mu_N \frac{B}{\hbar} t\right)
\right].
\end{aligned}
$$

- True `g` and `A₂` can either be drawn uniformly from the displayed ranges or fixed manually.
- Only events inside the observration window`[t_min, t_max]` are retained, so the accepted sample can contain fewer than the raw number of generated events.
- Detector efficiencies are hard-coded to 1 for both detectors.

**Data analysis**

- The prior over the displayed `g × A₂` range is flat.
- The binning-independent likelihood is given by:

$$
\begin{aligned}
p(i_k|t_k)=\frac{  W\left(\theta(i_k),t_k\right)}
{\sum_{j=0}^{I-1}   W\left(\theta(j),t_k\right)},\\[0.4em]
\mathcal L(g,A_2|\mathcal D) = \prod_{k=0}^{K-1} p(i_k|t_k)
\end{aligned}
$$

- The binning-dependent likelihood is given by:

$$
\begin{aligned}
R(t)=\frac{p(0,t)-p(1,t)}{p(0,t)+p(1,t)},\\[0.4em]
\mathcal L(g,A_2|\mathcal D) \propto \exp\left(-\frac{1}{2}\sum_{\tilde k}\frac{(R^{exp}_{\tilde k}-R^{theo}(t_{\tilde k}))^2}{(\Delta R_{\tilde k}^{exp})^2}\right)
\end{aligned}
$$

- Bins with zero counts in either detector are excluded from the binnning dependent analysis. The number of bins can be adjusted with a slider.
- The posterior is computed on a uniform grid in `g × A₂` space, and the MAP is found by locating the grid point with the maximum posterior probability. The number of grid points can be adjusted.
- The posterior snapshots are cumulative: The analysis uses all accepted events up to the selected snapshot.

**Interpretation**

- In the marginalized 1D posteriors, the 68% and 95% highest-posterior-density (HPD) regions are shaded in light green. The MAP is marked with a red cross, and the true value is marked with a dashed line.
- Note the different character of the top-right figure. For the binning-independent analysis, it serves only as a visualization and a different binning has no influcence on the posterior.For the binning-dependent analysis, the top-right figure is used to compute the likelihood and a different bin size will result in a different posterior.
- A local Gaussian can be fit to the g marginal posterior to visualize a classical uncertainty estimate.
