"""
broadening.py
=============
Converts the ideal stick pattern from bragg.py into a continuous diffraction
pattern by convolving each peak with a user-chosen peak shape function.

Three peak shapes are available:

    "gaussian"      — symmetric bell curve.  Simple, fast, not physically
                      accurate but useful for quick visualisation.

    "lorentzian"    — heavier tails than Gaussian.  Closer to instrumental
                      broadening contributions alone.

    "pseudo-voigt"  — weighted sum of Gaussian and Lorentzian:
                          pV(x) = η·L(x) + (1−η)·G(x)
                      This is the standard approximation to the true Voigt
                      profile (which is a convolution of G and L, expensive
                      to compute exactly).  Recommended for realistic
                      simulation; η ≈ 0.5–0.7 matches most lab diffractometers.

Peak width — the Scherrer equation
-----------------------------------
All three shapes share the same FWHM (full width at half maximum), computed
from the Scherrer equation:

    FWHM(θ) = K · λ / (L · cos θ)

where:
    K ≈ 0.9394  — Scherrer constant (shape factor for spherical crystallites)
    λ           — X-ray wavelength in Å (default Cu Kα = 1.5406 Å)
    L           — crystallite size in Å (input by user; e.g. 500 Å = 50 nm)
    θ           — Bragg angle in radians

FWHM is angle-dependent: peaks at high 2θ are broader than low-angle peaks.
This is physically correct and visible in real diffractograms.

Usage
-----
    from cif_reader import fetch_structure
    from bragg import compute_peaks
    from broadening import compute_pattern, plot_pattern

    cs     = fetch_structure("mp-390")          # TiO2 anatase
    peaks  = compute_peaks(cs)

    # Choose any of: "gaussian", "lorentzian", "pseudo-voigt"
    two_theta, intensity = compute_pattern(peaks, shape="pseudo-voigt",
                                           crystallite_size_nm=50.0)
    plot_pattern(two_theta, intensity, peaks, crystal=cs,
                 shape="pseudo-voigt", crystallite_size_nm=50.0)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from bragg import BraggPeak, CU_KALPHA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHERRER_K = 0.9394          # shape factor for spherical crystallites
DEFAULT_ETA = 0.6             # pseudo-Voigt mixing: 0 = pure G, 1 = pure L
TWO_THETA_POINTS = 4000       # number of points in the output pattern

PeakShape = Literal["gaussian", "lorentzian", "pseudo-voigt"]


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_pattern(
    peaks: list[BraggPeak],
    shape: PeakShape = "pseudo-voigt",
    crystallite_size_nm: float = 50.0,
    wavelength: float = CU_KALPHA,
    eta: float = DEFAULT_ETA,
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
    add_noise: bool = False,
    noise_level: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convolve Bragg stick peaks with a peak shape to produce a continuous
    diffraction pattern.

    Parameters
    ----------
    peaks : list[BraggPeak]
        Output of bragg.compute_peaks().
    shape : "gaussian" | "lorentzian" | "pseudo-voigt"
        Peak shape function to use.
    crystallite_size_nm : float
        Average crystallite size in nanometres.  Smaller → broader peaks.
        Typical range: 5–500 nm.  Default: 50 nm.
    wavelength : float
        X-ray wavelength in Å.  Default: Cu Kα = 1.5406 Å.
    eta : float
        Pseudo-Voigt mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian).
        Only used when shape="pseudo-voigt".  Default: 0.6.
    two_theta_min, two_theta_max : float
        Angular range of the output pattern in degrees.
    add_noise : bool
        If True, adds Poisson-like noise to simulate detector counting statistics.
    noise_level : float
        Noise amplitude as a fraction of the maximum intensity.  Default: 0.02.

    Returns
    -------
    two_theta : np.ndarray, shape (N,)
        2θ angles in degrees.
    intensity : np.ndarray, shape (N,)
        Normalised intensity (strongest feature = 100).

    Examples
    --------
    >>> two_theta, intensity = compute_pattern(peaks, shape="gaussian",
    ...                                        crystallite_size_nm=30.0)
    >>> two_theta, intensity = compute_pattern(peaks, shape="pseudo-voigt",
    ...                                        eta=0.7)
    """
    shape = shape.lower().replace(" ", "-")
    if shape not in ("gaussian", "lorentzian", "pseudo-voigt"):
        raise ValueError(
            f"Unknown shape {shape!r}. Choose 'gaussian', 'lorentzian', or 'pseudo-voigt'."
        )

    # Convert crystallite size: nm → Å (Scherrer equation uses Å)
    L_angstrom = crystallite_size_nm * 10.0

    two_theta = np.linspace(two_theta_min, two_theta_max, TWO_THETA_POINTS)
    intensity  = np.zeros(TWO_THETA_POINTS)

    for peak in peaks:
        theta_rad = np.radians(peak.two_theta / 2.0)
        fwhm = _scherrer_fwhm(theta_rad, L_angstrom, wavelength)

        if shape == "gaussian":
            profile = _gaussian(two_theta, peak.two_theta, fwhm)
        elif shape == "lorentzian":
            profile = _lorentzian(two_theta, peak.two_theta, fwhm)
        else:  # pseudo-voigt
            profile = _pseudo_voigt(two_theta, peak.two_theta, fwhm, eta)

        intensity += peak.intensity * profile

    # Normalise to 100
    max_i = intensity.max()
    if max_i > 0:
        intensity = (intensity / max_i) * 100.0

    # Optional noise
    if add_noise:
        rng = np.random.default_rng()
        noise = rng.normal(0, noise_level * 100.0, size=intensity.shape)
        intensity = np.clip(intensity + noise, 0, None)

    logger.info(
        "Pattern computed: shape=%s, L=%.1f nm, η=%.2f",
        shape, crystallite_size_nm, eta if shape == "pseudo-voigt" else float("nan"),
    )
    return two_theta, intensity


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pattern(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    peaks: list[BraggPeak],
    crystal=None,
    shape: PeakShape = "pseudo-voigt",
    crystallite_size_nm: float = 50.0,
    eta: float = DEFAULT_ETA,
    show_sticks: bool = True,
    show_hkl: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Plot the broadened XRD pattern with optional stick markers and hkl labels.

    Parameters
    ----------
    two_theta, intensity : arrays from compute_pattern()
    peaks     : BraggPeak list from bragg.compute_peaks() — used for stick markers
    crystal   : CrystalStructure (optional) — used for the plot title
    shape     : peak shape used, shown in the legend
    crystallite_size_nm : shown in the legend
    eta       : pseudo-Voigt mixing, shown in legend if shape="pseudo-voigt"
    show_sticks : draw vertical stick markers at each peak position
    show_hkl    : label each stick with its Miller indices
    save_path   : if given, save the figure to this path instead of showing it
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # --- Broadened pattern ---
    shape_labels = {
        "gaussian":    "Gaussian",
        "lorentzian":  "Lorentzian",
        "pseudo-voigt": f"Pseudo-Voigt (η={eta:.2f})",
    }
    label = f"{shape_labels.get(shape, shape)},  L={crystallite_size_nm:.0f} nm"
    ax.plot(two_theta, intensity, color="#2563eb", linewidth=1.4, label=label, zorder=3)

    # --- Stick markers ---
    if show_sticks and peaks:
        stick_x = [p.two_theta for p in peaks]
        stick_y = [p.intensity for p in peaks]
        ax.vlines(stick_x, 0, stick_y,
                  color="#dc2626", linewidth=0.8, alpha=0.6,
                  label="Bragg positions", zorder=2)

        # hkl labels on the tallest sticks (avoid crowding)
        if show_hkl:
            labeled = _select_peaks_to_label(peaks, max_labels=12)
            for p in labeled:
                h, k, l = p.hkl
                ax.text(
                    p.two_theta, p.intensity + 1.5,
                    f"({h}{k}{l})",
                    fontsize=6.5, ha="center", va="bottom",
                    color="#374151", rotation=90,
                )

    # --- Formatting ---
    formula = crystal.formula if crystal else "Unknown"
    sg      = crystal.space_group_symbol if crystal else ""
    ax.set_title(
        f"Simulated XRD — {formula}  [{sg}]",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xlabel("2θ (degrees)", fontsize=11)
    ax.set_ylabel("Intensity (arb. units)", fontsize=11)
    ax.set_xlim(two_theta[0], two_theta[-1])
    ax.set_ylim(bottom=-2)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Peak shape functions
# ---------------------------------------------------------------------------

def _scherrer_fwhm(theta_rad: float, L_angstrom: float, wavelength: float) -> float:
    """
    Compute the FWHM at angle θ using the Scherrer equation.

        FWHM = K · λ / (L · cosθ)

    Returns FWHM in degrees.

    Parameters
    ----------
    theta_rad   : Bragg angle θ in radians (half of 2θ)
    L_angstrom  : crystallite size in Å
    wavelength  : X-ray wavelength in Å
    """
    fwhm_rad = SCHERRER_K * wavelength / (L_angstrom * np.cos(theta_rad))
    return np.degrees(fwhm_rad)


def _gaussian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """
    Normalised Gaussian peak (peak height = 1).

        G(x) = exp( -4·ln2·(x − x0)² / fwhm² )

    The factor 4·ln2 ensures the half-maximum points are exactly at ±fwhm/2.
    """
    sigma_sq = (fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))) ** 2
    return np.exp(-((x - center) ** 2) / (2.0 * sigma_sq))


def _lorentzian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """
    Normalised Lorentzian peak (peak height = 1).

        L(x) = 1 / (1 + 4·(x − x0)²/fwhm²)

    Heavier tails than Gaussian — models instrumental broadening and
    defect scattering more accurately.
    """
    gamma = fwhm / 2.0
    return gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


def _pseudo_voigt(
    x: np.ndarray, center: float, fwhm: float, eta: float
) -> np.ndarray:
    """
    Pseudo-Voigt profile: weighted sum of Lorentzian and Gaussian.

        pV(x) = η · L(x) + (1 − η) · G(x)

    This approximates the true Voigt convolution (exact computation is slow).
    η = 0 → pure Gaussian; η = 1 → pure Lorentzian.
    Real XRD peaks typically have η ≈ 0.5–0.7.

    Parameters
    ----------
    eta : float in [0, 1]
        Mixing parameter.
    """
    eta = float(np.clip(eta, 0.0, 1.0))
    return eta * _lorentzian(x, center, fwhm) + (1.0 - eta) * _gaussian(x, center, fwhm)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _select_peaks_to_label(
    peaks: list[BraggPeak], max_labels: int = 12
) -> list[BraggPeak]:
    """
    Choose which peaks to label with hkl indices on the plot.
    Selects the strongest peaks while enforcing a minimum angular spacing
    so labels don't overlap.
    """
    sorted_by_intensity = sorted(peaks, key=lambda p: p.intensity, reverse=True)
    labeled: list[BraggPeak] = []
    for peak in sorted_by_intensity:
        if len(labeled) >= max_labels:
            break
        # Enforce minimum spacing of 2° between labels
        if all(abs(peak.two_theta - lp.two_theta) >= 2.0 for lp in labeled):
            labeled.append(peak)
    return labeled


# ---------------------------------------------------------------------------
# Entry point — interactive demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from CIF_fetch import fetch_structure, summarise
    from bragg import compute_peaks

    # --- User inputs ---
    mp_id              = input("Enter MP ID (e.g. mp-390): ").strip() or "mp-390"
    shape              = input("Peak shape — gaussian / lorentzian / pseudo-voigt [pseudo-voigt]: ").strip() or "pseudo-voigt"
    size_input         = input("Crystallite size in nm [50]: ").strip()
    crystallite_size   = float(size_input) if size_input else 50.0

    eta = DEFAULT_ETA
    if shape == "pseudo-voigt":
        eta_input = input(f"Pseudo-Voigt eta (0=Gaussian, 1=Lorentzian) [{DEFAULT_ETA}]: ").strip()
        eta = float(eta_input) if eta_input else DEFAULT_ETA

    # --- Fetch and compute ---
    print(f"\nFetching {mp_id} …")
    cs = fetch_structure(mp_id)
    print(summarise(cs))

    print(f"\nComputing Bragg peaks …")
    peaks = compute_peaks(cs)
    print(f"Found {len(peaks)} peaks")

    print(f"\nBroadening with {shape}, L={crystallite_size} nm …")
    two_theta, intensity = compute_pattern(
        peaks,
        shape=shape,
        crystallite_size_nm=crystallite_size,
        eta=eta,
    )

    plot_pattern(
        two_theta, intensity, peaks,
        crystal=cs,
        shape=shape,
        crystallite_size_nm=crystallite_size,
        eta=eta,
    )