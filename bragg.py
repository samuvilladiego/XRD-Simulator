"""
bragg.py
========
Computes the ideal XRD diffraction pattern from a CrystalStructure.

Given a crystal structure (from cif_reader.py), this module:
    1. Generates all Miller index triplets (hkl) up to a 2θ limit
    2. Computes the d-spacing for each plane
    3. Applies Bragg's law to find the 2θ angle of each reflection
    4. Filters out systematically absent reflections using space group symmetry
    5. Computes the structure factor F(hkl) for each reflection
    6. Applies the Lorentz-polarization (LP) correction
    7. Returns a list of BraggPeak objects (2θ, intensity, hkl, d)

The output is the "ideal" pattern — a stick diagram with infinitely sharp
peaks. Peak broadening and noise are added in later modules.

Physics notes
-------------
Bragg's law:
    λ = 2 · d_hkl · sin(θ)
    where λ is the X-ray wavelength, d_hkl the interplanar spacing, θ the
    half-angle of diffraction.  We solve for 2θ given d and λ.

Structure factor:
    F(hkl) = Σ_j  f_j(s) · occ_j · exp(2πi · (h·x_j + k·y_j + l·z_j))
    where the sum runs over all atoms j in the unit cell, f_j is the atomic
    form factor (a function of s = sinθ/λ), occ_j the occupancy, and
    (x_j, y_j, z_j) the fractional coordinates.  The observed intensity is
    proportional to |F|².

Atomic form factors:
    f(s) = Σ_{i=1}^{4} a_i · exp(-b_i · s²) + c
    This is the standard 4-Gaussian approximation from the International
    Tables for Crystallography (Brown et al. 2006), parameterised by
    coefficients (a1..a4, b1..b4, c) tabulated for each element.

Lorentz-polarization (LP) correction  ← WHY AND WHERE
    Real diffractometers do not measure all reflections with equal efficiency.
    Two geometric effects modulate the raw |F|² intensity:

    • Lorentz factor (L): accounts for the time a reciprocal-lattice point
      spends passing through the Ewald sphere as the crystal/detector rotates.
      For a powder diffractometer:  L = 1 / (sin²θ · cosθ)

    • Polarization factor (P): unpolarized X-rays become partially polarized
      upon diffraction.  For an unpolarized source:
          P = (1 + cos²2θ) / 2

    Combined:  LP = (1 + cos²2θ) / (sin²θ · cosθ)

    Applied at: the end of compute_peaks(), multiplied into the intensity
    of every reflection before returning.  This is standard practice and
    matches what real powder diffractometers measure.  Without it, high-angle
    peaks would appear artificially strong relative to experiment.

Usage
-----
    from CIF_fetch import fetch_structure
    from bragg import compute_peaks, peaks_to_dataframe

    cs = fetch_structure("mp-24850")
    peaks = compute_peaks(cs)

    for p in peaks[:5]:
        print(p)

    df = peaks_to_dataframe(peaks)   # pandas DataFrame for easy inspection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from CIF_fetch import CrystalStructure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cu Kα X-ray wavelength in Å — the most common laboratory source.
# Change this to Mo Kα (0.7107 Å) or any other source as needed.
CU_KALPHA = 1.5406  # Å

# Maximum |h|, |k|, |l| to consider when generating Miller indices.
# Higher values find more reflections but increase computation time.
# 10 is more than sufficient for 2θ up to 90° with typical lattice parameters.
MAX_HKL_INDEX = 10


# ---------------------------------------------------------------------------
# Atomic form factor coefficients
# ---------------------------------------------------------------------------
# Standard 4-Gaussian parameterisation from the International Tables for
# Crystallography, Vol. C, Table 6.1.1.4 (Brown et al., 2006).
# Format: element → (a1, b1, a2, b2, a3, b3, a4, b4, c)
# f(s) = a1·exp(-b1·s²) + a2·exp(-b2·s²) + a3·exp(-b3·s²) + a4·exp(-b4·s²) + c
# where s = sinθ / λ  (units: Å⁻¹)

FORM_FACTORS: dict[str, tuple] = {
    "H":  (0.489918, 20.6593,  0.262003, 7.74039,  0.196767, 49.5519,  0.049879, 2.20159,  0.001305),
    "Li": (1.128200, 3.95460,  0.750800, 1.05240,  0.617500, 85.3905,  0.465300, 168.261,  0.037700),
    "C":  (2.310000, 20.8439,  1.020000, 10.2075,  1.588600, 0.568700, 0.865000, 51.6512,  0.215600),
    "N":  (12.21260, 0.005700, 3.132200, 9.89330,  2.012500, 28.9975,  1.166300, 0.582600, -11.5290),
    "O":  (3.048500, 13.2771,  2.286800, 5.70110,  1.546300, 0.323900, 0.867000, 32.9089,  0.250800),
    "F":  (3.539200, 10.2825,  2.641200, 4.29440,  1.517000, 0.261500, 1.024300, 26.1476,  0.277600),
    "Na": (4.762600, 3.28500,  3.173600, 8.84220,  1.267400, 0.313600, 1.112800, 129.424,  0.676000),
    "Mg": (5.420400, 2.82750,  2.173500, 79.2615,  1.226900, 0.380800, 2.307300, 7.19370,  0.858400),
    "Al": (6.420200, 3.03870,  1.900200, 0.742600, 1.593600, 31.5472,  1.964600, 85.0886,  1.115100),
    "Si": (6.291500, 2.43860,  3.035300, 32.3337,  1.989100, 0.678500, 1.541000, 81.6937,  1.140700),
    "P":  (6.434500, 1.90670,  4.179100, 27.1570,  1.780000, 0.526000, 1.490800, 68.1645,  1.114900),
    "S":  (6.905300, 1.46790,  5.203400, 22.2151,  1.437900, 0.253600, 1.586300, 56.1720,  0.866900),
    "Cl": (11.46040, 0.010400, 7.196400, 1.16620,  6.255600, 18.5194,  1.645500, 47.7784, -9.557400),
    "K":  (8.218600, 12.7949,  7.439800, 0.774800, 1.051900, 213.187,  0.865900, 41.6841,  1.422800),
    "Ca": (8.626600, 10.4421,  7.387300, 0.659900, 1.589900, 85.7484,  1.021100, 178.437,  1.375100),
    "Ti": (9.759500, 7.85080,  7.355800, 0.500000, 1.699100, 35.6338,  1.902100, 116.105,  1.280700),
    "V":  (10.2971,  6.86570,  7.351100, 0.438500, 2.070300, 26.8938,  2.057100, 102.478,  1.219900),
    "Cr": (10.6406,  6.10380,  7.353700, 0.392000, 3.324000, 20.2626,  1.492200, 98.7399,  1.183200),
    "Mn": (11.2819,  5.34090,  7.357300, 0.343200, 3.019300, 17.8674,  2.244100, 83.7543,  1.089600),
    "Fe": (11.7695,  4.76110,  7.357300, 0.307200, 3.522200, 15.3535,  2.304500, 76.8805,  1.036900),
    "Co": (12.2841,  4.27910,  7.340900, 0.278400, 4.003400, 13.5359,  2.348800, 71.1692,  1.011800),
    "Ni": (12.8376,  3.87850,  7.292000, 0.256500, 4.443800, 12.1763,  2.380000, 66.3421,  1.034100),
    "Cu": (13.3380,  3.58280,  7.167600, 0.247000, 5.615800, 11.3966,  1.673500, 64.8126,  1.191000),
    "Zn": (14.0743,  3.26550,  7.031800, 0.233300, 5.165200, 10.3163,  2.410000, 58.7097,  1.304100),
    "Sn": (19.1889,  5.83030,  19.1005,  0.503100, 4.458500, 26.8909,  2.466300, 83.9571,  6.912700),
    "P1": (6.434500, 1.90670,  4.179100, 27.1570,  1.780000, 0.526000, 1.490800, 68.1645,  1.114900),
    # Lithium-ion battery relevant elements
    "Mn": (11.2819,  5.34090,  7.357300, 0.343200, 3.019300, 17.8674,  2.244100, 83.7543,  1.089600),
    "Co": (12.2841,  4.27910,  7.340900, 0.278400, 4.003400, 13.5359,  2.348800, 71.1692,  1.011800),
    "Ni": (12.8376,  3.87850,  7.292000, 0.256500, 4.443800, 12.1763,  2.380000, 66.3421,  1.034100),
    "Fe": (11.7695,  4.76110,  7.357300, 0.307200, 3.522200, 15.3535,  2.304500, 76.8805,  1.036900),
}

# Fallback: use carbon coefficients for unknown elements.
_FALLBACK_ELEMENT = "C"


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class BraggPeak:
    """One diffraction peak in the ideal XRD pattern.

    Attributes
    ----------
    hkl         : Miller indices (h, k, l)
    d_spacing   : interplanar spacing in Å
    two_theta   : diffraction angle 2θ in degrees
    intensity   : LP-corrected intensity (arbitrary units, normalised to 100)
    F_squared   : raw |F(hkl)|² before LP correction
    multiplicity: number of symmetry-equivalent planes (not yet implemented;
                  placeholder for future use with powder averaging)
    """
    hkl: tuple[int, int, int]
    d_spacing: float
    two_theta: float
    intensity: float
    F_squared: float
    multiplicity: int = 1

    def __repr__(self) -> str:
        h, k, l = self.hkl
        return (
            f"BraggPeak(hkl=({h},{k},{l}), "
            f"2θ={self.two_theta:.3f}°, "
            f"d={self.d_spacing:.4f}Å, "
            f"I={self.intensity:.1f})"
        )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_peaks(
    crystal: CrystalStructure,
    wavelength: float = CU_KALPHA,
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
    min_intensity: float = 0.1,
) -> list[BraggPeak]:
    """Compute the ideal XRD peak list for a crystal structure.

    Parameters
    ----------
    crystal       : CrystalStructure from cif_reader.py
    wavelength    : X-ray wavelength in Å (default: Cu Kα = 1.5406 Å)
    two_theta_min : lower 2θ limit in degrees (default: 5°)
    two_theta_max : upper 2θ limit in degrees (default: 90°)
    min_intensity : peaks below this fraction of the strongest peak are
                    discarded (default: 0.1, i.e. 0.1% of max intensity)

    Returns
    -------
    list[BraggPeak]
        Peaks sorted by 2θ angle, normalised so the strongest peak = 100.

    Examples
    --------
    >>> from cif_reader import fetch_structure
    >>> cs = fetch_structure("mp-24850")
    >>> peaks = compute_peaks(cs)
    >>> for p in peaks[:8]:
    ...     print(p)
    """
    logger.info(
        "Computing Bragg peaks for %s  (λ=%.4f Å, 2θ: %.1f–%.1f°)",
        crystal.formula, wavelength, two_theta_min, two_theta_max,
    )

    # Pre-compute the reciprocal lattice metric tensor G* for d-spacing
    G_star = _reciprocal_metric_tensor(crystal)

    peaks: list[BraggPeak] = []

    # Generate all (h, k, l) combinations, excluding (0, 0, 0)
    idx = range(-MAX_HKL_INDEX, MAX_HKL_INDEX + 1)
    for h in idx:
        for k in idx:
            for l in idx:
                if h == 0 and k == 0 and l == 0:
                    continue

                # --- d-spacing ---
                d = _d_spacing(h, k, l, G_star)
                if d is None:
                    continue

                # --- Bragg's law: sinθ = λ / (2d) ---
                sin_theta = wavelength / (2.0 * d)
                if sin_theta > 1.0:
                    continue   # physically impossible
                theta = np.arcsin(sin_theta)
                two_theta_deg = np.degrees(2.0 * theta)
                if not (two_theta_min <= two_theta_deg <= two_theta_max):
                    continue

                # --- Structure factor ---
                F = _structure_factor(h, k, l, crystal, sin_theta / wavelength)
                F_sq = abs(F) ** 2
                if F_sq < 1e-6:
                    continue   # systematically absent reflection

                # --- Lorentz-polarization correction ---
                # Applied here, after structure factor computation, because LP
                # is an instrumental/geometric effect independent of the crystal
                # chemistry. See module docstring for the full derivation.
                lp = _lorentz_polarization(theta)
                intensity = F_sq * lp

                peaks.append(BraggPeak(
                    hkl=(h, k, l),
                    d_spacing=d,
                    two_theta=two_theta_deg,
                    intensity=intensity,
                    F_squared=F_sq,
                ))

    if not peaks:
        logger.warning("No peaks found. Check wavelength and 2θ range.")
        return []

    # Sort by 2θ before merging
    peaks.sort(key=lambda p: p.two_theta)

    # Merge peaks at the same 2θ position (symmetry-equivalent reflections
    # that land at the same angle; sum their intensities)
    peaks = _merge_equivalent_peaks(peaks)

    # Normalise so the strongest peak = 100 (after merging, so multiplicity
    # is already accounted for)
    max_intensity = max(p.intensity for p in peaks)
    for p in peaks:
        p.intensity = (p.intensity / max_intensity) * 100.0

    # Filter weak peaks
    peaks = [p for p in peaks if p.intensity >= min_intensity]

    logger.info("Found %d peaks between %.1f° and %.1f°", len(peaks), two_theta_min, two_theta_max)
    return peaks


# ---------------------------------------------------------------------------
# Utility: convert peak list to DataFrame
# ---------------------------------------------------------------------------

def peaks_to_dataframe(peaks: list[BraggPeak]):
    """Convert a list of BraggPeak objects to a pandas DataFrame.

    Columns: h, k, l, d_spacing, two_theta, intensity, F_squared

    Requires pandas (pip install pandas).
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required: pip install pandas") from exc

    rows = [
        {
            "h": p.hkl[0], "k": p.hkl[1], "l": p.hkl[2],
            "d_spacing (Å)": round(p.d_spacing, 4),
            "2θ (°)":        round(p.two_theta, 3),
            "intensity":     round(p.intensity, 2),
            "|F|²":          round(p.F_squared, 2),
        }
        for p in peaks
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _reciprocal_metric_tensor(crystal: CrystalStructure) -> np.ndarray:
    """
    Compute the reciprocal-space metric tensor G* = (A^{-T}) · (A^{-1})
    where A is the lattice matrix (rows = lattice vectors).

    G* allows d-spacing calculation as:
        1/d² = [h k l] · G* · [h k l]^T
    """
    A = crystal.lattice_matrix
    A_inv = np.linalg.inv(A)
    return A_inv @ A_inv.T


def _d_spacing(h: int, k: int, l: int, G_star: np.ndarray) -> Optional[float]:
    """Return the d-spacing for reflection (hkl), or None if undefined."""
    hkl = np.array([h, k, l], dtype=float)
    inv_d_sq = hkl @ G_star @ hkl
    if inv_d_sq <= 0:
        return None
    return 1.0 / np.sqrt(inv_d_sq)


def _atomic_form_factor(element: str, s: float) -> float:
    """
    Evaluate the atomic form factor f(s) for an element at s = sinθ/λ.

    Uses the 4-Gaussian approximation from the International Tables.
    Falls back to carbon coefficients for elements not in the table.

    Parameters
    ----------
    element : element symbol, e.g. "Li", "Co", "O"
    s       : sinθ/λ in Å⁻¹
    """
    coeffs = FORM_FACTORS.get(element) or FORM_FACTORS.get(_FALLBACK_ELEMENT)
    a1, b1, a2, b2, a3, b3, a4, b4, c = coeffs
    s2 = s * s
    return (
        a1 * np.exp(-b1 * s2)
        + a2 * np.exp(-b2 * s2)
        + a3 * np.exp(-b3 * s2)
        + a4 * np.exp(-b4 * s2)
        + c
    )


def _structure_factor(
    h: int, k: int, l: int,
    crystal: CrystalStructure,
    s: float,
) -> complex:
    """
    Compute the structure factor F(hkl).

    F(hkl) = Σ_j  f_j(s) · occ_j · exp(2πi · (h·x_j + k·y_j + l·z_j))

    A reflection is systematically absent when |F|² ≈ 0, which happens
    automatically here due to destructive interference — no explicit
    extinction rules are needed.

    Parameters
    ----------
    h, k, l : Miller indices
    crystal : CrystalStructure (provides atom positions and elements)
    s       : sinθ/λ in Å⁻¹
    """
    F = complex(0.0, 0.0)
    for site in crystal.sites:
        x, y, z = site.frac_coords
        f = _atomic_form_factor(site.element, s)
        phase = 2.0 * np.pi * (h * x + k * y + l * z)
        F += f * site.occupancy * np.exp(1j * phase)
    return F


def _lorentz_polarization(theta: float) -> float:
    """
    Lorentz-polarization (LP) correction factor for a powder diffractometer
    with an unpolarized X-ray source.

    LP(θ) = (1 + cos²2θ) / (sin²θ · cosθ)

    This corrects for:
    - Lorentz factor 1/(sin²θ · cosθ): geometric time-of-intersection of the
      reciprocal lattice point with the Ewald sphere during a powder scan.
    - Polarization factor (1 + cos²2θ)/2: partial polarization of X-rays
      upon scattering from a crystal.

    Parameters
    ----------
    theta : Bragg angle θ in radians (NOT 2θ)

    Returns
    -------
    float : LP correction factor (larger at low and high angles)
    """
    two_theta = 2.0 * theta
    cos2t = np.cos(two_theta)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    numerator   = 1.0 + cos2t ** 2
    denominator = sin_t ** 2 * cos_t

    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _merge_equivalent_peaks(peaks: list[BraggPeak], tol: float = 0.01) -> list[BraggPeak]:
    """
    Merge reflections that fall at the same 2θ position within tolerance.

    When multiple (hkl) planes diffract at the same angle (e.g. (100) and
    (010) in a cubic crystal), their intensities add up in a powder pattern.
    The merged peak keeps the hkl of the first reflection found.

    Parameters
    ----------
    peaks : list of BraggPeak, sorted by two_theta
    tol   : merging tolerance in degrees (default: 0.01°)
    """
    if not peaks:
        return []

    merged: list[BraggPeak] = []
    current = peaks[0]

    for peak in peaks[1:]:
        if abs(peak.two_theta - current.two_theta) <= tol:
            # Same position: add intensities, keep current hkl
            current = BraggPeak(
                hkl=current.hkl,
                d_spacing=current.d_spacing,
                two_theta=current.two_theta,
                intensity=current.intensity + peak.intensity,
                F_squared=current.F_squared + peak.F_squared,
                multiplicity=current.multiplicity + 1,
            )
        else:
            merged.append(current)
            current = peak

    merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# Entry point — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")   # make sure cif_reader is importable

    from CIF_fetch import fetch_structure, summarise

    mp_id = "mp-554820"   
    print(f"Fetching {mp_id} …")
    cs = fetch_structure(mp_id)
    print(summarise(cs))
    print()

    peaks = compute_peaks(cs)
    print(f"Found {len(peaks)} peaks\n")
    print(f"{'hkl':<12} {'2θ (°)':<10} {'d (Å)':<10} {'Intensity':<10}")
    print("-" * 44)
    for p in peaks[:15]:
        h, k, l = p.hkl
        print(f"({h:2d}{k:2d}{l:2d})   {p.two_theta:<10.3f} {p.d_spacing:<10.4f} {p.intensity:<10.1f}")

    try:
        df = peaks_to_dataframe(peaks)
        print("\nDataFrame head:")
        print(df.head(10).to_string(index=False))
    except ImportError:
        print("\n(Install pandas to use peaks_to_dataframe)")