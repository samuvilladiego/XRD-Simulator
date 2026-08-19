# XRD Simulator

A Python-based X-ray diffraction pattern simulator for energy materials. Crystal structures are fetched directly from the Materials Project database and used to compute diffraction patterns with physically rigorous peak modelling.

The simulator produces both an **ideal pattern** (sharp stick peaks) and a **realistic pattern** (broadened continuous curve) with user-selectable peak shape and crystallite size.

---

## Physics

XRD works by shining X-rays on a crystalline powder. Lattice planes with spacing *d* diffract the beam at angles satisfying **Bragg's law**:

```
λ = 2d·sinθ
```

Peak intensity depends on the **structure factor** F(hkl) — the coherent sum of scattering from every atom in the unit cell, weighted by atomic form factors. Systematic absences (reflections forbidden by space group symmetry) emerge naturally from destructive interference and are not hardcoded.

A **Lorentz-polarization correction** is applied to each peak to account for the geometry of a powder diffractometer and the partial polarization of scattered X-rays.

Real peaks are broadened by finite crystallite size. The **Scherrer equation** gives the FWHM at each angle:

```
FWHM = K·λ / (L·cosθ)
```

where L is the crystallite size and K = 0.9394 for spherical grains.

---

## Project structure

```
XRD_Simulator/
├── CIF_fetch.py      — fetches crystal structures from the Materials Project API
├── bragg.py          — computes 2θ positions, structure factors, and LP-corrected intensities
├── broadening.py     — convolves stick peaks with a peak shape and plots the result
├── .env              — stores your API key (never uploaded to GitHub)
├── .gitignore
└── README.md
```

Each file is a self-contained module. `broadening.py` is the entry point for the full pipeline — it imports `bragg.py`, which imports `CIF_fetch.py`.

---

## Supported materials

| Formula | MP ID | Structure | Role |
|---|---|---|---|
| LiCoO₂ | mp-24850 | Layered oxide (R-3m) | Cathode |
| LiFePO₄ | mp-19017 | Olivine (Pnma) | Cathode |
| LiMn₂O₄ | mp-25015 | Spinel (Fd-3m) | Cathode |
| LiNiO₂ | mp-25582 | Layered oxide (R-3m) | Cathode |
| TiO₂ | mp-390 | Anatase (I4₁/amd) | Anode |
| SnO₂ | mp-856 | Rutile (P4₂/mnm) | Anode |
| Li₄Ti₅O₁₂ | mp-776280 | Spinel (Fd-3m) | Anode |
| V₂O₅ | mp-25279 | Layered (Pmmn) | Cathode / supercapacitor |
| Li₂O | mp-1960 | Rock-salt (Fm-3m) | Degradation product |
| Li₃PO₄ | mp-13725 | Orthorhombic (Pnma) | Solid electrolyte |

Any other material on the Materials Project can be used by entering its MP ID directly.

---

## Installation

```powershell
pip install mp-api gemmi numpy scipy matplotlib python-dotenv
```

---

## API key setup

1. Create a free account at [materialsproject.org](https://materialsproject.org)
2. Copy your API key from [next-gen.materialsproject.org/dashboard](https://next-gen.materialsproject.org/dashboard)
3. Create a file called `.env` in the project folder containing:

```
MP_API_KEY=your_key_here
```

The key is read automatically at runtime. It is listed in `.gitignore` and never uploaded to GitHub.

---

## Usage

Run the full pipeline from `broadening.py`:

```powershell
python broadening.py
```

You will be prompted for:

```
Enter MP ID (e.g. mp-390):
Peak shape — gaussian / lorentzian / pseudo-voigt:
Crystallite size in nm:
Pseudo-Voigt eta (0=Gaussian, 1=Lorentzian):
```

Press Enter to accept the defaults. The output is a matplotlib plot showing the broadened pattern (blue) with ideal stick peaks (red) and hkl labels on the strongest reflections.

To inspect the raw peak list without plotting, run:

```powershell
python bragg.py
```

---

## Peak shape models

All three models share the same FWHM from the Scherrer equation. The difference is in the shape of the profile:

| Model | Description |
|---|---|
| Gaussian | Symmetric bell curve. Simple but underestimates peak tails. |
| Lorentzian | Heavier tails. Models instrumental broadening more accurately. |
| Pseudo-Voigt | Weighted sum: η·Lorentzian + (1−η)·Gaussian. Standard in powder diffraction; η ≈ 0.5–0.7 matches most lab diffractometers. |

---

## Example output

Running the simulator on TiO₂ anatase (mp-390) with pseudo-Voigt broadening and L = 50 nm produces a pattern consistent with the ICDD reference card for anatase, with the strongest reflection at 2θ ≈ 25.3° corresponding to the (101) plane.

---

## Roadmap

- [ ] Noise model (Poisson counting statistics + background)
- [ ] CSV / JSON export of peak data
- [ ] Validation table against ICDD reference patterns
- [ ] Command-line interface (main.py)
- [ ] Multi-material overlay plots

---

## Dependencies

| Package | Purpose |
|---|---|
| mp-api | Materials Project API client |
| gemmi | CIF parsing and crystallography |
| numpy | Numerical computation |
| scipy | Signal processing |
| matplotlib | Plotting |
| python-dotenv | .env file handling |

---

## Author

Samuel Villadiego — Energy Systems Engineering, Universidad del Rosario
