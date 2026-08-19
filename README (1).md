# xrd-simulator

Simulate X-ray diffraction (XRD) patterns for energy materials directly from
crystal structures fetched via the [Materials Project API](https://materialsproject.org).

Produces both an **ideal** pattern (sharp delta-function peaks) and a
**realistic** pattern (Scherrer-broadened peaks with Poisson noise and
background), so you can see what a real diffractometer would measure.

## Project status

| Module | Status |
|---|---|
| `cif_reader` — fetch & parse CIF files | ✅ done |
| `bragg` — 2θ positions + structure factors | 🔲 next |
| `broadening` — Scherrer / Voigt peak shapes | 🔲 planned |
| `noise` — Poisson noise + background | 🔲 planned |
| `plot` — matplotlib output + CSV export | 🔲 planned |
| `cli` — command-line interface | 🔲 planned |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/xrd-simulator.git
cd xrd-simulator
pip install -e ".[dev]"
```

## Getting a Materials Project API key

1. Create a free account at <https://materialsproject.org>
2. Go to <https://next-gen.materialsproject.org/dashboard>
3. Copy your API key

Store it as an environment variable so you never paste it in code:

```bash
# Linux / macOS
export MP_API_KEY="your_key_here"

# Windows (Command Prompt)
set MP_API_KEY=your_key_here
```

---

## Usage

### Fetch a structure and inspect it

```python
from xrd_simulator.cif_reader import fetch_structure, summarise

# Use a formula — picks the lowest-energy stable polymorph automatically
cs = fetch_structure("LiCoO2")          # reads MP_API_KEY from environment
print(summarise(cs))

# Or use an explicit Materials Project ID for reproducibility
cs = fetch_structure("mp-24850")
```

Example output:
```
╔════════════════════════════════════════════════╗
║  LiCoO2  (mp-24850)                           ║
╠════════════════════════════════════════════════╣
║  Formula      : LiCoO2  (mp-24850)            ║
║  Space group  : R-3m  (#166)                  ║
║  Lattice (Å)  : a=2.8162  b=2.8162  c=14.0804 ║
║  Angles  (°)  : α=90.00  β=90.00  γ=120.00   ║
║  Volume       : 96.49 Å³                       ║
║  Atoms        : 4  (Co Li O)                  ║
╚════════════════════════════════════════════════╝
```

### Save the CIF file for offline use

```python
cs = fetch_structure("LiFePO4", save_cif="data/LiFePO4_mp-19017.cif")
```

### Load a locally saved CIF file

```python
from xrd_simulator.cif_reader import load_cif

cs = load_cif("data/LiFePO4_mp-19017.cif")
```

### Supported energy materials (built-in shortcuts)

| Formula | MP ID | Description |
|---|---|---|
| LiCoO2 | mp-24850 | Layered oxide cathode (R-3m) |
| LiFePO4 | mp-19017 | Olivine cathode (Pnma) |
| LiMn2O4 | mp-25015 | Spinel cathode (Fd-3m) |
| LiNiO2 | mp-25582 | Layered oxide cathode (R-3m) |
| TiO2 | mp-2657 | Anatase anode (I41/amd) |
| SnO2 | mp-856 | Rutile anode (P42/mnm) |
| Li4Ti5O12 | mp-776280 | Spinel anode (Fd-3m) |
| V2O5 | mp-25279 | Cathode / supercapacitor |

---

## Running the tests

```bash
# Fast unit tests (no API key needed)
pytest

# Including live API integration tests
pytest -m integration --mp-key YOUR_KEY
```

---

## Project structure

```
xrd-simulator/
├── xrd_simulator/
│   ├── __init__.py
│   ├── cif_reader.py     ← fetch & parse crystal structures
│   ├── bragg.py          ← Bragg peaks and structure factors (TODO)
│   ├── broadening.py     ← peak broadening models (TODO)
│   ├── noise.py          ← noise and background (TODO)
│   ├── pattern.py        ← assemble ideal/realistic patterns (TODO)
│   └── plot.py           ← matplotlib + CSV export (TODO)
├── tests/
│   └── test_cif_reader.py
├── data/                 ← put downloaded .cif files here
├── pyproject.toml
└── README.md
```

---

## Physics background

XRD works by shining X-rays on a crystalline sample.  The lattice planes
(characterised by Miller indices *hkl*) diffract the beam when the angle
satisfies **Bragg's law**: λ = 2*d*·sin*θ*, where *d* is the interplanar
spacing and λ is the X-ray wavelength (typically Cu Kα = 1.5406 Å).

Peak intensity depends on the **structure factor** F(hkl), which is the
coherent sum of scattering from every atom in the unit cell, weighted by
their **atomic form factors** *f*(sinθ/λ).

Real peaks are not perfectly sharp: finite crystallite size causes
**Scherrer broadening** (width ∝ 1/crystallite size), and non-uniform
strain causes additional asymmetric broadening (modelled with the
Williamson–Hall method).  Shot noise in the detector follows a **Poisson
distribution**, and there is a slowly varying background from fluorescence
and air scatter.

---

## License

MIT
