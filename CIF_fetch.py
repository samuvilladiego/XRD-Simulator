"""
cif_reader.py
=============
Fetches crystal structures from the Materials Project API and parses them
into a CrystalStructure dataclass ready for XRD simulation.

Public functions
----------------
    fetch_structure(formula_or_id)  →  CrystalStructure
        Download the conventional-standard structure for a formula or MP ID.

    load_cif(path)  →  CrystalStructure
        Parse a locally saved .cif file.

    summarise(crystal)  →  str
        Pretty-print a CrystalStructure.

Setup
-----
    pip install mp-api gemmi python-dotenv

    Create a .env file in your project folder containing:
        MP_API_KEY=your_key_here

    Get your key at https://next-gen.materialsproject.org/dashboard
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import gemmi
import numpy as np

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AtomSite:
    """One atom in the unit cell.

    Attributes
    ----------
    label     : unique site label, e.g. "Li1"
    element   : element symbol, e.g. "Li"
    frac_coords : fractional [x, y, z] coordinates
    occupancy : site occupancy (1.0 = fully occupied)
    """
    label: str
    element: str
    frac_coords: np.ndarray   # shape (3,)
    occupancy: float = 1.0


@dataclass
class CrystalStructure:
    """All crystallographic data needed for XRD simulation.

    Attributes
    ----------
    formula            : reduced chemical formula, e.g. "LiCoO2"
    material_id        : MP ID if fetched via API, else the CIF filename
    space_group_symbol : Hermann-Mauguin symbol, e.g. "R-3m"
    space_group_number : international space group number (1–230)
    a, b, c            : lattice parameters in Å
    alpha, beta, gamma : lattice angles in degrees
    lattice_matrix     : 3×3 Cartesian lattice vectors (rows)
    volume             : unit-cell volume in Å³
    sites              : all atoms in the conventional unit cell
    source_cif         : raw CIF text, kept for reproducibility
    """
    formula: str
    material_id: str
    space_group_symbol: str
    space_group_number: int
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    lattice_matrix: np.ndarray
    volume: float
    sites: list[AtomSite] = field(default_factory=list)
    source_cif: str = ""

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the conventional unit cell."""
        return len(self.sites)

    @property
    def elements(self) -> list[str]:
        """Sorted unique list of element symbols."""
        return sorted(set(s.element for s in self.sites))

    def __repr__(self) -> str:
        return (
            f"CrystalStructure({self.formula!r}, "
            f"SG #{self.space_group_number} {self.space_group_symbol!r}, "
            f"a={self.a:.4f} b={self.b:.4f} c={self.c:.4f} Å, "
            f"{self.n_atoms} atoms)"
        )


# ---------------------------------------------------------------------------
# Curated energy materials
# ---------------------------------------------------------------------------

# Maps common formula names to stable MP IDs so you get a specific polymorph
# rather than whatever the API considers most stable on a given day.
ENERGY_MATERIALS: dict[str, str] = {
    # Cathodes
    "LiCoO2":    "mp-24850",   # layered oxide, R-3m
    "LiFePO4":   "mp-19017",   # olivine, Pnma
    "LiMn2O4":   "mp-25015",   # spinel, Fd-3m
    "LiNiO2":    "mp-25582",   # layered oxide, R-3m
    # Anodes
    "TiO2":      "mp-2657",    # anatase, I41/amd
    "SnO2":      "mp-856",     # rutile, P42/mnm
    "Li4Ti5O12": "mp-776280",  # spinel, Fd-3m
    # Other
    "Li2O":      "mp-1960",
    "Li3PO4":    "mp-13725",
    "V2O5":      "mp-25279",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_structure(
    formula_or_id: str,
    api_key: Optional[str] = None,
    *,
    save_cif: Optional[Path | str] = None,
) -> CrystalStructure:
    """Download a crystal structure from the Materials Project.

    Parameters
    ----------
    formula_or_id : str
        A chemical formula ("LiCoO2") or MP ID ("mp-24850").
        Formulas are looked up in ENERGY_MATERIALS first; unknown formulas
        are searched directly and the lowest-energy result is used.
    api_key : str, optional
        MP API key. If omitted, reads the MP_API_KEY environment variable
        (set via a .env file or your shell).
    save_cif : path, optional
        If given, saves the downloaded CIF text to this path for offline use.

    Returns
    -------
    CrystalStructure

    Examples
    --------
    >>> cs = fetch_structure("LiCoO2")
    >>> cs = fetch_structure("mp-24850")
    >>> cs = fetch_structure("LiFePO4", save_cif="data/LiFePO4.cif")
    """
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise ImportError(
            "mp-api is not installed. Run:  pip install mp-api"
        ) from exc

    resolved_key = api_key or os.environ.get("MP_API_KEY")
    if not resolved_key:
        raise EnvironmentError(
            "No API key found. Add MP_API_KEY=your_key to your .env file\n"
            "or pass api_key=... directly to fetch_structure().\n"
            "Get your key at https://next-gen.materialsproject.org/dashboard"
        )

    # Resolve formula → MP ID if needed
    mp_id = ENERGY_MATERIALS.get(formula_or_id, formula_or_id)

    with MPRester(resolved_key) as mpr:

        # Unknown formula: search and take the most stable result
        if not mp_id.startswith("mp-"):
            docs = mpr.materials.summary.search(
                formula=mp_id,
                fields=["material_id", "formula_pretty", "energy_above_hull"],
            )
            if not docs:
                raise ValueError(
                    f"No material found for {formula_or_id!r}. "
                    "Try using the MP ID directly, e.g. 'mp-24850'."
                )
            docs.sort(key=lambda d: d.energy_above_hull)
            mp_id = str(docs[0].material_id)
            logger.info("Resolved %r → %s", formula_or_id, mp_id)

        # Fetch structure
        logger.info("Fetching %s from Materials Project …", mp_id)
        docs = mpr.materials.summary.search(
            material_ids=[mp_id],
            fields=["structure"],
        )
        if not docs:
            raise ValueError(f"No structure found for {mp_id!r}.")

        structure = docs[0].structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(structure)
        structure = sga.get_conventional_standard_structure()
        sg_symbol = sga.get_space_group_symbol()
        sg_number = sga.get_space_group_number()
        cif_text = structure.to(fmt="cif")


        if save_cif is not None:
            Path(save_cif).write_text(cif_text, encoding="utf-8")
            logger.info("CIF saved to %s", save_cif)

    crystal = _parse_cif_text(cif_text, material_id=mp_id,
                                  sg_symbol_override=sg_symbol,
                                  sg_number_override=sg_number)
    logger.info("Done: %s", crystal)
    return crystal


def load_cif(path: Path | str) -> CrystalStructure:
    """Parse a local .cif file and return a CrystalStructure.

    Parameters
    ----------
    path : path to a .cif file on disk

    Examples
    --------
    >>> cs = load_cif("data/LiCoO2_mp-24850.cif")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {path}")
    cif_text = path.read_text(encoding="utf-8")
    crystal = _parse_cif_text(cif_text, material_id=path.stem)
    logger.info("Loaded: %s", crystal)
    return crystal


def summarise(crystal: CrystalStructure) -> str:
    """Return a formatted summary of a CrystalStructure."""
    el_list = " ".join(crystal.elements)
    lines = [
        f"Formula      : {crystal.formula}  ({crystal.material_id})",
        f"Space group  : {crystal.space_group_symbol}  (#{crystal.space_group_number})",
        f"Lattice (Å)  : a={crystal.a:.4f}  b={crystal.b:.4f}  c={crystal.c:.4f}",
        f"Angles  (°)  : α={crystal.alpha:.2f}  β={crystal.beta:.2f}  γ={crystal.gamma:.2f}",
        f"Volume       : {crystal.volume:.2f} Å³",
        f"Atoms        : {crystal.n_atoms}  ({el_list})",
    ]
    width = max(len(l) for l in lines) + 4
    border = "═" * width
    header = f"  {crystal.formula}  ({crystal.material_id})"
    out = [f"╔{border}╗", f"║{header:<{width}}║", f"╠{border}╣"]
    for l in lines:
        out.append(f"║  {l:<{width - 2}}║")
    out.append(f"╚{border}╝")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Internal parser
# ---------------------------------------------------------------------------

def _parse_cif_text(cif_text: str, material_id: str = "",
                    sg_symbol_override: str = None,
                    sg_number_override: int = None) -> CrystalStructure:
    """Parse CIF text with gemmi and return a CrystalStructure."""
    doc = gemmi.cif.read_string(cif_text)
    block = doc.sole_block() if len(doc) == 1 else doc[0]

    def _float(tag: str) -> float:
        val = block.find_value(tag)
        if val is None:
            raise KeyError(f"CIF tag {tag!r} not found")
        return float(str(val).split("(")[0].strip())

    a     = _float("_cell_length_a")
    b     = _float("_cell_length_b")
    c     = _float("_cell_length_c")
    alpha = _float("_cell_angle_alpha")
    beta  = _float("_cell_angle_beta")
    gamma = _float("_cell_angle_gamma")

    sg_symbol = sg_symbol_override or _read_tag(block, [
        "_symmetry_space_group_name_H-M",
        "_space_group_name_H-M_alt",
        "_symmetry_space_group_name_H_M",
    ], default="P 1").strip().strip("'\"")

    if sg_number_override:
        sg_number = sg_number_override
    else:
        sg_number_raw = _read_tag(block, [
            "_symmetry_Int_Tables_number",
            "_space_group_IT_number",
        ], default=None)
        if sg_number_raw is not None:
            sg_number = int(str(sg_number_raw).strip())
        else:
            sg = gemmi.find_spacegroup_by_name(sg_symbol)
            sg_number = sg.number if sg else 1

    formula = _read_tag(block, [
        "_chemical_formula_sum",
        "_chemical_formula_reduced",
        "_chemical_formula_moiety",
    ], default="Unknown").strip().strip("'\"")
    formula = _reduce_formula(_compact_formula(formula))

    st = gemmi.make_small_structure_from_block(block)
    sites = _extract_sites_from_small_structure(st) or _extract_sites_raw(block)

    lattice_matrix = _build_lattice_matrix(a, b, c, alpha, beta, gamma)
    volume = abs(np.linalg.det(lattice_matrix))

    return CrystalStructure(
        formula=formula,
        material_id=material_id,
        space_group_symbol=sg_symbol,
        space_group_number=sg_number,
        a=a, b=b, c=c,
        alpha=alpha, beta=beta, gamma=gamma,
        lattice_matrix=lattice_matrix,
        volume=volume,
        sites=sites,
        source_cif=cif_text,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_tag(block: gemmi.cif.Block, tags: list[str], default=None):
    for tag in tags:
        val = block.find_value(tag)
        if val is not None:
            return val
    return default


def _compact_formula(formula: str) -> str:
    """Turn 'Li 1 Co 1 O 2' into 'LiCoO2'."""
    import re
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    result = ""
    for symbol, count in tokens:
        if not symbol:
            continue
        result += symbol
        if count and count != "1":
            result += count
    return result or formula.strip()

from math import gcd
from functools import reduce

def _reduce_formula(formula: str) -> str:
    """Reduce Ti4O8 → TiO2 by dividing all counts by their GCD."""
    import re
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    tokens = [(s, n) for s, n in tokens if s]
    counts = [int(n) if n else 1 for _, n in tokens]
    divisor = reduce(gcd, counts)
    result = ""
    for symbol, count in tokens:
        n = (int(count) if count else 1) // divisor
        result += symbol + (str(n) if n > 1 else "")
    return result


def _extract_sites_from_small_structure(st: gemmi.SmallStructure) -> list[AtomSite]:
    sites = []
    for i, site in enumerate(st.sites):
        element = site.type_symbol.strip() or "X"
        label = site.label.strip() or f"{element}{i}"
        frac = np.array([site.fract.x, site.fract.y, site.fract.z], dtype=float)
        occ = float(site.occ) if site.occ else 1.0
        sites.append(AtomSite(label=label, element=element, frac_coords=frac, occupancy=occ))
    return sites


def _extract_sites_raw(block: gemmi.cif.Block) -> list[AtomSite]:
    """Fallback: read atom positions directly without symmetry expansion."""
    sites = []
    try:
        table = block.find(["_atom_site_"])
    except Exception:
        logger.warning("Could not find _atom_site_ loop in block %s", block.name)
        return sites
    for row in table:
        try:
            label = str(row[0]).strip()
            element = label[0].upper() + (label[1].lower() if len(label) > 1 and label[1].isalpha() else "")
            x = float(str(row[1]).split("(")[0])
            y = float(str(row[2]).split("(")[0])
            z = float(str(row[3]).split("(")[0])
            sites.append(AtomSite(label=label, element=element, frac_coords=np.array([x, y, z])))
        except Exception as exc:
            logger.debug("Skipping malformed atom_site row: %s", exc)
    return sites


def _build_lattice_matrix(
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
) -> np.ndarray:
    """Convert unit-cell parameters to a 3×3 Cartesian lattice matrix.

    Convention: a along x, b in the xy plane, c completes the right-hand system.
    Same convention as pymatgen and VESTA.
    """
    α, β, γ = np.radians(alpha), np.radians(beta), np.radians(gamma)
    cos_α, cos_β, cos_γ = np.cos(α), np.cos(β), np.cos(γ)
    sin_γ = np.sin(γ)
    cx = cos_β
    cy = (cos_α - cos_β * cos_γ) / sin_γ
    cz = np.sqrt(max(1.0 - cx**2 - cy**2, 0.0))
    return np.array([
        [a,          0.0,       0.0],
        [b * cos_γ,  b * sin_γ, 0.0],
        [c * cx,     c * cy,    c * cz],
    ], dtype=float)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cs = fetch_structure("mp-554820")
    print(summarise(cs))