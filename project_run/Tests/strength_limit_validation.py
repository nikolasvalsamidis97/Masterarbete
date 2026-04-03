import pathlib
import sys

import astropy.units as u
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Molecule import Molecule
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES

WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
B_MOLECULE = 1 * u.km / u.s
PROFILE_TYPE = "Voigt"

TEST_STAR_KEY = "M9"
TEST_T_ATM = 1 * u.K
TEST_DISTANCE = 1.0 * u.AU
TEST_NCOLS = np.array([0.0]) / u.cm**2

MOLECULES_TO_TEST = ["NO"]
STRENGTH_LIMIT = 1e-8

OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "Tables" / "cutoff_validation"
OUTPUT_FILE = OUTPUT_DIR / f"{MOLECULES_TO_TEST[0]} strength_limit_validation.txt"


def get_test_star():
    s = STAR_TEMPLATES[TEST_STAR_KEY]
    return Star(
        s["path"],
        s["radius"],
        s["mass"],
        vsini=s["vsini"],
        epsilon=s["epsilon"],
    )


def build_molecule(species: str):
    cfg = MOLECULE_TEMPLATES[species]
    mol = Molecule(species, WAVEMIN, WAVEMAX)

    if cfg["source"] == "exomol":
        mol.fetch_exomol(**cfg["fetch_kwargs"])
    elif cfg["source"] == "hitran":
        fetch_kwargs = dict(cfg["fetch_kwargs"])
        mol.fetch_hitran(
            molecule_name=species,
            isotope=fetch_kwargs.get("isotope", 1),
            localdatabase=fetch_kwargs.get("localdatabase"),
            path=fetch_kwargs.get("path"),
            databank_name=fetch_kwargs.get("databank_name", f"HITRAN-{species}"),
        )
    else:
        raise ValueError(f"Unknown source for {species}: {cfg['source']}")

    return mol


def compute_photon_pressure(species: str, strength_limit=None):
    mol = build_molecule(species)

    profile = BroadeningProfileMolecule(
        mol,
        B_MOLECULE,
        profileType=PROFILE_TYPE,
    )

    if strength_limit is None:
        profile.temp_strength_rel_cutoff = 0.0
    else:
        profile.temp_strength_rel_cutoff = float(strength_limit)

    star = get_test_star()
    pp = PhotonPressure(profile, star)

    F_ph_tot, _, _, _ = pp.calc_PhotonPressure_molecule(
        TEST_NCOLS,
        TEST_T_ATM,
        TEST_DISTANCE,
        chunk_size=1,
        lam_chunk_size=100000,
        verbose=False,
    )

    return float(F_ph_tot[0, 0].to_value(u.N))


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for species in MOLECULES_TO_TEST:
        print(species)
        f.write(f"{species}\n")

        print("cutoff = 0.0")
        f.write("cutoff = 0.0\n")
        f_all = compute_photon_pressure(species, strength_limit=0.0)
        print(f_all)
        f.write(f"{f_all}\n")

        print(f"cutoff = {STRENGTH_LIMIT}")
        f.write(f"cutoff = {STRENGTH_LIMIT}\n")
        f_limited = compute_photon_pressure(species, strength_limit=STRENGTH_LIMIT)
        print(f_limited)
        f.write(f"{f_limited}\n")

        abs_diff = abs(f_all - f_limited)
        rel_diff = abs_diff / abs(f_all) if f_all != 0.0 else np.nan

        print("abs diff")
        print(abs_diff)
        print("rel diff")
        print(rel_diff)
        print()

        f.write("abs diff\n")
        f.write(f"{abs_diff}\n")
        f.write("rel diff\n")
        f.write(f"{rel_diff}\n\n")

print(f"Saved results to {OUTPUT_FILE}")