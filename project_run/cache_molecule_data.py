import pathlib
import sys

import astropy.units as u
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from project_classes.Molecule import Molecule
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES
DEFAULT_LOCAL_DATABASE = "exomol_data"

WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
B_MOLECULE = 1 * u.km / u.s
PROFILE_TYPE = "Voigt"
SELECTED_MOLECULES = ["CO"]
TEST_STAR_KEY = "A0"
TEST_T_ATM = 1000 * u.K
TEST_DISTANCE = 1.0 * u.AU
TEST_NCOLS = np.array([0.0]) / u.cm**2

def get_test_star():
    if TEST_STAR_KEY not in STAR_TEMPLATES:
        raise KeyError(f"Unknown test star '{TEST_STAR_KEY}'. Available templates: {list(STAR_TEMPLATES)}")

    s = STAR_TEMPLATES[TEST_STAR_KEY]
    return Star(
        s["path"],
        s["radius"],
        s["mass"],
        vsini=s["vsini"],
        epsilon=s["epsilon"],
    )

def main():
    print("Starting molecule fetch-only cache script")
    print(f"Default local database folder: {DEFAULT_LOCAL_DATABASE}")
    test_star = get_test_star()
    print(f"Using test star: {TEST_STAR_KEY}")
    print(f"Test atmospheric temperature: {TEST_T_ATM}")
    print(f"Test distance: {TEST_DISTANCE}")

    for species in SELECTED_MOLECULES:
        if species not in MOLECULE_TEMPLATES:
            raise KeyError(f"Unknown molecule template '{species}'. Available templates: {list(MOLECULE_TEMPLATES)}")
        config = MOLECULE_TEMPLATES[species]

        source = config["source"]
        fetch_kwargs = dict(config["fetch_kwargs"])
        localdatabase = fetch_kwargs.get("localdatabase", DEFAULT_LOCAL_DATABASE)

        print("-" * 70)
        print(f"Fetching species: {species}")
        print(f"Source: {source}")
        print(f"Path: {fetch_kwargs.get('path')}")
        print(f"Database: {fetch_kwargs.get('database')}")
        print(f"Local database folder: {localdatabase}")

        try:
            mol = Molecule(species, WAVEMIN, WAVEMAX)

            if source == "exomol":
                mol.fetch_exomol(**fetch_kwargs)
            elif source == "hitran":
                isotope = fetch_kwargs.get("isotope", 1)
                localdatabase = fetch_kwargs.get("localdatabase", DEFAULT_LOCAL_DATABASE)
                local_path = fetch_kwargs.get("path", None)
                databank_name = fetch_kwargs.get("databank_name", f"HITRAN-{species}")
                mol.fetch_hitran(
                    molecule_name=species,
                    isotope=isotope,
                    localdatabase=localdatabase,
                    path=local_path,
                    databank_name=databank_name,
                )
            else:
                raise ValueError(f"Unknown molecule source: {source}")

            print(f"Finished fetching {species}")
            print(f"Building BroadeningProfileMolecule for {species}")
            profile = BroadeningProfileMolecule(
                mol,
                B_MOLECULE,
                profileType=PROFILE_TYPE,
            )
            print(f"Finished BroadeningProfileMolecule setup for {species}")

            print(f"Building PhotonPressure object for {species}")
            pp = PhotonPressure(profile, test_star)
            print(f"Calculating photon pressure for {species}")
            F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure_molecule(
                TEST_NCOLS,
                TEST_T_ATM,
                TEST_DISTANCE,
                chunk_size=1,
                lam_chunk_size=20000,
            )
            print(f"Finished photon pressure calculation for {species}")
            print(f"F_ph_tot = {F_ph_tot}")
            print(f"F_ph_tot_err = {F_ph_tot_err}")
            _ = profile
            _ = pp
        except Exception as exc:
            print(f"Failed to fetch {species}: {exc}")


if __name__ == "__main__":
    main()
