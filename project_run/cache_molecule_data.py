import pathlib
import sys
import time

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
# SELECTED_MOLECULES = ["CO", "NO", "SO", "SiO", "CO2", "H2O"]
SELECTED_MOLECULES = ["CO"]
TEST_STAR_KEY = "A0"
TEST_T_ATM = 1000 * u.K
TEST_DISTANCE = 1.0 * u.AU
TEST_NCOLS = np.array([0.0]) / u.cm**2

from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_WORKERS = min(4, len(SELECTED_MOLECULES))

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


def cache_one_species(species: str):
    t0 = time.perf_counter()

    test_star = get_test_star()

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
        print(f"[{species}] Stage 1/5: creating Molecule object")
        mol = Molecule(species, WAVEMIN, WAVEMAX)
        print(f"[{species}] Stage 1/5 done: Molecule object created")

        print(f"[{species}] Stage 2/5: starting fetch from {source}")
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
        print(f"[{species}] Stage 2/5 done: fetch complete")

        print(f"[{species}] Stage 3/5: building BroadeningProfileMolecule")
        profile = BroadeningProfileMolecule(
            mol,
            B_MOLECULE,
            profileType=PROFILE_TYPE,
        )
        print(f"[{species}] Stage 3/5 done: BroadeningProfileMolecule ready")

        print(f"[{species}] Stage 4/5: building PhotonPressure object")
        pp = PhotonPressure(profile, test_star)
        print(f"[{species}] Stage 4/5 done: PhotonPressure object ready")
        print(f"[{species}] Stage 5/5: calculating photon pressure")
        F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure_molecule(
            TEST_NCOLS,
            TEST_T_ATM,
            TEST_DISTANCE,
            chunk_size=1,
            lam_chunk_size=100000,
            verbose=True,
        )
        print(f"[{species}] Stage 5/5 done: photon pressure calculation finished")
        print(f"[{species}] F_ph_tot = {F_ph_tot}")
        print(f"[{species}] F_ph_tot_err = {F_ph_tot_err}")
        _ = profile
        _ = pp
        elapsed = time.perf_counter() - t0
        return {"species": species, "ok": True, "error": None, "elapsed_s": elapsed}
    except Exception as exc:
        print(f"[{species}] FAILED during pipeline: {exc}")
        elapsed = time.perf_counter() - t0
        return {"species": species, "ok": False, "error": str(exc), "elapsed_s": elapsed}


def main():
    print("Starting molecule fetch-only cache script")
    print(f"Default local database folder: {DEFAULT_LOCAL_DATABASE}")
    print(f"Using test star: {TEST_STAR_KEY}")
    print(f"Test atmospheric temperature: {TEST_T_ATM}")
    print(f"Test distance: {TEST_DISTANCE}")
    print(f"Selected molecules: {SELECTED_MOLECULES}")
    print(f"Max workers: {MAX_WORKERS}")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(cache_one_species, species): species for species in SELECTED_MOLECULES}
        for future in as_completed(future_map):
            species = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"species": species, "ok": False, "error": str(exc), "elapsed_s": None}
            results.append(result)

    print("-" * 70)
    print("Summary")
    total_elapsed = 0.0
    for result in results:
        elapsed_s = result.get("elapsed_s", None)
        elapsed_text = "unknown" if elapsed_s is None else f"{elapsed_s:.3f} s"
        if elapsed_s is not None:
            total_elapsed += elapsed_s

        if result["ok"]:
            print(f"  {result['species']}: OK | compute time = {elapsed_text}")
        else:
            print(f"  {result['species']}: FAILED -> {result['error']} | compute time = {elapsed_text}")

    print(f"Total summed compute time across molecules: {total_elapsed:.3f} s")


if __name__ == "__main__":
    main()
