import pathlib
import sys

import astropy.units as u

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from project_classes.Molecule import Molecule
from project_func.Templates.Molecules.molecules_template import MOLECULE_TEMPLATES
DEFAULT_LOCAL_DATABASE = "exomol_data"

WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
SELECTED_MOLECULES = ["CO", "O2", "SiO"]

def main():
    print("Starting molecule fetch-only cache script")
    print(f"Default local database folder: {DEFAULT_LOCAL_DATABASE}")

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
        except Exception as exc:
            print(f"Failed to fetch {species}: {exc}")


if __name__ == "__main__":
    main()
