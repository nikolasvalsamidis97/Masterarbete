import pathlib
import sys

import astropy.units as u

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from project_classes.Molecule import Molecule
DEFAULT_LOCAL_DATABASE = "exomol_data"

WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA

molecule_configs = {
    "CO2": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "CO2/12C-16O2/Dozen",
            "database": "Dozen",
            "localdatabase": "exomol_data",
        },
    },
}

def main():
    print("Starting molecule fetch-only cache script")
    print(f"Default local database folder: {DEFAULT_LOCAL_DATABASE}")

    for species, config in molecule_configs.items():
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
            else:
                raise ValueError(f"Unknown molecule source: {source}")

            print(f"Finished fetching {species}")
        except Exception as exc:
            print(f"Failed to fetch {species}: {exc}")


if __name__ == "__main__":
    main()
