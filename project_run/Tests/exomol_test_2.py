from astropy import units as u
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Molecule import Molecule


WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA

MOLECULE_TEMPLATES = {
    "CO": {
        "source": "exomol",
        "fetch_kwargs": {
            "path": "CO/12C-16O/Li2015",
            "database": "Li2015",
            "localdatabase": "exomol_data",
        },
    },
    "O2": {
        "source": "hitran",
        "fetch_kwargs": {
            "isotope": 1,
        },
    },
}


def test_fetch(species: str) -> None:
    if species not in MOLECULE_TEMPLATES:
        raise KeyError(f"Unknown molecule template '{species}'. Available: {list(MOLECULE_TEMPLATES)}")

    config = MOLECULE_TEMPLATES[species]
    source = config["source"]
    fetch_kwargs = dict(config["fetch_kwargs"])

    mol = Molecule(species, WAVEMIN, WAVEMAX)

    print("-" * 60)
    print(f"Testing molecule: {species}")
    print(f"Source: {source}")
    print(f"Fetch kwargs: {fetch_kwargs}")

    if source == "exomol":
        mol.fetch_exomol(**fetch_kwargs)
        print(f"ExoMol fetch succeeded for {species}")
    elif source == "hitran":
        isotope = fetch_kwargs.get("isotope", 1)
        df = mol.fetch_hitran(species, isotope=isotope)
        print(df.head())
        print(f"HITRAN fetch succeeded for {species}; rows = {len(df)}")
        mol.pandas_to_numpy()
        print(f"HITRAN numpy conversion succeeded for {species}")
    else:
        raise ValueError(f"Unknown source '{source}' for molecule '{species}'")


if __name__ == "__main__":
    test_fetch("O2")
