

from astropy import units as u
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from matplotlib import pyplot as plt
import astropy.constants as const
from project_classes.Atom import Atom
from project_classes.Molecule import Molecule
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star

wavemin = 150 * u.AA
wavemax = 50000 * u.AA
Npts = 150
b = 1 * u.km / u.s
Ncol = np.array([0.0]) * u.cm**-2
T_mol = 10 * u.K
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Lehtmets"

neutral_species = {
  "Neon": "Ne I",
  "Sodium": "Na I",
  "Magnesium": "Mg I",
  "Iron": "Fe I",
}

ionized_species = {
  "Neon": "Ne II",
  "Sodium": "Na II",
  "Magnesium": "Mg II",
  "Iron": "Fe II",
}

doubly_ionized_species = {
  "Neon": "Ne III",
  "Sodium": "Na III",
  "Magnesium": "Mg III",
  "Iron": "Fe III",
}

molecule_species = {
  "CO": {
    "source": "exomol",
    "fetch_kwargs": {
      "path": "CO/12C-16O/Li2015",
      "database": "Li2015",
      "localdatabase": "exomol_data",
    },
  },
  "NO": {
    "source": "exomol",
    "fetch_kwargs": {
      "path": "NO/14N-16O/XABC",
      "database": "XABC",
      "localdatabase": "exomol_data",
    },
  },
  "SiO": {
    "source": "exomol",
    "fetch_kwargs": {
      "path": "SiO/28Si-16O/SiOUVenIR",
      "database": "SiOUVenIR",
      "localdatabase": "exomol_data",
    },
  },
  "SO": {
    "source": "exomol",
    "fetch_kwargs": {
      "path": "SO/32S-16O/SOLIS",
      "database": "SOLIS",
      "localdatabase": "exomol_data",
    },
  },
}

stellar_models = {
  "K1": {
    "Teff": 5000 * u.K,
    "path": "Templates/TS/Spectral_type/K/K1/lte050-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 0.82 * const.R_sun,
    "mass": 0.82 * const.M_sun,
    "vsini": 3 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "G8": {
    "Teff": 5200 * u.K,
    "path": "Templates/TS/Spectral_type/G/G8/lte052-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 0.88 * const.R_sun,
    "mass": 0.88 * const.M_sun,
    "vsini": 5 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "G6": {
    "Teff": 5400 * u.K,
    "path": "Templates/TS/Spectral_type/G/G6/lte054-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 0.93 * const.R_sun,
    "mass": 0.93 * const.M_sun,
    "vsini": 5 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "G4": {
    "Teff": 5600 * u.K,
    "path": "Templates/TS/Spectral_type/G/G4/lte056-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 0.98 * const.R_sun,
    "mass": 0.98 * const.M_sun,
    "vsini": 5 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "G1": {
    "Teff": 5800 * u.K,
    "path": "Templates/TS/Spectral_type/G/G1/lte058-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 1.05 * const.R_sun,
    "mass": 1.05 * const.M_sun,
    "vsini": 5 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "F8": {
    "Teff": 6000 * u.K,
    "path": "Templates/TS/Spectral_type/F/F8/lte060-4.0-0.0a+0.2.BT-NextGen.7.dat.txt",
    "radius": 1.25 * const.R_sun,
    "mass": 1.25 * const.M_sun,
    "vsini": 20 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "F6": {
    "Teff": 6200 * u.K,
    "path": "Templates/TS/Spectral_type/F/F6/lte062-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 1.35 * const.R_sun,
    "mass": 1.35 * const.M_sun,
    "vsini": 20 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "F5": {
    "Teff": 6400 * u.K,
    "path": "Templates/TS/Spectral_type/F/F5/lte064-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 1.4 * const.R_sun,
    "mass": 1.4 * const.M_sun,
    "vsini": 20 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "F4": {
    "Teff": 6600 * u.K,
    "path": "Templates/TS/Spectral_type/F/F4/lte066-4.0-0.0a+0.2.BT-NextGen.7.dat.txt",
    "radius": 1.45 * const.R_sun,
    "mass": 1.45 * const.M_sun,
    "vsini": 20 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "F2": {
    "Teff": 6800 * u.K,
    "path": "Templates/TS/Spectral_type/F/F2/lte068-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 1.5 * const.R_sun,
    "mass": 1.5 * const.M_sun,
    "vsini": 20 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "F1": {
    "Teff": 7000 * u.K,
    "path": "Templates/TS/Spectral_type/F/F1/lte070-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 1.55 * const.R_sun,
    "mass": 1.55 * const.M_sun,
    "vsini": 20 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "F0": {
    "Teff": 7200 * u.K,
    "path": "Templates/TS/Spectral_type/F/F0/lte072-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 1.6 * const.R_sun,
    "mass": 1.6 * const.M_sun,
    "vsini": 20 * u.km / u.s,
    "epsilon": 0.6 * u.dimensionless_unscaled,
  },
  "A9": {
    "Teff": 7400 * u.K,
    "path": "Templates/TS/Spectral_type/A/A9/lte074-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 1.7 * const.R_sun,
    "mass": 1.8 * const.M_sun,
    "vsini": 120 * u.km / u.s,
    "epsilon": 0.5 * u.dimensionless_unscaled,
  },
  "A8": {
    "Teff": 7600 * u.K,
    "path": "Templates/TS/Spectral_type/A/A8/lte076-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 1.75 * const.R_sun,
    "mass": 1.85 * const.M_sun,
    "vsini": 120 * u.km / u.s,
    "epsilon": 0.5 * u.dimensionless_unscaled,
  },
  "A7": {
    "Teff": 7800 * u.K,
    "path": "Templates/TS/Spectral_type/A/A7/lte078-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 1.8 * const.R_sun,
    "mass": 1.9 * const.M_sun,
    "vsini": 120 * u.km / u.s,
    "epsilon": 0.5 * u.dimensionless_unscaled,
  },
  "A6": {
    "Teff": 8000 * u.K,
    "path": "Templates/TS/Spectral_type/A/A6/lte080-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 1.85 * const.R_sun,
    "mass": 1.95 * const.M_sun,
    "vsini": 120 * u.km / u.s,
    "epsilon": 0.5 * u.dimensionless_unscaled,
  },
  "A5": {
    "Teff": 8200 * u.K,
    "path": "Templates/TS/Spectral_type/A/A5/lte082-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 1.9 * const.R_sun,
    "mass": 2.0 * const.M_sun,
    "vsini": 120 * u.km / u.s,
    "epsilon": 0.5 * u.dimensionless_unscaled,
  },
  "A4": {
    "Teff": 8600 * u.K,
    "path": "Templates/TS/Spectral_type/A/A4/lte086-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 2.0 * const.R_sun,
    "mass": 2.1 * const.M_sun,
    "vsini": 120 * u.km / u.s,
    "epsilon": 0.5 * u.dimensionless_unscaled,
  },
  "A3": {
    "Teff": 8800 * u.K,
    "path": "Templates/TS/Spectral_type/A/A3/lte088-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 2.1 * const.R_sun,
    "mass": 2.2 * const.M_sun,
    "vsini": 120 * u.km / u.s,
    "epsilon": 0.5 * u.dimensionless_unscaled,
  },
  "A2": {
    "Teff": 9000 * u.K,
    "path": "Templates/TS/Spectral_type/A/A2/lte090-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 2.2 * const.R_sun,
    "mass": 2.3 * const.M_sun,
    "vsini": 120 * u.km / u.s,
    "epsilon": 0.5 * u.dimensionless_unscaled,
  },
  "A1": {
    "Teff": 9400 * u.K,
    "path": "Templates/TS/Spectral_type/A/A1/lte094-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 2.3 * const.R_sun,
    "mass": 2.4 * const.M_sun,
    "vsini": 120 * u.km / u.s,
    "epsilon": 0.5 * u.dimensionless_unscaled,
  },
  "A0": {
    "Teff": 10000 * u.K,
    "path": "Templates/TS/Spectral_type/A/A0/lte100-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 2.5 * const.R_sun,
    "mass": 2.5 * const.M_sun,
    "vsini": 120 * u.km / u.s,
    "epsilon": 0.5 * u.dimensionless_unscaled,
  },
  "B9": {
    "Teff": 11000 * u.K,
    "path": "Templates/TS/Spectral_type/B/B9/lte110-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 3.1 * const.R_sun,
    "mass": 3.0 * const.M_sun,
    "vsini": 80 * u.km / u.s,
    "epsilon": 0.4 * u.dimensionless_unscaled,
  },
  "B8": {
    "Teff": 12000 * u.K,
    "path": "Templates/TS/Spectral_type/B/B8/lte120-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 3.4 * const.R_sun,
    "mass": 4.0 * const.M_sun,
    "vsini": 80 * u.km / u.s,
    "epsilon": 0.4 * u.dimensionless_unscaled,
  },
  "B7": {
    "Teff": 13000 * u.K,
    "path": "Templates/TS/Spectral_type/B/B7/lte130-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 3.7 * const.R_sun,
    "mass": 5.0 * const.M_sun,
    "vsini": 80 * u.km / u.s,
    "epsilon": 0.4 * u.dimensionless_unscaled,
  },
  "B6": {
    "Teff": 14000 * u.K,
    "path": "Templates/TS/Spectral_type/B/B6/lte140-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 3.9 * const.R_sun,
    "mass": 5.5 * const.M_sun,
    "vsini": 80 * u.km / u.s,
    "epsilon": 0.4 * u.dimensionless_unscaled,
  },
  "B5": {
    "Teff": 15000 * u.K,
    "path": "Templates/TS/Spectral_type/B/B5/lte150-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 4.1 * const.R_sun,
    "mass": 6.0 * const.M_sun,
    "vsini": 80 * u.km / u.s,
    "epsilon": 0.4 * u.dimensionless_unscaled,
  },
  "B4": {
    "Teff": 17000 * u.K,
    "path": "Templates/TS/Spectral_type/B/B4/lte170-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
    "radius": 4.3 * const.R_sun,
    "mass": 7.0 * const.M_sun,
    "vsini": 80 * u.km / u.s,
    "epsilon": 0.4 * u.dimensionless_unscaled,
  },
}


def scalar_value(x):
  if isinstance(x, u.Quantity):
    return float(np.squeeze(x.value))
  return float(np.squeeze(x))


def make_atom_dict(species_dict):
  return {element: Atom(species, wavemin, wavemax) for element, species in species_dict.items()}


def make_broadening_dict(atom_dict):
  return {element: BroadeningProfile(atom, b, Npts, 'Voigt') for element, atom in atom_dict.items()}


def make_molecule(species, config, wav_min, wav_max):
  molecule = Molecule(species, wav_min, wav_max)
  source = config["source"].lower()
  fetch_kwargs = config["fetch_kwargs"]

  if source == "exomol":
    molecule.fetch_exomol(**fetch_kwargs)
  elif source == "hitran":
    molecule.fetch_hitran(**fetch_kwargs)
  else:
    raise ValueError(f"Unknown molecule source: {config['source']}")

  return molecule


neutral_atoms = make_atom_dict(neutral_species)
ionized_atoms = make_atom_dict(ionized_species)
doubly_ionized_atoms = make_atom_dict(doubly_ionized_species)

neutral_broad = make_broadening_dict(neutral_atoms)
ionized_broad = make_broadening_dict(ionized_atoms)
doubly_ionized_broad = make_broadening_dict(doubly_ionized_atoms)

molecules = {
  species: make_molecule(species, config, wavemin, wavemax)
  for species, config in molecule_species.items()
}
molecule_broad = {
  species: BroadeningProfileMolecule(molecule, b, profileType='Voigt')
  for species, molecule in molecules.items()
}
for species in molecule_broad:
  molecule_broad[species].temp_strength_rel_cutoff = 0.0

stars = {
  name: Star(
    model["path"],
    model["radius"],
    model["mass"],
    vsini=model["vsini"],
    epsilon=model["epsilon"],
  )
  for name, model in stellar_models.items()
}

element_order = ["Neon", "Sodium", "Magnesium", "Iron"]
panel_labels = ["A", "B", "C", "D"]

teff_vals = []
beta_neutral = {element: [] for element in element_order}
beta_ionized = {element: [] for element in element_order}
beta_doubly_ionized = {element: [] for element in element_order}
molecule_order = ["CO", "NO", "SiO", "SO"]
beta_molecules = {species: [] for species in molecule_order}

for star_name, model in stellar_models.items():
  star = stars[star_name]
  Teff = model["Teff"]
  teff_vals.append(Teff.to_value(u.K))
  print(f"Calculating beta for {star_name} at T_eff = {Teff}")

  for element in element_order:
    pp_neutral = PhotonPressure(neutral_broad[element], star)
    F_ph_tot, F_ph_tot_err, _, _ = pp_neutral.calc_PhotonPressure(Ncol, Teff, star.radius)
    beta, _ = pp_neutral.beta_Values(F_ph_tot, F_ph_tot_err, star.mass, star.radius)
    beta_neutral[element].append(scalar_value(beta))

    pp_ionized = PhotonPressure(ionized_broad[element], star)
    F_ph_tot, F_ph_tot_err, _, _ = pp_ionized.calc_PhotonPressure(Ncol, Teff, star.radius)
    beta, _ = pp_ionized.beta_Values(F_ph_tot, F_ph_tot_err, star.mass, star.radius)
    beta_ionized[element].append(scalar_value(beta))

    pp_doubly = PhotonPressure(doubly_ionized_broad[element], star)
    F_ph_tot, F_ph_tot_err, _, _ = pp_doubly.calc_PhotonPressure(Ncol, Teff, star.radius)
    beta, _ = pp_doubly.beta_Values(F_ph_tot, F_ph_tot_err, star.mass, star.radius)
    beta_doubly_ionized[element].append(scalar_value(beta))

  for species in molecule_order:
    pp_mol = PhotonPressure(molecule_broad[species], star)
    F_ph_tot, F_ph_tot_err, _, _ = pp_mol.calc_PhotonPressure(Ncol, T_mol, star.radius)
    beta, _ = pp_mol.beta_Values(F_ph_tot, F_ph_tot_err, star.mass, star.radius)
    beta_molecules[species].append(scalar_value(beta))

fig, axes = plt.subplots(1, 4, figsize=(14, 4.8), sharey=True)

for ax, element, panel_label in zip(axes, element_order, panel_labels):
  ax.plot(teff_vals, beta_neutral[element], color='black', marker='o', linestyle='-', label='Neutral')
  ax.plot(teff_vals, beta_ionized[element], color='orange', marker='v', linestyle='--', label='Singly ionised')
  ax.plot(teff_vals, beta_doubly_ionized[element], color='blue', marker='s', linestyle='-.', label='Doubly ionised')
  ax.axhline(1.0, color='black', linestyle='--', alpha=0.7)
  ax.set_yscale('log')
  ax.set_title(element)
  ax.text(0.03, 0.92, panel_label, transform=ax.transAxes, fontsize=18, fontweight='bold')
  ax.set_xlim(min(teff_vals) - 200, max(teff_vals) + 200)
  ax.grid(False)

axes[0].set_ylabel('Beta ratio')
axes[1].set_xlabel('Effective temperature [K]')
axes[-1].legend(loc='lower right')

plt.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_DIR / "beta_vs_teff_atoms.pdf", dpi=300)
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(14, 4.8), sharey=True)

for ax, species, panel_label in zip(axes, molecule_order, ["A", "B", "C", "D"]):
  ax.plot(teff_vals, beta_molecules[species], color='red', marker='o', linestyle='-', label=species)
  ax.axhline(1.0, color='black', linestyle='--', alpha=0.7)
  ax.set_yscale('log')
  ax.set_title(species)
  ax.text(0.03, 0.92, panel_label, transform=ax.transAxes, fontsize=18, fontweight='bold')
  ax.set_xlim(min(teff_vals) - 200, max(teff_vals) + 200)
  ax.grid(False)

axes[0].set_ylabel('Beta ratio')
axes[1].set_xlabel('Effective temperature [K]')

plt.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_DIR / "beta_vs_teff_molecules.pdf", dpi=300)
plt.show()
