

from astropy import units as u
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import astropy.constants as const
from project_classes.Molecule import Molecule
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star

wavemin = 150 * u.AA
wavemax = 50000 * u.AA
b = 1 * u.km / u.s
Ncol = np.array([0.0]) * u.cm**-2

# Fixed star for the atmospheric-temperature table.
fixed_star_model = {
  "Star": "A0",
  "Teff": 10000 * u.K,
  "path": "TS/Spectral_type/A/A0/lte100-4.0-0.0a+0.0.BT-NextGen.7.dat.txt",
  "radius": 2.5 * const.R_sun,
  "mass": 2.5 * const.M_sun,
  "vsini": 120 * u.km/u.s,
  "epsilon": 0.5 * u.dimensionless_unscaled,
}

atm_temperature_order = [1, 100, 300, 500, 750, 1000, 2000, 3000]

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
  "SO": {
    "source": "exomol",
    "fetch_kwargs": {
      "path": "SO/32S-16O/SOLIS",
      "database": "SOLIS",
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
    "CH": {
    "source": "exomol",
    "fetch_kwargs": {
      "path": "CH/12C-1H/MoLLIST",
      "database": "MoLLIST",
      "localdatabase": "exomol_data",
    },
  },
  "OH": {
    "source": "exomol",
    "fetch_kwargs": {
      "path": "OH/16O-1H/MYTHOS",
      "database": "MYTHOS",
      "localdatabase": "exomol_data",
    },
  },
  "NH": {
    "source": "exomol",
    "fetch_kwargs": {
      "path": "NH/14N-1H/kNigHt",
      "database": "kNigHt",
      "localdatabase": "exomol_data",
    },
  },
  "HCl": {
    "source": "exomol",
    "fetch_kwargs": {
      "path": "HCl/1H-35Cl/HITRAN-HCl",
      "database": "HITRAN-HCl",
      "localdatabase": "exomol_data",
    },
  },
  "CO2": {
    "source": "exomol",
    "fetch_kwargs": {
      "path": "CO2/12C-16O2/Dozen",
      "database": "Dozen",
      "localdatabase": "exomol_data",
    },
  },
}


def scalar_value(x):
  if isinstance(x, u.Quantity):
    return float(np.squeeze(x.value))
  return float(np.squeeze(x))


def format_beta_txt(beta, beta_err):
  beta_abs = abs(beta)
  if beta_abs < 1e-30:
    return "0"

  if beta_abs >= 100:
    return f"{beta:.0f}±{beta_err:.0f}"
  if beta_abs >= 10:
    return f"{beta:.1f}±{beta_err:.1f}"
  if beta_abs >= 1:
    return f"{beta:.2f}±{beta_err:.2f}"

  exponent = int(np.floor(np.log10(beta_abs)))
  mantissa = beta / (10**exponent)
  err_mantissa = beta_err / (10**exponent)
  return f"({mantissa:.1f}±{err_mantissa:.1f})10^{exponent}"


def format_beta_latex(beta, beta_err):
  beta_abs = abs(beta)
  if beta_abs < 1e-30:
    return r"$0$"

  if beta_abs >= 100:
    return rf"${beta:.0f} \pm {beta_err:.0f}$"
  if beta_abs >= 10:
    return rf"${beta:.1f} \pm {beta_err:.1f}$"
  if beta_abs >= 1:
    return rf"${beta:.2f} \pm {beta_err:.2f}$"

  exponent = int(np.floor(np.log10(beta_abs)))
  mantissa = beta / (10**exponent)
  err_mantissa = beta_err / (10**exponent)
  return rf"$({mantissa:.1f} \pm {err_mantissa:.1f})10^{{{exponent}}}$"


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


output_dir = pathlib.Path(__file__).resolve().parents[2] / "Tables" / "Beta_vs_temp"
output_dir.mkdir(parents=True, exist_ok=True)

star = Star(
  fixed_star_model["path"],
  fixed_star_model["radius"],
  fixed_star_model["mass"],
  vsini=fixed_star_model["vsini"],
  epsilon=fixed_star_model["epsilon"],
)

available_broad = {}
missing_species = set()

for species, config in molecule_species.items():
  try:
    molecule = make_molecule(species, config, wavemin, wavemax)
    broad = BroadeningProfileMolecule(molecule, b, profileType='Voigt')
    broad.temp_strength_rel_cutoff = 1e-8
    available_broad[species] = broad
    print(f"Loaded {species}")
  except Exception as exc:
    missing_species.add(species)
    print(f"Missing {species}: {exc}")
    continue

rows_txt = []
rows_latex = []
for species in molecule_species.keys():
  row_txt = {"Molecule": species}
  row_latex = {"Molecule": species}
  for T_atm in atm_temperature_order:
    row_txt[f"{T_atm} K"] = "-"
    row_latex[f"{T_atm} K"] = "-"
  rows_txt.append(row_txt)
  rows_latex.append(row_latex)

row_lookup_txt = {row["Molecule"]: row for row in rows_txt}
row_lookup_latex = {row["Molecule"]: row for row in rows_latex}

for T_atm_val in atm_temperature_order:
  T_atm = T_atm_val * u.K
  column_name = f"{T_atm_val} K"
  print(f"Calculating molecular beta table column for T_atm = {T_atm}")

  for species in molecule_species.keys():
    if species in missing_species:
      continue

    try:
      pp = PhotonPressure(available_broad[species], star)
      F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(Ncol, T_atm, star.radius)
      beta, beta_err = pp.beta_Values(F_ph_tot, F_ph_tot_err, star.mass, star.radius)
      beta_val = scalar_value(beta)
      beta_err_val = scalar_value(beta_err)
      row_lookup_txt[species][column_name] = format_beta_txt(beta_val, beta_err_val)
      row_lookup_latex[species][column_name] = format_beta_latex(beta_val, beta_err_val)
    except Exception as exc:
      row_lookup_txt[species][column_name] = "-"
      row_lookup_latex[species][column_name] = "-"
      print(f"Failed {species} at T_atm = {T_atm_val} K: {exc}")

beta_df_txt = pd.DataFrame(rows_txt)
ordered_columns = ["Molecule"] + [f"{T_atm} K" for T_atm in atm_temperature_order]
beta_df_txt = beta_df_txt[ordered_columns]
beta_df_txt.to_csv(output_dir / "Beta_vs_atmTemp_molecule.txt", index=False)
print(f"Saved beta table to {output_dir / 'Beta_vs_atmTemp_molecule.txt'}")

beta_df_latex = pd.DataFrame(rows_latex)
beta_df_latex = beta_df_latex[ordered_columns]
latex_table = beta_df_latex.to_latex(
  index=False,
  escape=False,
  longtable=True,
  column_format="l" + "c" * len(beta_df_latex.columns[1:])
)
with open(output_dir / "Beta_vs_atmTemp_molecule.tex", "w", encoding="utf-8") as f:
  f.write(latex_table)
print(f"Saved LaTeX beta table to {output_dir / 'Beta_vs_atmTemp_molecule.tex'}")