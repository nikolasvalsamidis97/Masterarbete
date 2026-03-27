from astropy import units as u
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import astropy.constants as const
from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star

wavemin = 150 * u.AA
wavemax = 50000 * u.AA
Npts = 150
b = 1 * u.km / u.s
Ncol = np.array([0.0]) * u.cm**-2

# All atomic species from H I to Fe III.
# Missing species are handled gracefully and marked with "-" in the output table.
atom_species = [
  "H I", "H II", "H III",
  "He I", "He II", "He III",
  "Li I", "Li II", "Li III",
  "Be I", "Be II", "Be III",
  "B I", "B II", "B III",
  "C I", "C II", "C III",
  "N I", "N II", "N III",
  "O I", "O II", "O III",
  "F I", "F II", "F III",
  "Ne I", "Ne II", "Ne III",
  "Na I", "Na II", "Na III",
  "Mg I", "Mg II", "Mg III",
  "Al I", "Al II", "Al III",
  "Si I", "Si II", "Si III",
  "P I", "P II", "P III",
  "S I", "S II", "S III",
  "Cl I", "Cl II", "Cl III",
  "Ar I", "Ar II", "Ar III",
  "K I", "K II", "K III",
  "Ca I", "Ca II", "Ca III",
  "Sc I", "Sc II", "Sc III",
  "Ti I", "Ti II", "Ti III",
  "V I", "V II", "V III",
  "Cr I", "Cr II", "Cr III",
  "Mn I", "Mn II", "Mn III",
  "Fe I", "Fe II", "Fe III",
]

stellar_models = {
  "B4": {"Teff": 19000 * u.K, "path": "TS/Spectral_type/B/B4/lte170-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 4.3 * const.R_sun, "mass": 7.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "B5": {"Teff": 17000 * u.K, "path": "TS/Spectral_type/B/B5/lte150-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 4.1 * const.R_sun, "mass": 6.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "B7": {"Teff": 13000 * u.K, "path": "TS/Spectral_type/B/B7/lte130-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 3.7 * const.R_sun, "mass": 5.0 * const.M_sun, "vsini": 80 * u.km/u.s, "epsilon": 0.4 * u.dimensionless_unscaled},
  "A0": {"Teff": 10000 * u.K, "path": "TS/Spectral_type/A/A0/lte100-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 2.5 * const.R_sun, "mass": 2.5 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
  "A6": {"Teff": 8000 * u.K, "path": "TS/Spectral_type/A/A6/lte080-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 1.85 * const.R_sun, "mass": 1.95 * const.M_sun, "vsini": 120 * u.km/u.s, "epsilon": 0.5 * u.dimensionless_unscaled},
  "F8": {"Teff": 6000 * u.K, "path": "TS/Spectral_type/F/F8/lte060-4.0-0.0a+0.2.BT-NextGen.7.dat.txt", "radius": 1.25 * const.R_sun, "mass": 1.25 * const.M_sun, "vsini": 20 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "K1": {"Teff": 5000 * u.K, "path": "TS/Spectral_type/K/K1/lte050-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.82 * const.R_sun, "mass": 0.82 * const.M_sun, "vsini": 3 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "K5": {"Teff": 4500 * u.K, "path": "TS/Spectral_type/K/K5/lte044-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.70 * const.R_sun, "mass": 0.70 * const.M_sun, "vsini": 3 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
  "K7": {"Teff": 4000 * u.K, "path": "TS/Spectral_type/K/K7/lte040-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", "radius": 0.65 * const.R_sun, "mass": 0.65 * const.M_sun, "vsini": 3 * u.km/u.s, "epsilon": 0.6 * u.dimensionless_unscaled},
}

column_order = [4000, 4500, 5000, 6000, 8000, 10000, 13000, 17000, 19000]


def scalar_value(x):
  if isinstance(x, u.Quantity):
    return float(np.squeeze(x.value))
  return float(np.squeeze(x))


def format_beta_csv(beta, beta_err):
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


# Build stars once.
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

# Build atomic objects if available.
available_atoms = {}
available_broad = {}
missing_species = set()

for species in atom_species:
  try:
    atom = Atom(species, wavemin, wavemax)
    broad = BroadeningProfile(atom, b, Npts, 'Voigt')
    available_atoms[species] = atom
    available_broad[species] = broad
    print(f"Loaded {species}")
  except Exception as exc:
    missing_species.add(species)
    print(f"Missing {species}: {exc}")

# Prepare table structure.
rows_csv = []
rows_latex = []
for species in atom_species:
  row_csv = {"Ion": species}
  row_latex = {"Ion": species.replace(" ", r"~")}
  for Teff in column_order:
    row_csv[f"{Teff} K"] = "-"
    row_latex[f"{Teff} K"] = "-"
  rows_csv.append(row_csv)
  rows_latex.append(row_latex)

row_lookup_csv = {row["Ion"]: row for row in rows_csv}
row_lookup_latex = {row["Ion"].replace(r"~", " "): row for row in rows_latex}

for star_name, model in stellar_models.items():
  Teff = int(round(model["Teff"].to_value(u.K)))
  column_name = f"{Teff} K"
  star = stars[star_name]
  print(f"Calculating beta table column for {star_name} at T_eff = {Teff} K")

  for species in atom_species:
    if species in missing_species:
      continue

    try:
      pp = PhotonPressure(available_broad[species], star)
      F_ph_tot, F_ph_tot_err, _, _ = pp.calc_PhotonPressure(Ncol, model["Teff"], star.radius)
      beta, beta_err = pp.beta_Values(F_ph_tot, F_ph_tot_err, star.mass, star.radius)
      beta_val = scalar_value(beta)
      beta_err_val = scalar_value(beta_err)
      row_lookup_csv[species][column_name] = format_beta_csv(beta_val, beta_err_val)
      row_lookup_latex[species][column_name] = format_beta_latex(beta_val, beta_err_val)
    except Exception as exc:
      row_lookup_csv[species][column_name] = "-"
      row_lookup_latex[species][column_name] = "-"
      print(f"Failed {species} at {Teff} K: {exc}")

beta_df_csv = pd.DataFrame(rows_csv)
ordered_columns = ["Ion"] + [f"{Teff} K" for Teff in column_order]
beta_df_csv = beta_df_csv[ordered_columns]
beta_df_csv.to_csv("Tables/Beta_vs_temp/Beta_vs_Teff.csv", index=False)
print(f"Saved beta table to Tables/Beta_vs_temp/Beta_vs_Teff.csv")

beta_df_latex = pd.DataFrame(rows_latex)
beta_df_latex = beta_df_latex[ordered_columns]
latex_table = beta_df_latex.to_latex(
  index=False,
  escape=False,
  longtable=True,
  column_format="l" + "c" * len(beta_df_latex.columns[1:])
)
with open("Tables/Beta_vs_temp/Beta_vs_Teff.tex", "w", encoding="utf-8") as f:
  f.write(latex_table)
print(f"Saved LaTeX beta table to Tables/Beta_vs_temp/Beta_vs_Teff.tex")

stars_rows = []
for star_name in stellar_models:
  model = stellar_models[star_name]
  stars_rows.append({
    "Teff~(K)": f"{int(round(model['Teff'].to_value(u.K)))}",
    "Radius~($R_{\\odot}$)": f"{model['radius'].to_value(const.R_sun):.2f}",
    "Mass~($M_{\\odot}$)": f"{model['mass'].to_value(const.M_sun):.1f}",
    "vsini~(km/s)": f"{model['vsini'].to_value(u.km / u.s):.0f}",
    "epsilon": f"{model['epsilon'].value:.1f}",
  })

stars_df = pd.DataFrame(stars_rows)
stars_df["Teff_sort"] = stars_df["Teff~(K)"].astype(int)
stars_df = stars_df.sort_values("Teff_sort")
stars_df = stars_df.drop(columns=["Teff_sort"])
stars_df.to_csv("Tables/Beta_vs_temp/Stars_used.csv", index=False)
print(f"Saved stars table to Tables/Beta_vs_temp/Stars_used.csv")

stars_latex = stars_df.to_latex(index=False, escape=False, longtable=True)
with open("Tables/Beta_vs_temp/Stars_used.tex", "w", encoding="utf-8") as f:
  f.write(stars_latex)
print(f"Saved LaTeX stars table to Tables/Beta_vs_temp/Stars_used.tex")