import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from project_classes.Atom import Molecule
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
import numpy as np

############################################################################################
######################################### KODGUIDE #########################################
# 1. Hämtar molekyldata
Na = Molecule('Na I', 1000 * u.AA, 9000*u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
b = 1 * u.km/u.s        # b = v_D
Npts = 150
Na_broadening = BroadeningProfile(Na, b , Npts, 'Voigt')
vsini = 130 * u.km / u.s
epsilon = 0 * u.dimensionless_unscaled

# 3. Hämta teoretiskt stjärnspectra
star = Star('TS/models_1770121505/bt-nextgen-agss2009/lte078-4.0-2.0a+0.4.BT-NextGen.7.dat.txt', 
            const.R_sun.value * u.m, 1.75*const.M_sun.value * u.kg, vsini, epsilon)

d_to_object = 19 * u.pc

# β-pic example for an comparison with observed magnitudes
targets = {
  "2MASS": {
    "J": 3.669, "H": 3.544, "Ks": 3.526
    },
  "Gaia":  {
    "G": 3.823242
    },
  "GCPD": {
    "U": 4.13, "B": 4.03, "V": 3.86, "R": 3.74, "I": 3.58
  },
}

MAGSYS = {
  "2MASS": "vegamag",
  "Gaia":  "vegamag",
  "GCPD":  "vegamag",
}

PHOTCALID = {
  "2MASS": {
    "J":  "2MASS/2MASS.J/Vega",
    "H":  "2MASS/2MASS.H/Vega",
    "Ks": "2MASS/2MASS.Ks/Vega",
  },
  "Gaia": {
    "G":  "GAIA/GAIA3.G/Vega",
  },
  "GCPD": {
    "U": "GCPD/Johnson.U/Vega",
    "B": "GCPD/Johnson.B/Vega",
    "V": "GCPD/Johnson.V/Vega",
    "R": "GCPD/Johnson.R/Vega",
    "I": "GCPD/Cousins.I/Vega",
  },
}

print("Old stellar radius before scaling:", star.radius.to(u.R_sun))

k_vals, lam_pivots = star.scale_factors_from_targets(targets, d_to_object, magsys=MAGSYS, use_rot=True, photcalid_map=PHOTCALID)
print("Scale factors k for each band:")
for survey, k in k_vals.items():
  print(f"  {survey}: {k}")

print(f"New stellar radius after scaling: {star.radius.to(u.R_sun)}")

survey_band_keys = list(k_vals.keys())
surveys = [key.split("_", 1)[0] for key in survey_band_keys]
bands = [key.split("_", 1)[1] for key in survey_band_keys]

unique_surveys = sorted(set(surveys))
unique_bands = sorted(set(bands))

marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H", "8", "p"]
survey_markers = {s: marker_cycle[i % len(marker_cycle)] for i, s in enumerate(unique_surveys)}

color_map = plt.get_cmap("tab20")
band_colors = {b: color_map(i % color_map.N) for i, b in enumerate(unique_bands)}

lam_pivot_vals = {k: lam_pivots[k].to_value(u.AA) for k in lam_pivots}
sorted_keys = sorted(survey_band_keys, key=lambda k: lam_pivot_vals[k])

x = np.arange(len(sorted_keys))
y = [k_vals[k] for k in sorted_keys]

fig, ax = plt.subplots(figsize=(10, 4))
for i, key in enumerate(sorted_keys):
  s, b = key.split("_", 1)
  ax.scatter(
    x[i],
    y[i],
    marker=survey_markers[s],
    color=band_colors[b],
    s=80,
    edgecolors="black",
    linewidths=0.5,
  )

ax.set_xticks(x)
ax.set_xticklabels(sorted_keys, rotation=45, ha="right")
ax.set_ylabel("Scale factor k")
ax.set_title("k-values ordered by pivot wavelength")

from matplotlib.lines import Line2D
survey_handles = [
  Line2D(
    [0],
    [0],
    marker=survey_markers[s],
    color="black",
    label=s,
    linestyle="None",
    markersize=8,
  )
  for s in unique_surveys
]
band_handles = [
  Line2D(
    [0],
    [0],
    marker="o",
    color="none",
    label=b,
    markerfacecolor=band_colors[b],
    markeredgecolor="black",
    markersize=8,
  )
  for b in unique_bands
]

legend1 = ax.legend(handles=survey_handles, title="Survey", loc="upper left", bbox_to_anchor=(1.02, 1))
ax.add_artist(legend1)
ax.legend(handles=band_handles, title="Band", loc="lower left", bbox_to_anchor=(1.02, 0))

plt.tight_layout()
plt.show()

print("Pivot wavelengths for each band:")
for key, lam_pivot in lam_pivots.items():
  print(f"  {key}: {lam_pivot.to(u.AA)}")
