from astropy import units as u
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))
from project_classes.Molecule import Molecule
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from project_classes.Planet import Planet
from project_classes.PlanetarySystem import PlanetarySystem
import numpy as np
from matplotlib import pyplot as plt
import time

wavemax = 50000 * u.AA 
wavemin = 150 * u.AA
A_min = 0 * u.s**-1 



CO = Molecule("CO", wavemin, wavemax, A_min)
NO = Molecule("NO", wavemin, wavemax, A_min)

t0 = time.perf_counter()
CO_df = CO.fetch_exomol(path="CO/12C-16O/Li2015", database="Li2015", localdatabase="exomol_data")
t1 = time.perf_counter()
print(f"CO fetch_exomol time: {t1 - t0:.2f} s")

t0 = time.perf_counter()
CO_df_pandas = CO.pandas_to_numpy()
t1 = time.perf_counter()
print(f"CO pandas_to_numpy time: {t1 - t0:.2f} s")

t0 = time.perf_counter()
NO_df = NO.fetch_exomol(path="NO/14N-16O/XABC", database="XABC", localdatabase="exomol_data")
t1 = time.perf_counter()
print(f"NO fetch_exomol time: {t1 - t0:.2f} s")

t0 = time.perf_counter()
CO_broad = BroadeningProfileMolecule(
    molecule=CO,
    b=1 * u.km / u.s,
    lam_min=wavemin,
    lam_max=wavemax,
    dlam = 0.1 * u.AA,
    profileType="Voigt",
)
t1 = time.perf_counter()
print(f"CO BroadeningProfileMolecule init time: {t1 - t0:.2f} s")

t0 = time.perf_counter()
NO_broad = BroadeningProfileMolecule(
    molecule=NO,
    b=1 * u.km / u.s,
    lam_min=wavemin,
    lam_max=wavemax,
    dlam = 0.1 * u.AA,
    profileType="Voigt",
)
t1 = time.perf_counter()
print(f"NO BroadeningProfileMolecule init time: {t1 - t0:.2f} s")

T_atm = 300 * u.K
sun = Star("Templates/TS/Spectral_type/A/A0/lte100-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", 1 * u.R_sun, 1 * u.M_sun, vsini=10 * u.km / u.s, epsilon=0.2 * u.dimensionless_unscaled)
earth = Planet(1 * u.R_earth, 1 * u.M_earth, T_atm, mu=28.97 * u.dimensionless_unscaled, P0=1 * u.bar)
system = PlanetarySystem(sun, earth, 0.1 * u.au)

CO_pp = PhotonPressure(CO_broad, sun)
NO_pp = PhotonPressure(NO_broad, sun)

N_cols = np.logspace(10, 25, 100) * u.cm**(-2)
print(f"Running photon pressure for {CO_broad.molecule.species} at T = {T_atm}")
t0 = time.perf_counter()
pp_CO, _,_,_ = CO_pp.calc_PhotonPressure(N_cols, T_atm, system.distance)
t1 = time.perf_counter()
print(f"CO photon pressure time: {t1 - t0:.2f} s")

print(f"Running photon pressure for {NO_broad.molecule.species} at T = {T_atm}")
t0 = time.perf_counter()
pp_NO, _,_,_ = NO_pp.calc_PhotonPressure(N_cols, T_atm, system.distance)
t1 = time.perf_counter()

print(f"NO photon pressure time: {t1 - t0:.2f} s")

# Check how much weighted central strength is kept by the cutoff
for broad_obj in [CO_broad, NO_broad]:
    weights = broad_obj.boltzmann_line_weights(T_atm)
    sig0_val = np.abs(broad_obj.sig_0[:, 0].to_value(u.cm**2 * u.km / u.s))
    weighted_strength = np.abs(weights * sig0_val)

    max_weighted_strength = np.nanmax(weighted_strength)
    strength_cutoff = broad_obj.temp_strength_rel_cutoff * max_weighted_strength
    keep_mask = weighted_strength >= strength_cutoff

    total_strength = np.nansum(weighted_strength)
    kept_strength = np.nansum(weighted_strength[keep_mask])
    removed_strength = total_strength - kept_strength
    kept_fraction = kept_strength / total_strength if total_strength > 0.0 else np.nan
    removed_fraction = removed_strength / total_strength if total_strength > 0.0 else np.nan

    print(f"\n{broad_obj.molecule.species} weighted-strength summary at T = {T_atm}")
    print(f"Cutoff = {broad_obj.temp_strength_rel_cutoff:.1e} * max(weight * sig0)")
    print(f"Kept lines = {np.sum(keep_mask)}/{len(weighted_strength)}")
    print(f"Kept weighted-strength fraction = {kept_fraction:.6e}")
    print(f"Removed weighted-strength fraction = {removed_fraction:.6e}")

plt.plot(N_cols, pp_CO[0], label="CO")
plt.plot(N_cols, pp_NO[0], label="NO")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Column density (cm^-2)")
plt.ylabel("Photon pressure (N)")
plt.title("Photon pressure vs column density for CO and NO")
plt.legend()
plt.grid()
plt.show()