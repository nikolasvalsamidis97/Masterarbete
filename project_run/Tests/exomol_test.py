from radis.io.exomol import fetch_exomol
from astropy import units as u
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from project_classes.Molecule import Molecule
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from project_classes.Planet import Planet
from project_classes.PlanetarySystem import PlanetarySystem
import numpy as np
from matplotlib import pyplot as plt

wavemax = 50000 * u.AA 
wavemin = 150 * u.AA
A_min = 0 * u.s**-1 


CO = Molecule("CO", wavemin, wavemax, A_min)

CO_df = CO.fetch_exomol(path="CO/12C-16O/Li2015", database="Li2015", localdatabase="exomol_data")
CO_df_pandas = CO.pandas_to_numpy()
print(CO.data_numpy["A_ul"].shape)

# H2O = Molecule("H2O", wavemin, wavemax, A_min)
# H2O_df = H2O.fetch_exomol(path="H2O/1H2-16O/POKAZATEL", database="POKAZATEL", localdatabase="exomol_data")
# H2O_df_pandas = H2O.pandas_to_numpy()


# NO = Molecule("NO", wavemin, wavemax, A_min)
# NO_df = NO.fetch_exomol(path="NO/14N-16O/XABC", database="XABC", localdatabase="exomol_data")
# NO_df_pandas = NO.pandas_to_numpy()
# print(NO.data_numpy["A_ul"])

# SO = Molecule("SO", wavemin, wavemax, A_min)
# SO_df = SO.fetch_exomol(path="SO/32S-16O/SOLIS", database="SOLIS", localdatabase="exomol_data")
# SO_df_pandas = SO.pandas_to_numpy()
# print(SO.data_numpy["A_ul"].shape)

# # HITRAN
# O2 = Molecule("O2", wavemin, wavemax, A_min)

# O2_df = O2.fetch_hitran(molecule_name="O2")  # O2 = 7 in HITRAN
# O2_df_numpy = O2.pandas_to_numpy()

# print(O2.data_numpy["g_l"])
T_atm = 300 * u.K
CO_broad = BroadeningProfileMolecule(
    molecule=CO,
    b=1 * u.km / u.s,
    lam_min=wavemin,
    lam_max=wavemax,
    dlam=0.01 * u.AA,
    profileType="Voigt",
    Temp_atm=T_atm,
)

sun = Star("TS/Spectral_type/A/A0/lte100-4.0-0.0a+0.0.BT-NextGen.7.dat.txt", 1 * u.R_sun, 1 * u.M_sun, vsini=10 * u.km / u.s, epsilon=0.2 * u.dimensionless_unscaled)
earth = Planet(1 * u.R_earth, 1 * u.M_earth, T_atm, mu=28.97 * u.dimensionless_unscaled, P0=1 * u.bar)
system = PlanetarySystem(sun, earth, 0.1 * u.au)

CO_pp = PhotonPressure(CO_broad, sun)

N_cols = np.logspace(10, 25, 100) * u.cm**(-2)

pp_CO, _,_,_ = CO_pp.calc_PhotonPressure(N_cols, T_atm, system.distance)

plt.plot(N_cols, pp_CO[0])
plt.xlabel("Column Density [cm^-2]")
plt.ylabel("Photon Pressure [N]")
plt.xscale("log")
plt.yscale("log")
plt.show()