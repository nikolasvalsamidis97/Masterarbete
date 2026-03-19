from radis.io.exomol import fetch_exomol
from astropy import units as u
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from project_classes.Molecule import Molecule

wavemax = 50000 * u.AA 
wavemin = 150 * u.AA
A_min = 0 * u.s**-1 


CO = Molecule("CO", wavemin, wavemax, A_min)

CO_df = CO.fetch_exomol(path="CO/12C-16O/Li2015", database="Li2015", localdatabase="exomol_data")
CO_df_pandas = CO.pandas_to_numpy()
print(CO.data_numpy["A_ul"])

# H2O = Molecule("H2O", wavemin, wavemax, A_min)
# H2O_df = H2O.fetch_exomol(path="H2O/1H2-16O/POKAZATEL", database="POKAZATEL", localdatabase="exomol_data")
# H2O_df_pandas = H2O.pandas_to_numpy()


