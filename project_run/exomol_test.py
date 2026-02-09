from radis.io.exomol import fetch_exomol
from astropy import units as u
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from project_classes.Molecule import Molecule



CO_species = "CO"
CO_wavemin = 150 * u.AA
CO_wavemax = 50000 * u.AA 
A_min = 0 * u.s**-1 
CO_path = "CO/12C-16O/Li2015"
CO_database = "Li2015"
CO_localdatabase = "exomol_data"



CO = Molecule(CO_species, CO_wavemin, CO_wavemax, A_min, CO_path, CO_database, CO_localdatabase)








