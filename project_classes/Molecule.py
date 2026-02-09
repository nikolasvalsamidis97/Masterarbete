import astropy.units as u
from project_func.errors import _not_quantity
from molmass import Formula
import pandas as pd
from radis.io.exomol import fetch_exomol
from radis.api.exomolapi import MdbExomol, get_exomol_full_isotope_name
import numpy as np

class Molecule:
  def __init__(self, species: "str", lam_min, lam_max, A_ul_min, path: str, database: str, localdatabase: str):
      self.species = species
      self.lam_min = lam_min.to(u.AA) if isinstance(lam_min, u.Quantity) else _not_quantity("lam_min")
      self.lam_max = lam_max.to(u.AA) if isinstance(lam_max, u.Quantity) else _not_quantity("lam_max")
      self.wavenum_min = (1 / self.lam_max).to(u.cm**-1)
      self.wavenum_max = (1 / self.lam_min).to(u.cm**-1)
      self.A_ul_min = A_ul_min.to(1/u.s) if isinstance(A_ul_min, u.Quantity) else _not_quantity("A_ul_min")
      self.path = path
      self.database = database
      self.localdatabase = localdatabase

      self.data = self.fetch_exomol()
      self.mass = Formula(self.species).mass * u.u

      self.i_upper, self.i_lower, self.A_ul, self.A_ul_err, self.lam0, self.g_u, self.g_l, self.j_l, self.j_u = self.pandas_to_numpy()
      

      self.A_ul_err = np.zeros_like(self.A_ul) * u.s**-1


  def fetch_exomol(self):
      """
      Fetches data from ExoMol database using the radis package. The data is filtered according to the input parameters and stored in a pandas dataframe.

      ** Returns **
      data:         
      """

      path = self.path
      nurange = [self.wavenum_min.value, self.wavenum_max.value]

      mdb = MdbExomol(
          path=path,
          molecule=self.species,
          database=self.database,
          local_databases=self.localdatabase,  # folder where it will download/cache
          nurange=nurange,
          engine="pytables",              # easiest for pandas workflow
          skip_optional_data=True,
      )

      # get local cached trans files, then load them
      mgr = mdb.get_datafile_manager()
      local_files = [mgr.cache_file(f) for f in mdb.trans_file]

      cols = ["i_upper","i_lower","A","nu_lines","elower","gup","jlower","jupper","Sij0"]
      df = mdb.load(
          local_files,
          columns=cols,
          lower_bound=[("nu_lines", self.wavenum_min.value)],
          upper_bound=[("nu_lines", self.wavenum_max.value)],
          output="pandas",              # returns pandas
      )

      states = mgr.read(mgr.cache_file(mdb.states_file))   # pandas df, has columns i, g, J, E
      gmap = dict(zip(states["i"], states["g"]))
      df["glower"] = df["i_lower"].map(gmap)

      return df

  def pandas_to_numpy(self):
      """
      Numpy arrays with dimensions (N_lines, None) ex. (16,)
      """
      i_upper = pd.to_numeric(self.data['i_upper']).to_numpy().reshape(-1, 1)
      i_lower = pd.to_numeric(self.data['i_lower']).to_numpy().reshape(-1, 1)
      A_ul = pd.to_numeric(self.data['A']).to_numpy().reshape(-1, 1) / u.s
      A_ul_err = np.zeros_like(A_ul) * u.s**-1
      wav_cm1 = pd.to_numeric(self.data["nu_lines"]).to_numpy().reshape(-1, 1) / u.cm
      lam0 = (1 / wav_cm1).to(u.AA)
      g_u = pd.to_numeric(self.data["gup"]).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled
      g_l = pd.to_numeric(self.data["glower"]).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled
      j_l = pd.to_numeric(self.data["jlower"]).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled
      j_u = pd.to_numeric(self.data["jupper"]).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled

      return i_upper, i_lower, A_ul, A_ul_err, lam0, g_u, g_l, j_l, j_u
      