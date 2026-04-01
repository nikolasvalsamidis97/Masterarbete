import astropy.units as u
from project_func.errors import _not_quantity
from molmass import Formula
import pandas as pd
from radis.io.exomol import fetch_exomol
from radis.api.exomolapi import MdbExomol, get_exomol_full_isotope_name
import numpy as np
import pathlib
from radis import SpectrumFactory
from radis.io.hitran import fetch_hitran

class Molecule:
  def __init__(self, species: "str", lam_min, lam_max, A_ul_min = 0 * u.s**(-1)):
      self.species = species
      self.lam_min = lam_min.to(u.AA) if isinstance(lam_min, u.Quantity) else _not_quantity("lam_min")
      self.lam_max = lam_max.to(u.AA) if isinstance(lam_max, u.Quantity) else _not_quantity("lam_max")
      self.wavenum_min = (1 / self.lam_max).to(u.cm**-1)
      self.wavenum_max = (1 / self.lam_min).to(u.cm**-1)
      self.A_ul_min = A_ul_min.to(1/u.s) if isinstance(A_ul_min, u.Quantity) else _not_quantity("A_ul_min")
      self.mass = Formula(self.species).mass * u.u
      self.data = None


  def fetch_exomol(self, path, database, localdatabase):
      """
      Fetches data from ExoMol database using the radis package. The data is filtered according to the input parameters and stored in a pandas dataframe.

      ** Returns **
      data:         
      """

      nurange = [self.wavenum_min.value, self.wavenum_max.value]

      mdb = MdbExomol(
          path=path,
          molecule=self.species,
          database=database,
          local_databases=localdatabase,  # folder where it will download/cache
          nurange=nurange,
          engine="pytables",              # easiest for pandas workflow
          skip_optional_data=True,
      )

      # get local cached trans files, then load them
      mgr = mdb.get_datafile_manager()
      local_files = [mgr.cache_file(f) for f in mdb.trans_file]

      cols = ["A", "nu_lines", "elower", "gup", "i_lower"]  # A_ul, wavenumber, lower state energy, upper state degeneracy, lower state index
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

      self.data = df

      return df

  
  def fetch_hitran(
      self,
      molecule_name=None,
      isotope=1,
      localdatabase=None,
      path=None,
      databank_name=None,
      cache=True,
      engine="default",
      output="pandas",
  ):
    """
    Fetch line-by-line data from HITRAN.

    Parameters
    ----------
    molecule_name : str or None
        HITRAN molecule name. If None, use self.species.
    isotope : int or str
        HITRAN isotope selector.
    localdatabase : str or None
        Root local folder where RADIS should create/cache the HITRAN HDF5 files.
        If None, RADIS uses its own default configuration.
    path : str or None
        Optional relative subfolder inside ``localdatabase``. If both are given,
        the actual cache folder becomes ``localdatabase/path``.
    databank_name : str or None
        Optional RADIS databank registration name.
    cache : bool or str
        Passed to radis.io.hitran.fetch_hitran.
    engine : str
        HDF engine passed to radis.io.hitran.fetch_hitran.
    output : str
        Output format passed to radis.io.hitran.fetch_hitran.
    """

    molecule_name = self.species if molecule_name is None else molecule_name

    fetch_kwargs = dict(
      molecule=molecule_name,
      isotope=str(isotope),
      load_wavenum_min=float(self.wavenum_min.value),
      load_wavenum_max=float(self.wavenum_max.value),
      columns=None,
      cache=cache,
      engine=engine,
      output=output,
    )

    if localdatabase is not None:
      local_path = pathlib.Path(localdatabase)
      if path is not None:
        local_path = local_path / path
      local_path.mkdir(parents=True, exist_ok=True)
      fetch_kwargs["local_databases"] = str(local_path)
    if databank_name is not None:
      fetch_kwargs["databank_name"] = databank_name

    df = fetch_hitran(**fetch_kwargs)

    print(df.columns)

    df = df.rename(columns={
      "A": "A",
      "wav": "nu_lines",
      "El": "elower",
      "gp": "gup",
      "gpp": "glower",
    })
    df = df[["A", "nu_lines", "elower", "gup", "glower"]].copy()

    self.data = df
    return df

  def pandas_to_numpy(self):
    """
    Minimal numpy arrays needed for the molecular opacity pipeline.
    """
    A_ul = pd.to_numeric(self.data["A"]).to_numpy().reshape(-1, 1) / u.s
    A_ul_err = np.zeros_like(A_ul.value) / u.s

    wav_cm1 = pd.to_numeric(self.data["nu_lines"]).to_numpy().reshape(-1, 1) / u.cm
    lam0 = (1 / wav_cm1).to(u.AA)

    E_l = pd.to_numeric(self.data["elower"]).to_numpy().reshape(-1, 1) / u.cm
    E_l = E_l.to(u.eV, equivalencies=u.spectral())

    g_u = pd.to_numeric(self.data["gup"]).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled
    g_l = pd.to_numeric(self.data["glower"]).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled

    self.data_numpy = {
        "A_ul": A_ul,
        "A_ul_err": A_ul_err,
        "lam0": lam0,
        "E_l": E_l,
        "g_u": g_u,
        "g_l": g_l,
    }

    return A_ul, A_ul_err, lam0, E_l, g_u, g_l