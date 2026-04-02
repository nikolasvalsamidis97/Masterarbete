import astropy.units as u
from project_func.errors import _not_quantity
from molmass import Formula
from radis.api.exomolapi import MdbExomol
import numpy as np
import pandas as pd
import pathlib
import tables
from radis.io.hitran import fetch_hitran

class Molecule:
  def __init__(self, species: "str", lam_min, lam_max):
    self.species = species
    self.lam_min = lam_min.to(u.AA) if isinstance(lam_min, u.Quantity) else _not_quantity("lam_min")
    self.lam_max = lam_max.to(u.AA) if isinstance(lam_max, u.Quantity) else _not_quantity("lam_max")
    self.wavenum_min = (1 / self.lam_max).to(u.cm**-1)
    self.wavenum_max = (1 / self.lam_min).to(u.cm**-1)
    self.mass = Formula(self.species).mass * u.u
    self.cache_info = None
    self.cache_ready = False
    self.source = None
    self._exomol_mdb = None
    self._exomol_mgr = None

  def _build_exomol_mdb(self, path, database, localdatabase):
      nurange = [self.wavenum_min.value, self.wavenum_max.value]
      mdb = MdbExomol(
          path=path,
          molecule=self.species,
          database=database,
          local_databases=localdatabase,
          nurange=nurange,
          engine="pytables",
          skip_optional_data=True,
      )
      return mdb

  def fetch_exomol(self, path, database, localdatabase, verbose=False):
      """
      Cache/setup ExoMol files only.

      This method no longer assembles one giant pandas dataframe. Instead, it:
      - builds the ExoMol database handle,
      - resolves/caches the needed transition files,
      - resolves/caches the states file,
      - stores enough metadata for later file-by-file loading in the
        broadening/cross-section stage.
      """

      nurange = [self.wavenum_min.value, self.wavenum_max.value]
      if verbose:
          print(f"[{self.species}] fetch_exomol: building MdbExomol")
          print(f"[{self.species}] fetch_exomol: nurange = {nurange}")
          print(f"[{self.species}] fetch_exomol: localdatabase = {localdatabase}")
          print(f"[{self.species}] fetch_exomol: path = {path}")
          print(f"[{self.species}] fetch_exomol: database = {database}")

      mdb = self._build_exomol_mdb(path, database, localdatabase)
      if verbose:
          print(f"[{self.species}] fetch_exomol: MdbExomol created")

      if verbose:
          print(f"[{self.species}] fetch_exomol: getting datafile manager")
      mgr = mdb.get_datafile_manager()
      if verbose:
          print(f"[{self.species}] fetch_exomol: datafile manager ready")
          print(f"[{self.species}] fetch_exomol: number of transition files listed = {len(mdb.trans_file)}")

      local_trans_files = []
      for i, f in enumerate(mdb.trans_file, start=1):
          if verbose:
              print(f"[{self.species}] fetch_exomol: caching trans file {i}/{len(mdb.trans_file)} -> {f}")
          local_trans_files.append(mgr.cache_file(f))
      if verbose:
          print(f"[{self.species}] fetch_exomol: all transition files cached/resolved")

      if verbose:
          print(f"[{self.species}] fetch_exomol: caching states file")
      local_states_file = mgr.cache_file(mdb.states_file)
      if verbose:
          print(f"[{self.species}] fetch_exomol: states file cached/resolved")

      self.source = "exomol"
      self._exomol_mdb = mdb
      self._exomol_mgr = mgr
      self.cache_ready = True
      self.cache_info = {
          "source": "exomol",
          "path": path,
          "database": database,
          "localdatabase": localdatabase,
          "nurange": nurange,
          "local_trans_files": local_trans_files,
          "local_states_file": local_states_file,
          "states_file": mdb.states_file,
          "transition_columns": ["A", "nu_lines", "elower", "gup", "i_lower"],
      }
      if verbose:
          print(f"[{self.species}] fetch_exomol: cache metadata stored on Molecule")

      return self.cache_info

  def load_exomol_transition_dataframe(self, local_file):
      if self.cache_info is None or self.cache_info.get("source") != "exomol":
          raise ValueError("ExoMol cache is not prepared for this molecule.")
      if self._exomol_mdb is None:
          raise ValueError("ExoMol MdbExomol handle is missing on this Molecule object.")

      cols = self.cache_info["transition_columns"]
      return self._exomol_mdb.load(
          [local_file],
          columns=cols,
          lower_bound=[("nu_lines", self.wavenum_min.value)],
          upper_bound=[("nu_lines", self.wavenum_max.value)],
          output="pandas",
      )
  
  def _load_exomol_transition_chunk_h5(self, local_file):
      required_cols = {"A", "nu_lines", "elower", "gup", "i_lower"}
      nu_min = float(self.wavenum_min.value)
      nu_max = float(self.wavenum_max.value)

      with tables.open_file(local_file, mode="r") as h5:
          table_node = None
          for node in h5.walk_nodes("/", classname="Table"):
              table_node = node
              colnames = set(getattr(node, "colnames", []))
              if required_cols.issubset(colnames):
                  break

          if table_node is None:
              raise ValueError(
                  f"Could not find any HDF5 table in {local_file}"
              )

          colnames = list(getattr(table_node, "colnames", []))

          # Layout 1: direct named columns
          if required_cols.issubset(set(colnames)):
              nu_vals = np.asarray(table_node.col("nu_lines"), dtype=np.float64).reshape(-1)
              mask = (nu_vals >= nu_min) & (nu_vals <= nu_max)
              if not np.any(mask):
                  return {
                      "A_vals": np.empty(0, dtype=np.float64),
                      "nu_vals": np.empty(0, dtype=np.float64),
                      "elower_vals": np.empty(0, dtype=np.float64),
                      "gup_vals": np.empty(0, dtype=np.float64),
                      "i_lower_vals": np.empty(0, dtype=np.int64),
                      "lam0_vals": np.empty(0, dtype=np.float64),
                  }

              A_vals = np.asarray(table_node.col("A"), dtype=np.float64).reshape(-1)[mask]
              nu_vals = nu_vals[mask]
              elower_vals = np.asarray(table_node.col("elower"), dtype=np.float64).reshape(-1)[mask]
              gup_vals = np.asarray(table_node.col("gup"), dtype=np.float64).reshape(-1)[mask]
              i_lower_vals = np.asarray(table_node.col("i_lower")).reshape(-1)[mask].astype(np.int64, copy=False)
              return {
                  "A_vals": A_vals,
                  "nu_vals": nu_vals,
                  "elower_vals": elower_vals,
                  "gup_vals": gup_vals,
                  "i_lower_vals": i_lower_vals,
                  "lam0_vals": 1.0e8 / nu_vals,
              }

          # Layout 2: pandas block table, e.g. /df/table with values_block_* columns.
          # Use pandas directly on the local HDF5 file instead of trying to manually
          # decode the block packing, which can vary between files.
          key = table_node._v_parent._v_pathname
          where = f"nu_lines >= {nu_min} & nu_lines <= {nu_max}"
          df = pd.read_hdf(
              local_file,
              key=key,
              where=where,
              columns=["A", "nu_lines", "elower", "gup", "i_lower"],
          )
          if len(df) == 0:
              return {
                  "A_vals": np.empty(0, dtype=np.float64),
                  "nu_vals": np.empty(0, dtype=np.float64),
                  "elower_vals": np.empty(0, dtype=np.float64),
                  "gup_vals": np.empty(0, dtype=np.float64),
                  "i_lower_vals": np.empty(0, dtype=np.int64),
                  "lam0_vals": np.empty(0, dtype=np.float64),
              }

          nu_vals = np.asarray(df["nu_lines"], dtype=np.float64).reshape(-1)
          return {
              "A_vals": np.asarray(df["A"], dtype=np.float64).reshape(-1),
              "nu_vals": nu_vals,
              "elower_vals": np.asarray(df["elower"], dtype=np.float64).reshape(-1),
              "gup_vals": np.asarray(df["gup"], dtype=np.float64).reshape(-1),
              "i_lower_vals": np.asarray(df["i_lower"]).reshape(-1).astype(np.int64, copy=False),
              "lam0_vals": 1.0e8 / nu_vals,
          }

  def load_exomol_transition_chunk(self, local_file):
        if self.cache_info is None or self.cache_info.get("source") != "exomol":
            raise ValueError("ExoMol cache is not prepared for this molecule.")

        try:
            return self._load_exomol_transition_chunk_h5(local_file)
        except (tables.exceptions.HDF5ExtError, tables.NoSuchNodeError, ValueError, OSError) as exc:
            print(f"[{self.species}] fallback to pandas for {local_file}: {type(exc).__name__}: {exc}")
            df = self.load_exomol_transition_dataframe(local_file)
            if len(df) == 0:
                return {
                    "A_vals": np.empty(0, dtype=np.float64),
                    "nu_vals": np.empty(0, dtype=np.float64),
                    "elower_vals": np.empty(0, dtype=np.float64),
                    "gup_vals": np.empty(0, dtype=np.float64),
                    "i_lower_vals": np.empty(0, dtype=np.int64),
                    "lam0_vals": np.empty(0, dtype=np.float64),
                }

            nu_vals = np.asarray(df["nu_lines"], dtype=np.float64).reshape(-1)
            return {
                "A_vals": np.asarray(df["A"], dtype=np.float64).reshape(-1),
                "nu_vals": nu_vals,
                "elower_vals": np.asarray(df["elower"], dtype=np.float64).reshape(-1),
                "gup_vals": np.asarray(df["gup"], dtype=np.float64).reshape(-1),
                "i_lower_vals": np.asarray(df["i_lower"]).reshape(-1).astype(np.int64, copy=False),
                "lam0_vals": 1.0e8 / nu_vals,
            }

  def load_exomol_states_dataframe(self):
      if self.cache_info is None or self.cache_info.get("source") != "exomol":
          raise ValueError("ExoMol cache is not prepared for this molecule.")
      if self._exomol_mgr is None:
          raise ValueError("ExoMol datafile manager is missing on this Molecule object.")
      return self._exomol_mgr.read(self.cache_info["local_states_file"])

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
      verbose=False,
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
    if verbose:
        print(f"[{self.species}] fetch_hitran: molecule_name = {molecule_name}")
        print(f"[{self.species}] fetch_hitran: isotope = {isotope}")
        print(f"[{self.species}] fetch_hitran: localdatabase = {localdatabase}")
        print(f"[{self.species}] fetch_hitran: path = {path}")
        print(f"[{self.species}] fetch_hitran: databank_name = {databank_name}")

    if verbose:
        print(f"[{self.species}] fetch_hitran: building fetch kwargs")
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
    if verbose:
        print(f"[{self.species}] fetch_hitran: final fetch kwargs keys = {list(fetch_kwargs.keys())}")
        if "local_databases" in fetch_kwargs:
            print(f"[{self.species}] fetch_hitran: effective local_databases = {fetch_kwargs['local_databases']}")

    if verbose:
        print(f"[{self.species}] fetch_hitran: calling RADIS fetch_hitran")
    df = fetch_hitran(**fetch_kwargs)
    if verbose:
        print(f"[{self.species}] fetch_hitran: RADIS fetch_hitran returned, rows = {len(df)}")

    if verbose:
        print(f"[{self.species}] fetch_hitran: original dataframe columns = {list(df.columns)}")

    self.source = "hitran"
    self.cache_ready = True
    self.cache_info = {
      "source": "hitran",
      "molecule_name": molecule_name,
      "isotope": isotope,
      "localdatabase": localdatabase,
      "path": path,
      "databank_name": databank_name,
      "cache": cache,
      "engine": engine,
      "output": output,
    }
    if verbose:
        print(f"[{self.species}] fetch_hitran: cache metadata stored on Molecule")
    del df
    return self.cache_info
