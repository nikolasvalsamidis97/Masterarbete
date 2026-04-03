import astropy.units as u
from project_func.errors import _not_quantity
from molmass import Formula
from radis.api.exomolapi import MdbExomol
import numpy as np
import pandas as pd
import pathlib
import time
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
      t_h5_total_start = time.perf_counter()
      t_open_start = time.perf_counter()

      with tables.open_file(local_file, mode="r") as h5:
          t_open = time.perf_counter() - t_open_start
          t_find_table_start = time.perf_counter()

          table_node = None
          for node in h5.walk_nodes("/", classname="Table"):
              table_node = node
              colnames = set(getattr(node, "colnames", []))
              if required_cols.issubset(colnames):
                  break

          t_find_table = time.perf_counter() - t_find_table_start

          if table_node is None:
              raise ValueError(
                  f"Could not find any HDF5 table in {local_file}"
              )

          colnames = list(getattr(table_node, "colnames", []))

          # Layout 1: direct named columns
          if required_cols.issubset(set(colnames)):
              t_direct_start = time.perf_counter()
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
              t_direct = time.perf_counter() - t_direct_start
              t_h5_total = time.perf_counter() - t_h5_total_start
              print(
                  f"[{self.species}] _load_exomol_transition_chunk_h5 direct-table timing: "
                  f"open = {t_open:.2f} s, find_table = {t_find_table:.2f} s, "
                  f"read/filter = {t_direct:.2f} s, total = {t_h5_total:.2f} s, file = {local_file}"
              )
              return {
                  "A_vals": A_vals,
                  "nu_vals": nu_vals,
                  "elower_vals": elower_vals,
                  "gup_vals": gup_vals,
                  "i_lower_vals": i_lower_vals,
                  "lam0_vals": 1.0e8 / nu_vals,
              }

          # Layout 2: pandas block table, e.g. /df/table with values_block_* columns.
          # First try a lower-level PyTables path to avoid the heavy pd.read_hdf(...) cost.
          t_rows_start = time.perf_counter()
          rows = table_node.read_where(f"(nu_lines >= {nu_min}) & (nu_lines <= {nu_max})")
          t_read_rows = time.perf_counter() - t_rows_start

          if len(rows) == 0:
              print(
                  f"[{self.species}] _load_exomol_transition_chunk_h5 pandas-table timing: "
                  f"open = {t_open:.2f} s, find_table = {t_find_table:.2f} s, "
                  f"read_rows = {t_read_rows:.2f} s, rows = 0, file = {local_file}"
              )
              return {
                  "A_vals": np.empty(0, dtype=np.float64),
                  "nu_vals": np.empty(0, dtype=np.float64),
                  "elower_vals": np.empty(0, dtype=np.float64),
                  "gup_vals": np.empty(0, dtype=np.float64),
                  "i_lower_vals": np.empty(0, dtype=np.int64),
                  "lam0_vals": np.empty(0, dtype=np.float64),
              }

          t_unpack_start = time.perf_counter()
          nu_vals = np.asarray(rows["nu_lines"], dtype=np.float64).reshape(-1)

          key = table_node._v_parent._v_pathname
          with pd.HDFStore(local_file, mode="r") as store:
              storer = store.get_storer(key.lstrip("/"))
              logical_columns = list(storer.non_index_axes[0][1])
              block_items_map = {}
              for name in getattr(table_node, "colnames", []):
                  if name.startswith("values_block_"):
                      items_attr = getattr(storer.attrs, f"{name}_items", None)
                      if items_attr is not None:
                          block_items_map[name] = list(items_attr)

          float_blocks = []
          int_blocks = []
          for name in colnames:
              if name == "index" or name == "nu_lines":
                  continue
              if not name.startswith("values_block_"):
                  continue

              values = np.asarray(rows[name])
              if values.ndim == 1:
                  values = values.reshape(-1, 1)
              elif values.ndim > 2:
                  values = values.reshape(values.shape[0], -1)

              if np.issubdtype(values.dtype, np.floating):
                  float_blocks.append((name, values))
              elif np.issubdtype(values.dtype, np.integer):
                  int_blocks.append((name, values))
              else:
                  raise ValueError(
                      f"Unsupported pandas block dtype for {name} in {local_file}: {values.dtype}"
                  )

          float_blocks.sort(key=lambda x: x[0])
          int_blocks.sort(key=lambda x: x[0])

          float_concat = np.concatenate([v for _, v in float_blocks], axis=1) if float_blocks else np.empty((len(rows), 0), dtype=np.float64)
          int_concat = np.concatenate([v for _, v in int_blocks], axis=1) if int_blocks else np.empty((len(rows), 0), dtype=np.int64)

          print(
              f"[{self.species}] pandas-block debug: logical_columns = {logical_columns}, block_items_map = {block_items_map}, colnames = {colnames}, "
              f"float_blocks = {[(name, arr.shape, str(arr.dtype)) for name, arr in float_blocks]}, "
              f"int_blocks = {[(name, arr.shape, str(arr.dtype)) for name, arr in int_blocks]}, "
              f"float_concat.shape = {float_concat.shape}, int_concat.shape = {int_concat.shape}, file = {local_file}"
          )

          # Use the exact pandas block-item metadata to reconstruct logical columns.
          block_column_map = {}
          for name, values in float_blocks + int_blocks:
              items = block_items_map.get(name)
              if items is None:
                  continue
              if values.shape[1] != len(items):
                  raise ValueError(
                      f"Pandas block item metadata mismatch for {name} in {local_file}: "
                      f"values.shape[1] = {values.shape[1]}, len(items) = {len(items)}"
                  )
              for j, item_name in enumerate(items):
                  block_column_map[item_name] = values[:, j]

          if {"A", "elower", "gup", "i_lower"}.issubset(block_column_map.keys()):
              A_vals = np.asarray(block_column_map["A"], dtype=np.float64)
              elower_vals = np.asarray(block_column_map["elower"], dtype=np.float64)
              gup_vals = np.asarray(block_column_map["gup"], dtype=np.float64)
              i_lower_vals = np.asarray(block_column_map["i_lower"]).reshape(-1).astype(np.int64, copy=False)

              if len(rows) <= 10000:
                  where = f"nu_lines >= {nu_min} & nu_lines <= {nu_max}"
                  df_check = pd.read_hdf(
                      local_file,
                      key=key,
                      where=where,
                      columns=["A", "nu_lines", "elower", "gup", "i_lower"],
                  )
                  n_show = min(5, len(df_check), len(nu_vals))
                  print(
                      f"[{self.species}] pandas-unpack check file = {local_file}, n_show = {n_show}, "
                      f"A_low = {A_vals[:n_show].tolist()}, A_pd = {np.asarray(df_check['A'], dtype=np.float64)[:n_show].tolist()}"
                  )
                  print(
                      f"[{self.species}] pandas-unpack check file = {local_file}, n_show = {n_show}, "
                      f"nu_low = {nu_vals[:n_show].tolist()}, nu_pd = {np.asarray(df_check['nu_lines'], dtype=np.float64)[:n_show].tolist()}"
                  )
                  print(
                      f"[{self.species}] pandas-unpack check file = {local_file}, n_show = {n_show}, "
                      f"elower_low = {elower_vals[:n_show].tolist()}, elower_pd = {np.asarray(df_check['elower'], dtype=np.float64)[:n_show].tolist()}"
                  )
                  print(
                      f"[{self.species}] pandas-unpack check file = {local_file}, n_show = {n_show}, "
                      f"gup_low = {gup_vals[:n_show].tolist()}, gup_pd = {np.asarray(df_check['gup'], dtype=np.float64)[:n_show].tolist()}"
                  )
                  print(
                      f"[{self.species}] pandas-unpack check file = {local_file}, n_show = {n_show}, "
                      f"i_lower_low = {i_lower_vals[:n_show].tolist()}, i_lower_pd = {np.asarray(df_check['i_lower']).reshape(-1).astype(np.int64, copy=False)[:n_show].tolist()}"
                  )

              t_unpack = time.perf_counter() - t_unpack_start
              t_h5_total = time.perf_counter() - t_h5_total_start
              print(
                  f"[{self.species}] _load_exomol_transition_chunk_h5 pandas-table timing: "
                  f"open = {t_open:.2f} s, find_table = {t_find_table:.2f} s, "
                  f"read_rows = {t_read_rows:.2f} s, unpack = {t_unpack:.2f} s, rows = {len(rows)}, file = {local_file}"
              )
              print(
                  f"[{self.species}] _load_exomol_transition_chunk_h5 pandas-table total = {t_h5_total:.2f} s, file = {local_file}"
              )
              return {
                  "A_vals": A_vals,
                  "nu_vals": nu_vals,
                  "elower_vals": elower_vals,
                  "gup_vals": gup_vals,
                  "i_lower_vals": i_lower_vals,
                  "lam0_vals": 1.0e8 / nu_vals,
              }

          # Safe fallback if the block layout is not the expected one.
          print(
              f"[{self.species}] pandas-block unpack did not match expected layout; "
              f"falling back to pd.read_hdf for file = {local_file}"
          )
          t_pandas_start = time.perf_counter()
          key = table_node._v_parent._v_pathname
          where = f"nu_lines >= {nu_min} & nu_lines <= {nu_max}"
          df = pd.read_hdf(
              local_file,
              key=key,
              where=where,
              columns=["A", "nu_lines", "elower", "gup", "i_lower"],
          )
          t_read_hdf = time.perf_counter() - t_pandas_start
          print(
              f"[{self.species}] _load_exomol_transition_chunk_h5 pandas fallback timing: "
              f"open = {t_open:.2f} s, find_table = {t_find_table:.2f} s, "
              f"read_rows = {t_read_rows:.2f} s, read_hdf = {t_read_hdf:.2f} s, rows = {len(df)}, key = {key}, file = {local_file}"
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
          t_h5_total = time.perf_counter() - t_h5_total_start
          print(
              f"[{self.species}] _load_exomol_transition_chunk_h5 pandas fallback total = {t_h5_total:.2f} s, file = {local_file}"
          )
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
