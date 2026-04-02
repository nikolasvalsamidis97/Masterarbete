from project_classes.Molecule import Molecule
from project_func.errors import _not_quantity
from astropy import constants as const
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.special import wofz
from radis.io.hitran import fetch_hitran
import pathlib


class BroadeningProfileMolecule:
  def __init__(self,
               molecule: Molecule,
               b,
               lam_min=None,
               lam_max=None,
               dlam=None,
               profileType: str = 'Voigt',
               line_weights=None,
               Temp_atm=None,
               ):
    """
    Build a TOTAL molecular cross-section spectrum on one shared wavelength grid.

    This class now assumes the Molecule object is a cache/setup object only.
    Actual line data are loaded on demand, chunk-by-chunk, during the stitched
    cross-section build.
    """
    self.molecule = molecule
    self.b = b.to(u.km / u.s) if isinstance(b, u.Quantity) else _not_quantity("b (broadening parameter)")
    self.profileType = profileType.lower()
    self.init_timers = {}
    self.line_weights_init = line_weights
    self.Temp_atm_init = Temp_atm

    if not getattr(self.molecule, "cache_ready", False):
      raise ValueError(
        "Molecule cache is not prepared. Run fetch_exomol(...) or fetch_hitran(...) "
        "before creating BroadeningProfileMolecule."
      )

    t0 = time.perf_counter()
    self.c_kms = const.c.to_value(u.km / u.s)
    self.init_timers["set_constants"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    self.lam_min = self.set_lam_min(lam_min)
    self.init_timers["set_lam_min"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    self.lam_max = self.set_lam_max(lam_max)
    self.init_timers["set_lam_max"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    self.dlam = self.set_dlam(dlam)
    self.init_timers["set_dlam"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    self.lam_grid = self.wavelength_grid()
    self.init_timers["wavelength_grid"] = time.perf_counter() - t0

    self.lam_grid_val = self.lam_grid.to_value(u.AA)

    self.line_weights = None
    self.sigmaArray = None
    self.sigmaArray_err = None
    self.sigma_total = None
    self.sigma_total_err = None
    self.temperature_cache = {}
    self._states_gmap = None

    # Compatibility placeholders for older code paths that still expect the
    # previous monolithic-line-array attributes to exist on this class.
    self.A_ul = None
    self.A_ul_err = None
    self.lam0 = None
    self.E_l = None
    self.g_u = None
    self.g_l = None
    self.sig_0 = None
    self.sig_0_err = None
    self.lorentz_FWHM_v = None
    self.lorentz_FWHM_v_err = None
    self.gauss_FWHM_v = None
    self.vlim = None

    # Temporary speed hack: skip lines whose weighted central strength is tiny
    # compared to the strongest weighted line in the current build.
    self.temp_strength_rel_cutoff = 1e-8

    self.init_timers["total_init_profile_setup"] = np.sum(list(self.init_timers.values()))
    print(f"{self.molecule.species} BroadeningProfileMolecule init timing:")
    for key, value in self.init_timers.items():
      print(f"  {key}: {value:.2f} s")

  def _temperature_cache_key(self, Temp_atm):
    T_val = float(np.asarray(Temp_atm.to_value(u.K)).reshape(-1)[0])
    return (T_val, float(self.temp_strength_rel_cutoff), self.profileType)

  def set_lam_min(self, lam_min):
    if lam_min is not None:
      return lam_min.to(u.AA) if isinstance(lam_min, u.Quantity) else _not_quantity("lam_min")
    return (1 / self.molecule.wavenum_max).to(u.AA)

  def set_lam_max(self, lam_max):
    if lam_max is not None:
      return lam_max.to(u.AA) if isinstance(lam_max, u.Quantity) else _not_quantity("lam_max")
    return (1 / self.molecule.wavenum_min).to(u.AA)

  def set_dlam(self, dlam):
    if dlam is not None:
      return dlam.to(u.AA) if isinstance(dlam, u.Quantity) else _not_quantity("dlam")

    rep = 0.5 * (self.lam_min + self.lam_max)
    doppler_sigma = (rep * (self.b / const.c)).to(u.AA)
    dlam_auto = (doppler_sigma / 3.0).to(u.AA)
    floor = 1e-5 * u.AA
    return np.maximum(dlam_auto, floor)

  def wavelength_grid(self):
    npts = int(np.floor(((self.lam_max - self.lam_min) / self.dlam).decompose().value)) + 1
    grid = self.lam_min + np.arange(npts) * self.dlam
    return grid.to(u.AA)

  def calc_central_crossection(self, A_ul, A_ul_err, lam0, g_u, g_l):
    sig0 = (A_ul * (lam0**3 / (8 * np.pi)) * (g_u / g_l))
    sig0 = sig0.to(u.cm**2 * u.km / u.s)

    with np.errstate(divide='ignore', invalid='ignore'):
      frac = np.where(A_ul.value != 0, (A_ul_err / A_ul).to_value(u.dimensionless_unscaled), 0.0)
    sig0_err = sig0 * frac
    return sig0, sig0_err

  def FWHM_lorentz(self, A_ul, A_ul_err, lam0):
    lorentz_FWHM_v = lam0 * (A_ul / (2 * np.pi))
    lorentz_FWHM_v_err = lam0 * (A_ul_err / (2 * np.pi))
    return lorentz_FWHM_v.to(u.km / u.s), lorentz_FWHM_v_err.to(u.km / u.s)

  def FWHM_gauss(self, lam0):
    gauss_scalar = (2 * np.sqrt(np.log(2)) * self.b).to(u.km / u.s)
    gauss = np.full_like(lam0.value, gauss_scalar.value) * gauss_scalar.unit
    return gauss

  def set_vlim(self, gauss_FWHM_v, lorentz_FWHM_v):
    vlim = np.maximum(6 * gauss_FWHM_v, 25 * lorentz_FWHM_v)
    return vlim.to(u.km / u.s)

  def profile_from_velocity(self, v_val, lorentz_val, lorentz_err_val, voigt_sigma_val, voigt_gamma_val, voigt_gamma_err_val):
    if self.profileType == 'lorentz':
      L = lorentz_val
      dL = lorentz_err_val
      phi_val = (1.0 / np.pi) * (0.5 * L) / (v_val**2 + (0.5 * L)**2)
      phi_err_val = np.abs((1.0 / np.pi) * 0.5 * (v_val**2 - L**2 / 4.0) / (v_val**2 + L**2 / 4.0)**2) * dL
      return phi_val, phi_err_val

    if self.profileType == 'gauss':
      b_val = self.b.to_value(u.km / u.s)
      phi_val = (1.0 / (b_val * np.sqrt(np.pi))) * np.exp(-(v_val / b_val)**2)
      phi_err_val = np.zeros_like(phi_val)
      return phi_val, phi_err_val

    sigma = voigt_sigma_val
    gamma = voigt_gamma_val
    dgamma = voigt_gamma_err_val

    z = (v_val + 1j * gamma) / (sigma * np.sqrt(2.0))
    phi_val = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))

    if np.isclose(dgamma, 0.0, equal_nan=True):
      phi_err_val = np.zeros_like(phi_val)
    else:
      gamma_m = max(gamma - dgamma, 1e-30)
      gamma_p = gamma + dgamma
      z_m = (v_val + 1j * gamma_m) / (sigma * np.sqrt(2.0))
      z_p = (v_val + 1j * gamma_p) / (sigma * np.sqrt(2.0))
      phi_m = np.real(wofz(z_m)) / (sigma * np.sqrt(2.0 * np.pi))
      phi_p = np.real(wofz(z_p)) / (sigma * np.sqrt(2.0 * np.pi))
      dphi_dL = (phi_p - phi_m) / ((gamma_p - gamma_m) * 2.0)
      phi_err_val = np.abs(dphi_dL) * (2.0 * dgamma)

    return phi_val, phi_err_val

  def _get_exomol_state_gmap(self):
    if self._states_gmap is None:
      states = self.molecule.load_exomol_states_dataframe()
      self._states_gmap = dict(zip(states["i"], states["g"]))
    return self._states_gmap

  def _load_hitran_dataframe(self):
    info = self.molecule.cache_info
    molecule_name = info["molecule_name"]
    isotope = info["isotope"]
    localdatabase = info["localdatabase"]
    path = info["path"]
    databank_name = info["databank_name"]
    cache = info["cache"]
    engine = info["engine"]
    output = info["output"]

    fetch_kwargs = dict(
      molecule=molecule_name,
      isotope=str(isotope),
      load_wavenum_min=float(self.molecule.wavenum_min.value),
      load_wavenum_max=float(self.molecule.wavenum_max.value),
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
    df = df.rename(columns={
      "A": "A",
      "wav": "nu_lines",
      "El": "elower",
      "gp": "gup",
      "gpp": "glower",
    })
    df = df[["A", "nu_lines", "elower", "gup", "glower"]].copy()
    return df

  def iter_line_dataframes(self, verbose=False):
    source = getattr(self.molecule, "source", None)
    if source == "exomol":
      gmap = self._get_exomol_state_gmap()
      local_files = self.molecule.cache_info["local_trans_files"]
      for i, local_file in enumerate(local_files, start=1):
        if verbose:
          print(f"[{self.molecule.species}] iter_line_dataframes: loading file {i}/{len(local_files)}")
        df_chunk = self.molecule.load_exomol_transition_dataframe(local_file)
        if len(df_chunk) == 0:
          continue
        df_chunk["glower"] = df_chunk["i_lower"].map(gmap)
        df_chunk = df_chunk[["A", "nu_lines", "elower", "gup", "glower"]].copy()
        yield i, len(local_files), df_chunk
    elif source == "hitran":
      if verbose:
        print(f"[{self.molecule.species}] iter_line_dataframes: loading HITRAN dataframe")
      df = self._load_hitran_dataframe()
      yield 1, 1, df
    else:
      raise ValueError(f"Unsupported molecule source for BroadeningProfileMolecule: {source}")

  def _compute_chunk_weights(self, df_chunk, weights, Temp_atm, partition_Z=None):
    if (weights is not None) and (Temp_atm is not None):
      raise ValueError("Give either weights or Temp_atm, not both")

    if Temp_atm is not None:
      T = Temp_atm.to(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")
      kb_eV = const.k_B.to(u.eV / u.K)
      E_l = pd_to_numeric(df_chunk["elower"]).reshape(-1, 1) * (1 / u.cm)
      E_l = E_l.to(u.eV, equivalencies=u.spectral())
      g_l = pd_to_numeric(df_chunk["glower"]).reshape(-1, 1) * u.dimensionless_unscaled
      boltz = (g_l * np.exp(-(E_l / (kb_eV * T)).decompose().value)).reshape(-1)
      return boltz / partition_Z

    if weights is None:
      return np.ones(len(df_chunk), dtype=np.float64)

    return np.asarray(weights, dtype=np.float64).reshape(-1)

  def _compute_partition_function(self, Temp_atm, verbose=False):
    T = Temp_atm.to(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")
    kb_eV = const.k_B.to(u.eV / u.K)

    unique_states = set()
    for _, _, df_chunk in self.iter_line_dataframes(verbose=verbose):
      el_vals = pd_to_numeric(df_chunk["elower"])
      gl_vals = pd_to_numeric(df_chunk["glower"])
      pairs = np.unique(np.column_stack((el_vals, gl_vals)), axis=0)
      for pair in pairs:
        unique_states.add((float(pair[0]), float(pair[1])))

    if len(unique_states) == 0:
      raise ValueError("No lower-state information found while building partition function")

    states_arr = np.array(list(unique_states), dtype=float)
    El_unique = states_arr[:, 0].reshape(-1, 1) * (1 / u.cm)
    El_unique = El_unique.to(u.eV, equivalencies=u.spectral())
    gl_unique = states_arr[:, 1].reshape(-1, 1) * u.dimensionless_unscaled

    boltz_unique = gl_unique * np.exp(-(El_unique / (kb_eV * T)).decompose().value)
    Z = np.nansum(boltz_unique)
    if not np.isfinite(Z) or Z == 0:
      raise ValueError("Partition function is zero or non-finite for the requested temperature")
    return float(Z)

  def build_total_crossection(self, weights=None, Temp_atm=None, store_weights=False, verbose=False, progress_every=100000):
    if (weights is not None) and (Temp_atm is not None):
      raise ValueError("Give either weights or Temp_atm, not both")

    cache_key = None
    if Temp_atm is not None:
      cache_key = self._temperature_cache_key(Temp_atm)
      if cache_key in self.temperature_cache:
        cached = self.temperature_cache[cache_key]
        if verbose:
          print(f"Using cached molecular cross-section for {self.molecule.species} at T = {Temp_atm:.3g}")
        if store_weights:
          self.line_weights = cached["weights"].copy()
        return cached["sigma"].copy(), cached["sigma_err"].copy()

    t_start = time.perf_counter()
    sigma_total_val = np.zeros(self.lam_grid_val.shape, dtype=np.float64)
    sigma_total_err2_val = np.zeros(self.lam_grid_val.shape, dtype=np.float64)

    partition_Z = None
    if Temp_atm is not None:
      if verbose:
        print(f"Computing partition function for {self.molecule.species} at T = {Temp_atm:.3g}")
      partition_Z = self._compute_partition_function(Temp_atm, verbose=verbose)
      if verbose:
        print(f"Partition function ready for {self.molecule.species}: Z = {partition_Z:.6e}")

    if store_weights:
      self.line_weights = None

    line_offset = 0
    processed_lines = 0

    for chunk_index, n_chunks, df_chunk in self.iter_line_dataframes(verbose=verbose):
      n_chunk = len(df_chunk)
      if n_chunk == 0:
        continue

      if verbose:
        print(f"[{self.molecule.species}] build_total_crossection: processing chunk {chunk_index}/{n_chunks}, rows = {n_chunk}")

      if Temp_atm is not None:
        weights_chunk = self._compute_chunk_weights(df_chunk, None, Temp_atm, partition_Z=partition_Z)
      elif weights is None:
        weights_chunk = np.ones(n_chunk, dtype=np.float64)
      else:
        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        weights_chunk = weights_arr[line_offset:line_offset + n_chunk]
        if len(weights_chunk) != n_chunk:
          raise ValueError("weights must have one value per molecular line across all chunks")

      A_ul, A_ul_err, lam0, E_l, g_u, g_l = self.molecule.pandas_to_numpy(data=df_chunk)
      sig0, sig0_err = self.calc_central_crossection(A_ul, A_ul_err, lam0, g_u, g_l)
      lorentz_FWHM_v, lorentz_FWHM_v_err = self.FWHM_lorentz(A_ul, A_ul_err, lam0)
      gauss_FWHM_v = self.FWHM_gauss(lam0)
      vlim = self.set_vlim(gauss_FWHM_v, lorentz_FWHM_v)

      lam0_val = lam0[:, 0].to_value(u.AA)
      sig0_val = sig0[:, 0].to_value(u.cm**2 * u.km / u.s)
      sig0_err_val = sig0_err[:, 0].to_value(u.cm**2 * u.km / u.s)
      lorentz_val = lorentz_FWHM_v[:, 0].to_value(u.km / u.s)
      lorentz_err_val = lorentz_FWHM_v_err[:, 0].to_value(u.km / u.s)
      gauss_val = gauss_FWHM_v[:, 0].to_value(u.km / u.s)
      vlim_val = vlim[:, 0].to_value(u.km / u.s)
      voigt_sigma_val = gauss_val / (2.0 * np.sqrt(2.0 * np.log(2.0)))
      voigt_gamma_val = 0.5 * lorentz_val
      voigt_gamma_err_val = 0.5 * lorentz_err_val

      weighted_strength = np.abs(weights_chunk * sig0_val)
      max_weighted_strength = np.nanmax(weighted_strength) if np.any(np.isfinite(weighted_strength)) else 0.0
      strength_cutoff = self.temp_strength_rel_cutoff * max_weighted_strength if max_weighted_strength > 0.0 else 0.0

      halfwidth_val = lam0_val * vlim_val / self.c_kms
      lam_lo_val = lam0_val - halfwidth_val
      lam_hi_val = lam0_val + halfwidth_val
      i0_arr = np.searchsorted(self.lam_grid_val, lam_lo_val, side='left')
      i1_arr = np.searchsorted(self.lam_grid_val, lam_hi_val, side='right')

      for i in range(n_chunk):
        weight_i = weights_chunk[i]
        if not np.isfinite(weight_i) or weight_i == 0.0:
          continue
        if weighted_strength[i] < strength_cutoff:
          continue

        i0 = i0_arr[i]
        i1 = i1_arr[i]
        if i1 <= i0:
          continue

        lam_local_val = self.lam_grid_val[i0:i1]
        v_local_val = self.c_kms * ((lam_local_val - lam0_val[i]) / lam0_val[i])
        phi_val, phi_err_val = self.profile_from_velocity(
          v_local_val,
          lorentz_val[i],
          lorentz_err_val[i],
          voigt_sigma_val[i],
          voigt_gamma_val[i],
          voigt_gamma_err_val[i],
        )

        sigma_line_val = sig0_val[i] * phi_val
        sigma_line_err_val = sig0_val[i] * phi_err_val + sig0_err_val[i] * phi_val

        sigma_total_val[i0:i1] += weight_i * sigma_line_val
        sigma_total_err2_val[i0:i1] += (weight_i * sigma_line_err_val) ** 2

        processed_lines += 1
        if verbose and (processed_lines == 1 or processed_lines % progress_every == 0):
          elapsed = time.perf_counter() - t_start
          print(f"Cross-section build progress for {self.molecule.species}: processed line {processed_lines}, elapsed = {elapsed:.2f} s")

      line_offset += n_chunk

    sigma_total = sigma_total_val * u.cm**2
    sigma_total_err = np.sqrt(sigma_total_err2_val) * u.cm**2

    if cache_key is not None:
      self.temperature_cache[cache_key] = {
        "sigma": sigma_total.copy(),
        "sigma_err": sigma_total_err.copy(),
        "weights": np.array([], dtype=np.float64),
      }

    if verbose:
      total_time = time.perf_counter() - t_start
      print(f"Finished total cross-section for {self.molecule.species} in {total_time:.2f} s")

    return sigma_total, sigma_total_err

  def set_line_weights(self, weights):
    self.sigmaArray, self.sigmaArray_err = self.build_total_crossection(
      weights=weights,
      store_weights=True,
    )
    self.sigma_total = self.sigmaArray
    self.sigma_total_err = self.sigmaArray_err
    return self.sigmaArray, self.sigmaArray_err

  def clear_line_weights(self):
    self.sigmaArray, self.sigmaArray_err = self.build_total_crossection(
      weights=None,
      store_weights=True,
    )
    self.sigma_total = self.sigmaArray
    self.sigma_total_err = self.sigmaArray_err
    return self.sigmaArray, self.sigmaArray_err

  def apply_boltzmann_weights(self, Temp_atm):
    self.sigmaArray, self.sigmaArray_err = self.build_total_crossection(
      Temp_atm=Temp_atm,
      store_weights=True,
      verbose=True,
    )
    self.sigma_total = self.sigmaArray
    self.sigma_total_err = self.sigmaArray_err
    return self.sigmaArray, self.sigmaArray_err

  def plot_total_crossection(self, xscale: str = 'linear', yscale: str = 'log', xlim: tuple = None):
    lam = self.lam_grid.to_value(u.AA)
    sig = self.sigmaArray.to_value(u.cm**2)
    sig_err = self.sigmaArray_err.to_value(u.cm**2)

    plt.figure(figsize=(10, 4))
    plt.plot(lam, sig)
    plt.fill_between(lam, np.maximum(sig - sig_err, 0.0), sig + sig_err, alpha=0.3, color='red', label='error')
    plt.xlabel(f"Wavelength [{self.lam_grid.unit}]")
    plt.ylabel(r"Total cross-section $\sigma_{\rm tot}$ [cm$^2$]")
    plt.title(f"{self.molecule.species} total molecular cross-section")
    if xlim is not None:
      plt.xlim(xlim)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.legend()
    plt.tight_layout()
    plt.show()


def pd_to_numeric(series):
  return np.asarray(series, dtype=np.float64).reshape(-1)