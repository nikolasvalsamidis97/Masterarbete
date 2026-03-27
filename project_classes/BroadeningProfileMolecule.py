from project_classes.Molecule import Molecule
from project_func.errors import _not_quantity
from astropy import constants as const
from astropy import units as u
from astropy.modeling.models import Voigt1D
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.special import wofz


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

    Parameters
    ----------
    molecule : Molecule
        Molecule object with loaded line data.
    b : Quantity
        Doppler broadening parameter [km/s].
    lam_min, lam_max : Quantity, optional
        Global wavelength range. If omitted, uses the molecular line range.
    dlam : Quantity, optional
        Wavelength spacing of the global grid. If omitted, an adaptive value
        based on the narrowest Doppler width is used.
    profileType : str
        'lorentz', 'gauss', or 'voigt'.
    line_weights : array-like, optional
        Optional per-line weights for later stitched-spectrum construction.
    Temp_atm : Quantity, optional
        Optional temperature for later Boltzmann-weighted stitched-spectrum construction.
    """
    self.molecule = molecule
    self.b = b.to(u.km / u.s) if isinstance(b, u.Quantity) else _not_quantity("b (broadening parameter)")
    self.profileType = profileType.lower()
    self.init_timers = {}
    self.line_weights_init = line_weights
    self.Temp_atm_init = Temp_atm

    if getattr(self.molecule, "data", None) is None:
      raise ValueError("Molecule.data is empty. Load molecular line data before creating BroadeningProfileMolecule.")

    # Ensure numpy line arrays exist on the molecule object
    self.A_ul, self.A_ul_err, self.lam0, self.E_l, self.g_u, self.g_l = self.molecule.pandas_to_numpy()

    t0 = time.perf_counter()
    self.sig_0, self.sig_0_err = self.calc_central_crossection()
    self.init_timers["calc_central_crossection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    self.lorentz_FWHM_v, self.lorentz_FWHM_v_err = self.FWHM_lorentz()
    self.init_timers["FWHM_lorentz"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    self.gauss_FWHM_v = self.FWHM_gauss()
    self.init_timers["FWHM_gauss"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    self.vlim = self.set_vlim()
    self.init_timers["set_vlim"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    self._precompute_numeric_arrays()
    self.init_timers["precompute_numeric_arrays"] = time.perf_counter() - t0

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

    t0 = time.perf_counter()
    self.line_bounds = self.precompute_line_bounds()
    self.init_timers["precompute_line_bounds"] = time.perf_counter() - t0

    self.line_weights = None
    self.sigmaArray = None
    self.sigmaArray_err = None
    self.sigma_total = None
    self.sigma_total_err = None
    self.temperature_cache = {}

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
    if lam_min is None:
      return np.nanmin(self.lam0).to(u.AA)
    return lam_min.to(u.AA) if isinstance(lam_min, u.Quantity) else _not_quantity("lam_min")
  
  def set_vlim(self):
    vlim = np.maximum(6 * self.gauss_FWHM_v, 25 * self.lorentz_FWHM_v)
    return vlim.to(u.km / u.s)

  def set_lam_max(self, lam_max):
    if lam_max is None:
      return np.nanmax(self.lam0).to(u.AA)
    return lam_max.to(u.AA) if isinstance(lam_max, u.Quantity) else _not_quantity("lam_max")

  def set_dlam(self, dlam):
    if dlam is not None:
      return dlam.to(u.AA) if isinstance(dlam, u.Quantity) else _not_quantity("dlam")

    lam0 = self.lam0.to(u.AA)
    doppler_sigma = (lam0 * (self.b / const.c)).to(u.AA)
    narrowest = np.nanmin(doppler_sigma)

    # Sample the narrowest Doppler sigma with ~6 points as a safe default.
    rep = np.nanpercentile(doppler_sigma.value, 10) * u.AA
    dlam_auto = (rep / 3.0).to(u.AA)

    # Prevent pathologically tiny grids.
    floor = 1e-5 * u.AA
    return np.maximum(dlam_auto, floor)

  def wavelength_grid(self):
    npts = int(np.floor(((self.lam_max - self.lam_min) / self.dlam).decompose().value)) + 1
    grid = self.lam_min + np.arange(npts) * self.dlam
    return grid.to(u.AA)

  def precompute_line_bounds(self):
    lam_grid_val = self.lam_grid.to_value(u.AA)
    halfwidth_val = self.lam0_val * self.vlim_val / self.c_kms
    lam_lo_val = self.lam0_val - halfwidth_val
    lam_hi_val = self.lam0_val + halfwidth_val

    i0 = np.searchsorted(lam_grid_val, lam_lo_val, side='left')
    i1 = np.searchsorted(lam_grid_val, lam_hi_val, side='right')
    return np.column_stack((i0, i1)).astype(np.int64)

  def calc_central_crossection(self):
    sig0 = (self.A_ul * (self.lam0**3 / (8 * np.pi)) * (self.g_u / self.g_l))
    sig0 = sig0.to(u.cm**2 * u.km / u.s)

    with np.errstate(divide='ignore', invalid='ignore'):
      frac = np.where(self.A_ul.value != 0, (self.A_ul_err / self.A_ul).to_value(u.dimensionless_unscaled), 0.0)
    sig0_err = sig0 * frac
    return sig0, sig0_err

  def FWHM_lorentz(self):
    lorentz_FWHM_v = self.lam0 * (self.A_ul / (2 * np.pi))
    lorentz_FWHM_v_err = self.lam0 * (self.A_ul_err / (2 * np.pi))
    return lorentz_FWHM_v.to(u.km / u.s), lorentz_FWHM_v_err.to(u.km / u.s)

  def FWHM_gauss(self):
    gauss_scalar = (2 * np.sqrt(np.log(2)) * self.b).to(u.km / u.s)
    gauss = np.full_like(self.lam0.value, gauss_scalar.value) * gauss_scalar.unit
    return gauss

  def line_window_halfwidth_lam(self, idx: int):
    v_halfwidth = self.vlim[idx, 0]
    dlam = (self.lam0[idx, 0] * (v_halfwidth / const.c)).to(u.AA)
    return dlam

  def lambda_offsets_to_velocity(self, lam_local, lam0):
    return (const.c * ((lam_local - lam0) / lam0)).to(u.km / u.s)

  def profile_from_velocity(self, v_val, idx: int):
    """
    Numeric profile evaluation working directly on float velocity arrays in km/s.
    """
    if self.profileType == 'lorentz':
      L = self.lorentz_val[idx]
      dL = self.lorentz_err_val[idx]
      phi_val = (1.0 / np.pi) * (0.5 * L) / (v_val**2 + (0.5 * L)**2)
      phi_err_val = np.abs((1.0 / np.pi) * 0.5 * (v_val**2 - L**2 / 4.0) / (v_val**2 + L**2 / 4.0)**2) * dL
      return phi_val, phi_err_val

    if self.profileType == 'gauss':
      b_val = self.b.to_value(u.km / u.s)
      phi_val = (1.0 / (b_val * np.sqrt(np.pi))) * np.exp(-(v_val / b_val)**2)
      phi_err_val = np.zeros_like(phi_val)
      return phi_val, phi_err_val

    sigma = self.voigt_sigma_val[idx]
    gamma = self.voigt_gamma_val[idx]
    dgamma = self.voigt_gamma_err_val[idx]

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
  def _precompute_numeric_arrays(self):
    self.c_kms = const.c.to_value(u.km / u.s)
    self.lam0_val = self.lam0[:, 0].to_value(u.AA)
    self.sig0_val = self.sig_0[:, 0].to_value(u.cm**2 * u.km / u.s)
    self.sig0_err_val = self.sig_0_err[:, 0].to_value(u.cm**2 * u.km / u.s)
    self.lorentz_val = self.lorentz_FWHM_v[:, 0].to_value(u.km / u.s)
    self.lorentz_err_val = self.lorentz_FWHM_v_err[:, 0].to_value(u.km / u.s)
    self.gauss_val = self.gauss_FWHM_v[:, 0].to_value(u.km / u.s)
    self.vlim_val = self.vlim[:, 0].to_value(u.km / u.s)
    self.voigt_sigma_val = self.gauss_val / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    self.voigt_gamma_val = 0.5 * self.lorentz_val
    self.voigt_gamma_err_val = 0.5 * self.lorentz_err_val


  def build_total_crossection(self, weights=None, Temp_atm=None, store_weights=False, verbose=False, progress_every=100000):
    """
    Build the stitched molecular cross-section spectrum on the shared wavelength grid.

    Parameters
    ----------
    weights : array-like, optional
        Optional per-line weights. Must have one value per molecular line.
    Temp_atm : Quantity, optional
        If given, Boltzmann line weights are computed automatically from this
        temperature and applied during the stitched-spectrum build.
    store_weights : bool, optional
        If True, store the active weights on self.line_weights.
    verbose : bool, optional
        If True, print sparse progress information while building the spectrum.
    progress_every : int, optional
        Print progress every N processed molecular lines.

    Notes
    -----
    Temporary speed hack: lines with weighted central strength
    (weight_i * sig0_i) below
    self.temp_strength_rel_cutoff * max(weights * sig0)
    are skipped.
    """
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

      if verbose:
        print(f"Computing Boltzmann weights for {self.molecule.species} at T = {Temp_atm:.3g}")
      weights_arr = self.boltzmann_line_weights(Temp_atm)
    elif weights is None:
      weights_arr = np.ones(len(self.lam0), dtype=np.float64)
    else:
      weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
      if len(weights_arr) != len(self.lam0):
        raise ValueError("weights must have one value per molecular line")

    if store_weights:
      self.line_weights = None if (Temp_atm is None and weights is None) else weights_arr.copy()

    t_start = time.perf_counter()

    lam_grid_val = self.lam_grid_val
    lam0_val = self.lam0_val
    sig0_val = self.sig0_val
    sig0_err_val = self.sig0_err_val

    weighted_strength = np.abs(weights_arr * sig0_val)
    max_weighted_strength = np.nanmax(weighted_strength) if np.any(np.isfinite(weighted_strength)) else 0.0
    if max_weighted_strength > 0.0:
      strength_cutoff = self.temp_strength_rel_cutoff * max_weighted_strength
    else:
      strength_cutoff = 0.0

    sigma_total_val = np.zeros(lam_grid_val.shape, dtype=np.float64)
    sigma_total_err2_val = np.zeros(lam_grid_val.shape, dtype=np.float64)

    n_lines = len(lam0_val)
    c_kms = self.c_kms

    if verbose:
      n_active_lines = int(np.sum(weighted_strength >= strength_cutoff)) if max_weighted_strength > 0.0 else 0
      print(f"Building total cross-section for {self.molecule.species} with {n_lines} lines")
      print(
        f"Temporary weighted-strength cutoff = {self.temp_strength_rel_cutoff:.1e}; "
        f"active lines = {n_active_lines}/{n_lines}"
      )

    for i in range(n_lines):
      weight_i = weights_arr[i]
      if not np.isfinite(weight_i) or weight_i == 0.0:
        continue
      if weighted_strength[i] < strength_cutoff:
        continue

      i0 = self.line_bounds[i, 0]
      i1 = self.line_bounds[i, 1]
      if i1 <= i0:
        continue

      lam_local_val = lam_grid_val[i0:i1]
      v_local_val = c_kms * ((lam_local_val - lam0_val[i]) / lam0_val[i])
      phi_val, phi_err_val = self.profile_from_velocity(v_local_val, i)

      sigma_line_val = sig0_val[i] * phi_val
      sigma_line_err_val = sig0_val[i] * phi_err_val + sig0_err_val[i] * phi_val

      sigma_total_val[i0:i1] += weight_i * sigma_line_val
      sigma_total_err2_val[i0:i1] += (weight_i * sigma_line_err_val) ** 2

      if verbose and ((i + 1) == 1 or (i + 1) % progress_every == 0 or (i + 1) == n_lines):
        elapsed = time.perf_counter() - t_start
        print(f"Cross-section build progress for {self.molecule.species}: line {i + 1}/{n_lines}, elapsed = {elapsed:.2f} s")

    sigma_total = sigma_total_val * u.cm**2
    sigma_total_err = np.sqrt(sigma_total_err2_val) * u.cm**2

    if cache_key is not None:
      self.temperature_cache[cache_key] = {
        "sigma": sigma_total.copy(),
        "sigma_err": sigma_total_err.copy(),
        "weights": weights_arr.copy(),
      }

    if verbose:
      total_time = time.perf_counter() - t_start
      print(f"Finished total cross-section for {self.molecule.species} in {total_time:.2f} s")

    return sigma_total, sigma_total_err

  def set_line_weights(self, weights):
    """
    Store per-line weights and rebuild the stitched molecular spectrum.
    """
    weights_arr = np.asarray(weights, dtype=float).reshape(-1)
    if len(weights_arr) != len(self.lam0):
      raise ValueError("weights must have one value per molecular line")

    self.sigmaArray, self.sigmaArray_err = self.build_total_crossection(
      weights=weights_arr,
      store_weights=True,
    )
    self.sigma_total = self.sigmaArray
    self.sigma_total_err = self.sigmaArray_err
    return self.sigmaArray, self.sigmaArray_err

  def clear_line_weights(self):
    """
    Rebuild the stitched molecular spectrum without line weights.
    """
    self.sigmaArray, self.sigmaArray_err = self.build_total_crossection(
      weights=None,
      store_weights=True,
    )
    self.sigma_total = self.sigmaArray
    self.sigma_total_err = self.sigmaArray_err
    return self.sigmaArray, self.sigmaArray_err

  def boltzmann_line_weights(self, Temp_atm):
    """
    Build normalized lower-state Boltzmann weights for each molecular line.
    Lines sharing the same lower state get the same population factor.
    """
    T = Temp_atm.to(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")
    kb_eV = const.k_B.to(u.eV / u.K)

    El = self.E_l.reshape(-1, 1)
    gl = self.g_l.reshape(-1, 1)

    Eg = np.column_stack([El.value.reshape(-1), gl.value.reshape(-1)])
    Eg_unique, _, inv = np.unique(Eg, axis=0, return_index=True, return_inverse=True)
    El_unique = Eg_unique[:, 0].reshape(-1, 1) * u.eV
    gl_unique = Eg_unique[:, 1].reshape(-1, 1) * u.dimensionless_unscaled

    boltz_unique = gl_unique * np.exp(-(El_unique / (kb_eV * T)).decompose().value)
    Z = np.nansum(boltz_unique)
    if not np.isfinite(Z) or Z == 0:
      raise ValueError("Partition function is zero or non-finite for the requested temperature")

    weights = (boltz_unique[inv] / Z).reshape(-1)
    return weights

  def apply_boltzmann_weights(self, Temp_atm):
    """
    Compute Boltzmann line weights and rebuild the stitched molecular spectrum.
    """
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

  def plot_window_for_line(self, line: int):
    lam_center = self.lam0[line, 0]
    halfwidth = self.line_window_halfwidth_lam(line)
    lam_lo = (lam_center - halfwidth).to_value(u.AA)
    lam_hi = (lam_center + halfwidth).to_value(u.AA)

    lam = self.lam_grid.to_value(u.AA)
    sig = self.sigmaArray.to_value(u.cm**2)

    mask = (lam >= lam_lo) & (lam <= lam_hi)

    plt.figure(figsize=(9, 4))
    plt.plot(lam[mask], sig[mask])
    plt.axvline(lam_center.to_value(u.AA), linestyle='--')
    plt.xlabel(f"Wavelength [{self.lam_grid.unit}]")
    plt.ylabel(r"Total cross-section $\sigma_{\rm tot}$ [cm$^2$]")
    plt.title(f"{self.molecule.species} line window around {lam_center:.4f}")
    plt.tight_layout()
    plt.show()