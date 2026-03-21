

from project_classes.Molecule import Molecule
from project_func.errors import _not_quantity
from astropy import constants as const
from astropy import units as u
from astropy.modeling.models import Voigt1D
import numpy as np
from matplotlib import pyplot as plt


class BroadeningProfileMolecule:
  def __init__(self,
               molecule: Molecule,
               b,
               lam_min=None,
               lam_max=None,
               dlam=None,
               profileType: str = 'Voigt',
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
    """
    self.molecule = molecule
    self.b = b.to(u.km / u.s) if isinstance(b, u.Quantity) else _not_quantity("b (broadening parameter)")
    self.profileType = profileType.lower()

    if getattr(self.molecule, "data", None) is None:
      raise ValueError("Molecule.data is empty. Load molecular line data before creating BroadeningProfileMolecule.")

    # Ensure numpy line arrays exist on the molecule object
    self.A_ul, self.A_ul_err, self.lam0, self.E_l, self.g_u, self.g_l = self.molecule.pandas_to_numpy()

    self.sig_0, self.sig_0_err = self.calc_central_crossection()
    self.lorentz_FWHM_v, self.lorentz_FWHM_v_err = self.FWHM_lorentz()
    self.gauss_FWHM_v = self.FWHM_gauss()
    self.vlim = self.set_vlim()

    self.lam_min = self.set_lam_min(lam_min)
    self.lam_max = self.set_lam_max(lam_max)
    self.dlam = self.set_dlam(dlam)
    self.lam_grid = self.wavelength_grid()

    self.sigma_total, self.sigma_total_err = self.build_total_crossection()

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
    dlam_auto = (narrowest / 6.0).to(u.AA)

    # Prevent pathologically tiny grids.
    floor = 1e-5 * u.AA
    return np.maximum(dlam_auto, floor)

  def wavelength_grid(self):
    npts = int(np.floor(((self.lam_max - self.lam_min) / self.dlam).decompose().value)) + 1
    grid = self.lam_min + np.arange(npts) * self.dlam
    return grid.to(u.AA)

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

  def profile_from_velocity(self, v, idx: int):
    if self.profileType == 'lorentz':
      L = self.lorentz_FWHM_v[idx, 0]
      phi = (1 / np.pi) * (0.5 * L) / (v**2 + (0.5 * L)**2)
      phi_err = np.abs((1 / np.pi) * 0.5 * (v**2 - L**2 / 4.0) / (v**2 + L**2 / 4.0)**2) * self.lorentz_FWHM_v_err[idx, 0]
      return phi.to(u.s / u.km), phi_err.to(u.s / u.km)

    if self.profileType == 'gauss':
      phi = ((1 / (self.b * np.sqrt(np.pi))) * np.exp(-(v / self.b)**2)).to(u.s / u.km)
      phi_err = np.zeros_like(phi.value) * phi.unit
      return phi, phi_err

    L = self.lorentz_FWHM_v[idx, 0]
    dL = self.lorentz_FWHM_v_err[idx, 0]
    G = self.gauss_FWHM_v[idx, 0]

    def phi_from_L(Lval):
      amp_L = 2 / (np.pi * Lval)
      model = Voigt1D(x_0=0.0, amplitude_L=amp_L, fwhm_G=G, fwhm_L=Lval)
      return model(v)

    phi = phi_from_L(L)

    if np.allclose(dL.value, 0.0, equal_nan=True):
      phi_err = np.zeros_like(phi.value) * phi.unit
    else:
      Lm = np.maximum(L - dL, 1e-30 * (u.km / u.s))
      Lp = L + dL
      dphi_dL = (phi_from_L(Lp) - phi_from_L(Lm)) / (Lp - Lm)
      phi_err = np.abs(dphi_dL) * dL

    return phi.to(u.s / u.km), phi_err.to(u.s / u.km)

  def build_total_crossection(self):
    lam_grid = self.lam_grid
    sigma_total = np.zeros(lam_grid.shape) * u.cm**2
    sigma_total_err2 = np.zeros(lam_grid.shape) * (u.cm**4)

    lam_grid_val = lam_grid.to_value(u.AA)
    lam0_val = self.lam0[:, 0].to_value(u.AA)     # Strip units

    for i in range(len(lam0_val)):
      lam_center = self.lam0[i, 0]                        # Get center line wavelength
      halfwidth = self.line_window_halfwidth_lam(i)       # Get cutoff window
      lam_lo = (lam_center - halfwidth).to_value(u.AA)    # Lower cutoff
      lam_hi = (lam_center + halfwidth).to_value(u.AA)    # Upper cutoff

      i0 = np.searchsorted(lam_grid_val, lam_lo, side='left')   # Find where cutoff window sits in the global grid
      i1 = np.searchsorted(lam_grid_val, lam_hi, side='right')

      if i1 <= i0:        # Skip empty windows
        continue

      lam_local = lam_grid[i0:i1]                                         # Small wavelength grid around lam0 (the cutoff window)  
      v_local = self.lambda_offsets_to_velocity(lam_local, lam_center)    # convert to velocity offsets (might change later)
      phi_local, phi_local_err = self.profile_from_velocity(v_local, i)   # Broaden that line

      sigma_line = (self.sig_0[i, 0] * phi_local).to(u.cm**2)             # Convert to line cross-section
      sigma_line_err = (self.sig_0[i, 0] * phi_local_err + self.sig_0_err[i, 0] * phi_local).to(u.cm**2)  # Error in the line cross-section

      sigma_total[i0:i1] += sigma_line            # Stitch to global grid
      sigma_total_err2[i0:i1] += sigma_line_err**2  # Error stitch
    sigma_total_err = np.sqrt(sigma_total_err2)
    return sigma_total, sigma_total_err

  def plot_total_crossection(self, xscale: str = 'linear', yscale: str = 'log', xlim: tuple = None):
    lam = self.lam_grid.to_value(u.AA)
    sig = self.sigma_total.to_value(u.cm**2)
    sig_err = self.sigma_total_err.to_value(u.cm**2)

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
    sig = self.sigma_total.to_value(u.cm**2)

    mask = (lam >= lam_lo) & (lam <= lam_hi)

    plt.figure(figsize=(9, 4))
    plt.plot(lam[mask], sig[mask])
    plt.axvline(lam_center.to_value(u.AA), linestyle='--')
    plt.xlabel(f"Wavelength [{self.lam_grid.unit}]")
    plt.ylabel(r"Total cross-section $\sigma_{\rm tot}$ [cm$^2$]")
    plt.title(f"{self.molecule.species} line window around {lam_center:.4f}")
    plt.tight_layout()
    plt.show()