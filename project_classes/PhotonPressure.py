from project_classes import BroadeningProfile
from project_classes import Star
from project_func.errors import _not_quantity
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
import numpy as np


class PhotonPressure:
  
  def __init__(self, broadeing_profile: BroadeningProfile, star: Star):

    self.broad_prof = broadeing_profile
    self.lam_sym = broadeing_profile.lam_sym
    self.crossection_sym = broadeing_profile.sigmaArray_sym
    self.crossection_err_sym = broadeing_profile.sigmaArray_sym_err
    
    self.star = star
    self.flux_star = star.flux_star
    self.lam_star = star.lam_star
    self.flux_star_interp = self.get_interp_Spectra()
    self.lam_star_interp  = self.lam_sym

    self.F_ph_tot, self.F_ph_tot_err, self.F_ph_perline, self.F_ph_perline_err = None, None, None, None

  def get_interp_Spectra(self):
    """
    Interpolates the stars spectra over a profile with different amount of datapoints.
    The interpolation is done over a asymmetric spectra so the symmetrical lambdagrid has to be used
    profile: Class= Broadeing_profile
    """
    lam_sym = self.lam_sym.value          # (Nlines, Npts)
    lam_star = self.lam_star.value
    flux_star = self.flux_star.value
    L = lam_sym.shape[0]

    F_star_interp = np.empty_like(lam_sym, dtype=float)
    for line in range(L):
      lam_L = lam_sym[line]
      F_star_interp[line] = np.interp(lam_L, lam_star, flux_star) # losing units

    F_star_interp *= u.erg/(u.cm**2)/u.s/u.AA         # adding back units

    return F_star_interp
  
  def plot_interp_Spectra(self, line: int):
    lam0 = self.broad_prof.molecule.lam0[line,0]
    lam = self.lam_sym
    flux = self.flux_star
    lam_star = self.lam_star
    flux_interp = self.flux_star_interp
    zoom = 1
    vlim = self.broad_prof.vlim
    b = self.broad_prof.b
    N = self.broad_prof.N
    text = fr'$v_{{lim}}$ = {vlim}, b = {b}, N = {N}'

    plt.figure(figsize=(9, 4)) 
    plt.plot(lam_star, flux, label = f'Star theoretical')
    plt.plot(lam[line,:], flux_interp[line,:], label = 'Interpolated flux')
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes,
         va='top', ha='left', bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))
    plt.ticklabel_format(axis='x', style='plain', useOffset=False)
    plt.title(f"Pure spectra vs interpolated flux, {self.broad_prof.molecule.species} {lam0}")
    plt.xlabel(f"Wavelength [{lam0.unit}]")
    plt.xlim(lam0.value -zoom, lam0.value +zoom)
    plt.ylabel(f"Flux [{flux.unit}]")
    plt.legend()
    plt.show()
    return 0

  def transmission(self, column_density):
    N = column_density
    sigma = self.crossection_sym.to(u.cm**2)
    sigma_err = self.crossection_err_sym.to(u.cm**2)
    tau = N * sigma                             # Optical depth τ
    trans = np.exp(-tau)                        # Transmission  T = exp(-τ)
    absorbtion = 1 - trans                      # Absorbtion    A = 1 - exp(-1)

    tau_err = N * sigma_err
    trans_err = np.exp(-tau) * tau_err
    
    return trans, trans_err
  
  def calc_PhotonPressure(self, column_density):
    N_col = column_density.to(u.cm**(-2)) if isinstance(column_density, u.Quantity) else _not_quantity("column_density")
    Trans, Trans_err = self.transmission(N_col)
    
    T = Trans
    T_err = Trans_err
    sig = self.crossection_sym
    sig_err = self.crossection_err_sym
    Flux = self.flux_star_interp
    lam = self.lam_sym
    I = Flux * sig * T

    F_ph_perline = (np.trapz(I, lam) / const.c).to(u.N)
    F_ph_tot = np.nansum(F_ph_perline)

    N = N_col
    dA = self.broad_prof.molecule.A_ul_err

    factor = (1-(N*sig))
    dF_dA = np.trapz((Flux * T * factor * sig_err)/ const.c, lam)

    F_ph_perline_err = (np.abs(dF_dA)).to(u.N)
    F_ph_tot_err = np.sqrt(np.nansum(F_ph_perline_err**2))

    return F_ph_tot, F_ph_tot_err, F_ph_perline, F_ph_perline_err
