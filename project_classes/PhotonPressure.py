from project_classes import BroadeningProfile
from project_classes import Star
from project_func.errors import _not_quantity
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
import numpy as np


class PhotonPressure:
  
  def __init__(self, broadeing_profile: BroadeningProfile, star: Star):
    """
    Creates a photon pressure object for a Star object
    """

    self.broad_prof = broadeing_profile
    self.lam_sym = broadeing_profile.lam_sym
    self.crossection_sym = broadeing_profile.sigmaArray_sym
    self.crossection_err_sym = broadeing_profile.sigmaArray_sym_err
    
    self.star = star
    self.flux_star = star.flux_star_rot
    self.lam_star = star.lam_star
    self.flux_star_interp = self.get_interp_Spectra()
    self.lam_star_interp  = self.lam_sym

    self.F_ph_tot, self.F_ph_tot_err, self.F_ph_perline, self.F_ph_perline_err = None, None, None, None

    self.E_l = broadeing_profile.molecule.E_l
    self.g_l = broadeing_profile.molecule.g_l
    self.J_l = broadeing_profile.molecule.J_l
    self.fik = broadeing_profile.molecule.fik

  def get_interp_Spectra(self):
    """
    Interpolates the stars spectra over a profile with different amount of datapoints.
    The interpolation is done over a asymmetric spectra so the symmetrical lambdagrid has to be used
    profile: Class= Broadeing_profile
    """
    lam_sym = self.lam_sym.to_value(u.AA)          # (Nlines, Npts)
    lam_star = self.lam_star.to_value(u.AA)
    flux_star = self.flux_star.to_value(self.flux_star.unit)
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
    N = np.atleast_1d(column_density.to(u.cm**-2)) 

    sigma = self.crossection_sym.to(u.cm**2)
    sigma_err = self.crossection_err_sym.to(u.cm**2)

    tau = sigma[:, :, None] * N[None, None, :]                            # Optical depth τ
    trans = np.exp(-tau)                        # Transmission  T = exp(-τ)
    absorbtion = 1 - trans                      # Absorbtion    A = 1 - exp(-1)

    tau_err = sigma_err[:, :, None] * N[None, None, :] 
    trans_err = np.exp(-tau) * tau_err
    
    print(f"Transmission coefficient calculated with shape: {trans.shape}")
    return trans, trans_err
  
  def excitation_weights(self, Temp_atm):
    kb_eV = const.k_B.to(u.eV/u.K)
    El = self.E_l
    gl =  self.g_l
    T = Temp_atm

    # Find all levels for calculating the partition function for all T
    Eg = np.column_stack([El.value, gl])
    mask = np.isfinite(Eg).all(axis=1) 
    Eg = Eg[mask]
    Eg_unique, idx, inv = np.unique(Eg, axis=0, return_index=True, return_inverse=True)
    El_unique = (Eg_unique[:,0]).reshape(-1,1) * u.eV
    gl_unique = (Eg_unique[:,1]).reshape(-1,1) * u.dimensionless_unscaled

    # Calculate the boltzmann factor, for all levels available and calculate the partition function
    El_kb_unique = El_unique / kb_eV
    exp_unique = np.exp(-El_kb_unique/T)
    boltz_unique = gl_unique * np.exp(-El_kb_unique / T)
    Z = np.nansum((gl_unique * exp_unique), axis=0)

    # Calculate the weights for each line (assigning weights back into same shape array, corresponding to the amount of lines)
    w_line = boltz_unique[inv] / Z

    return w_line

  def calc_PhotonPressure(self, column_density, Temp_atm, distance):
    """
    Calculates the photon pressure and the per line pressure

    ** Inputs **
    column_density:           Array of column densities                   [cm-2]
    Temp_atm:                 Planetary atmospheric temperature           [K]
    distance                  Distance between object and its star        [length]

    ** Returns **
    F_ph_tot:                 Total photon pressure                       [N]           [N_temp, N_col]
    F_ph_tot_err:             Error in the total photon pressure -||-     [N]           - || -
    F_ph_perline,             Per line photon pressure                    [N]           [N_lines, N_temp, N_col]
    F_ph_perline_err          Error per line                              [N]           - || -

    """
    N_col = column_density.to(u.cm**(-2)) if isinstance(column_density, u.Quantity) else _not_quantity("column_density")
    Trans, Trans_err = self.transmission(N_col)
    d = distance
    R_star = self.star.radius
    omega = (R_star/d)**2
    
    sig = self.crossection_sym
    sig_err = self.crossection_err_sym
    Flux = self.flux_star_interp * omega
    lam = self.lam_sym

    Temp = Temp_atm.to(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")
    weights = self.excitation_weights(Temp)

    # Trans: (lines, lam, Ncol)
    # Flux:  (lines, lam)
    # sig:   (lines, lam)
    # weights: (lines, Temp)

    I = Flux[:, :, None] * sig[:, :, None] * Trans                          # (lines, lam, Ncol)

    F_ph_perline = (np.trapz(I, lam[:, :, None], axis=1) / const.c).to(u.N)           # (lines, Ncol)                   
    F_ph_perline = F_ph_perline[:, None, :] * weights[:, :, None]                     # (lines, Temp, Ncol)

    print(f"Per line photon pressure calculated with shape: {F_ph_perline.shape}") # (lines, Temp, Ncol)

    F_ph_tot = np.nansum(F_ph_perline, axis = 0)                            # (Temp, Ncol)

    print(f"Total photon pressure calculated with shape: {F_ph_tot.shape}")

    N = N_col
    # dA = self.broad_prof.molecule.A_ul_err

    factor = (1-(N[None, None, :]*sig[:, :, None]))
    sig_err = sig_err[:, :, None]
    dF_dA = np.trapz((Flux[:, :, None] * Trans * factor * sig_err)/ const.c, lam[:, :, None], axis = 1)

    F_ph_perline_err = (np.abs(dF_dA)).to(u.N)[:, None, :] * weights[:, :, None]
    F_ph_tot_err = np.sqrt(np.nansum(F_ph_perline_err**2, axis=0))

    return F_ph_tot, F_ph_tot_err, F_ph_perline, F_ph_perline_err

  def beta_Values(self, F_ph_tot, F_ph_tot_err):
    """
    Calculates the beta ratio for a given photon pressure

    ** Inputs **
    F_ph_tot:               Total photon pressure         [N]         [N_temp, N_col]
    F_ph_tot_err            Error of above value          [N]         - || -

    ** Returns **
    beta:                   Beta values                   [Unitless]  [N_temp, N_col]
    beta_err                Errors in beta                [Unitless]  - || -  
    """
    mass_star = self.star.mass
    mass_species = self.broad_prof.molecule.mass
    radius = self.star.radius
    G = const.G.cgs

    F_ph = F_ph_tot
    F_ph_err = F_ph_tot_err
    F_grav = ((G * mass_star * mass_species) / (radius)**2).to(u.N)

    beta = F_ph / F_grav
    beta_err = F_ph_err/F_grav

    print(f"Beta values calculated successfully with the shape: {beta.shape}")
    return(beta, beta_err)
