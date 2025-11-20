from project_classes.Molecule import Molecule
from project_func.errors import _not_quantity
from astropy import constants as const
from astropy import units as u
import numpy as np
from astropy.modeling.models import Voigt1D
from matplotlib import pyplot as plt

class BroadeningProfile:
  
  def __init__(self, molecule: Molecule, b, vlim, N:int, profileType:str = 'Voigt'):
    """
    Contains both the broadeing profile and calculates the crossection using the profile
    molecule:    Molecule
    b:           broadening parameter                 [km/s]
    vlim:        maximum broading                     [km/s]
    N:           resolution of the velocity grid      [int]
    profileType: Type of broadening.                  Ex. "Lorentz", "Gauss" or "Voigt"
    """
    self.molecule = molecule
    self.b = b.to(u.km /u.s) if isinstance(b, u.Quantity) else _not_quantity("b (broadening parameter)")
    self.vlim = vlim.to(u.km/u.s) if isinstance(vlim, u.Quantity) else _not_quantity("vlim")
    self.N = N
    self.profileType = profileType
    self.v_grid = self.velocity_Grid()
    self.lam_grid = self.v_to_lam()
    self.lorentz_FWHM_v, self.lorentz_FWHM_v_err = self.FWHM_lorentz()
    self.gauss_FWHM_v = self.FWHM_gauss()
    self.profileArray, self.profileArray_err  = self.profile_Array()
    self.sigmaArray, self.sigmaArray_err  = self.Crossection_Array()

    # For interpolating in the photon pressure classs
    self.v_grid_sym, self.sigmaArray_sym = self.half_to_symmetric_v(self.sigmaArray)
    _, self.sigmaArray_sym_err = self.half_to_symmetric_v(self.sigmaArray_err)
    self.lam_sym = self.half_to_symmetric_lam()
    
    
  def velocity_Grid(self):
    """
    A fixed velocity grid, shared by all profiles
    v_grid [km/s]
    """
    v_grid = (np.linspace(0, 1, self.N).reshape(1, -1))**(2) * self.vlim  # (1,N)
    return v_grid.to(u.km/u.s)
  
  def FWHM_lorentz(self):
    """
    Calculates the FWHM, lorentzian for the object
    lorentz_FWHM_v  [km/s]
    """
    lorentz_FWHM_v = self.molecule.lam0 * ((self.molecule.A_ul)/ (2*np.pi))
    lorentz_FWHM_v_err = self.molecule.lam0 * ((self.molecule.A_ul_err)/ (2*np.pi))
    return lorentz_FWHM_v.to(u.km/u.s), lorentz_FWHM_v_err.to(u.km/u.s)
  def FWHM_gauss(self):
    """
    gauss_FWHM_v    [km/s]
    """
    gauss_FWHM_v_scalar = (2 * np.sqrt(np.log(2)) * self.b).to(u.km/u.s)

    # make an array with the SAME SHAPE as lam0 (16, 1)
    gauss_FWHM_v = (np.full_like(self.molecule.lam0.value,
                                 gauss_FWHM_v_scalar.value)
                    * gauss_FWHM_v_scalar.unit)
    return gauss_FWHM_v
  
  def lorentz_Profile(self):
    phi = (1/ np.pi) * (0.5*self.lorentz_FWHM_v) / ( (self.v_grid**2) + ((0.5*self.lorentz_FWHM_v)**2) )
    dphi_dL = ((1/np.pi) * 0.5 * (self.v_grid**2 - (self.lorentz_FWHM_v**2)/4.0) / (self.v_grid**2 + (self.lorentz_FWHM_v**2)/4.0)**2)
    phi_err = np.abs(dphi_dL) * self.lorentz_FWHM_v_err
    return phi.to(u.s/u.km), phi_err.to(u.s/u.km)
  def gauss_Profile(self):
    phi_single = ((1/(self.b * np.sqrt(np.pi))) *
                  np.exp(-(self.v_grid/self.b)**2)).to(1/(u.km/u.s))
    n_lines = self.molecule.lam0.shape[0]
    phi = (np.repeat(phi_single.value, n_lines, axis=0) * phi_single.unit)
    phi_err = np.full_like(phi.value, np.nan) * phi.unit
    return phi.to(u.s/u.km), phi_err.to(u.s/u.km)
  def voigt_Profile(self):
    L  = self.lorentz_FWHM_v
    dL = self.lorentz_FWHM_v_err
    G  = self.gauss_FWHM_v
    v  = self.v_grid

    def phi_from_L(Lval):
      amp_L = 2 / (np.pi * Lval)
      model = Voigt1D(x_0=0.0, amplitude_L=amp_L, fwhm_G=G, fwhm_L=Lval)
      return model(v)

    phi = phi_from_L(L)

    eps = dL
    Lm = L - eps
    Lp = L + eps
    dphi_dL = (phi_from_L(Lp) - phi_from_L(Lm)) / (Lp - Lm)
    phi_err = np.abs(dphi_dL) * dL

    return phi.to(u.s/u.km), phi_err.to(u.s/u.km)

  def profile_Array(self):
    """
    Returns a half symmetric (normalized to 0.5) broadening profile. If the number of lines > 1 an array, for each, line is returned
    phi [1/km/s]
    """
    
    if self.profileType == 'lorentz':
      return self.lorentz_Profile()
    elif self.profileType == 'gauss':
      return self.gauss_Profile()
    else:
      return self.voigt_Profile()

  def half_to_symmetric_v(self, array):
    """
    Returns symmetric axes for half symmetric arrays. For plotting
    """
    v_sym = np.concatenate((-self.v_grid[:, :0:-1], self.v_grid), axis=1)
    array_sym = np.concatenate((array[:, :0:-1], array), axis=1)
    return v_sym, array_sym
  def half_to_symmetric_lam(self):
    """
    Returns the full grid of wavelengths, for plotting only
    """
    x = self.lam_grid
    c = x[:, [0]]
    dx = x - c
    left = c - dx[:, 1:][:, ::-1]
    return np.hstack([left, x])
  
  def v_to_lam(self):
    """
    Converts a velocity grid back to wavelength
    """
    lam = (self.molecule.lam0 * (1 + (self.v_grid/const.c))).to(u.AA)
    return lam

  def plot_Symmetric_Profile(self, line: int, domain: str = 'velocity'):
    v_sym, phi_sym = self.half_to_symmetric_v(self.profileArray)
    _, phi_sym_err = self.half_to_symmetric_v(self.profileArray_err)
    lam_sym = self.half_to_symmetric_lam()

    v = v_sym[0,:].to_value(u.km/u.s)
    lam = lam_sym[line,:].to_value(u.AA)
    phi = phi_sym[line,:].to_value(u.s/u.km)
    phi_err = phi_sym_err[line,:].to_value(u.s/u.km)
    if domain == 'velocity':
      plt.figure(figsize=(9, 4)) 
      plt.plot(v, phi)
      plt.fill_between(v, phi - phi_err, phi + phi_err, alpha=0.3, label='error', color = 'red')
      plt.ticklabel_format(axis='x', style='plain', useOffset=False)
      plt.title(f"{self.molecule.species} {self.molecule.lam0[line, 0]}, b={self.b}")
      plt.xlabel(f"Relative velocity v [{self.molecule.lam0[line, 0].unit}]")
      plt.ylabel(f"{self.profileType}-Profile [{phi_sym.unit}]")
      plt.legend()
      plt.show()
    elif domain == 'wavelength':
      plt.figure(figsize=(9, 4)) 
      plt.plot(lam, phi)
      plt.fill_between(lam, phi - phi_err, phi + phi_err, alpha=0.3, label='error', color = 'red')
      plt.ticklabel_format(axis='x', style='plain', useOffset=False)
      plt.title(f"{self.molecule.species} {self.molecule.lam0[line, 0]}, b={self.b}")
      plt.xlabel(f"λ [{self.molecule.lam0[line, 0].unit}]")
      plt.ylabel(f"{self.profileType}-Profile [{(lam_sym**(-1)).unit}]")
      plt.legend()
      plt.show()
  def plot_Symmetric_Crossection(self, line: int, domain: str = 'velocity'):
    v_sym, sig_sym = self.half_to_symmetric_v(self.sigmaArray)
    _, sig_sym_err = self.half_to_symmetric_v(self.sigmaArray_err)
    lam_sym = self.half_to_symmetric_lam()

    v = v_sym[0,:].to_value(u.km/u.s)
    lam = lam_sym[line,:].to_value(u.AA)
    sig = sig_sym[line,:].to_value(u.cm**2)
    sig_err = sig_sym_err[line,:].to_value(u.cm**2)
    
    if domain == 'velocity':
      plt.figure(figsize=(9, 4)) 
      plt.plot(v, sig)
      plt.fill_between(v, sig - sig_err, sig + sig_err, alpha=0.3, label='error', color = 'red')
      plt.ticklabel_format(axis='x', style='plain', useOffset=False)
      plt.title(f"{self.molecule.species} {self.molecule.lam0[line, 0]}, b={self.b}")
      plt.xlabel(f"Relative velocity v [{self.molecule.lam0[line, 0].unit}]")
      plt.ylabel(f"Crossection σ [{sig_sym.unit}]")
      plt.legend()
      plt.show()
    elif domain == 'wavelength':
      plt.figure(figsize=(9, 4)) 
      plt.plot(lam, sig)
      plt.fill_between(lam, sig - sig_err, sig + sig_err, alpha=0.3, label='error', color = 'red')
      plt.ticklabel_format(axis='x', style='plain', useOffset=False)
      plt.title(f"{self.molecule.species} {self.molecule.lam0[line, 0]}, b={self.b}")
      plt.xlabel(f"λ [{lam_sym.unit}]")
      plt.ylabel(f"Crossection σ [{sig_sym.unit}]")
      plt.legend()
      plt.show()
      return 0

  def Crossection_Array(self):
    """
    Returns the crossection for all lines.
    Note: Using uncorrelated Aul assumption. If lorentzbroadening is heave then you should use correlated error-propagation
    """
    # Correlated error
    phi      = self.profileArray           # array, units 1/velocity
    phi_err  = self.profileArray_err       # array, same shape/units
    sig_0    = self.molecule.sig_0         # scalar (area*velocity)
    A        = self.molecule.A_ul          # scalar (s^-1)
    dA       = self.molecule.A_ul_err      # scalar
    lam0     = self.molecule.lam0          # wavelength
    dL       = self.lorentz_FWHM_v_err     # scalar (same units as L)

    # Full cross-section
    sig_array = phi * sig_0

    # Correlated propagation via A_ul:
    # |dphi/dL| from voigt error: phi_err = |dphi/dL| * dL
    dphi_dL_mag = phi_err / dL

    # Chain rule to A_ul
    dL_dA   = lam0 / (2 * np.pi)
    dphi_dA_mag = dphi_dL_mag * dL_dA
    
    # dσ/dA magnitude (sum of the two contributions)
    dsig0_dA = sig_0 / A
    dsig_dA_mag = sig_0 * dphi_dA_mag + phi * dsig0_dA

    # Final error
    sig_array_err = np.abs(dsig_dA_mag) * dA

    ## Uncorrelated error
    # phi = self.profileArray
    # phi_err = self.profileArray_err
    # sig_0 = self.molecule.sig_0
    # sig_0_err = self.molecule.sig_0_err

    # # Full cross-section
    # sig_array = phi * sig_0
    # # Final error
    # sig_array_err = np.sqrt((sig_0 * phi_err)**2 + (phi * sig_0_err)**2)
    return sig_array, sig_array_err
  