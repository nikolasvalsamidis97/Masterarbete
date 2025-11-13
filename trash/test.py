from astropy import constants as const
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
from astropy.modeling.models import Voigt1D
from astroquery.nist import Nist
import pandas as pd
from molmass import Formula
from astropy.table import Table
from astropy.io.votable import parse_single_table

# ========================================= Helpers ========================================= #
def _not_quantity(name: str):
  raise TypeError(f"{name} must be an astropy Quantity with units (e.g., 500*u.nm).")
# =========================================================================================== #


class Molecule:
  
  def __init__(self, species: str, lam_min, lam_max, A_ul_min = 0 / u.s):
    """
    species:        Chemical formula of molecule
    lam_min:        Minimum data wavelength                     Quantity
    lam_max:        Maximum data wavelength                     Quantity
    A_ul_min:       Minimum value of spontaneous deexitation    Quantity
    """
    self.species = species
    self.lam_min = lam_min.to(u.AA) if isinstance(lam_min, u.Quantity) else _not_quantity("lam_min")
    self.lam_max = lam_max.to(u.AA) if isinstance(lam_max, u.Quantity) else _not_quantity("lam_max")
    self.A_ul_min = A_ul_min.to(1/u.s) if isinstance(A_ul_min, u.Quantity) else _not_quantity("A_ul_min")

    self.data = self.set_Nist_Data(self.species, self.lam_min, self.lam_max, self.A_ul_min)
    self.mass = Formula(self.species).mass * u.u
    self.A_ul, self.A_ul_err, self.lam0, self.g_u, self.g_l, self.E_u, self.E_l = self.pandas_to_numpy(self.data)
    self.sig_0, self.sig_0_err = self.calc_central_crossection()

  def get_Name(self):
    print(self.species)

  def pandas_to_numpy(self, data):
    """
    Numpy arrays with dimensions (N_lines, None) ex. (16,)
    """
    Aul = pd.to_numeric(data['A_ul']).to_numpy().reshape(-1, 1) / u.s
    Aul_err = pd.to_numeric(data['Acc']).to_numpy().reshape(-1, 1) / u.s
    lam0 = pd.to_numeric(data['lam_obs']).to_numpy().reshape(-1, 1) * u.AA
    gu = pd.to_numeric(data['g_u']).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled
    gl = pd.to_numeric(data['g_l']).to_numpy().reshape(-1, 1) * u.dimensionless_unscaled
    Eu = pd.to_numeric(data['E_u']).to_numpy().reshape(-1, 1) * u.eV
    El = pd.to_numeric(data['E_l']).to_numpy().reshape(-1, 1) * u.eV

    return Aul, Aul_err, lam0, gu, gl, Eu, El

  def set_Nist_Data(self,
                    species,
                    wav_min, 
                    wav_max, 
                    A_ul):
    """
    wav_min:    Minimum wavelength as float in Angstrom
    wav_max:    Maximum wavelength as float in Angstrom
    A_ul:       Minimum A_ul as float in 1/s
    """

    tab = Nist.query(wav_min, 
                    wav_max, 
                    linename = species, 
                    energy_level_unit = 'eV',
                    wavelength_type='vacuum', 
                    output_order='wavelength')
    
    df = tab.to_pandas()
    # Ion identification
    spec = df.get('Spectrum')

    # Wavelength in nm. If observed wavelength is missing i will use Ritz
    lam_obs = pd.to_numeric(df['Observed'], errors='coerce')
    lam_ritz = pd.to_numeric(df['Ritz'], errors='coerce')
    lam_obs = lam_obs.fillna(lam_ritz)

    A_ul = pd.to_numeric(df['Aki'], errors='coerce')
    

    Acc = df['Acc.']
    ACC_FRAC = {                     # Map onto Acc-code. Source: 'https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html#OUTACC' Search for "estimated accuracy"
    'AAA': 0.003,
    'AA': 0.01,
    'A+': 0.02,
    'A': 0.03,
    'B+': 0.07,
    'B': 0.10,
    'C+': 0.18,
    'C': 0.25,
    'D+': 0.40,
    'D': 0.50,
    'E': 0.50
    }
    Acc = Acc.map(ACC_FRAC) * A_ul
    

    eiek = df['Ei           Ek'].str.split('-', n=1, expand=True)
    ei = eiek[0].str.strip(' []?') # lower energy (Ei)
    ek = eiek[1].str.strip(' []?') # upper energy (Ek)
    Ei = pd.to_numeric(ei, errors='coerce')
    Ek = pd.to_numeric(ek, errors='coerce')

    gigk = df['gi   gk'].str.split('-', n=1, expand=True)
    gi = gigk[0].str.strip()
    gk = gigk[1].str.strip()
    Gi = pd.to_numeric(gi, errors='coerce')
    Gk = pd.to_numeric(gk, errors='coerce')

    ji = df['Lower level'].str.split('|',n=2 , expand=True)[2].str.strip()
    jk = df['Upper level'].str.split('|',n=2 , expand=True)[2].str.strip()

    def parse_j(series: pd.Series) -> pd.Series:                                  # For turning spin values (string) into floats
      s = series.astype(str).str.strip().str.strip('()')
      mask = s.str.contains('/', regex=False, na=False)
      out = pd.to_numeric(s, errors='coerce')
      if mask.any():
        parts = s[mask].str.split('/', n=1, expand=True)
        num = pd.to_numeric(parts[0].str.strip(), errors='coerce')
        den = pd.to_numeric(parts[1].str.strip(), errors='coerce')
        out.loc[mask] = num / den
      return out
    
    Ji = parse_j(ji)
    Jk = parse_j(jk)

    if species == 'H':
      spec = 'H'

    output = pd.DataFrame({
                          'Ion'       : spec,             # string
                          'lam_obs'   : lam_obs,          # Å
                          'A_ul'      : A_ul,             # 1/s
                          'Acc'       : Acc,              # Accuracy of A_ul
                          'E_l'       : Ei,               # eV
                          'E_u'       : Ek,               # eV
                          'J_l'       : Ji,               # dimless
                          'J_u'       : Jk,               # dimless
                          'g_l'       : Gi,               # dimless
                          'g_u'       : Gk,               # dimless
                          'transition': df['Transition'].astype(str)
                          })

    # Sorting the values with respect to observed wavelength
    out_f = (output
          .sort_values('lam_obs', kind='mergesort')
          .drop_duplicates(subset=['lam_obs', 'transition', 'A_ul'])
          .reset_index(drop=True))

    data = out_f

    return data

  def calc_central_crossection(self):
    # Returns σ_0 in cm^2 km/s such that s = integral(σ_0 * φ) [cm^2]
  
    sig0 = (self.A_ul * (self.lam0**3/(8 * np.pi)) * (self.g_u/self.g_l))                 # σ_v = σ_λ = σ_0 * φ_λ = σ_0 * φ_v * c/λ
    sig0 = sig0.to(u.cm**2 * u.km / u.s)
    sig0_err = sig0 * (self.A_ul_err/self.A_ul)

    return sig0, sig0_err
    
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
    return (2 * np.sqrt(np.log(2)) * self.b ).to(u.km/u.s)
  
  def lorentz_Profile(self):
    phi = (1/ np.pi) * (0.5*self.lorentz_FWHM_v) / ( (self.v_grid**2) + ((0.5*self.lorentz_FWHM_v)**2) )
    dphi_dL = ((1/np.pi) * 0.5 * (self.v_grid**2 - (self.lorentz_FWHM_v**2)/4.0) / (self.v_grid**2 + (self.lorentz_FWHM_v**2)/4.0)**2)
    phi_err = np.abs(dphi_dL) * self.lorentz_FWHM_v_err
    return phi.to(u.s/u.km), phi_err.to(u.s/u.km)
  def gauss_Profile(self):
    phi = ((1/(self.b * np.sqrt(np.pi))) * np.exp(-(self.v_grid/self.b)**2)).to(1/(u.km/u.s))
    phi_err = np.NaN
    return phi.to(u.s/u.km), phi_err
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
  
class Star:
  
  def __init__(self, path: str, distance, radius):
    """
    path: str       Filepath for theoretical spectra
    """
    self.path = path
    self.distance = distance.to(u.au) if isinstance(distance, u.Quantity) else _not_quantity("distance")
    self.radius = radius.to(u.m) if isinstance(radius, u.Quantity) else _not_quantity("radius")
    self.lam_star, self.flux_star = self.read_Spectra()
    
  def read_Spectra(self):
    """
    Reads a spectra from a file and returns the flux in vacuum
    """
    VOtab = Table.read(self.path, format='votable')

    lam = VOtab['WAVELENGTH'].value          #u.AA
    flux = VOtab['FLUX'].value              #(u.erg/u.s/(u.cm**2)/u.AA)
    flux = flux * (u.erg/u.s/(u.cm**2)/u.AA)
    lam = self.air_to_vacuum(lam) * u.AA
    omega = (self.radius.to(u.m)/self.distance.to(u.m))**2
    flux *= omega

    return lam, flux 
  
  def air_to_vacuum(self, lam_air_A):
    s2 = (1e4/lam_air_A)**2
    n_minus_1 = 1e-8*(8342.13 + 2406030/(130 - s2) + 15997/(38.9 - s2))
    n = 1 + n_minus_1
    return lam_air_A * n

class PhotonPressure:
  
  def __init__(self, broadeing_profile: BroadeningProfile, star: Star):

    self.broad_prof = broadeing_profile
    self.lam_sym = broadeing_profile.lam_sym
    self.crossection_sym = broadeing_profile.sigmaArray_sym
    self.crossection_err_sym = broadeing_profile.sigmaArray_sym_err
    
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

############################################################################################
######################################### KODGUIDE #########################################
# 1. Hämtar molekyldata
Na = Molecule('Na', 5800 * u.AA, 6000*u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
b = 1 * u.km/u.s
vlim = 10 * u.km/u.s
Npts = 1000
Na_broadening = BroadeningProfile(Na, b , vlim, Npts, 'Voigt')
## 2.5 Möjlighet att plotta profil och tvärsnitt för en linje. För att se linje: print(Na.data)
#print(Na.data)
line = 5
domain1 = 'velocity'
domain2 = 'wavelength'
Na_broadening.plot_Symmetric_Profile(line ,domain1)
Na_broadening.plot_Symmetric_Crossection(line, domain2)

# 3. Hämta teoretiskt stjärnspectra
star = Star('Kod och data/TS/models_1758706196/bt-nextgen-agss2009/lte063-1.0-0.0a+0.0.BT-NextGen.7.dat.xml', 1*u.au, const.R_sun.value * u.m)

# 4. Skapa object för strålningstryck
Na_Ph = PhotonPressure(Na_broadening, star)

# 4.5 Exempelplott för strplningstryck som funktion av kolumndensitet
Ncols = np.logspace(7, 25, 100) * u.cm**-2

F_tot = []
F_tot_err = []
for N in Ncols:
    F_ph_tot, F_ph_tot_err, _, _ = Na_Ph.calc_PhotonPressure(N)
    F_tot.append(F_ph_tot.to(u.N).value)
    F_tot_err.append(F_ph_tot_err.to(u.N).value)

# x-axis values (unitless array for mpl)
x = Ncols.to(1/u.cm**2).value
y = np.array(F_tot)
yerr = np.array(F_tot_err)


plt.figure(figsize=(7,4))
plt.errorbar(x, y, yerr=yerr, fmt='-', capsize=3, lw=1.5)
plt.xscale('log')
plt.yscale('log')   # optional; use linear if you prefer
plt.xlabel(r'$N_{\rm col}\ [{\rm cm^{-2}}]$')
plt.ylabel('Photon force per absorber [N]')
plt.title('Photon force vs column density')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

