


# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------------------------------- Masterarbete ------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------------------ #

# ------- Modules ------- #

from astropy import constants as const
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
from astropy.modeling.models import Voigt1D
from astroquery.nist import Nist
import pandas as pd
from scipy.interpolate import interp1d
import time
from specutils.utils.wcs_utils import air_to_vac

# ------------------------ #
#start = time.perf_counter()



# ------- Loading in Atomic data ------- #
def get_nist_lines(species: str, 
                   wav_min: float, 
                   wav_max: float, 
                   A_ul: float | None=None) -> pd.DataFrame: # minimum A_ul
  
  tab = Nist.query(wav_min * u.AA, 
                   wav_max * u.AA, 
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


  return out_f, df
# -------------------------------------- #

# ---------------------- Helpers ---------------------- #

def pandas_to_numpy(lines):
  Aul = pd.to_numeric(lines['A_ul']).to_numpy().reshape(-1,1) / u.s
  Aul_err = pd.to_numeric(lines['Acc']).to_numpy().reshape(-1,1) / u.s
  lam0 = pd.to_numeric(lines['lam_obs']).to_numpy().reshape(-1,1) * u.AA
  gu = pd.to_numeric(lines['g_u']).to_numpy().reshape(-1,1) * u.dimensionless_unscaled
  gl = pd.to_numeric(lines['g_l']).to_numpy().reshape(-1,1) * u.dimensionless_unscaled
  return Aul, Aul_err, lam0, gu, gl

def ask_plot_kind():
  print("\nWhat plot do you want?")
  print("[1] Profiles (Φ)")
  print("[2] Cross sections (σ)")
  print("[q] Quit")
  while True:
    ans = input("Enter choice [1/2/q]: ").strip().lower()
    if ans in ("1", "2", "q"):
      return ans
    print("Please enter 1, 2, or q.")

def show_lam_choices(lines_df):
  lam = lines_df["lam_obs"].to_numpy(dtype=float)
  print("\nWhat lines do you want to plot?")
  print("(Enter indices like 1,3,5; ranges like 2-6; or 'all')\n")
  for i, w in enumerate(lam, start=1):
    print(f"  [{i:>2}] {w:.6f} Å")
  return lam

def parse_selection(n_items):
  while True:
    s = input("Selection: ").strip().lower()
    if s == "all":
      return list(range(1, n_items + 1))
    # allow commas and ranges like 2-5
    parts = s.replace(" ", "").split(",")
    idxs = []
    ok = True
    for p in parts:
      if "-" in p:
        try:
          a, b = [int(x) for x in p.split("-")]
          if a > b: a, b = b, a
          idxs.extend(range(a, b + 1))
        except Exception:
          ok = False
          break
      elif p:
        try:
          idxs.append(int(p))
        except ValueError:
          ok = False
          break
    if ok and idxs and all(1 <= i <= n_items for i in idxs):
      # dedupe, keep order
      seen = set(); ordered = []
      for i in idxs:
        if i not in seen:
          ordered.append(i); seen.add(i)
      return ordered
    print("Invalid selection. Try '1,3,5', '2-6', or 'all'.")

# ------------------------------------------------------------- #

# ------- Calculating the central wavelengt crossection ------- #
def crossection(lam0, A_ul, A_err, gl, gu):
  # Returns σ_0 in cm^2 km/s such that s = integral(σ_0 * φ) [cm^2]
  
  sig0 = (A_ul * (lam0**3/(8 * np.pi)) * (gu/gl))                 # σ_v = σ_λ = σ_0 * φ_λ = σ_0 * φ_v * c/λ
  sig0 = sig0.to(u.cm**2 * u.km / u.s)
  sig0_err = sig0 * (A_err/A_ul)

  return sig0, sig0_err

# ------------------------------------------------------------- #

# ------------------------------------------------------------- #
def air_to_vacuum(lam_air_A):
    s2 = (1e4/lam_air_A)**2
    n_minus_1 = 1e-8*(8342.13 + 2406030/(130 - s2) + 15997/(38.9 - s2))
    n = 1 + n_minus_1
    return lam_air_A * n
# ------------------------------------------------------------- #

# ------- Retrieving theoretical spectra ------- #
from astropy.table import Table
startpath = 'Kod och data/TS/models_1758706196/bt-nextgen-agss2009/'
def read_Spectra(path: str):

  VOtab = Table.read(path, format='votable')

  lam = VOtab['WAVELENGTH'].value          #u.AA
  flux = VOtab['FLUX'].value              #(u.erg/u.s/(u.cm**2)/u.AA)
  lam = air_to_vacuum(lam)
  return lam * u.AA, flux * (u.erg/u.s/(u.cm**2)/u.AA)

# ---------------------------------------------- #

# -------- Creating the line profiles -------- #
# Calculating profiles and FWHM in wavelength domain
def vel_grid(vlim, N):
  vgrid = (np.linspace(0, 1, N).reshape(1, -1))**(2) * vlim  # (1,N)
  return vgrid

def lorentz_FWHM(A_ul, gam_coll, lam0):
  FWHM_v = lam0 * ((A_ul + gam_coll)/ (2*np.pi))
  return FWHM_v.to(u.km /u.s)

def gaussian_FWHM(b):
  FWHM_v = 2 * np.sqrt(np.log(2)) * b    # b = sqrt((2kT)/(m) + X^2), Δλ_FWHM = 2 * sqrt(ln(2)) * Δλ

  return FWHM_v.to(u.km/u.s)

def lorentz_Profile(v, fwhm_l):

  phi = (1/ np.pi) * (0.5*fwhm_l) / ( (v**2) + ((0.5*fwhm_l)**2) )

  return phi.to(u.s/u.km)

def gaussian_Profile(v, b):
  
  phi = ((1/(b * np.sqrt(np.pi))) * np.exp(-(v/b)**2)).to(1/(u.km/u.s))

  return phi.to(u.s/u.km)
  
def voigt_Profile(v, FWHM_L_v, FWHM_G_v):

  voigt = Voigt1D(x_0=0,
                  amplitude_L= 2 / (np.pi * FWHM_L_v),
                  fwhm_G=FWHM_G_v,
                  fwhm_L=FWHM_L_v)

  phi = voigt(v)

  return phi.to(u.s/u.km)
# --------------------------------------------- #

# --------------------- Converting from v to lambda ------------------------ #
def v_to_lam(lam0, v):
  lam = (lam0 * (1 + (v/const.c))).to(u.AA)
  return lam
# -------------------------------------------------------------------------- #

# ----------------------- Profile Array -------------------------- #

def profile_array(b, fwhm_l, fwhm_g, v, profile:str = 'voigt'):
  
  if profile == 'lorentz':
    phi_array = lorentz_Profile(v, fwhm_l)
  elif profile == 'gauss':
    phi_array = gaussian_Profile(v, b)
  else:
    phi_array = voigt_Profile(v, fwhm_l, fwhm_g)

  return phi_array

def half_to_symmetric(array, v):
  v_sym = np.concatenate((-v[:, :0:-1], v), axis=1)
  array_sym = np.concatenate((array[:, :0:-1], array), axis=1)

  return v_sym, array_sym

# ---------------------------------------------------------------- #

# -------------------------------Crossection Array--------------------------------- #

def crossection_Array(profileArray, sig_0, sig_0_err):

  sig_array = profileArray * sig_0
  sig_array_err = profileArray * sig_0_err

  return sig_array, sig_array_err
# --------------------------------------------------------------------------------- #

# ------------------------------- Radiation Force --------------------------------- #

def radiation_Force(
    species: str,
    lam_min: float,
    lam_max: float,
    R_star: float,
    d_star: float,
    N_col: float,
    star_path: str,
    vlim: float = 20 * u.km/u.s,
    Npoints: int = 100,
    b: float = 0.005 * u.km/u.s,
    gam_col: int = 0 /u.s,
    profile: str = 'voigt',
    plot: bool = False,

):
  # Loading lines
  lines, df = get_nist_lines(species, lam_min, lam_max)

  # Loading constants
  A_ul, Aul_err, lam0, g_u, g_l = pandas_to_numpy(lines)
  sig_0, sig_0_err = crossection(lam0, A_ul, Aul_err, g_l, g_u)
  fwhm_L = lorentz_FWHM(A_ul, gam_col, lam0)
  fwhm_G = gaussian_FWHM(b)

  # Creating velocity grid and arrays for the profiles
  v = vel_grid(vlim, Npoints)
  phi_array = profile_array(b, fwhm_L, fwhm_G, v, profile)
  sig_array, sig_array_err = crossection_Array(phi_array, sig_0, sig_0_err)
  v_sym, phi_sym = half_to_symmetric(phi_array, v)
  v_sym, sig_sym = half_to_symmetric(sig_array, v)
  lam_grid = v_to_lam(lam0, v)                                    # (7, 1000)
  F_star_interp = np.empty_like(sig_array.value, dtype=float)     # (7, 1000)
  L = lam_grid.shape[0]                                           # Amount of lines. used for looping

  # For plotting profiles and crossections. Activated by 'plot='True''
  if plot:
    kind = ask_plot_kind()
    if kind == "q":
        print("Plotting cancelled.")
    else:
        _ = show_lam_choices(lines)                # prints numbered lam_obs list
        idxs_1based = parse_selection(len(lines))  # user picks by numbers
        idxs = [i - 1 for i in idxs_1based]      # convert to 0-based

        if kind == "1":  # PROFILES φ(v)
            for i in idxs:
                plt.figure()
                plt.plot(v_sym[0,:], phi_sym[i])
                plt.title(f"Profile φ(v) = {float(lines['lam_obs'].iloc[i]):.3f} Å")
                plt.xlabel("v [km s$^{-1}$]")
                plt.ylabel("φ(v)")
                plt.tight_layout()
                plt.show()

        elif kind == "2":  # CROSS SECTIONS σ(λ)
            for i in idxs:
                plt.figure()
                plt.plot(v_sym[0,:], sig_sym[i])   # cm^2
                plt.title(f"Profile σ(v) = {float(lines['lam_obs'].iloc[i]):.3f} km s^-1")
                plt.xlabel("v  [km/s]")
                plt.ylabel("σ(v)  [cm$^2$]")
                plt.tight_layout()
                plt.show()
      
  # Retrieving the spectra from the star. This is defined as the flux at the surface of the star, F_λ_surf
  lam_star, flux_star = read_Spectra(star_path)       # (401417,)

  # This is the surface flux (4*π*H_λ)  we multiply by Ω to get a distance dependence (not exactly solid angle)
  omega = (R_star.to(u.m)/d_star.to(u.m))**2
  flux_star *= omega

  # Interpolate the spectra such that the spectra has equally number of lambda-values as the sigma array
  for line in range(L):
    lam_L = lam_grid[line]
    F_star_interp[line] = np.interp(lam_L, lam_star, flux_star) # losing units

  F_star_interp *= u.erg/(u.cm**2)/u.s/u.AA         # adding back units
  
  tau = N_col * sig_array
  absorbed = np.exp(-tau)

  F_ph_perline = (np.trapz(F_star_interp * sig_array * absorbed, lam_grid) *  2/const.c).to(u.N)   # calculating radiation force per line. * 2 because half profile

  F_ph_tot = np.nansum(F_ph_perline)  # total radiation force

  # For propagating the error
  # Build trapezoid weights w with same shape as lam_grid
  dlam = np.diff(lam_grid, axis=1)               # (L, M-1)
  w = np.empty_like(lam_grid)
  w[:, 0]    = 0.5 * dlam[:, 0]
  w[:, -1]   = 0.5 * dlam[:, -1]
  w[:, 1:-1] = 0.5 * (dlam[:, :-1] + dlam[:, 1:])

  F_ph_err_perline = ((2/const.c) * np.sqrt(np.sum((w * F_star_interp * sig_array_err)**2, axis=1))).to(u.N)
  F_ph_err_tot = np.sqrt(np.nansum(F_ph_err_perline**2))


  print(F_ph_tot)
  print(F_ph_err_tot)

  # # Checker for verifying the interpolated curve follows the flux-curve from the star
  # plt.plot(lam_star, flux_star)
  # plt.plot(lam_grid[5], F_star_interp[5])
  # plt.xlim(5891, 5892)
  # plt.xlabel("λ [Å]")
  # plt.ylabel("Interpolated flux")
  # plt.show()



  return F_ph_tot, F_ph_err_tot


# --------------------------------------------------------------------------------- #


path = 'Kod och data/TS/models_1758706196/bt-nextgen-agss2009/lte060-1.0-0.0a+0.0.BT-NextGen.7.dat.xml'
#lam_star, flux_star = read_Spectra('lte060-3.5-1.5a+0.4.BT-NextGen.7.dat.xml')
R_sun = const.R_sun.value * u.m
radiation_Force('Na', 5800, 5900, R_sun, 1*u.au, 1/(u.cm**2), path)




