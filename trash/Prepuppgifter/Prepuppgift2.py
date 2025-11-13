from astropy import constants as const
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
from astropy.modeling.models import Voigt1D

#========= Constants for Na =========#
m_Na = 22.989769*u.u

# 589.1583264 1st
A_1 = 6.16e7/u.s
lam_1 = 589.1583264 * u.nm
gl_1 = 2
gu_1 = 4

# 589.7558147 2nd
A_2 = 6.14e7/u.s
lam_2 = 589.7558147 * u.nm
gl_2 = 4
gu_2 = 4
#=====================================#

#=================Const====================#
T_sun = 5778 * u.K
l_cent = (590 * u.nm).to(u.m)
d_to_sun = 1 * const.au
R_sun = const.R_sun
h = const.h
C = const.c
k = const.k_B
#=========================================#

#============ Distances =============#
d1 = 0.1 * u.au
d2 = u.au
d3 = 10 * u.au
#====================================#

#================Crossection====================#
def crossection(A_ul, wav, gl, gu):
  sig = A_ul * (wav**4/(8 * np.pi * const.c)) * (gu/gl)
  return sig.to(u.cm**2 * u.AA)

sig_1 = crossection(A_1, lam_1, gl_1, gu_1)
sig_2 = crossection(A_2, lam_2, gl_2, gu_2)
#===============================================#


#================Planck=================#
def planck(T, lam, R, d):
  Omega = np.pi * (R/d)**2
  first = (2 * h * C**2)/((lam)**5)
  exp = (h * C)/(lam * k * T)
  second = (np.e**(exp) - 1)**(-1)
  B = first * second
  Flux = B * Omega
  return Flux.to(u.erg / u.cm**2 / u.s / u.AA)

F_sun1 = planck(T_sun, l_cent, R_sun, d_to_sun * 0.1)
F_sun2 = planck(T_sun, l_cent, R_sun, d_to_sun * 1)
F_sun3 = planck(T_sun, l_cent, R_sun, d_to_sun * 10)
#=======================================#

#========== Gravitation at 0.1 1 and 10 au ==========#
def gravforce(mass, d):
  F_grav = (const.G * const.M_sun * mass) / (d**2)
  return F_grav.to(u.N)

F_g1 = gravforce(m_Na, d1)
F_g2 = gravforce(m_Na, d2)
F_g3 = gravforce(m_Na, d3)
#=====================================================#

#====================Radpressure======================#  Only at line center, No broadening
def radPressure(sig, flux):
  F_rad = sig / C*flux
  return F_rad.to(u.N)

radPressure1 = radPressure(sig_1+sig_2, F_sun1)
radPressure2 = radPressure(sig_1+sig_2, F_sun2)
radPressure3 = radPressure(sig_1+sig_2, F_sun3)
#=====================================================#
print("")
print("Without broadening, we get β, at 0,1, 1 and 10 AU ")
print(f"β1 = {(radPressure1/F_g1):.3f}, β2 = {(radPressure2/F_g2):.3f}, β3 = {(radPressure3/F_g3):.3f}")
print("")
#============================================= Part 2 ===========================================#
#====================================== Include line broadeing ==================================#

def lorentz_fwhm(Aul, lam0):
  
  lam0 = lam0.to(u.AA)
  dnu = Aul / (2*np.pi)
  dlam = ((lam0**2)/C) * dnu
  dlam = dlam.to(u.AA)

  return dlam

def lorentz_profile(Aul, lam, lam0):
  
  dlam = lorentz_fwhm(Aul, lam0)
  lam = lam.to(u.AA)
  lam0 = lam0.to(u.AA)
  phi = (1/np.pi) * ( (dlam/2) / ((lam-lam0)**2 + (dlam/2)**2) )
  
  return phi

def gauss_fwhm(lam0, b):
  lam0 = lam0.to(u.AA)
  dlam_g = (2*np.sqrt(np.log(2)) * lam0 * (b/const.c)).to(u.AA)
  return dlam_g

def voigt_profile(lam, lam0, fwhm_L, fwhm_G):
  lam = lam.to(u.AA).value
  lam0 = lam0.to(u.AA).value

  voigt = Voigt1D(x_0=lam0,
                  amplitude_L=1.0,
                  fwhm_L=fwhm_L.to(u.AA).value,
                  fwhm_G=fwhm_G.to(u.AA).value)
  
  y = voigt(lam)
  area = np.trapz(y, lam)
  phi_norm = (y / area) / u.AA
  return phi_norm

def radPressure_lorentz(Aul, lam0, sigma, d, T = T_sun, R = R_sun,  nwidth=3000, N=60001):
  
  # Wavelength grid around lam0
  dlam = lorentz_fwhm(Aul, lam0) # in Å
  lam0 = lam0.to(u.AA)
  lamgrid = lam0 + np.linspace(-nwidth, nwidth, N) * dlam # (-3000+lam0)*dlam .... lam0 .... (3000+lam0)dlam
  
  # Profile and crossection
  phi = lorentz_profile(Aul, lamgrid, lam0) # [1/Å]
  sig = (sigma * phi).to(u.cm**2) # Sigma is strongest at peak. More absorbtion in line center

  # Spectral flux over wavelength grid
  F_lam = planck(T, lamgrid.to(u.m), R, d)

  # integrating 
  integ = (sig * F_lam).to(u.erg / u.s / u.AA)
  integral = np.trapz(integ.value, lamgrid.to(u.AA).value) * (u.erg / u.s) # cm2 * erg/cm2/s AA Falls out after integration

  # Photon pressure
  F_ph = (integral / C).to(u.N) # erg/s * s/m = N
  return F_ph

def radPressure_voigt(Aul, lam0, sigma, d, b, Ncol, T = T_sun, R = R_sun, nwidth=3000, N=60001):
  
  dlam_g = gauss_fwhm(lam0, b)
  dlam_l = lorentz_fwhm(Aul, lam0)

  dlg = dlam_g.value
  dll = dlam_l.value
  dl = (dlam_g if dlg >= dll else dlam_l)# Choosing the larger width
  lam0 = lam0.to(u.AA)
 
  lamgrid = lam0 + (np.linspace(-nwidth, nwidth, N) * dl)

  voigt = voigt_profile(lamgrid, lam0, fwhm_L=dlam_l, fwhm_G=dlam_g)
  sig = (sigma * voigt).to(u.cm**2)
  tau = (sig * Ncol.to(1/u.cm**2)).value

  F_lam = planck(T, lamgrid.to(u.m), R, d)

  integ = (sig * F_lam * np.exp(-tau)).to(u.erg / u.s / u.AA)
  F_lam_obs = np.trapz(integ.value, lamgrid.value) * (u.erg / u.s)

  F_ph = (F_lam_obs/C).to(u.N)

  return F_ph



radP1_lorentz = radPressure_lorentz(A_1, lam_1, sig_1,d1) + radPressure_lorentz(A_2, lam_2, sig_2, d1)
radP2_lorentz = radPressure_lorentz(A_1, lam_1, sig_1,d2) + radPressure_lorentz(A_2, lam_2, sig_2, d2)
radP3_lorentz = radPressure_lorentz(A_1, lam_1, sig_1,d3) + radPressure_lorentz(A_2, lam_2, sig_2, d3)



print("With Lorenz-Broadening we get β, for 0.1, 1, and 10 AU")
print(f"β1 = {(radP1_lorentz/F_g1):.3f}, β2 = {(radP2_lorentz/F_g2):.3f}, β3 = {(radP3_lorentz/F_g3):.3f}")

#============================================= Part 3 ===========================================#
#====================================== Optical depth dependence ==================================#


# ------------------ Computing arrays ------------------- #

def precomp_arrays(Aul, lam0, sigma0, d, T = T_sun, R = R_sun, nwidth=3000, Npoints=60001): # Npoints not to confuse
  
  dlam = lorentz_fwhm(Aul, lam0)
  lam0 = lam0.to(u.AA)

  lam_grid = lam0 + np.linspace(-nwidth, nwidth, Npoints) * dlam  # Using the FWHM for a convenient width
  phi = lorentz_profile(Aul, lam_grid, lam0)
  sig_lam = (sigma0 * phi).to(u.cm**2)
  F_lam = planck(T, lam_grid, R, d).to(u.erg / u.cm**2 / u.s / u.AA)

  return lam_grid, sig_lam, F_lam


# --------------- Calculating the optical depth dependent radpressure --------------- #

def radPressure_tau(Aul, lam0, sig0, d, T, R, N_col, nwidth=3000, Npoints=60001):
  
  lam_grid, sig_lam, F_lam = precomp_arrays(Aul, lam0, sig0, d, T, R, nwidth=3000, Npoints=60001)

  lam_grid = lam_grid.to(u.AA)
  sig_lam = sig_lam.to(u.cm**2)
  F_lam = F_lam.to(u.erg / u.cm**2 / u.s / u.AA)

  N_col = N_col.to(1/u.cm**2)
  tau_lam = (sig_lam * N_col)

  F_obs_lam_integrand = sig_lam * F_lam * np.exp(-tau_lam)

  F_obs_lam_value = np.trapz(F_obs_lam_integrand, lam_grid)
  

  return (F_obs_lam_value/C).to(u.N)

# --------------- calculating betas for an array of column densities --------------- #

def beta_tau(b, d, Ncol, Fg, T=T_sun, R=R_sun, nwidth=3000, Npoints= 600001):
  
  tot = []

  for Nval in Ncol:
    F1 = radPressure_voigt(A_1, lam_1, sig_1, d, b, Nval, T_sun, R_sun)
    F2 = radPressure_voigt(A_2, lam_2, sig_2, d, b, Nval, T_sun, R_sun)

    tot.append(((F1 + F2)/Fg).to(u.dimensionless_unscaled).value) # Adding both sigmas here

  return tot

N_grid = np.logspace(0, 30, 66)*(1/u.cm**2)

b = 1 * u.km / u.s
b1 = 2 * u.km / u.s
b2 = 3 * u.km / u.s
b3 = 4 * u.km / u.s
b4 = 5 * u.km / u.s
betaTau_d1 = beta_tau(b, d1, N_grid, F_g1)
betaTau_d2 = beta_tau(b, d2, N_grid, F_g2)
betaTau_d3 = beta_tau(b, d3, N_grid, F_g3)

betaTau_b1 = beta_tau(b1, d1, N_grid, F_g1)
betaTau_b2 = beta_tau(b2, d1, N_grid, F_g1)
betaTau_b3 = beta_tau(b3, d1, N_grid, F_g1)
betaTau_b4 = beta_tau(b4, d1, N_grid, F_g1)

print("\nBetas from flux with optical depth at N=0")
print(f" β1(N_min)≈{betaTau_d1[0]:.3f}")
print(f" β2(N_min)≈{betaTau_d2[0]:.3f}")
print(f" β3(N_min)≈{betaTau_d3[0]:.3f}")

plt.figure()
plt.loglog(N_grid, betaTau_d1, label = 'b = 1')
plt.loglog(N_grid, betaTau_b1, label = 'b = 2')
plt.loglog(N_grid, betaTau_b2, label = 'b = 3')
plt.loglog(N_grid, betaTau_b3, label = 'b = 4')
plt.loglog(N_grid, betaTau_b4, label = 'b = 5')
plt.xlabel(r'N [cm$^{-2}$]')
plt.ylabel(r'$\beta = F_{\mathrm{rad}}/F_{\mathrm{grav}}$')
plt.legend()
plt.tight_layout()
plt.show()

def plot_voigt_transmission_D1(b=1.0*u.km/u.s, nwidth=300, Npts=60001):
  """Na D1 Voigt transmission for 8 columns (N=1e10..1e17 cm^-2)."""
  # Build λ-grid scaled by the larger of the Gaussian/Lorentz FWHM
  fwhm_G = gauss_fwhm(lam_1, b)
  fwhm_L = lorentz_fwhm(A_1, lam_1)
  step = fwhm_G if fwhm_G >= fwhm_L else fwhm_L
  lam0 = lam_1.to(u.AA)
  lam = lam0 + np.linspace(-nwidth, nwidth, Npts) * step


  # Voigt profile (normalized in λ); per-λ cross section
  phiV = voigt_profile(lam, lam0, fwhm_L=fwhm_L, fwhm_G=fwhm_G) # [1/Å]
  sigma_lambda = (sig_1 * phiV).to(u.cm**2)


  Ns = (np.logspace(10, 17, 8) / u.cm**2)


  plt.figure()
  for Ncol in Ns:
    tau = (sigma_lambda * Ncol).to(u.dimensionless_unscaled).value
    T = np.exp(-tau)
    plt.plot(lam.value, T, label=f'1e{int(np.log10(Ncol.value))}')
  plt.xlabel(r'$\lambda$ [Å]')
  plt.ylabel(r'Transmission $e^{-\tau_\lambda}$')
  plt.title(f'Na D1 Voigt transmission (b={b.to(u.km/u.s).value:.1f} km/s)')
  plt.ylim(-0.02, 1.02)
  plt.legend(title='N [cm$^{-2}$]')
  plt.grid(True, alpha=0.3)
  plt.tight_layout()




def plot_voigt_transmission_D2(b=1.0*u.km/u.s, nwidth=300, Npts=60001):
  """Na D2 Voigt transmission for 8 columns (N=1e10..1e17 cm^-2)."""
  fwhm_G = gauss_fwhm(lam_2, b)
  fwhm_L = lorentz_fwhm(A_2, lam_2)
  step = fwhm_G if fwhm_G >= fwhm_L else fwhm_L
  lam0 = lam_2.to(u.AA)
  lam = lam0 + np.linspace(-nwidth, nwidth, Npts) * step


  phiV = voigt_profile(lam, lam0, fwhm_L=fwhm_L, fwhm_G=fwhm_G) # [1/Å]
  sigma_lambda = (sig_2 * phiV).to(u.cm**2)


  Ns = (np.logspace(10, 17, 8) / u.cm**2)


  plt.figure()
  for Ncol in Ns:
    tau = (sigma_lambda * Ncol).to(u.dimensionless_unscaled).value
    T = np.exp(-tau)
    plt.plot(lam.value, T, label=f'1e{int(np.log10(Ncol.value))}')
  plt.xlabel(r'$\lambda$ [Å]')
  plt.ylabel(r'Transmission $e^{-\tau_\lambda}$')
  plt.title(f'Na D2 Voigt transmission (b={b.to(u.km/u.s).value:.1f} km/s)')
  plt.ylim(-0.02, 1.02)
  plt.legend(title='N [cm$^{-2}$]')
  plt.grid(True, alpha=0.3)
  plt.tight_layout()


plot_voigt_transmission_D1(b=1.0*u.km/u.s, nwidth=300, Npts=60001)
plot_voigt_transmission_D2(b=1.0*u.km/u.s, nwidth=300, Npts=60001)
plt.show()
