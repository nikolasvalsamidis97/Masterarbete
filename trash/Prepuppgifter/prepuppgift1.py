from astropy import constants as const
from astropy import units as u
import numpy as np


#========= Constants for Na =========#
m_Na = 22.989769*u.u

# 589.1583264
A_5891 = 6.16e7/u.s
w_5891 = 589.1583264 * u.nm
g_l_5891 = 2
g_u_5891 = 4

# 589.7558147
A_5897 = 6.14e7/u.s
w_5897 = 589.7558147 * u.nm
g_l_5897 = 4
g_u_5897 = 4

#=====================================#


#============ Distances =============#
d1 = 0.1 * u.au
d2 = u.au
d3 = 10 * u.au
#====================================#

#========== Gravitation at 0.1 1 and 10 au ==========#
def gravforce(mass, d):
  F_grav = (const.G * const.M_sun * mass) / (d**2)
  return F_grav.to(u.N)

F_grav_01 = gravforce(m_Na, d1)
F_grav_1 = gravforce(m_Na, d2)
F_grav_10 = gravforce(m_Na, d3)

#=====================================================#


#================ Radiation Pressure ================#

# Crossection
def crossection(A_ul, wav, gl, gu):
  sig = A_ul * (wav**4/(8 * np.pi * const.c)) * (gu/gl)
  return sig.to(u.cm**2 * u.AA)

sig_5891 = crossection(A_5891, w_5891, g_l_5891, g_u_5891)
sig_5897 = crossection(A_5897, w_5897, g_l_5897, g_u_5897)

print(f"Crossection for Na-5891 Å: {sig_5891:.3e}, Na-5891 Å: {sig_5897:.3e}")
#=====================================================#


#====================== Solar flux ======================#

T_sun = 5778 * u.K
l_cent = (590 * u.nm).to(u.m)
d_to_sun = 1 * const.au
R_sun = const.R_sun
h = const.h
C = const.c
k = const.k_B

def planck(T, lam, R, d):
  Omega = np.pi * (R/d)**2
  first = (2 * h * C**2)/((lam)**5)
  exp = (h * C)/(lam * k * T)
  second = (np.e**(exp) - 1)**(-1)
  B = first * second
  Flux = B * Omega
  return Flux.to(u.erg / u.cm**2 / u.s / u.AA)

F_sun_01 = planck(T_sun, l_cent, R_sun, d_to_sun * 0.1)
F_sun_1 = planck(T_sun, l_cent, R_sun, d_to_sun * 1)
F_sun_10 = planck(T_sun, l_cent, R_sun, d_to_sun * 10)

#========================================================#

#================== Radiation pressure ==================#

def radPressure(sig, flux):
  F_rad = sig / C*flux
  return F_rad.to(u.N)

# all transitions
sig_all = sig_5891 + sig_5897

F_rad_01 = radPressure(sig_all,F_sun_01)
F_rad_1 = radPressure(sig_all,F_sun_1)
F_rad_10 = radPressure(sig_all,F_sun_10)
#========================================================#

#=============Lorentz Profile===============#
def lorenz(lam, lam0, A_ul):
  lam  = lam.to(lam0.unit)
  lam0 = lam0.to(lam0.unit)

  dnu = A_ul/(4*np.pi)  # goto wavelength
  gamma = (lam0**2/C) * dnu # now in wavlength domain
  x = (lam - lam0)
  phi = (1/np.pi) * (gamma / (x**2 + gamma**2))

  num = np.trapz(phi.to(1/lam0.unit).value, lam.to(lam0.unit).value)
  phi = (phi / num)

  return phi

def rad_lorentz(sig_int, lam0, A_ul, T, R, d, offset=0.1*u.AA, N=20001):
  lam = np.linspace(
    (lam0 - offset).to(lam0.unit).value, 
    (lam0 + offset).to(lam0.unit).value, 
    N
    )
  phi = lorenz(lam, lam0, A_ul)
  sig_lam = sig_int.to(u.cm**2 * lam0.unit) * phi.to(u.cm**2)
  F_lam = planck(T, lam, R, d).to(u.erg/u.cm**2/u.s/lam.unit)

  # numerical integral
  integ = (sig_lam * F_lam).to(u.erg/u.s/lam.unit)
  I = np.trapz(integ.value, lam.to(lam0.unit).value) * u.erg/u.s

  F_rad = (I/C).to(u.N)

  return F_rad
#========================================================#

#================== Results ==================#
Beta_01AU = F_rad_01/F_grav_01
Beta_1AU = F_rad_1/F_grav_1
Beta_10AU = F_rad_10/F_grav_10

print(f"0.1 AU: F_grav = {F_grav_01:.2e} F_rad = {F_rad_01:.2e} and Beta = {Beta_01AU.round(2)}")
print(f"1 AU: F_grav = {F_grav_1:.2e} F_rad = {F_rad_1:.2e} and Beta = {Beta_1AU.round(2)}")
print(f"10 AU: F_grav = {F_grav_10:.2e} F_rad = {F_rad_10:.2e} and Beta = {Beta_10AU.round(2)}")

#=============================================#

F_rad_01_Lorentz = (
    rad_lorentz(sig_5891, w_5891, A_5891, T_sun, R_sun, 0.1*u.au) +
    rad_lorentz(sig_5897, w_5897, A_5897, T_sun, R_sun, 0.1*u.au)
)
Beta_01AU_Lorentz = (F_rad_01_Lorentz / F_grav_01).decompose()
print(Beta_01AU_Lorentz)