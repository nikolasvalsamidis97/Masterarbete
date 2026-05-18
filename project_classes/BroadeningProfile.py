from project_classes.Atom import Atom
from project_utils.errors import _not_quantity
from astropy import constants as const
from astropy import units as u
import numpy as np
from astropy.modeling.models import Voigt1D
from scipy.special import wofz
from matplotlib import pyplot as plt

class BroadeningProfile:
    
    def __init__(self, molecule: Atom, b, N:int, profileType:str = 'Voigt'):
        """
        Contains both the broadeing profile and calculates the crossection using the profile
        molecule:    Molecule
        b:           broadening parameter                 [km/s]
        N:           resolution of the velocity grid      [int]
        profileType: Type of broadening.                  Ex. "Lorentz", "Gauss" or "Voigt"
        """
        self.molecule = molecule
        self.b = b.to(u.km /u.s) if isinstance(b, u.Quantity) else _not_quantity("b (broadening parameter)")
        self.vlim = self.set_vlim()
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
        
        
    def set_vlim(self):
        fwhm_l, _ = self.FWHM_lorentz()
        fwhm_g = self.FWHM_gauss()
        vlim = np.maximum(6 * fwhm_g, 25 * fwhm_l)
        return vlim.to(u.km/u.s)
    
    def velocity_Grid(self):
        """
        Per-line velocity grids using the same quadratic spacing, but with a per-line
        vlim so each line gets the same window logic as the molecule pipeline.
        v_grid [km/s]
        """
        base = (np.linspace(0, 1, self.N).reshape(1, -1))**2
        v_grid = base * self.vlim
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
        phi = ((1/(self.b * np.sqrt(np.pi))) *
               np.exp(-(self.v_grid/self.b)**2)).to(1/(u.km/u.s))
        phi_err = np.zeros_like(phi.value) * phi.unit
        return phi.to(u.s/u.km), phi_err.to(u.s/u.km)

    def voigt_Profile(self):
        L = self.lorentz_FWHM_v
        dL = self.lorentz_FWHM_v_err
        G = self.gauss_FWHM_v
        v = self.v_grid

        sigma = (G / (2.0 * np.sqrt(2.0 * np.log(2.0)))).to(u.km/u.s)
        gamma = (0.5 * L).to(u.km/u.s)
        dgamma = (0.5 * dL).to(u.km/u.s)

        v_val = v.to_value(u.km/u.s)
        sigma_val = sigma.to_value(u.km/u.s)
        gamma_val = gamma.to_value(u.km/u.s)
        dgamma_val = dgamma.to_value(u.km/u.s)

        z = (v_val + 1j * gamma_val) / (sigma_val * np.sqrt(2.0))
        phi_val = np.real(wofz(z)) / (sigma_val * np.sqrt(2.0 * np.pi))

        if np.all(np.isclose(dgamma_val, 0.0, equal_nan=True)):
            phi_err_val = np.zeros_like(phi_val)
        else:
            gamma_m = np.maximum(gamma_val - dgamma_val, 1e-30)
            gamma_p = gamma_val + dgamma_val
            z_m = (v_val + 1j * gamma_m) / (sigma_val * np.sqrt(2.0))
            z_p = (v_val + 1j * gamma_p) / (sigma_val * np.sqrt(2.0))
            phi_m = np.real(wofz(z_m)) / (sigma_val * np.sqrt(2.0 * np.pi))
            phi_p = np.real(wofz(z_p)) / (sigma_val * np.sqrt(2.0 * np.pi))
            delta_gamma = (gamma_p - gamma_m) * 2.0
            dphi_dL = np.divide(
                phi_p - phi_m,
                delta_gamma,
                out=np.zeros_like(phi_p),
                where=np.abs(delta_gamma) > 0.0,
            )
            phi_err_val = np.abs(dphi_dL) * (2.0 * dgamma_val)

        phi = phi_val * (u.s/u.km)
        phi_err = phi_err_val * (u.s/u.km)
        return phi.to(u.s/u.km), phi_err.to(u.s/u.km)

    def profile_Array(self):
        """
        Returns a half symmetric (normalized to 0.5) broadening profile. If the number of lines > 1 an array, for each, line is returned
        phi [1/km/s]
        """

        profile_type = str(self.profileType).strip().lower()

        if profile_type in ('lorentz', 'lorentzian'):
            return self.lorentz_Profile()
        elif profile_type in ('gauss', 'gaussian'):
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
        dphi_dL_unit = phi_err.unit / dL.unit
        dphi_dL_val = np.divide(
            phi_err.to_value(phi_err.unit),
            dL.to_value(dL.unit),
            out=np.zeros_like(phi_err.value, dtype=float),
            where=np.abs(dL.to_value(dL.unit)) > 0,
        )
        dphi_dL_mag = dphi_dL_val * dphi_dL_unit

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
  
