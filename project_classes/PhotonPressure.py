from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule
from project_classes.Star import Star
from project_func.errors import _not_quantity
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
import time


class PhotonPressure:
  
  def __init__(self, broadeing_profile: BroadeningProfile | BroadeningProfileMolecule, star: Star):
    """
    Creates a photon pressure object for a Star object
    Molecules run by full crossection spectrum
    Atoms run for line by line
    """

    self.broad_prof = broadeing_profile
    self.star = star
    self.flux_star = star.flux_star_rot
    self.lam_star = star.lam_star

    if isinstance(broadeing_profile, BroadeningProfileMolecule):
      self.mode = "molecule"

      self.lam_grid = broadeing_profile.lam_grid
      self.sigma_total = broadeing_profile.sigmaArray
      self.sigma_total_err = broadeing_profile.sigmaArray_err

      self.lam_sym = None
      self.crossection_sym = None
      self.crossection_err_sym = None
      self.flux_star_interp = None
      self.lam_star_interp = None
      self._flux_star_interp_molecule = None

      self.E_l = None
      self.g_l = None

    else:
      self.mode = "atom"

      self.broad_prof = broadeing_profile
      self.lam_sym = broadeing_profile.lam_sym
      self.crossection_sym = broadeing_profile.sigmaArray_sym
      self.crossection_err_sym = broadeing_profile.sigmaArray_sym_err
      
      self.star = star
      self.flux_star = star.flux_star_rot
      self.lam_star = star.lam_star
      self.flux_star_interp = self.get_interp_Spectra()
      self._flux_star_interp_molecule = None
      self.lam_star_interp = self.lam_sym

      self.E_l = broadeing_profile.molecule.E_l
      self.g_l = broadeing_profile.molecule.g_l

    self.F_ph_tot, self.F_ph_tot_err, self.F_ph_perline, self.F_ph_perline_err = None, None, None, None
    self.last_calc_time_molecule = None
  

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
  
  def get_interp_Spectra_molecule(self):
    """
    Interpolates the stellar spectrum onto the shared molecular wavelength grid.
    Cached after the first build because the star and molecular wavelength grid
    do not change within a PhotonPressure object.
    """
    if self._flux_star_interp_molecule is not None:
      return self._flux_star_interp_molecule

    lam_grid = self.lam_grid.to_value(u.AA)
    lam_star = self.lam_star.to_value(u.AA)
    flux_star = self.flux_star.to_value(self.flux_star.unit)

    F_star_interp = np.interp(lam_grid, lam_star, flux_star)
    F_star_interp *= self.flux_star.unit

    self._flux_star_interp_molecule = F_star_interp
    return self._flux_star_interp_molecule
  
  def plot_interp_Spectra(self, line: int):
    if self.mode != "atom":
      raise ValueError("plot_interp_Spectra(line) is only available for atom mode")

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

    # tau_err = sigma_err[:, :, None] * N[None, None, :] 
    # trans_err = np.exp(-tau) * tau_err
    trans_err = None
    
    return trans, trans_err 

  def transmission_molecule(self, column_density, sigma_total=None):
    """
    Transmission on the shared molecular wavelength grid.
    """
    N = np.atleast_1d(column_density.to(u.cm**-2))

    sigma = self.sigma_total if sigma_total is None else sigma_total
    sigma = sigma.to(u.cm**2)

    tau = sigma[:, None] * N[None, :]
    trans = np.exp(-tau)
    trans_err = None

    return trans, trans_err
  
  def excitation_weights(self, Temp_atm):
    kb_eV = const.k_B.to(u.eV/u.K)
    El = self.E_l
    gl =  self.g_l
    T = Temp_atm
    
    # Find all levels for calculating the partition function for all T
    Eg = np.column_stack([El.value, gl])
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

  def calc_PhotonPressure(self, column_density, Temp_atm, distance, chunk_size=1):
    """
    Chunked over N_col to reduce memory.
    Does NOT store/return per-line arrays (returns None for them).

  
    ** Inputs **
    column_density:           Array of column densities                   [cm-2]
    Temp_atm:                 Planetary atmospheric temperature           [K]
    distance                  Distance between object and its gravitational source        [length]

    ** Returns **
    F_ph_tot:                 Total photon pressure                       [N]           [N_temp, N_col]
    F_ph_tot_err:             Error in the total photon pressure -||-     [N]           - || -
    F_ph_perline,             Per line photon pressure                    [N]           [N_lines, N_temp, N_col]
    F_ph_perline_err          Error per line                              [N]           - || -

    """
    if self.mode == "molecule":
      return self.calc_PhotonPressure_molecule(column_density, Temp_atm, distance, chunk_size=chunk_size)
    
    N_col = column_density.to(u.cm**(-2)) if isinstance(column_density, u.Quantity) else _not_quantity("column_density")
    Temp = Temp_atm.to(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")

    d = distance
    R_star = self.star.radius
    omega = (R_star / d) ** 2

    sig = self.crossection_sym                 # (lines, lam)
    sig_err = self.crossection_err_sym         # (lines, lam)
    Flux = self.flux_star_interp * omega       # (lines, lam)
    lam = self.lam_sym                         # (lines, lam)

    weights = self.excitation_weights(Temp)    # (lines, Temp)

    n_T = weights.shape[1]
    n_col = N_col.shape[0]

    # Allocate only total outputs
    F_ph_tot = np.zeros((n_T, n_col)) * u.N
    F_ph_tot_err2 = np.zeros((n_T, n_col)) * (u.N**2)
  

    #### Stripping units before loop. Comment this part for old calculations with units
    Flux_unit = Flux.unit
    sig_unit = sig.unit
    sig_err_unit = sig_err.unit
    lam_unit = lam.unit

    force_unit = (Flux_unit * sig_unit * lam_unit / const.c.unit)
    err_unit = (Flux_unit * sig_err_unit * lam_unit / const.c.unit)

    Flux = np.asarray(Flux.value, dtype=np.float64)
    sig = np.asarray(sig.value, dtype=np.float64)
    sig_err = np.asarray(sig_err.value, dtype=np.float64)
    lam = np.asarray(lam.value, dtype=np.float64)

    weights = np.asarray(weights, dtype=np.float64)
    ##### 

    # Chunk loop over N_col
    N_chunks = 0
    for j0 in range(0, n_col, chunk_size):
      N_chunks += 1
      j1 = min(j0 + chunk_size, n_col)
      N_chunk = N_col[j0:j1]  # (chunk,)

      # Transmission for this chunk only
      Trans, Trans_err = self.transmission(N_chunk)   # (lines, lam, chunk)
      Trans = np.asarray(Trans, dtype=np.float64)     # Comment this for old calculations with units
      
      # Integrand (lines, lam, chunk)
      I_chunk = Flux[:, :, None] * sig[:, :, None] * Trans

      # Per-line force for this chunk (lines, chunk)
      # F_line_chunk = (trapezoid(I_chunk, lam[:, :, None], axis=1) / const.c).to(u.N) # Comment this for new calculations without units
      F_line_chunk = ((trapezoid(I_chunk, lam[:, :, None], axis=1) / const.c.to_value(u.m / u.s)) * force_unit).to(u.N)  # Comment this for old calculations with units

      # Apply excitation weights -> (lines, Temp, chunk)
      F_line_T_chunk =F_line_chunk[:, None, :] * weights[:, :, None]

      # Total force -> (Temp, chunk)
      F_ph_tot[:, j0:j1] = np.nansum(F_line_T_chunk, axis=0)

      ######## Comment for new calculations without units
      # --- Error propagation (chunked) ---
      # factor = (1 - (N_chunk[None, None, :] * sig[:, :, None]))  # (lines, lam, chunk)

      # dF_dA = trapezoid(
      #   (Flux[:, :, None] * Trans * factor * sig_err[:, :, None]) / const.c,
      #   lam[:, :, None],
      #   axis=1
      # )  # (lines, chunk)
      ########

      ######## Comment this for old calculations with units
      N_col_val = N_col.to_value(1 / u.cm**2)
      N_chunk_val = N_col_val[j0:j1] 
      factor = (1.0 - (N_chunk_val[None, None, :] * sig[:, :, None]))

      dF_dA = trapezoid(
        (Flux[:, :, None] * Trans * factor * sig_err[:, :, None]) / const.c.to_value(u.m/u.s),
        lam[:, :, None],
        axis=1
      )  # (lines, chunk)
      dF_dA = (dF_dA * Flux_unit * sig_err_unit * lam_unit / const.c.unit).to(u.N)
      #########

      F_line_err_T_chunk = (np.abs(dF_dA)).to(u.N)[:, None, :] * weights[:, :, None]  # (lines, Temp, chunk)
      F_ph_tot_err2[:, j0:j1] = np.nansum(F_line_err_T_chunk**2, axis=0)

      # print(f"Chunk {N_chunks} completed")

    F_ph_tot_err = np.sqrt(F_ph_tot_err2)

    # print(f"Total photon pressure has been calculated in {N_chunks} chunks")

    # --- Store only totals on the object ---
    self.F_ph_tot = F_ph_tot
    self.F_ph_tot_err = F_ph_tot_err
    self.F_ph_perline = None
    self.F_ph_perline_err = None

    return F_ph_tot, F_ph_tot_err, None, None


  def calc_PhotonPressure_molecule(self, column_density, Temp_atm, distance, chunk_size=1, lam_chunk_size=100000, verbose=True):
    """
    Calculates total photon pressure for a molecule using a stitched molecular
    spectrum built for the requested atmospheric temperature.

    The calculation is chunked over both column density and wavelength to reduce
    memory usage for very large molecular wavelength grids.
    """
    N_col = column_density.to(u.cm**(-2)) if isinstance(column_density, u.Quantity) else _not_quantity("column_density")
    N_col = np.atleast_1d(N_col)
    Temp = Temp_atm.to(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")
    Temp = np.atleast_1d(Temp)

    d = distance
    R_star = self.star.radius
    omega = (R_star / d) ** 2

    t_start_total = time.perf_counter()

    if verbose:
      print(f"Building weighted molecular cross-section for {self.broad_prof.molecule.species} at T = {Temp[0]:.3g}")
    t_start_sigma = time.perf_counter()
    self.broad_prof.apply_boltzmann_weights(Temp[0])
    self.sigma_total = self.broad_prof.sigmaArray
    self.sigma_total_err = self.broad_prof.sigmaArray_err
    t_end_sigma = time.perf_counter()
    if verbose:
      print(
        f"Finished weighted molecular cross-section for {self.broad_prof.molecule.species}; "
        f"wavelength points = {self.lam_grid.shape[0]}, "
        f"build time = {t_end_sigma - t_start_sigma:.2f} s"
      )

    lam = self.lam_grid
    Flux = self.get_interp_Spectra_molecule() * omega
    sigma = self.sigma_total
    sigma_err = self.sigma_total_err

    n_col = N_col.shape[0]
    n_lam = lam.shape[0]
    n_lam_chunks = int(np.ceil(n_lam / lam_chunk_size))
    n_col_chunks = int(np.ceil(n_col / chunk_size))
    n_total_chunks = n_lam_chunks * n_col_chunks

    F_ph_tot = np.zeros(n_col) * u.N
    F_ph_tot_err2 = np.zeros(n_col) * (u.N**2)

    Flux_unit = Flux.unit
    sigma_unit = sigma.unit
    sigma_err_unit = sigma_err.unit
    lam_unit = lam.unit
    force_unit = (Flux_unit * sigma_unit * lam_unit / const.c.unit)

    Flux_val = np.asarray(Flux.value, dtype=np.float64)
    sigma_val = np.asarray(sigma.value, dtype=np.float64)
    sigma_err_val = np.asarray(sigma_err.value, dtype=np.float64)
    lam_val = np.asarray(lam.value, dtype=np.float64)
    N_col_val = np.asarray(N_col.to_value(1 / u.cm**2), dtype=np.float64)

    chunk_counter = 0

    for j0 in range(0, n_col, chunk_size):
      j1 = min(j0 + chunk_size, n_col)
      N_chunk_val = N_col_val[j0:j1]

      F_chunk_sum = np.zeros(j1 - j0, dtype=np.float64)
      F_chunk_err2_sum = np.zeros(j1 - j0, dtype=np.float64)

      for i0 in range(0, n_lam, lam_chunk_size):
        i1 = min(i0 + lam_chunk_size, n_lam)

        lam_chunk = lam_val[i0:i1]
        Flux_chunk = Flux_val[i0:i1]
        sigma_chunk = sigma_val[i0:i1]
        sigma_err_chunk = sigma_err_val[i0:i1]

        tau_chunk = sigma_chunk[:, None] * N_chunk_val[None, :]
        Trans_chunk = np.exp(-tau_chunk)

        integrand_chunk = Flux_chunk[:, None] * sigma_chunk[:, None] * Trans_chunk
        F_chunk_sum += trapezoid(integrand_chunk, lam_chunk[:, None], axis=0) / const.c.to_value(u.m / u.s)

        factor_chunk = 1.0 - (sigma_chunk[:, None] * N_chunk_val[None, :])
        dF_dsigma_chunk = trapezoid(
          (Flux_chunk[:, None] * Trans_chunk * factor_chunk * sigma_err_chunk[:, None]) / const.c.to_value(u.m / u.s),
          lam_chunk[:, None],
          axis=0
        )
        F_chunk_err2_sum += dF_dsigma_chunk**2

        chunk_counter += 1

      F_ph_tot[j0:j1] = (F_chunk_sum * force_unit).to(u.N)
      F_ph_tot_err2[j0:j1] = (F_chunk_err2_sum * (Flux_unit * sigma_err_unit * lam_unit / const.c.unit)**2).to(u.N**2)

      if verbose:
        col_chunk_idx = j0 // chunk_size + 1
        print(
          f"Completed N_col chunk {col_chunk_idx}/{n_col_chunks} "
          f"for {self.broad_prof.molecule.species}"
        )

    F_ph_tot_err = np.sqrt(F_ph_tot_err2)

    t_end_total = time.perf_counter()
    self.last_calc_time_molecule = t_end_total - t_start_total
    if verbose:
      print(
        f"Finished molecule photon pressure for {self.broad_prof.molecule.species} "
        f"in {self.last_calc_time_molecule:.2f} s"
      )

    self.F_ph_tot = F_ph_tot[None, :]
    self.F_ph_tot_err = F_ph_tot_err[None, :]
    self.F_ph_perline = None
    self.F_ph_perline_err = None

    return self.F_ph_tot, self.F_ph_tot_err, None, None


  def beta_Values(self, F_ph_tot, F_ph_tot_err, mass_body, r):
    """
    Calculates the beta ratio for a given photon pressure

    ** Inputs **
    F_ph_tot:               Total photon pressure         [N]         [N_temp, N_col]
    F_ph_tot_err            Error of above value          [N]         - || -
    mass_body:              Mass of the central gravitating body
    r:                      Distance from the host to the absorbing species [length]

    ** Returns **
    beta:                   Beta values                   [Unitless]  [N_temp, N_col]
    beta_err                Errors in beta                [Unitless]  - || -  
    """
    mass_body = mass_body.to(u.g) if isinstance(mass_body, u.Quantity) else _not_quantity("mass_body")
    mass_species = self.broad_prof.molecule.mass
    d = r.to(u.cm) if isinstance(r, u.Quantity) else _not_quantity("r")
    G = const.G.cgs

    F_ph = F_ph_tot
    F_ph_err = F_ph_tot_err
    F_grav = ((G * mass_body * mass_species) / (d)**2).to(u.N)

    beta = F_ph / F_grav
    beta_err = F_ph_err/F_grav

    # print(f"Beta values calculated successfully with the shape: {beta.shape}")
    return(beta, beta_err)
  
  def tau_one_height(self, z, Ncol, Temp_atm):
    """
    Find the height where tau ~ 1 using the strongest *populated* line-center cross section.
    """
    if self.mode != "atom":
      raise ValueError("tau_one_height() is only available for atom mode")

    z = z.to(u.km) if isinstance(z, u.Quantity) else _not_quantity("z")
    Ncol = Ncol.to(1 / u.cm**2) if isinstance(Ncol, u.Quantity) else _not_quantity("Ncol")
    Temp = Temp_atm.to(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")

    # population weights for each lower level
    w_line = self.excitation_weights(Temp)[:, 0]    # Important! Otherwise it may choose a strong line which is unpopulated

    # line-center cross section for each line
    sigma_center = self.broad_prof.sigmaArray[:, 0]   # [N_lines]

    # effective cross section = strongest populated line
    sigma_eff = np.nanmax(sigma_center * w_line)    # Which populated line is the strongest absorber? This is used to approximate τ = Ncol * σ_eff

    # optical depth profile
    tau_z = (Ncol * sigma_eff).decompose()          # optical depth at each height, using the effective cross section

    # height where tau is closest to 1
    idx = np.argmin(np.abs(tau_z.value - 1.0))

    return z[idx], tau_z[idx], tau_z, sigma_eff
