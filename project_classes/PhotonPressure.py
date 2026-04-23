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
    _molecule_flux_interp_cache = {}
    
    def __init__(self, broadening_profile: BroadeningProfile | BroadeningProfileMolecule, star: Star):
        """
        Creates a photon pressure object for a Star object
        Molecules run by full crossection spectrum
        Atoms run for line by line
        """

        self.broad_prof = broadening_profile
        self.star = star
        self.flux_star = star.flux_star_rot
        self.lam_star = star.lam_star

        if isinstance(broadening_profile, BroadeningProfileMolecule):
            self.mode = "molecule"

            self.lam_grid = broadening_profile.lam_grid
            self._flux_star_interp_molecule = None
            self._announced_beta_calc_conditions = set()

        else:
            self.mode = "atom"

            self.broad_prof = broadening_profile
            self.lam_sym = broadening_profile.lam_sym
            self.crossection_sym = broadening_profile.sigmaArray_sym
            self.crossection_err_sym = broadening_profile.sigmaArray_sym_err
            
            self.star = star
            self.flux_star = star.flux_star_rot
            self.lam_star = star.lam_star
            self.flux_star_interp = self.get_interp_Spectra()
            self._flux_star_interp_molecule = None
            self.lam_star_interp = self.lam_sym

            self.E_l = broadening_profile.molecule.E_l
            self.g_l = broadening_profile.molecule.g_l

        self.F_ph_tot, self.F_ph_tot_err, self.F_ph_perline, self.F_ph_perline_err = None, None, None, None
        self.last_calc_time_molecule = None
    

    def get_interp_Spectra(self):
        """
        Interpolates the stars spectra over a profile with different amount of datapoints.
        The interpolation is done over a asymmetric spectra so the symmetrical lambdagrid has to be used
        profile: Class= broadening_profile
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
    
    def _molecule_flux_cache_key(self):
        lam_grid_val = self.lam_grid.to_value(u.AA)
        lam_star_val = self.lam_star.to_value(u.AA)

        star_path = getattr(self.star, "path", None)
        if star_path is None and hasattr(self.star, "header") and isinstance(self.star.header, dict):
            teff_entry = self.star.header.get("teff")
            if isinstance(teff_entry, dict) and "value" in teff_entry:
                star_path = ("teff", float(teff_entry["value"]))
            elif teff_entry is not None:
                star_path = ("teff", float(teff_entry))

        if star_path is None:
            star_path = ("star_id", id(self.star))

        return (
            star_path,
            len(lam_star_val),
            float(lam_star_val[0]),
            float(lam_star_val[-1]),
            len(lam_grid_val),
            float(lam_grid_val[0]),
            float(lam_grid_val[-1]),
        )

    def get_interp_Spectra_molecule(self):
        """
        Interpolates the stellar spectrum onto the shared molecular wavelength grid.
        Cached both on the object and in a class-level cache so repeated
        PhotonPressure objects for the same star/grid can reuse the interpolation.
        """
        if self._flux_star_interp_molecule is not None:
            return self._flux_star_interp_molecule

        cache_key = self._molecule_flux_cache_key()
        if cache_key in PhotonPressure._molecule_flux_interp_cache:
            self._flux_star_interp_molecule = PhotonPressure._molecule_flux_interp_cache[cache_key]
            return self._flux_star_interp_molecule

        lam_grid = self.lam_grid.to_value(u.AA)
        lam_star = self.lam_star.to_value(u.AA)
        flux_star = self.flux_star.to_value(self.flux_star.unit)

        F_star_interp = np.interp(lam_grid, lam_star, flux_star)
        F_star_interp *= self.flux_star.unit

        self._flux_star_interp_molecule = F_star_interp
        PhotonPressure._molecule_flux_interp_cache[cache_key] = F_star_interp
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

    def transmission(self, column_density, sigma_override=None):
        N = np.atleast_1d(column_density.to(u.cm**-2))

        sigma = self.crossection_sym if sigma_override is None else sigma_override
        sigma = sigma.to(u.cm**2)

        if sigma.ndim == 2:
            tau = sigma[:, :, None] * N[None, None, :]
        elif sigma.ndim == 3:
            tau = sigma[:, :, :, None] * N[None, None, None, :]
        else:
            raise ValueError(f"Unexpected sigma dimensions in transmission(): {sigma.shape}")

        trans = np.exp(-tau)

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
        T = Temp_atm.to(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")
        T_val = np.atleast_1d(np.asarray(T.to_value(u.K), dtype=float))

        El_val = np.asarray(self.E_l.to_value(u.eV), dtype=float).reshape(-1)
        gl_val = np.asarray(self.g_l.to_value(u.dimensionless_unscaled), dtype=float).reshape(-1)
        kb_eV_per_K = const.k_B.to_value(u.eV / u.K)

        boltz_lower = gl_val[:, None] * np.exp(-El_val[:, None] / (kb_eV_per_K * T_val[None, :]))

        atom_obj = getattr(self.broad_prof, "molecule", None)
        if atom_obj is not None and hasattr(atom_obj, "partition_function"):
            Z = np.asarray(atom_obj.partition_function(T), dtype=float).reshape(1, -1)
        else:
            Z = np.nansum(boltz_lower, axis=0, keepdims=True)

        w_line = boltz_lower / Z
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

            # Transmission for this chunk only, now using weighted cross sections
            # Apply excitation weights inside the opacity.
            # sigma_weighted has shape (lines, lam, Temp)
            sigma_weighted = sig[:, :, None] * weights[:, None, :]
            sigma_weighted_q = sigma_weighted * sig_unit

            # Transmission for this chunk only, now using weighted cross sections.
            # Trans has shape (lines, lam, Temp, chunk)
            Trans, Trans_err = self.transmission(N_chunk, sigma_override=sigma_weighted_q)
            Trans = np.asarray(Trans, dtype=np.float64)

            # Integrand (lines, lam, Temp, chunk)
            I_chunk = Flux[:, :, None, None] * sigma_weighted[:, :, :, None] * Trans

            # Per-line force for this chunk (lines, Temp, chunk)
            F_line_chunk = ((trapezoid(I_chunk, lam[:, :, None, None], axis=1) / const.c.to_value(u.m / u.s)) * force_unit).to(u.N)

            # weights are already included in sigma_weighted
            F_ph_tot[:, j0:j1] = np.nansum(F_line_chunk, axis=0)

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
            factor = (1.0 - (sigma_weighted[:, :, :, None] * N_chunk_val[None, None, None, :]))

            dF_dA = trapezoid(
            (Flux[:, :, None, None] * Trans * factor * (sig_err[:, :, None, None] * weights[:, None, :, None])) / const.c.to_value(u.m/u.s),
            lam[:, :, None, None],
            axis=1
            )
            dF_dA = (dF_dA * Flux_unit * sig_err_unit * lam_unit / const.c.unit).to(u.N)

            F_ph_tot_err2[:, j0:j1] = np.nansum(np.abs(dF_dA).to(u.N)**2, axis=0)

            # print(f"Chunk {N_chunks} completed")

        F_ph_tot_err = np.sqrt(F_ph_tot_err2)

        # print(f"Total photon pressure has been calculated in {N_chunks} chunks")

        # --- Store only totals on the object ---
        self.F_ph_tot = F_ph_tot
        self.F_ph_tot_err = F_ph_tot_err
        self.F_ph_perline = None
        self.F_ph_perline_err = None

        return F_ph_tot, F_ph_tot_err, None, None


    def calc_PhotonPressure_molecule(self, column_density, Temp_atm, distance, chunk_size=1, lam_chunk_size=1000000, verbose=False):
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

        lam = self.lam_grid
        Flux = self.get_interp_Spectra_molecule() * omega

        n_T = Temp.shape[0]
        n_col = N_col.shape[0]
        n_lam = lam.shape[0]
        n_col_chunks = int(np.ceil(n_col / chunk_size))

        F_ph_tot = np.zeros((n_T, n_col)) * u.N

        Flux_unit = Flux.unit
        c_val = const.c.to_value(u.m / u.s)

        Flux_val = np.asarray(Flux.value, dtype=np.float64)
        lam_val = np.asarray(lam.value, dtype=np.float64)
        N_col_val = np.asarray(N_col.to_value(1 / u.cm**2), dtype=np.float64)

        for t_idx, temp in enumerate(Temp):
            temp_key = float(temp.to_value(u.K))
            t_start_sigma = time.perf_counter()
            if verbose:
                print(f"Building weighted molecular cross-section for {self.broad_prof.molecule.species} at T = {temp:.3g}")
            self.broad_prof.apply_boltzmann_weights(temp, verbose=verbose)
            self.sigma_total = self.broad_prof.sigmaArray
            self.sigma_total_err = self.broad_prof.sigmaArray_err
            sigma = self.sigma_total
            sigma_unit = sigma.unit
            lam_unit = lam.unit
            force_unit = (Flux_unit * sigma_unit * lam_unit / const.c.unit)
            sigma_val = np.asarray(sigma.value, dtype=np.float64)
            base_integrand_val = (Flux_val * sigma_val) / c_val

            dist_key = float(d.to_value(u.AU))
            star_teff = self.star.header["teff"]["value"]
            announce_key = (temp_key, dist_key, float(star_teff))
            if announce_key not in self._announced_beta_calc_conditions:
                print(
                    f"[{self.broad_prof.molecule.species}] calculating betas at d = {dist_key:.2f} AU "
                    f"for T_atm={temp.to_value(u.K):.0f} K and T_eff={float(star_teff):.0f} K"
                )
                self._announced_beta_calc_conditions.add(announce_key)
            t_end_sigma = time.perf_counter()
            if verbose:
                print(
                    f"Finished weighted molecular cross-section for {self.broad_prof.molecule.species}; "
                    f"wavelength points = {self.lam_grid.shape[0]}, "
                    f"build time = {t_end_sigma - t_start_sigma:.2f} s"
                )

            for j0 in range(0, n_col, chunk_size):
                j1 = min(j0 + chunk_size, n_col)
                N_chunk_val = N_col_val[j0:j1]

                F_chunk_sum = np.zeros(j1 - j0, dtype=np.float64)

                for i0 in range(0, n_lam, lam_chunk_size):
                    i1 = min(i0 + lam_chunk_size, n_lam)

                    lam_chunk = lam_val[i0:i1]
                    sigma_chunk = sigma_val[i0:i1]
                    base_integrand_chunk = base_integrand_val[i0:i1]

                    if len(N_chunk_val) == 1:
                        N_val = N_chunk_val[0]
                        tau_chunk = sigma_chunk * N_val
                        Trans_chunk = np.exp(-tau_chunk)

                        integrand_chunk = base_integrand_chunk * Trans_chunk
                        F_chunk_sum[0] += trapezoid(integrand_chunk, lam_chunk, axis=0)
                    else:
                        tau_chunk = sigma_chunk[:, None] * N_chunk_val[None, :]
                        Trans_chunk = np.exp(-tau_chunk)

                        integrand_chunk = base_integrand_chunk[:, None] * Trans_chunk
                        F_chunk_sum += trapezoid(integrand_chunk, lam_chunk[:, None], axis=0)

                F_ph_tot[t_idx, j0:j1] = (F_chunk_sum * force_unit).to(u.N)

                if verbose and n_col_chunks > 1:
                    col_chunk_idx = j0 // chunk_size + 1
                    print(
                        f"Completed N_col chunk {col_chunk_idx}/{n_col_chunks} "
                        f"for {self.broad_prof.molecule.species} at T={temp.to_value(u.K):.0f} K"
                    )

        F_ph_tot_err = np.zeros((n_T, n_col)) * u.N

        t_end_total = time.perf_counter()
        self.last_calc_time_molecule = t_end_total - t_start_total
        if verbose:
            print(
                f"Finished molecule photon pressure for {self.broad_prof.molecule.species} "
                f"in {self.last_calc_time_molecule:.2f} s"
            )

        self.F_ph_tot = F_ph_tot
        self.F_ph_tot_err = F_ph_tot_err
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
