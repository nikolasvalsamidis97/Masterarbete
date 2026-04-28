from project_classes.Molecule import Molecule
from project_func.errors import _not_quantity
from astropy import constants as const
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.special import wofz
from radis.io.hitran import fetch_hitran
import pathlib
import hashlib
import json


class BroadeningProfileMolecule:
    def __init__(self,
                 molecule: Molecule,
                 b,
                 lam_min=None,
                 lam_max=None,
                 dlam=None,
                 profileType: str = 'Voigt',
                 verbose=False,
                 ):
        """
        Build a TOTAL molecular cross-section spectrum on one shared wavelength grid.

        This class now assumes the Molecule object is a cache/setup object only.
        Actual line data are loaded on demand, chunk-by-chunk, during the stitched
        cross-section build.
        """
        self.molecule = molecule
        self.b = b.to(u.km / u.s) if isinstance(b, u.Quantity) else _not_quantity("b (broadening parameter)")
        self.profileType = profileType.lower()
        self.init_timers = {}

        self._announced_weight_build_temps = set()
        self._announced_weight_ready_temps = set()
        self._announced_beta_calc_temps = set()

        if not getattr(self.molecule, "cache_ready", False):
            raise ValueError(
                "Molecule cache is not prepared. Run fetch_exomol(...) or fetch_hitran(...) "
                "before creating BroadeningProfileMolecule."
            )

        t0 = time.perf_counter()
        self.c_kms = const.c.to_value(u.km / u.s)
        self.kb_eV_per_K = const.k_B.to_value(u.eV / u.K)
        self.invcm_to_eV = (1.0 * (1 / u.cm)).to_value(u.eV, equivalencies=u.spectral())
        self.init_timers["set_constants"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.lam_min = self.set_lam_min(lam_min)
        self.init_timers["set_lam_min"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.lam_max = self.set_lam_max(lam_max)
        self.init_timers["set_lam_max"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.dlam = self.set_dlam(dlam)
        self.init_timers["set_dlam"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.lam_grid = self.wavelength_grid()
        self.init_timers["wavelength_grid"] = time.perf_counter() - t0

        self.lam_grid_val = self.lam_grid.to_value(u.AA)

        self.sigmaArray = None
        self.sigmaArray_err = None
        self.sigma_total = None
        self.sigma_total_err = None
        self._sigma_cache_by_temp = {}
        self.partition_function_cache = {}
        self._states_g_lookup = None
        self._weight_cache_root = self._build_weight_cache_root()
        self._weight_cache_root.mkdir(parents=True, exist_ok=True)

        # Temporary speed hack: skip lines whose weighted central strength is tiny
        # compared to the strongest weighted line in the current build.
        self.temp_strength_rel_cutoff = 1e-8

        self.init_timers["total_init_profile_setup"] = np.sum(list(self.init_timers.values()))
        if verbose:
            print(f"{self.molecule.species} BroadeningProfileMolecule init timing:")
            for key, value in self.init_timers.items():
                print(f"  {key}: {value:.2f} s")

    def _temperature_cache_key(self, Temp_atm):
        T_val = float(np.asarray(Temp_atm.to_value(u.K)).reshape(-1)[0])
        return T_val

    def set_lam_min(self, lam_min):
        if lam_min is not None:
            return lam_min.to(u.AA) if isinstance(lam_min, u.Quantity) else _not_quantity("lam_min")
        return (1 / self.molecule.wavenum_max).to(u.AA)

    def set_lam_max(self, lam_max):
        if lam_max is not None:
            return lam_max.to(u.AA) if isinstance(lam_max, u.Quantity) else _not_quantity("lam_max")
        return (1 / self.molecule.wavenum_min).to(u.AA)

    def set_dlam(self, dlam):
        if dlam is not None:
            return dlam.to(u.AA) if isinstance(dlam, u.Quantity) else _not_quantity("dlam")

        # rep = 0.5 * self.lam_min
        rep = 0.5 * (self.lam_min + self.lam_max)
        doppler_sigma = (rep * (self.b / const.c)).to(u.AA)
        dlam_auto = (doppler_sigma / 3.0).to(u.AA)
        floor = 1e-5 * u.AA
        return np.maximum(dlam_auto, floor)

    def wavelength_grid(self):
        npts = int(np.floor(((self.lam_max - self.lam_min) / self.dlam).decompose().value)) + 1
        grid = self.lam_min + np.arange(npts) * self.dlam
        return grid.to(u.AA)

    def calc_central_crossection(self, A_vals, lam0_val, gup_vals, glower_vals):
        """
        Numeric line-center cross section in [cm^2 km / s].
        Inputs are plain NumPy arrays:
        - A_vals      : Einstein A [1/s]
        - lam0_val    : wavelength [Angstrom]
        - gup_vals    : upper degeneracy
        - glower_vals : lower degeneracy
        """
        sig0_val = A_vals * ((lam0_val ** 3) / (8.0 * np.pi)) * (gup_vals / glower_vals)
        sig0_val *= 1.0e-29
        sig0_err_val = np.zeros_like(sig0_val)
        return sig0_val, sig0_err_val

    def FWHM_lorentz(self, A_vals, lam0_val):
        """
        Numeric Lorentz FWHM in [km/s].
        Inputs are plain NumPy arrays:
        - A_vals   : Einstein A [1/s]
        - lam0_val : wavelength [Angstrom]
        """
        lorentz_val = lam0_val * (A_vals / (2.0 * np.pi)) * 1.0e-13
        lorentz_err_val = np.zeros_like(lorentz_val)
        return lorentz_val, lorentz_err_val

    def FWHM_gauss(self, lam0_val):
        """
        Numeric Gaussian FWHM in [km/s].
        The result is constant across lines for fixed b, but shaped like lam0_val.
        """
        gauss_scalar = self.b.to_value(u.km / u.s) * (2.0 * np.sqrt(np.log(2.0)))
        return np.full_like(lam0_val, gauss_scalar)

    def set_vlim(self, gauss_val, lorentz_val):
        """
        Numeric profile half-window in [km/s].
        """
        return np.maximum(6.0 * gauss_val, 25.0 * lorentz_val)

    def profile_from_velocity_batch(self, v_val, lorentz_val, lorentz_err_val, voigt_sigma_val, voigt_gamma_val, voigt_gamma_err_val):
        """
        Vectorized profile evaluation for a batch of lines.

        Parameters are expected to broadcast as:
        - v_val:                (n_lines, n_points)
        - lorentz_val:          (n_lines, 1)
        - lorentz_err_val:      (n_lines, 1)
        - voigt_sigma_val:      (n_lines, 1)
        - voigt_gamma_val:      (n_lines, 1)
        - voigt_gamma_err_val:  (n_lines, 1)
        """
        if self.profileType == 'lorentz':
            L = lorentz_val
            dL = lorentz_err_val
            phi_val = (1.0 / np.pi) * (0.5 * L) / (v_val**2 + (0.5 * L)**2)
            phi_err_val = np.abs((1.0 / np.pi) * 0.5 * (v_val**2 - L**2 / 4.0) / (v_val**2 + L**2 / 4.0)**2) * dL
            return phi_val, phi_err_val

        if self.profileType == 'gauss':
            b_val = self.b.to_value(u.km / u.s)
            phi_val = (1.0 / (b_val * np.sqrt(np.pi))) * np.exp(-(v_val / b_val)**2)
            phi_err_val = np.zeros_like(phi_val)
            return phi_val, phi_err_val

        sigma = voigt_sigma_val
        gamma = voigt_gamma_val
        dgamma = voigt_gamma_err_val

        z = (v_val + 1j * gamma) / (sigma * np.sqrt(2.0))
        phi_val = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))

        if np.all(np.isclose(dgamma, 0.0, equal_nan=True)):
            phi_err_val = np.zeros_like(phi_val)
        else:
            gamma_m = np.maximum(gamma - dgamma, 1e-30)
            gamma_p = gamma + dgamma
            z_m = (v_val + 1j * gamma_m) / (sigma * np.sqrt(2.0))
            z_p = (v_val + 1j * gamma_p) / (sigma * np.sqrt(2.0))
            phi_m = np.real(wofz(z_m)) / (sigma * np.sqrt(2.0 * np.pi))
            phi_p = np.real(wofz(z_p)) / (sigma * np.sqrt(2.0 * np.pi))
            dphi_dL = (phi_p - phi_m) / ((gamma_p - gamma_m) * 2.0)
            phi_err_val = np.abs(dphi_dL) * (2.0 * dgamma)

        return phi_val, phi_err_val

    def _get_exomol_state_lookup(self):
        if self._states_g_lookup is None:
            states = self.molecule.load_exomol_states_dataframe()
            state_i = np.asarray(states["i"]).reshape(-1).astype(np.int64, copy=False)
            state_g = np.asarray(states["g"], dtype=np.float64).reshape(-1)

            if len(state_i) == 0:
                raise ValueError(f"ExoMol states file for {self.molecule.species} is empty")

            max_i = int(np.max(state_i))
            lookup = np.full(max_i + 1, np.nan, dtype=np.float64)
            lookup[state_i] = state_g
            self._states_g_lookup = lookup

        return self._states_g_lookup

    def _prepare_chunk_dict(self, df_chunk):
        """
        Convert one molecular chunk dataframe into the minimal plain-NumPy arrays
        needed by the broadening pipeline.
        """
        chunk = {
            "A_vals": np.asarray(df_chunk["A"], dtype=np.float64).reshape(-1),
            "nu_vals": np.asarray(df_chunk["nu_lines"], dtype=np.float64).reshape(-1),
            "elower_vals": np.asarray(df_chunk["elower"], dtype=np.float64).reshape(-1),
            "gup_vals": np.asarray(df_chunk["gup"], dtype=np.float64).reshape(-1),
            "glower_vals": np.asarray(df_chunk["glower"], dtype=np.float64).reshape(-1),
        }
        chunk["lam0_vals"] = 1.0e8 / chunk["nu_vals"]
        chunk["chunk_cache_id"] = "hitran_full_range"
        return chunk

    def _build_weight_cache_root(self):
        info = getattr(self.molecule, "cache_info", {}) or {}
        localdatabase = info.get("localdatabase")
        path = info.get("path")

        if localdatabase is not None:
            root = pathlib.Path(localdatabase)
            if path:
                parts = pathlib.Path(path).parts
                if len(parts) > 0:
                    root = root / parts[0]
            return root / "weight_cache"

        return pathlib.Path("Tables") / "weight_cache" / self.molecule.species

    def _format_cache_temperature_label(self, Temp_atm):
        T_val = Temp_atm.to_value(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")
        return f"{T_val:g}K"


    def _weight_cache_manifest_path(self, Temp_atm):
        temp_label = self._format_cache_temperature_label(Temp_atm)
        return self._weight_cache_root / f"weight_cache_{temp_label}.txt"

    def _update_weight_cache_manifest(self, chunk_cache_id, Temp_atm, cache_file):
        T_val = Temp_atm.to_value(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")
        manifest_path = self._weight_cache_manifest_path(Temp_atm)
        record = {
            "species": self.molecule.species,
            "temperature_K": float(T_val),
            "chunk_cache_id": str(chunk_cache_id),
            "cache_file": cache_file.name,
        }

        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _weight_cache_filename(self, chunk_cache_id, Temp_atm):
        temp_label = self._format_cache_temperature_label(Temp_atm)
        cache_key = f"{chunk_cache_id}|{temp_label}"
        cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:16]
        return self._weight_cache_root / f"weight_cache_{temp_label}_{cache_hash}.npy"

    def _load_cached_chunk_weights(self, chunk_cache_id, Temp_atm):
        cache_file = self._weight_cache_filename(chunk_cache_id, Temp_atm)
        if cache_file.exists():
            return np.load(cache_file)
        return None

    def _save_cached_chunk_weights(self, chunk_cache_id, Temp_atm, weights_chunk):
        cache_file = self._weight_cache_filename(chunk_cache_id, Temp_atm)
        is_new_file = not cache_file.exists()
        np.save(cache_file, np.asarray(weights_chunk, dtype=np.float64))
        if is_new_file:
            self._update_weight_cache_manifest(chunk_cache_id, Temp_atm, cache_file)

    def _load_hitran_dataframe(self):
        info = self.molecule.cache_info
        molecule_name = info["molecule_name"]
        isotope = info["isotope"]
        localdatabase = info["localdatabase"]
        path = info["path"]
        databank_name = info["databank_name"]
        cache = info["cache"]
        engine = info["engine"]
        output = info["output"]

        fetch_kwargs = dict(
            molecule=molecule_name,
            isotope=str(isotope),
            load_wavenum_min=float(self.molecule.wavenum_min.value),
            load_wavenum_max=float(self.molecule.wavenum_max.value),
            columns=None,
            cache=cache,
            engine=engine,
            output=output,
        )

        if localdatabase is not None:
            local_path = pathlib.Path(localdatabase)
            if path is not None:
                local_path = local_path / path
            local_path.mkdir(parents=True, exist_ok=True)
            fetch_kwargs["local_databases"] = str(local_path)
        if databank_name is not None:
            fetch_kwargs["databank_name"] = databank_name

        df = fetch_hitran(**fetch_kwargs)
        df = df.rename(columns={
            "A": "A",
            "wav": "nu_lines",
            "El": "elower",
            "gp": "gup",
            "gpp": "glower",
        })
        return df

    def iter_line_dataframes(self, verbose=False):
        source = getattr(self.molecule, "source", None)
        if source == "exomol":
            states_g_lookup = self._get_exomol_state_lookup()
            local_files = self.molecule.cache_info["local_trans_files"]
            for i, local_file in enumerate(local_files, start=1):
                t_file_start = time.perf_counter()
                if verbose:
                    print(
                        f"\n[{self.molecule.species}] Loading chunk {i}/{len(local_files)}"
                    )

                t_load_start = time.perf_counter()
                chunk = self.molecule.load_exomol_transition_chunk(local_file)
                chunk["chunk_cache_id"] = str(local_file)
                t_load = time.perf_counter() - t_load_start

                if len(chunk["A_vals"]) == 0:
                    if verbose:
                        t_file_total = time.perf_counter() - t_file_start
                        print(
                            f"[{self.molecule.species}] iter_line_dataframes: file {i}/{len(local_files)} empty after load, "
                            f"load = {t_load:.2f} s, total = {t_file_total:.2f} s"
                        )
                    continue

                t_map_start = time.perf_counter()
                i_lower_vals = chunk.pop("i_lower_vals")
                if len(i_lower_vals) == 0:
                    chunk["glower_vals"] = np.empty(0, dtype=np.float64)
                else:
                    max_idx = len(states_g_lookup) - 1
                    valid = (i_lower_vals >= 0) & (i_lower_vals <= max_idx)
                    glower_vals = np.full(len(i_lower_vals), np.nan, dtype=np.float64)
                    glower_vals[valid] = states_g_lookup[i_lower_vals[valid]]
                    chunk["glower_vals"] = glower_vals
                t_map = time.perf_counter() - t_map_start

                if verbose:
                    t_file_total = time.perf_counter() - t_file_start
                    print(
                        f"[{self.molecule.species}] Chunk {i}/{len(local_files)} ready "
                        f"(load: {t_load:.2f} s, map: {t_map:.2f} s, total: {t_file_total:.2f} s)"
                    )

                yield i, len(local_files), chunk
        elif source == "hitran":
            if verbose:
                print(f"[{self.molecule.species}] iter_line_dataframes: loading HITRAN dataframe")
            df = self._load_hitran_dataframe()
            chunk = self._prepare_chunk_dict(df)
            del df
            yield 1, 1, chunk
        else:
            raise ValueError(f"Unsupported molecule source for BroadeningProfileMolecule: {source}")

    def _compute_chunk_weights(self, chunk, weights, Temp_atm, partition_Z=None):
        if (weights is not None) and (Temp_atm is not None):
            raise ValueError("Give either weights or Temp_atm, not both")

        if Temp_atm is not None:
            T = Temp_atm.to_value(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")
            E_l_eV = chunk["elower_vals"] * self.invcm_to_eV
            boltz = chunk["glower_vals"] * np.exp(-E_l_eV / (self.kb_eV_per_K * T))
            return boltz / partition_Z

        if weights is None:
            return np.ones(len(chunk["A_vals"]), dtype=np.float64)

        return np.asarray(weights, dtype=np.float64).reshape(-1)

    def _compute_partition_function(self, Temp_atm, verbose=False):
        T = Temp_atm.to_value(u.K) if isinstance(Temp_atm, u.Quantity) else _not_quantity("Temp_atm")
        source = getattr(self.molecule, "source", None)

        if source == "exomol":
            if verbose:
                print(f"[{self.molecule.species}] partition function: using ExoMol states file directly")

            states = self.molecule.load_exomol_states_dataframe()
            if len(states) == 0:
                raise ValueError("ExoMol states file is empty; cannot build partition function")

            El_states_eV = np.asarray(states["E"], dtype=np.float64).reshape(-1) * self.invcm_to_eV
            g_states = np.asarray(states["g"], dtype=np.float64).reshape(-1)

            boltz_states = g_states * np.exp(-El_states_eV / (self.kb_eV_per_K * T))
            Z = np.nansum(boltz_states)
            if not np.isfinite(Z) or Z == 0:
                raise ValueError("Partition function is zero or non-finite for the requested temperature")
            return float(Z)

        if source == "hitran":
            if verbose:
                print(f"[{self.molecule.species}] partition function: using HITRAN TIPS data")

            info = getattr(self.molecule, "cache_info", {}) or {}
            molecule_name = info.get("molecule_name", self.molecule.species)
            isotope = int(info.get("isotope", 1))

            try:
                from radis.levels.partfunc import PartFuncTIPS

                partfunc = PartFuncTIPS(molecule_name, isotope, verbose=False)
                Z = float(partfunc.at(T))
                if not np.isfinite(Z) or Z <= 0.0:
                    raise ValueError("Partition function is zero or non-finite for the requested temperature")
                return Z
            except Exception as exc:
                if verbose:
                    print(
                        f"[{self.molecule.species}] HITRAN TIPS partition function unavailable "
                        f"({type(exc).__name__}: {exc}); falling back to scanned states"
                    )

                unique_states = set()
                for _, _, chunk in self.iter_line_dataframes(verbose=verbose):
                    lower_states = np.column_stack((chunk["elower_vals"], chunk["glower_vals"]))
                    upper_states = np.column_stack((
                        chunk["elower_vals"] + chunk["nu_vals"],
                        chunk["gup_vals"],
                    ))
                    states = np.vstack((lower_states, upper_states))
                    finite_states = states[np.all(np.isfinite(states), axis=1)]
                    for energy, degeneracy in np.unique(finite_states, axis=0):
                        unique_states.add((float(energy), float(degeneracy)))

                if len(unique_states) == 0:
                    raise ValueError("No finite HITRAN state information found while building partition function")

                states_arr = np.array(list(unique_states), dtype=np.float64)
                E_unique_eV = states_arr[:, 0] * self.invcm_to_eV
                g_unique = states_arr[:, 1]

                boltz_unique = g_unique * np.exp(-E_unique_eV / (self.kb_eV_per_K * T))
                Z = np.nansum(boltz_unique)
                if not np.isfinite(Z) or Z <= 0.0:
                    raise ValueError("Partition function is zero or non-finite for the requested temperature")
                return float(Z)

        raise ValueError(f"Unsupported molecule source for partition function: {source}")

    def build_total_crossection(self, weights=None, Temp_atm=None, verbose=False, progress_every=1000000, line_batch_size=2048):
        if (weights is not None) and (Temp_atm is not None):
            raise ValueError("Give either weights or Temp_atm, not both")

        cache_key = None
        if Temp_atm is not None:
            cache_key = self._temperature_cache_key(Temp_atm)

        t_start = time.perf_counter()
        sigma_total_val = np.zeros(self.lam_grid_val.shape, dtype=np.float64)
        sigma_total_err2_val = np.zeros(self.lam_grid_val.shape, dtype=np.float64)

        partition_Z = None
        if Temp_atm is not None:
            if cache_key not in self._announced_weight_build_temps:
                print(f"[{self.molecule.species}] starting calculating weights for T={Temp_atm.to_value(u.K):.0f}K")
                self._announced_weight_build_temps.add(cache_key)

            if cache_key in self.partition_function_cache:
                partition_Z = self.partition_function_cache[cache_key]
            else:
                partition_Z = self._compute_partition_function(Temp_atm, verbose=verbose)
                self.partition_function_cache[cache_key] = partition_Z

        line_offset = 0
        processed_lines = 0
        last_progress_bucket = 0

        for chunk_index, n_chunks, chunk in self.iter_line_dataframes(verbose=verbose):
            t_chunk_start = time.perf_counter()
            n_chunk = len(chunk["A_vals"])
            if n_chunk == 0:
                continue

            chunk_cache_id = chunk.get("chunk_cache_id")

            elapsed = time.perf_counter() - t_start
            if verbose:
                print(
                    f"[{self.molecule.species}] Chunk {chunk_index}/{n_chunks}: "
                    f"{n_chunk:,} lines loaded after {elapsed:.2f} s"
                )
            if verbose and Temp_atm is not None and chunk_cache_id is not None and chunk_index == 1:
                cache_file = self._weight_cache_filename(chunk_cache_id, Temp_atm)
                manifest_path = self._weight_cache_manifest_path(Temp_atm)
                cache_state = "found" if cache_file.exists() else "building"
                print(
                    f"[{self.molecule.species}] Weight cache at this temperature: {cache_state} "
                    f"({cache_file}) | manifest: {manifest_path}"
                )

            t_weights_start = time.perf_counter()
            if Temp_atm is not None:
                chunk_cache_id = chunk.get("chunk_cache_id")
                weights_chunk = None if chunk_cache_id is None else self._load_cached_chunk_weights(chunk_cache_id, Temp_atm)
                if weights_chunk is not None:
                    weights_chunk = np.asarray(weights_chunk, dtype=np.float64).reshape(-1)
                    if weights_chunk.shape[0] != n_chunk:
                        if verbose:
                            print(
                                f"[{self.molecule.species}] Ignoring stale weight cache for chunk {chunk_cache_id}: "
                                f"cached length {weights_chunk.shape[0]} != current length {n_chunk}"
                            )
                        weights_chunk = None
                if weights_chunk is None:
                    weights_chunk = self._compute_chunk_weights(chunk, None, Temp_atm, partition_Z=partition_Z)
                    if chunk_cache_id is not None:
                        self._save_cached_chunk_weights(chunk_cache_id, Temp_atm, weights_chunk)
            elif weights is None:
                weights_chunk = np.ones(n_chunk, dtype=np.float64)
            else:
                weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
                weights_chunk = weights_arr[line_offset:line_offset + n_chunk]
                if len(weights_chunk) != n_chunk:
                    raise ValueError("weights must have one value per molecular line across all chunks")
            t_weights = time.perf_counter() - t_weights_start

            t_profile_prep_start = time.perf_counter()
            A_vals = chunk["A_vals"]
            lam0_val = chunk["lam0_vals"]
            gup_vals = chunk["gup_vals"]
            glower_vals = chunk["glower_vals"]

            sig0_val, sig0_err_val = self.calc_central_crossection(
                A_vals,
                lam0_val,
                gup_vals,
                glower_vals,
            )

            lorentz_val, lorentz_err_val = self.FWHM_lorentz(A_vals, lam0_val)
            gauss_val = self.FWHM_gauss(lam0_val)
            vlim_val = self.set_vlim(gauss_val, lorentz_val)
            voigt_sigma_val = gauss_val / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            voigt_gamma_val = 0.5 * lorentz_val
            voigt_gamma_err_val = 0.5 * lorentz_err_val
            t_profile_prep = time.perf_counter() - t_profile_prep_start

            t_filter_start = time.perf_counter()
            weighted_strength = np.abs(weights_chunk * sig0_val)
            max_weighted_strength = np.nanmax(weighted_strength) if np.any(np.isfinite(weighted_strength)) else 0.0
            strength_cutoff = self.temp_strength_rel_cutoff * max_weighted_strength if max_weighted_strength > 0.0 else 0.0

            halfwidth_val = lam0_val * vlim_val / self.c_kms
            lam_lo_val = lam0_val - halfwidth_val
            lam_hi_val = lam0_val + halfwidth_val
            i0_arr = np.searchsorted(self.lam_grid_val, lam_lo_val, side='left')
            i1_arr = np.searchsorted(self.lam_grid_val, lam_hi_val, side='right')

            active_mask = np.isfinite(weights_chunk)
            active_mask &= (weights_chunk != 0.0)
            active_mask &= (i1_arr > i0_arr)
            if strength_cutoff > 0.0:
                active_mask &= (weighted_strength >= strength_cutoff)

            active_idx = np.where(active_mask)[0]
            n_active = len(active_idx)
            t_filter = time.perf_counter() - t_filter_start
            if verbose:
                print(
                    f"[{self.molecule.species}] Chunk {chunk_index}/{n_chunks}: "
                    f"{n_active:,}/{n_chunk:,} active lines"
                )

            t_accum_start = time.perf_counter()
            for b0 in range(0, n_active, line_batch_size):
                b1 = min(b0 + line_batch_size, n_active)
                batch_idx = active_idx[b0:b1]
                batch_size = len(batch_idx)
                if batch_size == 0:
                    continue

                i0_batch = i0_arr[batch_idx]
                i1_batch = i1_arr[batch_idx]
                widths = i1_batch - i0_batch
                max_width = int(np.max(widths))
                offsets = np.arange(max_width, dtype=np.int64)[None, :]
                grid_idx = i0_batch[:, None] + offsets
                valid_mask = offsets < widths[:, None]

                safe_grid_idx = np.where(valid_mask, grid_idx, 0)
                lam_local_val = self.lam_grid_val[safe_grid_idx]

                lam0_batch = lam0_val[batch_idx][:, None]
                v_local_val = self.c_kms * ((lam_local_val - lam0_batch) / lam0_batch)

                phi_val, phi_err_val = self.profile_from_velocity_batch(
                    v_local_val,
                    lorentz_val[batch_idx][:, None],
                    lorentz_err_val[batch_idx][:, None],
                    voigt_sigma_val[batch_idx][:, None],
                    voigt_gamma_val[batch_idx][:, None],
                    voigt_gamma_err_val[batch_idx][:, None],
                )

                sigma_line_val = sig0_val[batch_idx][:, None] * phi_val
                sigma_line_err_val = (
                    sig0_val[batch_idx][:, None] * phi_err_val
                    + sig0_err_val[batch_idx][:, None] * phi_val
                )

                weighted_sigma_val = weights_chunk[batch_idx][:, None] * sigma_line_val
                weighted_sigma_err2_val = (weights_chunk[batch_idx][:, None] * sigma_line_err_val) ** 2

                flat_idx = safe_grid_idx[valid_mask]
                flat_sigma = weighted_sigma_val[valid_mask]
                flat_sigma_err2 = weighted_sigma_err2_val[valid_mask]

                sigma_total_val += np.bincount(flat_idx, weights=flat_sigma, minlength=sigma_total_val.size)
                sigma_total_err2_val += np.bincount(flat_idx, weights=flat_sigma_err2, minlength=sigma_total_err2_val.size)

                processed_lines += batch_size
                if verbose:
                    current_bucket = processed_lines // progress_every
                    if current_bucket > last_progress_bucket:
                        last_progress_bucket = current_bucket
                        elapsed = time.perf_counter() - t_start
                        print(
                            f"[{self.molecule.species}] Progress: {processed_lines:,} active lines processed "
                            f"in {elapsed:.2f} s"
                        )

            t_accum = time.perf_counter() - t_accum_start
            if verbose:
                t_chunk_total = time.perf_counter() - t_chunk_start
                print(
                    f"[{self.molecule.species}] Chunk {chunk_index}/{n_chunks} finished in {t_chunk_total:.2f} s "
                    f"(weights: {t_weights:.2f} s, prep: {t_profile_prep:.2f} s, "
                    f"filter: {t_filter:.2f} s, accumulate: {t_accum:.2f} s)\n"
                )
            line_offset += n_chunk

        if Temp_atm is not None and cache_key not in self._announced_weight_ready_temps:
            print(f"[{self.molecule.species}] Finished caching data for T={Temp_atm.to_value(u.K):.0f}K")
            self._announced_weight_ready_temps.add(cache_key)

        sigma_total = sigma_total_val * u.cm**2
        sigma_total_err = np.sqrt(sigma_total_err2_val) * u.cm**2

        if verbose:
            total_time = time.perf_counter() - t_start
            print(f"[{self.molecule.species}] Total cross-section finished in {total_time:.2f} s")

        return sigma_total, sigma_total_err

    def apply_boltzmann_weights(self, Temp_atm, verbose=False):
        temp_key = self._temperature_cache_key(Temp_atm)

        if temp_key in self._sigma_cache_by_temp:
            sigma_cached, sigma_err_cached = self._sigma_cache_by_temp[temp_key]
            self.sigmaArray = sigma_cached
            self.sigmaArray_err = sigma_err_cached
            self.sigma_total = self.sigmaArray
            self.sigma_total_err = self.sigmaArray_err
            return self.sigmaArray, self.sigmaArray_err

        self.sigmaArray, self.sigmaArray_err = self.build_total_crossection(
            Temp_atm=Temp_atm,
            verbose=verbose,
        )
        self._sigma_cache_by_temp[temp_key] = (self.sigmaArray, self.sigmaArray_err)
        self.sigma_total = self.sigmaArray
        self.sigma_total_err = self.sigmaArray_err
        return self.sigmaArray, self.sigmaArray_err

    def clear_temperature_cache(self, keep_current: bool = False):
        self._sigma_cache_by_temp.clear()
        self.partition_function_cache.clear()
        if not keep_current:
            self.sigmaArray = None
            self.sigmaArray_err = None
            self.sigma_total = None
            self.sigma_total_err = None

    def plot_total_crossection(self, xscale: str = 'linear', yscale: str = 'log', xlim: tuple = None):
        lam = self.lam_grid.to_value(u.AA)
        sig = self.sigmaArray.to_value(u.cm**2)
        sig_err = self.sigmaArray_err.to_value(u.cm**2)

        plt.figure(figsize=(10, 4))
        plt.plot(lam, sig)
        plt.fill_between(lam, np.maximum(sig - sig_err, 0.0), sig + sig_err, alpha=0.3, color='red', label='error')
        plt.xlabel(f"Wavelength [{self.lam_grid.unit}]")
        plt.ylabel(r"Total cross-section $\sigma_{\rm tot}$ [cm$^2$]")
        plt.title(f"{self.molecule.species} total molecular cross-section")
        if xlim is not None:
            plt.xlim(xlim)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.legend()
        plt.tight_layout()
        plt.show()
