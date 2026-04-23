import pathlib
import sys

import numpy as np
import astropy.units as u
from astropy import constants as const
from matplotlib import pyplot as plt
from scipy.integrate import trapezoid
from scipy.special import wofz

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from project_func.Templates.Stars.stars_templates import STAR_TEMPLATES, infer_teff_from_star_template
from project_classes.BroadeningProfileMolecule import BroadeningProfileMolecule

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
ATOM_SPECIES = "Na I"
STAR_KEY = "A5"
B_VALUE = 1.0 * u.km / u.s
TEMP_ATM = 5000.0 * u.K
DISTANCE = 1.0 * u.AU
WAVEMIN = 150 * u.AA
WAVEMAX = 50000 * u.AA
NPTS_ATOM = 300
NCOL_VALUES = np.logspace(6, 29, 15) / u.cm**2
MOLECULE_GRID_REFINEMENT = 10.0

MAKE_PLOT = True
SAVE_PLOT = True
OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "Plots" / "Tests"
OUTPUT_NAME = "atom_vs_molecule_pipeline_test.pdf"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_star(star_key: str) -> Star:
    params = STAR_TEMPLATES[star_key]
    return Star(
        params["path"],
        params["radius"],
        params["mass"],
        vsini=params["vsini"],
        epsilon=params["epsilon"],
    )


def relative_difference(a, b):
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


def molecule_style_dlam(lam_min: u.Quantity, lam_max: u.Quantity, b_value: u.Quantity) -> u.Quantity:
    rep = 0.5 * (lam_min + lam_max)
    doppler_sigma = (rep * (b_value / const.c)).to(u.AA)
    dlam_auto = (doppler_sigma / MOLECULE_GRID_REFINEMENT).to(u.AA)
    floor = 1e-5 * u.AA
    return np.maximum(dlam_auto, floor)


def molecule_style_wavelength_grid(lam_min: u.Quantity, lam_max: u.Quantity, b_value: u.Quantity):
    dlam = molecule_style_dlam(lam_min, lam_max, b_value)
    npts = int(np.floor(((lam_max - lam_min) / dlam).decompose().value)) + 1
    lam_grid = (lam_min + np.arange(npts) * dlam).to(u.AA)
    return lam_grid, dlam

def molecule_style_partition_function_from_atom(atom_obj, temp_atm: u.Quantity) -> float:
    T = temp_atm.to_value(u.K)
    kb_eV_per_K = const.k_B.to_value(u.eV / u.K)
    E_l = np.asarray(atom_obj.E_l.to_value(u.eV), dtype=float).reshape(-1)
    g_l = np.asarray(atom_obj.g_l, dtype=float).reshape(-1)

    unique_states = np.unique(np.column_stack((E_l, g_l)), axis=0)
    E_unique = unique_states[:, 0]
    g_unique = unique_states[:, 1]

    boltz = g_unique * np.exp(-E_unique / (kb_eV_per_K * T))
    Z = np.nansum(boltz)
    if not np.isfinite(Z) or Z <= 0.0:
        raise ValueError("Molecule-style atomic partition function is zero or non-finite.")
    return float(Z)


def molecule_style_atomic_weights(atom_obj, temp_atm: u.Quantity):
    T = temp_atm.to_value(u.K)
    kb_eV_per_K = const.k_B.to_value(u.eV / u.K)
    E_l = np.asarray(atom_obj.E_l.to_value(u.eV), dtype=float).reshape(-1)
    g_l = np.asarray(atom_obj.g_l, dtype=float).reshape(-1)
    Z = molecule_style_partition_function_from_atom(atom_obj, temp_atm)

    weights = g_l * np.exp(-E_l / (kb_eV_per_K * T)) / Z
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return weights



class FixedSigmaMoleculeLikeProfile:
    def __init__(self, atom_profile: BroadeningProfile, lam_grid, sigma_total, sigma_total_err):
        self.molecule = atom_profile.molecule
        self.b = atom_profile.b
        self.profileType = atom_profile.profileType
        self.lam_grid = lam_grid
        self.lam_grid_val = lam_grid.to_value(u.AA)
        self.sigmaArray = sigma_total
        self.sigmaArray_err = sigma_total_err
        self.sigma_total = sigma_total
        self.sigma_total_err = sigma_total_err
        self._sigma_cache_by_temp = {}

    def apply_boltzmann_weights(self, Temp_atm, verbose=False):
        temp_key = float(np.asarray(Temp_atm.to_value(u.K)).reshape(-1)[0])
        self._sigma_cache_by_temp[temp_key] = (self.sigmaArray, self.sigmaArray_err)
        return self.sigmaArray, self.sigmaArray_err


# --------------------------------------------------------------------------
# Fake-atom molecule adapter classes for real BroadeningProfileMolecule usage
# --------------------------------------------------------------------------

class FakeAtomMoleculeAdapter:
    """
    Minimal adapter that makes an Atom look like a Molecule cache/setup object,
    so the real BroadeningProfileMolecule class can be exercised without touching
    production code.
    """
    def __init__(self, atom_obj, lam_min, lam_max):
        self.atom_obj = atom_obj
        self.species = atom_obj.species + " [fake-molecule]"
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.wavenum_min = (1 / lam_max).to(u.cm**-1)
        self.wavenum_max = (1 / lam_min).to(u.cm**-1)
        self.mass = atom_obj.mass
        self.cache_ready = True
        self.source = "atom_fake"
        self.cache_info = {"source": "atom_fake"}

        self.A_ul = atom_obj.A_ul
        self.lam0 = atom_obj.lam0
        self.g_u = atom_obj.g_u
        self.g_l = atom_obj.g_l
        self.E_l = atom_obj.E_l


class FakeAtomBroadeningProfileMolecule(BroadeningProfileMolecule):
    """
    Use the real BroadeningProfileMolecule machinery, but feed it one synthetic
    atom chunk so atoms can be processed as if they were molecules.
    """
    def __init__(self, atom_profile: BroadeningProfile, b, lam_min=None, lam_max=None, dlam=None, profileType: str = 'Voigt', verbose=False):
        self.atom_profile = atom_profile
        fake_molecule = FakeAtomMoleculeAdapter(
            atom_profile.molecule,
            atom_profile.molecule.lam_min,
            atom_profile.molecule.lam_max,
        )
        super().__init__(
            molecule=fake_molecule,
            b=b,
            lam_min=lam_min,
            lam_max=lam_max,
            dlam=dlam,
            profileType=profileType,
            verbose=verbose,
        )
        self.temp_strength_rel_cutoff = 0.0

    def iter_line_dataframes(self, verbose=False):
        atom_obj = self.atom_profile.molecule
        lam0_vals = np.asarray(atom_obj.lam0.to_value(u.AA), dtype=np.float64).reshape(-1)
        nu_vals = 1.0e8 / lam0_vals

        chunk = {
            "A_vals": np.asarray(atom_obj.A_ul.to_value(1 / u.s), dtype=np.float64).reshape(-1),
            "nu_vals": nu_vals,
            "elower_vals": np.asarray(
                atom_obj.E_l.to_value(1 / u.cm, equivalencies=u.spectral()),
                dtype=np.float64,
            ).reshape(-1),
            "gup_vals": np.asarray(atom_obj.g_u, dtype=np.float64).reshape(-1),
            "glower_vals": np.asarray(atom_obj.g_l, dtype=np.float64).reshape(-1),
            "lam0_vals": lam0_vals,
            "chunk_cache_id": f"fake_atom::{atom_obj.species}",
        }
        yield 1, 1, chunk

    def _compute_partition_function(self, Temp_atm, verbose=False):
        return molecule_style_partition_function_from_atom(self.atom_profile.molecule, Temp_atm)


def molecule_style_central_cross_section(A_vals, lam0_val, gup_vals, glower_vals):
    sig0_val = A_vals * ((lam0_val ** 3) / (8.0 * np.pi)) * (gup_vals / glower_vals)
    sig0_val *= 1.0e-29
    sig0_err_val = np.zeros_like(sig0_val)
    return sig0_val, sig0_err_val


def molecule_style_fwhm_lorentz(A_vals, lam0_val):
    lorentz_val = lam0_val * (A_vals / (2.0 * np.pi)) * 1.0e-13
    lorentz_err_val = np.zeros_like(lorentz_val)
    return lorentz_val, lorentz_err_val


def molecule_style_fwhm_gauss(b_value, lam0_val):
    gauss_scalar = b_value.to_value(u.km / u.s) * (2.0 * np.sqrt(np.log(2.0)))
    return np.full_like(lam0_val, gauss_scalar)


def molecule_style_vlim(gauss_val, lorentz_val):
    return np.maximum(6.0 * gauss_val, 25.0 * lorentz_val)


def molecule_style_profile_from_velocity_batch(profile_type, b_value, v_val, lorentz_val, lorentz_err_val, voigt_sigma_val, voigt_gamma_val, voigt_gamma_err_val):
    profile_type = str(profile_type).lower()

    if profile_type == 'lorentz':
        L = lorentz_val
        dL = lorentz_err_val
        phi_val = (1.0 / np.pi) * (0.5 * L) / (v_val**2 + (0.5 * L)**2)
        phi_err_val = np.abs((1.0 / np.pi) * 0.5 * (v_val**2 - L**2 / 4.0) / (v_val**2 + L**2 / 4.0)**2) * dL
        return phi_val, phi_err_val

    if profile_type == 'gauss':
        b_val = b_value.to_value(u.km / u.s)
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


def build_atom_total_sigma_on_common_grid(atom_profile: BroadeningProfile, star: Star, temp_atm: u.Quantity):
    """
    Build an atom on a molecule-style global wavelength grid using the same
    per-line accumulation philosophy as BroadeningProfileMolecule.build_total_crossection.
    """
    atom_obj = atom_profile.molecule
    weights = molecule_style_atomic_weights(atom_obj, temp_atm)

    lam_grid, dlam = molecule_style_wavelength_grid(WAVEMIN, WAVEMAX, B_VALUE)
    lam_grid_val = lam_grid.to_value(u.AA)

    A_vals = np.asarray(atom_obj.A_ul.to_value(1 / u.s), dtype=float).reshape(-1)
    lam0_val = np.asarray(atom_obj.lam0.to_value(u.AA), dtype=float).reshape(-1)
    gup_vals = np.asarray(atom_obj.g_u, dtype=float).reshape(-1)
    glower_vals = np.asarray(atom_obj.g_l, dtype=float).reshape(-1)

    sig0_val, sig0_err_val = molecule_style_central_cross_section(A_vals, lam0_val, gup_vals, glower_vals)
    lorentz_val, lorentz_err_val = molecule_style_fwhm_lorentz(A_vals, lam0_val)
    gauss_val = molecule_style_fwhm_gauss(B_VALUE, lam0_val)
    vlim_val = molecule_style_vlim(gauss_val, lorentz_val)
    voigt_sigma_val = gauss_val / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    voigt_gamma_val = 0.5 * lorentz_val
    voigt_gamma_err_val = 0.5 * lorentz_err_val

    halfwidth_val = lam0_val * vlim_val / const.c.to_value(u.km / u.s)
    lam_lo_val = lam0_val - halfwidth_val
    lam_hi_val = lam0_val + halfwidth_val
    i0_arr = np.searchsorted(lam_grid_val, lam_lo_val, side='left')
    i1_arr = np.searchsorted(lam_grid_val, lam_hi_val, side='right')

    active_mask = np.isfinite(weights)
    active_mask &= (weights > 0.0)
    active_mask &= np.isfinite(sig0_val)
    active_mask &= np.isfinite(lam0_val)
    active_mask &= np.isfinite(gup_vals)
    active_mask &= np.isfinite(glower_vals)
    active_mask &= (glower_vals > 0.0)
    active_mask &= (i1_arr > i0_arr)

    active_idx = np.where(active_mask)[0]
    if len(active_idx) == 0:
        raise ValueError("No active atomic lines survived for molecule-style accumulation.")

    sigma_total_val = np.zeros_like(lam_grid_val, dtype=np.float64)
    sigma_total_err2_val = np.zeros_like(lam_grid_val, dtype=np.float64)

    line_batch_size = 2048
    c_kms = const.c.to_value(u.km / u.s)

    for b0 in range(0, len(active_idx), line_batch_size):
        b1 = min(b0 + line_batch_size, len(active_idx))
        batch_idx = active_idx[b0:b1]
        if len(batch_idx) == 0:
            continue

        i0_batch = i0_arr[batch_idx]
        i1_batch = i1_arr[batch_idx]
        widths = i1_batch - i0_batch
        max_width = int(np.max(widths))
        offsets = np.arange(max_width, dtype=np.int64)[None, :]
        grid_idx = i0_batch[:, None] + offsets
        valid_mask = offsets < widths[:, None]

        safe_grid_idx = np.where(valid_mask, grid_idx, 0)
        lam_local_val = lam_grid_val[safe_grid_idx]

        lam0_batch = lam0_val[batch_idx][:, None]
        v_local_val = c_kms * ((lam_local_val - lam0_batch) / lam0_batch)

        phi_val, phi_err_val = molecule_style_profile_from_velocity_batch(
            atom_profile.profileType,
            B_VALUE,
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

        weighted_sigma_val = weights[batch_idx][:, None] * sigma_line_val
        weighted_sigma_err2_val = (weights[batch_idx][:, None] * sigma_line_err_val) ** 2

        flat_idx = safe_grid_idx[valid_mask]
        flat_sigma = weighted_sigma_val[valid_mask]
        flat_sigma_err2 = weighted_sigma_err2_val[valid_mask]

        sigma_total_val += np.bincount(flat_idx, weights=flat_sigma, minlength=sigma_total_val.size)
        sigma_total_err2_val += np.bincount(flat_idx, weights=flat_sigma_err2, minlength=sigma_total_err2_val.size)

    sigma_total_val = np.nan_to_num(sigma_total_val, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_total_err2_val = np.nan_to_num(sigma_total_err2_val, nan=0.0, posinf=0.0, neginf=0.0)

    sigma_total = sigma_total_val * u.cm**2
    sigma_total_err = np.sqrt(sigma_total_err2_val) * u.cm**2
    return lam_grid, dlam, sigma_total, sigma_total_err, len(active_idx)


def calc_photon_pressure_atom_stitched(atom_profile: BroadeningProfile, star: Star, column_density, temp_atm: u.Quantity, distance: u.Quantity):
    """
    Compute photon pressure for the atom using a molecule-style stitched spectrum
    on one common wavelength grid.
    """
    N_col = column_density.to(u.cm**-2)
    N_col = np.atleast_1d(N_col)

    lam_grid, dlam, sigma_total, sigma_total_err, used_lines = build_atom_total_sigma_on_common_grid(atom_profile, star, temp_atm)

    if not np.any(np.isfinite(sigma_total.to_value(u.cm**2)) & (sigma_total.to_value(u.cm**2) > 0.0)):
        raise ValueError("Stitched sigma_total contains no positive finite values.")

    omega = (star.radius / distance) ** 2
    lam_star = star.lam_star.to_value(u.AA)
    flux_star = (star.flux_star_rot * omega).to(u.erg / u.s / u.cm**2 / u.AA)
    flux_interp = np.interp(lam_grid.to_value(u.AA), lam_star, flux_star.to_value(flux_star.unit)) * flux_star.unit
    flux_interp = np.nan_to_num(flux_interp.to_value(flux_star.unit), nan=0.0, posinf=0.0, neginf=0.0) * flux_star.unit

    tau = np.nan_to_num((sigma_total[:, None] * N_col[None, :]).decompose().value, nan=np.inf, posinf=np.inf, neginf=0.0)
    trans = np.exp(-tau)

    flux_val = np.asarray(flux_interp.to_value(u.erg / u.s / u.cm**2 / u.AA), dtype=float)
    sigma_val = np.asarray(sigma_total.to_value(u.cm**2), dtype=float)
    sigma_err_val = np.asarray(sigma_total_err.to_value(u.cm**2), dtype=float)
    lam_val = np.asarray(lam_grid.to_value(u.AA), dtype=float)
    ncol_val = np.asarray(N_col.to_value(1 / u.cm**2), dtype=float)

    integrand = np.nan_to_num(flux_val[:, None] * sigma_val[:, None] * trans, nan=0.0, posinf=0.0, neginf=0.0)
    force_total_val = trapezoid(integrand, lam_val[:, None], axis=0) / const.c.to_value(u.cm / u.s)
    force_total_val = np.nan_to_num(force_total_val, nan=0.0, posinf=0.0, neginf=0.0)
    force_total = (force_total_val * u.dyne).to(u.N)

    factor = 1.0 - (sigma_val[:, None] * ncol_val[None, :])
    err_integrand = np.nan_to_num(
        (flux_val[:, None] * trans * factor * sigma_err_val[:, None]) / const.c.to_value(u.cm / u.s),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    dF_dA = trapezoid(
        err_integrand,
        lam_val[:, None],
        axis=0,
    )
    dF_dA = np.nan_to_num(dF_dA, nan=0.0, posinf=0.0, neginf=0.0)
    force_total_err = (np.abs(dF_dA) * u.dyne).to(u.N)

    return force_total, force_total_err, lam_grid, dlam, sigma_total, used_lines


def calc_photon_pressure_atom_molecule_branch(atom_profile: BroadeningProfile, star: Star, column_density, temp_atm: u.Quantity, distance: u.Quantity):
    lam_grid, dlam, sigma_total, sigma_total_err, used_lines = build_atom_total_sigma_on_common_grid(atom_profile, star, temp_atm)
    stitched_profile = FixedSigmaMoleculeLikeProfile(atom_profile, lam_grid, sigma_total, sigma_total_err)
    stitched_profile.apply_boltzmann_weights(temp_atm, verbose=False)

    pp_molecule = PhotonPressure(stitched_profile, star)
    force_total, force_total_err, _, _ = pp_molecule.calc_PhotonPressure_molecule(column_density, temp_atm, distance)
    return force_total, force_total_err, lam_grid, dlam, sigma_total, used_lines


# --------------------------------------------------------------------------
# Real BroadeningProfileMolecule with fake-atom class for direct molecule pipeline test
# --------------------------------------------------------------------------

def calc_photon_pressure_atom_real_fakeclass(atom_profile: BroadeningProfile, star: Star, column_density, temp_atm: u.Quantity, distance: u.Quantity):
    fake_dlam = molecule_style_dlam(WAVEMIN, WAVEMAX, B_VALUE)
    fake_profile = FakeAtomBroadeningProfileMolecule(
        atom_profile=atom_profile,
        b=atom_profile.b,
        lam_min=WAVEMIN,
        lam_max=WAVEMAX,
        dlam=fake_dlam,
        profileType=atom_profile.profileType,
        verbose=False,
    )
    fake_profile.temp_strength_rel_cutoff = 0.0
    fake_profile.apply_boltzmann_weights(temp_atm, verbose=False)

    pp_fake = PhotonPressure(fake_profile, star)
    force_total, force_total_err, _, _ = pp_fake.calc_PhotonPressure_molecule(column_density, temp_atm, distance)
    return force_total, force_total_err, fake_profile

# -----------------------------------------------------------------------------
# Main comparison
# -----------------------------------------------------------------------------
def main() -> None:
    star = make_star(STAR_KEY)
    atom = Atom(ATOM_SPECIES, WAVEMIN, WAVEMAX)
    atom_profile = BroadeningProfile(atom, B_VALUE, NPTS_ATOM, "Voigt")

    pp_atom = PhotonPressure(atom_profile, star)
    print(f"Atom profile sigmaArray_sym peak = {np.nanmax(atom_profile.sigmaArray_sym.to_value(u.cm**2)):.6e} cm^2")
    atom_weights = molecule_style_atomic_weights(atom_profile.molecule, TEMP_ATM)
    print(f"Molecule-style atomic weights: min={np.nanmin(atom_weights):.6e}, max={np.nanmax(atom_weights):.6e}, positive_count={np.count_nonzero(atom_weights > 0.0)}")
    F_atom, F_atom_err, _, _ = pp_atom.calc_PhotonPressure(NCOL_VALUES, TEMP_ATM, DISTANCE)
    beta_atom, beta_atom_err = pp_atom.beta_Values(F_atom, F_atom_err, star.mass, DISTANCE.to(u.cm))

    F_stitched_manual, F_stitched_manual_err, lam_grid, dlam, sigma_total, used_lines = calc_photon_pressure_atom_stitched(
        atom_profile,
        star,
        NCOL_VALUES,
        TEMP_ATM,
        DISTANCE,
    )

    try:
        F_stitched, F_stitched_err, fake_profile = calc_photon_pressure_atom_real_fakeclass(
            atom_profile,
            star,
            NCOL_VALUES,
            TEMP_ATM,
            DISTANCE,
        )
        lam_grid = fake_profile.lam_grid
        dlam = fake_profile.dlam
        sigma_total = fake_profile.sigmaArray
        used_lines = len(np.asarray(atom_profile.molecule.lam0).reshape(-1))
        stitched_source_label = "real BroadeningProfileMolecule fake-atom class"
    except Exception as exc:
        print(f"Falling back to manual molecule-style force integration because fake BroadeningProfileMolecule failed: {type(exc).__name__}: {exc}")
        try:
            F_stitched, F_stitched_err, _, _, _, _ = calc_photon_pressure_atom_molecule_branch(
                atom_profile,
                star,
                NCOL_VALUES,
                TEMP_ATM,
                DISTANCE,
            )
            stitched_source_label = "actual molecule photon-pressure branch from stitched adapter"
        except Exception as exc2:
            print(f"Falling back to manual molecule-style force integration because calc_PhotonPressure_molecule failed: {type(exc2).__name__}: {exc2}")
            F_stitched = F_stitched_manual
            F_stitched_err = F_stitched_manual_err
            stitched_source_label = "manual molecule-style force integration"
    beta_stitched, beta_stitched_err = pp_atom.beta_Values(F_stitched, F_stitched_err, star.mass, DISTANCE.to(u.cm))

    beta_atom_val = np.ravel(beta_atom.value)
    beta_stitched_val = np.ravel(beta_stitched.value)
    print(f"Atom finite beta count: {np.isfinite(beta_atom_val).sum()}")
    print(f"Stitched finite beta count: {np.isfinite(beta_stitched_val).sum()}")
    print(f"Atom finite force count: {np.isfinite(np.ravel(F_atom.value)).sum()}")
    print(f"Stitched finite force count: {np.isfinite(np.ravel(F_stitched.value)).sum()}")
    print(f"Peak stitched force = {np.nanmax(np.ravel(F_stitched.to_value(u.N))):.6e} N")
    print(f"Stitched force source = {stitched_source_label}")

    print("=" * 80)
    print(f"Atom pipeline vs molecule-pipeline fake-atom test for {ATOM_SPECIES}")
    print(f"Star key: {STAR_KEY} | T_eff = {infer_teff_from_star_template(STAR_KEY)} K")
    print(f"T_atm = {TEMP_ATM} | b = {B_VALUE} | d = {DISTANCE}")
    print(f"Common wavelength grid size = {len(lam_grid)}")
    print(f"Common wavelength step = {dlam.to_value(u.AA):.6e} AA")
    print(f"Used stitched lines = {used_lines}")
    print(f"Peak stitched sigma = {np.nanmax(sigma_total.to_value(u.cm**2)):.6e} cm^2")
    print(f"Integrated stitched sigma area = {trapezoid(sigma_total.to_value(u.cm**2), lam_grid.to_value(u.AA)):.6e} cm^2 AA")
    print("=" * 80)
    print(
        f"{'Ncol [cm^-2]':>14} | {'beta_atom':>14} | {'beta_stitched':>14} | {'rel diff':>12}"
    )
    print("-" * 80)

    for ncol, ba, bs in zip(NCOL_VALUES.to_value(1 / u.cm**2), beta_atom_val, beta_stitched_val):
        rdiff = relative_difference(ba, bs)
        print(f"{ncol:14.6e} | {ba:14.6e} | {bs:14.6e} | {rdiff:12.6e}")

    rel_diffs = np.array([relative_difference(a, b) for a, b in zip(beta_atom_val, beta_stitched_val)], dtype=float)
    max_rel_diff = np.nan if np.all(~np.isfinite(rel_diffs)) else np.nanmax(rel_diffs)
    print("-" * 80)
    print(f"Maximum relative beta difference = {max_rel_diff:.6e}")

    if MAKE_PLOT:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = NCOL_VALUES.to_value(1 / u.cm**2)
        ax.plot(x, beta_atom_val, marker="o", label="Atom pipeline")
        ax.plot(x, beta_stitched_val, marker="s", linestyle="--", label=f"Fake-atom molecule pipeline ({stitched_source_label})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N_{\rm col}$ [cm$^{-2}$]")
        ax.set_ylabel(r"$\beta$")
        ax.set_title(rf"{ATOM_SPECIES} | $T_{{\rm atm}}={TEMP_ATM.to_value(u.K):.0f}$ K | $b={B_VALUE.to_value(u.km/u.s):g}$ km s$^{{-1}}$")
        ax.grid(True, which="major", alpha=0.35)
        ax.legend(framealpha=0.9)
        fig.tight_layout()

        if SAVE_PLOT:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / OUTPUT_NAME
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {output_path}")

        plt.show()


if __name__ == "__main__":
    main()