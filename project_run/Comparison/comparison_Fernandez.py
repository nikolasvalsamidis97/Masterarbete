import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

PLOTS_ROOT = PROJECT_ROOT / "Plots"
SHOW_FIGURE = False
BT_NEXTGEN_PATH = PROJECT_ROOT / "TS" / "Spectral_type" / "A" / "A6" / "lte080-4.0-0.0a+0.0.BT-NextGen.7.dat.txt"
FERNANDEZ_PATH = PROJECT_ROOT / "TS" / "Spectra" / "HRspec_A5V_130.dat"


def log10_exponent_label(value, _position):
    if not np.isfinite(value) or value <= 0:
        return ""

    exponent = np.log10(value)
    rounded = round(exponent)
    if not np.isclose(exponent, rounded, atol=1e-10):
        return ""
    return f"{int(rounded)}"

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# This file compares the results from "Braking the gas in the β Pictoris debris disk" (Fernandez et al. 2006) with the results from this code.
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
beta_values_Fernandez = {
    "H I": (1.6e-3, 0.1e-3),
    "He I": (0.0, 0.0),

    "Li I": (900, 40),
    "Be I": (62, 7),
    "Be II": (124, 6),

    "B I": (30, 10),
    "B II": (0.07, 0.04),
    "B III": (19, 1),

    "C I": (3.3e-2, 0.1e-2),
    "C II": (2.3e-3, 0.2e-3),
    "C III": (8.5e-6, 0.9e-6),

    "N I": (2.1e-4, 0.1e-4),
    "N II": (7.5e-6, 0.5e-6),
    "N III": (7.0e-6, 1.0e-6),

    "O I": (3.3e-4, 0.2e-4),
    "O II": (3.1e-9, 0.7e-9),
    "O III": (6.5e-7, 0.6e-7),

    "F II": (3.5e-6, 0.9e-6),
    "F III": (5.0e-9, 1.0e-9),

    "Ne III": (9.0e-8, 2.0e-8),

    "Na I": (360, 20),

    "Mg I": (74, 8),
    "Mg II": (9, 2),
    "Mg III": (0.0, 0.0),

    "Al I": (53, 6),
    "Al II": (0.36, 0.05),
    "Al III": (12, 1),

    "Si I": (6.0, 0.6),
    "Si II": (9, 9),
    "Si III": (5.8e-4, 0.6e-4),

    "P I": (3.4, 0.6),
    "P II": (2.2e-3, 0.3e-3),
    "P III": (5.0e-4, 2.0e-4),

    "S I": (0.56, 0.09),
    "S II": (9.0e-5, 1.0e-5),
    "S III": (2.0e-4, 1.0e-4),

    "Cl I": (2.3e-3, 0.4e-3),
    "Cl II": (3.7e-7, 0.4e-7),
    "Cl III": (3.0e-6, 2.0e-6),

    "Ar I": (1.7e-6, 0.3e-6),
    "Ar III": (1.5e-7, 0.2e-7),

    "K I": (200, 20),
    "K III": (4.4e-4, 0.2e-4),

    "Ca I": (330, 40),
    "Ca II": (50, 10),

    "Sc I": (220, 20),
    "Sc II": (1.3e3, 0.4e3), 
    "Sc III": (9.0e-2, 3.0e-2),

    "Ti I": (97, 5),
    "Ti II": (28, 2),
    "Ti III": (5.0e-4, 0.1e-4),

    "V I": (72, 4),
    "V II": (4.4, 0.2),

    "Cr I": (93, 5),
    "Cr II": (6.0e-7, 3.0e-7),
    # "Cr III": ... excluded

    "Mn I": (28, 3),
    "Mn II": (7, 1),
    # "Mn III": ... excluded

    "Fe I": (27, 2),
    "Fe II": (5.0, 0.3),
    "Fe III": (3.0e-7, 0.6e-7),

    "Co I": (16, 1),
    "Co III": (4.0e-7, 2.0e-7),

    "Ni I": (26, 2),
    "Ni II": (7.0e-2, 2.0e-2),
    "Ni III": (3.0e-7, 2.0e-7),
}
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Creating an atom with line data for Na I. (All available lines from NIST in the range 150-50000 Å)
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
all_atoms_list = [
  "H I",
  "He I","He II",
  "Li I","Li II","Li III",
  "Be I","Be II","Be III",
  "B I","B II","B III",
  "C I","C II","C III",
  "N I","N II","N III",
  "O I","O II","O III",
  "F I","F II","F III",
  "Ne I","Ne II","Ne III",
  "Na I","Na II","Na III",
  "Mg I","Mg II","Mg III",
  "Al I","Al II","Al III",
  "Si I","Si II","Si III",
  "P I","P II","P III",
  "S I","S II","S III",
  "Cl I","Cl II","Cl III",
  "Ar I","Ar II","Ar III",
  "K I","K II","K III",
  "Ca I","Ca II","Ca III",
  "Sc I","Sc II","Sc III",
  "Ti I","Ti II","Ti III",
  "V I","V II","V III",
  "Cr I","Cr II", #"Cr III" --- IGNORE ---
  "Mn I","Mn II", #"Mn III" --- IGNORE ---
  "Fe I","Fe II", "Fe III",
  "Co I","Co II", "Co III",
  "Ni I","Ni II", "Ni III",
]
wav_min = 150 * u.AA
wav_max = 50000 * u.AA

atoms_all = {sp: Atom(sp, wav_min, wav_max) for sp in all_atoms_list}

for sp, atom in atoms_all.items():
  print(f"Number of lines for {sp}: {atom.lam0.shape[0]}")

atoms = {
    sp: atom
    for sp, atom in atoms_all.items()
    if atom.lam0.shape[0] > 0 and getattr(atom, "_unique_states", np.empty((0, 2))).size > 0
}

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Applying broadening to the atomic lines
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
b = 1 * u.km/u.s        # b = v_D
Npts = 150
broadening_profiles = {sp: BroadeningProfile(atom, b, Npts, 'Voigt') for sp, atom in atoms.items()}

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Creating synthetic star, using the same parameters as in Fernandez et al. 2006 (Teff = 8000 K, logg = 4.0, [Fe/H] = 0.0)
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
vsini = 130 * u.km / u.s
epsilon = 0.5 * u.dimensionless_unscaled
R0 = 1.75 * const.R_sun
M0 = 1.75 * const.M_sun                                                         # As in Fernandez et al. 2006 (e.g., Gray 1976)
beta_pic = Star(str(BT_NEXTGEN_PATH), 
               R0, M0, vsini, epsilon)
d_earth_to_pic = 19.3 * u.pc                                                                             # Distance to β Pic

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Calibration made with information from Tycho catalog and Fernandez et al. 2006
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
targets_pic = {
  "TYCHO": {
    "B": 4.056, "V": 3.870
  },
}

MAGSYS_pic = {
  "TYCHO": "vegamag",
}
PHOTCALID_pic = {
  "TYCHO": {
    "B": "TYCHO/TYCHO.B/Vega",
    "V": "TYCHO/TYCHO.V/Vega",
  },
}

# ------------------------------------------------------------------------------------------------------------------------------------------------ #
# Calibration of the stellar radius
# ------------------------------------------------------------------------------------------------------------------------------------------------ #
r_old = beta_pic.radius
k_vals, lam_pivot = beta_pic.scale_factors_from_targets(targets_pic, d_earth_to_pic, MAGSYS_pic, PHOTCALID_pic, use_rot=True)
r_new = beta_pic.radius
print(f"Old radius: {r_old.to(u.R_sun):.3f}, New radius: {r_new.to(u.R_sun):.3f}")

def compute_beta_arrays(star_obj, broadening_profiles, beta_values_Fernandez,
                        d_atom_to_pic, include_fluxcal_4pct=False):
    """Return aligned arrays: my_beta, my_err, fern_beta, fern_err (same species order)."""
    Temp_atm = [1] * u.K
    Ncol = [1] * u.cm**(-2)
    chunk_size = 1

    beta_vals = {}
    for sp, broad_prof in broadening_profiles.items():
        try:
            pp = PhotonPressure(broad_prof, star_obj)
            F, Ferr, a, b = pp.calc_PhotonPressure(Ncol, Temp_atm, d_atom_to_pic, chunk_size=chunk_size)
            if include_fluxcal_4pct:
                cal = 0.04
                Ferr = np.sqrt(Ferr**2 + (cal * F)**2)
            beta_vals[sp] = pp.beta_Values(F, Ferr, star_obj.mass, d_atom_to_pic)
        except Exception as exc:
            print(f"Skipping {sp}: {type(exc).__name__}: {exc}")
            continue

    # Align with Fernandez dict
    common = [k for k in beta_values_Fernandez.keys() if k in beta_vals]

    my_beta = np.array([beta_vals[k][0].to_value(u.dimensionless_unscaled).ravel()[0] for k in common], float)
    my_err  = np.array([beta_vals[k][1].to_value(u.dimensionless_unscaled).ravel()[0] for k in common], float)

    fern_beta = np.array([beta_values_Fernandez[k][0] for k in common], float)
    fern_err  = np.array([beta_values_Fernandez[k][1] for k in common], float)

    return common, my_beta, my_err, fern_beta, fern_err


def plot_compare(my_beta, my_err, fern_beta, fern_err, zoom_beta_gt_1=False, title=None, save_name=None, xlabel=r'$\log(\beta)$ (This work)', ylabel=r'$\log(\beta)$ (Fernandez et al. 2006)'):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.errorbar(
        my_beta, fern_beta,
        xerr=my_err, yerr=fern_err,
        fmt='o', color='black', ecolor='black',
        ms=2, capsize=3, elinewidth=1, markeredgewidth=0.5,
    )

    # limits
    if zoom_beta_gt_1:
        lo, hi = 1.0, 2e3
    else:
        lo, hi = 1e-5, 2e3

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
    ax.yaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # y=x line (works for log too)
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=13)

    if title:
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    if save_name is not None:
        output_path = PLOTS_ROOT / f"{save_name}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


# -----------------------------
# Stars (two cases)
# -----------------------------

alpha=5.24e-5
dist_for_spec=1 * const.au
# fern_radius = 1.75 * const.R_sun
fern_radius = r_new
fern_beta_pic = Star(str(FERNANDEZ_PATH),
               fern_radius, 1.75*const.M_sun, vsini, epsilon)
fern_beta_pic.convert_from_log10()
fern_beta_pic.flux_star_rot = fern_beta_pic.flux_star_unrot * alpha * (dist_for_spec / fern_beta_pic.radius)**2  # convert to surface flux

# -----------------------------
# Compute arrays for both cases
# -----------------------------
d_atom_to_pic = 1 * u.au

# A) Using your BT-NextGen model spectrum (no 4% fluxcal)
common_A, my_beta_A, my_err_A, fern_beta_A, fern_err_A = compute_beta_arrays(
    beta_pic, broadening_profiles, beta_values_Fernandez,
    d_atom_to_pic, include_fluxcal_4pct=False
)

# B) Using Fernandez spectrum + 4% flux calibration uncertainty
common_B, my_beta_B, my_err_B, fern_beta_B, fern_err_B = compute_beta_arrays(
    fern_beta_pic, broadening_profiles, beta_values_Fernandez,
    d_atom_to_pic, include_fluxcal_4pct=True
)

# -----------------------------
# Make 4 plots (same style, no legend)
# -----------------------------
# 1) Model spectrum, full range
plot_compare(my_beta_A, my_err_A, fern_beta_A, fern_err_A, zoom_beta_gt_1=False, title="Bt-NextGen model spectrum", save_name="Betacomp/Bt-NextGen")

# 2) Model spectrum, zoom β>1
plot_compare(my_beta_A, my_err_A, fern_beta_A, fern_err_A, zoom_beta_gt_1=True, title="Bt-NextGen model spectrum (β>1)", save_name="Betacomp/Bt-NextGen_zoom")

# 3) Fernandez spectrum (+4%), full range
plot_compare(my_beta_B, my_err_B, fern_beta_B, fern_err_B, zoom_beta_gt_1=False, title="Fernandez spectrum (+4% flux error) ", save_name="Betacomp/Fernandez")

# 4) Fernandez spectrum (+4%), zoom β>1
plot_compare(my_beta_B, my_err_B, fern_beta_B, fern_err_B, zoom_beta_gt_1=True, title="Fernandez spectrum (+4% flux error, β>1)", save_name="Betacomp/Fernandez_zoom")


# 5) Fernandez vs Model, (+4% flux error)
plot_compare(my_beta_A, my_err_A, my_beta_B, my_err_B, zoom_beta_gt_1=False, title="Bt-NextGen vs Fernandez Spectrum (+4% flux error)", save_name="Betacomp/Bt-NextGen_vs_Fernandez", xlabel=r'$\log(\beta)$ (Bt-NextGen)', ylabel=r'$\log(\beta)$ (Fernandez spectrum)')

# 5) Fernandez vs Model, (+4% flux error)
plot_compare(my_beta_A, my_err_A, my_beta_B, my_err_B, zoom_beta_gt_1=True, title="Bt-NextGen vs Fernandez Spectrum (+4% flux error, β>1)", save_name="Betacomp/Bt-NextGen_vs_Fernandez_zoom", xlabel=r'$\log(\beta)$ (Bt-NextGen)', ylabel=r'$\log(\beta)$ (Fernandez spectrum)')

# -----------------------------
# Paired (stacked) comparison figures (2 + 2 + 2)
# -----------------------------

def plot_compare_two_cases(
    my_beta_top, my_err_top, fern_beta_top, fern_err_top,
    my_beta_bot, my_err_bot, fern_beta_bot, fern_err_bot,
    zoom_beta_gt_1=False,
    save_name=None,
    xlabel=r'$\log(\beta)$ (This work)',
    ylabel=r'$\log(\beta)$ (Fernandez et al. 2006)',
):
    """
    Two-panel (vertical) comparison figure with shared axes:
    """

    # limits
    if zoom_beta_gt_1:
        lo, hi = 1.0, 2e3
    else:
        lo, hi = 1e-5, 2e3

    fig = plt.figure(figsize=(7, 14))
    gs = fig.add_gridspec(2, 1, hspace=0)

    pair_label_size = 15
    pair_tick_size = 15
    pair_panel_size = 14

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top, sharey=ax_top)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.98, hspace=0)
    
    def _panel(ax, my_beta, my_err, fern_beta, fern_err, panel_tag):
        ax.errorbar(
            my_beta, fern_beta,
            xerr=my_err, yerr=fern_err,
            fmt='o', color='black', ecolor='black',
            ms=2, capsize=3, elinewidth=1, markeredgewidth=0.5,
        )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.xaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
        ax.yaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        # y=x reference line
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)

        ax.tick_params(axis='both', which='major', labelsize=pair_tick_size)

        # panel label only
        ax.text(
            0.04, 0.94, panel_tag,
            transform=ax.transAxes,
            fontsize=pair_panel_size,
            va='top',
            ha='left'
        )

        # force square plotting box
        ax.set_box_aspect(1)

    _panel(ax_top, my_beta_top, my_err_top, fern_beta_top, fern_err_top, '(a)')
    _panel(ax_bot, my_beta_bot, my_err_bot, fern_beta_bot, fern_err_bot, '(b)')

    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(True)

    ax_top.set_anchor('S')
    ax_bot.set_anchor('N')

    # labels: only one x label
    ax_top.set_xlabel('')
    ax_top.set_ylabel(ylabel, fontsize=pair_label_size)
    ax_bot.set_xlabel(xlabel, fontsize=pair_label_size)
    ax_bot.set_ylabel(ylabel, fontsize=pair_label_size)

    if save_name is not None:
        output_path = PLOTS_ROOT / f"{save_name}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
    return fig, (ax_top, ax_bot)

# Paired figure: full range (unzoomed)
plot_compare_two_cases(
    my_beta_A, my_err_A, fern_beta_A, fern_err_A,
    my_beta_B, my_err_B, fern_beta_B, fern_err_B,
    zoom_beta_gt_1=False,
    save_name="Betacomp/Validation_pair_full",
    xlabel=r'$\log(\beta)$ (This work)',
    ylabel=r'$\log(\beta)$ (Fernandez et al. 2006)',
)

# Paired figure: zoom β>1
plot_compare_two_cases(
    my_beta_A, my_err_A, fern_beta_A, fern_err_A,
    my_beta_B, my_err_B, fern_beta_B, fern_err_B,
    zoom_beta_gt_1=True,
    save_name="Betacomp/Validation_pair_zoom",
    xlabel=r'$\log(\beta)$ (This work)',
    ylabel=r'$\log(\beta)$ (Fernandez et al. 2006)',
)
