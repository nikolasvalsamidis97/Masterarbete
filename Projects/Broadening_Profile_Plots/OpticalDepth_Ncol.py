import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star
from astropy import units as u
from astropy import constants as const
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


OUTPUT_PATH = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Broadening plots" / "BroadeningProfiles_combined.pdf"
DOPPLER_COMPARISON_OUTPUT_PATH = pathlib.Path(__file__).resolve().parent / "results" / "Plots" / "Broadening plots" / "BroadeningProfiles_doppler_0_5_comparison.pdf"
SUPTITLE_SIZE = 17
PANEL_TITLE_SIZE = 15
AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 14

# 1. Hämtar molekyldata
Na = Atom('Na I', 5800 * u.AA, 6000*u.AA)

# 2. Hämtar breddninsprofiler med molekylen breddningsparameter vlim och Npts samt typ av profil
bval = [0.1, 0.3] * u.km/u.s
DOPPLER_COMPARISON_BVAL = [0.0, 5.0] * u.km/u.s
vlim = 2 * u.km/u.s
Npts = 1000

## 2.5 Möjlighet att plotta profil och tvärsnitt för en linje. För att se linje: print(Na.data)
#print(Na.data)
line = 0
domain1 = 'velocity'
domain2 = 'wavelength'

Ncols = np.logspace(8, 16, 10) * u.cm**-2
column_colors = plt.get_cmap("tab10").colors

def normIntensity(N, sig):
  return np.exp(-(N*sig).to_value(u.dimensionless_unscaled))

def make_line_profile_plot(bvals, output_path, panel_xlims=None):
    fig, axes = plt.subplots(1, len(bvals), figsize=(10, 4), sharey=True, constrained_layout=True)

    # Common title for the whole figure
    fig.suptitle("Line profiles for various column densities", fontsize=SUPTITLE_SIZE)

    for panel_index, (ax, b) in enumerate(zip(axes, bvals)):
        # The zero-Doppler-width limit of a Voigt profile is the natural Lorentzian profile.
        profile_type = 'Lorentz' if np.isclose(b.to_value(u.km / u.s), 0.0) else 'Voigt'
        Na_broadening = BroadeningProfile(Na, b, Npts, profile_type)
        sig = Na_broadening.sigmaArray_sym[line, :]
        v = Na_broadening.v_grid_sym[0, :].to_value(u.km / u.s)

        for i, N in enumerate(Ncols):
            ax.plot(
                v,
                normIntensity(N, sig),
                linewidth=1.5,
                color=column_colors[i % len(column_colors)],
                alpha=0.95,
            )

        # Subtitle for each panel: broadening parameter
        ax.set_title(
            rf"$\Delta \mathrm{{v}}_\mathrm{{D}} = {b.to_value(u.km / u.s):.1f}\,\mathrm{{km\,s^{{-1}}}}$",
            fontsize=PANEL_TITLE_SIZE,
        )
        ax.set_xlabel(r"Relative velocity [$\mathrm{km\,s^{-1}}$]", fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

        # Fewer x-axis ticks
        ax.xaxis.set_major_locator(MaxNLocator(5))
        if panel_xlims is not None and panel_xlims[panel_index] is not None:
            ax.set_xlim(*panel_xlims[panel_index])

    # Shared y-label on the left
    axes[0].set_ylabel("Relative intensity", fontsize=AXIS_LABEL_SIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return fig

make_line_profile_plot(bval, OUTPUT_PATH)
make_line_profile_plot(
    DOPPLER_COMPARISON_BVAL,
    DOPPLER_COMPARISON_OUTPUT_PATH,
    panel_xlims=[None, (-15.0, 15.0)],
)

if plt.get_backend().lower() != "agg":
    plt.show()
