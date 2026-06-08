from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from astropy import constants as const
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from project_classes.Atom import Atom
from project_classes.BroadeningProfile import BroadeningProfile
from project_classes.PhotonPressure import PhotonPressure
from project_classes.Star import Star


PROJECT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "Plots"
    / "Photon pressure"
    / "F_ph_vs_Ncol_doppler_width_animation.gif"
)
SPECTRUM_PATH = (
    PROJECT_ROOT
    / "Templates"
    / "TS"
    / "Spectral_type"
    / "A"
    / "A6"
    / "lte080-4.0-0.0a+0.0.BT-NextGen.7.dat.txt"
)
Y_SCALE = 1.0e26
NUMERICAL_ZERO_DOPPLER_WIDTH_KM_S = 1.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate photon force versus column density as Doppler width changes."
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output GIF path.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=72,
        help="Number of linearly spaced Doppler-width frames.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="GIF playback rate.",
    )
    parser.add_argument(
        "--min-doppler-width",
        type=float,
        default=0.0,
        help="Minimum displayed Doppler width in km/s.",
    )
    parser.add_argument(
        "--max-doppler-width",
        type=float,
        default=5.0,
        help="Maximum displayed Doppler width in km/s.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.frames < 2:
        raise ValueError("--frames must be at least 2.")
    if args.fps < 1:
        raise ValueError("--fps must be at least 1.")
    if args.min_doppler_width < 0.0:
        raise ValueError("--min-doppler-width cannot be negative.")
    if args.max_doppler_width <= args.min_doppler_width:
        raise ValueError("--max-doppler-width must be greater than --min-doppler-width.")
    if args.output.suffix.lower() != ".gif":
        raise ValueError("--output must use the .gif extension.")


def log10_exponent_label(value: float, _position: float) -> str:
    if not np.isfinite(value) or value <= 0.0:
        return ""
    exponent = np.log10(value)
    rounded = round(exponent)
    return f"{int(rounded)}" if np.isclose(exponent, rounded, atol=1.0e-10) else ""


def compute_curves(doppler_widths: np.ndarray):
    atom = Atom("Na I", 300 * u.AA, 50000 * u.AA, 0 / u.s)
    column_density = np.logspace(8, 15, 100) * u.cm**-2
    temperature = 300 * u.K
    distance = 1 * u.au
    star = Star(
        str(SPECTRUM_PATH),
        1 * const.R_sun.value * u.m,
        1 * const.M_sun.value * u.kg,
        1 * u.km / u.s,
        0.0 * u.dimensionless_unscaled,
    )

    force_curves = []
    error_curves = []
    for frame_index, displayed_width in enumerate(doppler_widths, start=1):
        numerical_width = max(float(displayed_width), NUMERICAL_ZERO_DOPPLER_WIDTH_KM_S)
        broadening = BroadeningProfile(atom, numerical_width * u.km / u.s, 300, "Voigt")
        photon_pressure = PhotonPressure(broadening, star)
        force, force_error, _, _ = photon_pressure.calc_PhotonPressure(
            column_density,
            temperature,
            distance,
        )
        force_curves.append(force[0].to_value(u.N) * Y_SCALE)
        error_curves.append(force_error[0].to_value(u.N) * Y_SCALE)
        print(
            f"Computed Doppler-width frame {frame_index}/{len(doppler_widths)}: "
            f"{displayed_width:.3f} km/s"
        )

    return column_density.to_value(1 / u.cm**2), np.asarray(force_curves), np.asarray(error_curves)


def main() -> None:
    args = parse_args()
    validate_args(args)

    doppler_widths = np.linspace(
        args.min_doppler_width,
        args.max_doppler_width,
        args.frames,
    )
    column_density, force_curves, error_curves = compute_curves(doppler_widths)

    fig, axis = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.80, bottom=0.16)
    fig.suptitle(r"Photon force vs column density ($\mathrm{Na^0}$)", fontsize=17, y=0.97)
    doppler_text = fig.text(
        0.5,
        0.88,
        "",
        fontsize=16,
        ha="center",
        va="top",
    )
    width_colormap = LinearSegmentedColormap.from_list(
        "doppler_width_sweep",
        ["#2166ac", "#1a9850", "#fee08b", "#d73027"],
    )

    line, = axis.plot([], [], linewidth=2.0)
    fill_holder = [None]

    axis.set_xscale("log", base=10)
    axis.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    axis.xaxis.set_major_formatter(FuncFormatter(log10_exponent_label))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlim(float(column_density.min()), float(column_density.max()))
    ymax = float(np.nanmax(force_curves + error_curves))
    axis.set_ylim(0.0, 1.05 * ymax)
    axis.set_xlabel(
        r"$\log_{10}\!\left(N_{\mathrm{col}}\,[\mathrm{cm}^{-2}]\right)$",
        fontsize=18,
    )
    axis.set_ylabel(r"Photon force [$\mathrm{N}\times 10^{-26}$]", fontsize=18)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(True, which="both", alpha=0.3)

    def update(frame_index: int):
        if fill_holder[0] is not None:
            fill_holder[0].remove()

        force = force_curves[frame_index]
        force_error = error_curves[frame_index]
        frame_color = width_colormap(frame_index / (len(doppler_widths) - 1))
        line.set_data(column_density, force)
        line.set_color(frame_color)
        fill_holder[0] = axis.fill_between(
            column_density,
            force - force_error,
            force + force_error,
            color=frame_color,
            alpha=0.20,
            linewidth=0,
        )
        doppler_text.set_text(
            rf"$\Delta \mathrm{{v}}_\mathrm{{D}} = "
            rf"{doppler_widths[frame_index]:.2f}\,\mathrm{{km\,s^{{-1}}}}$"
        )
        doppler_text.set_color(frame_color)
        return line, fill_holder[0], doppler_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(doppler_widths),
        interval=1000 / args.fps,
        blit=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved photon-pressure animation to: {args.output}")


if __name__ == "__main__":
    main()
