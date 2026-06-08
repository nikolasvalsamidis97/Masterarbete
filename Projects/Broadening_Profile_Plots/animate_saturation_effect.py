from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.special import wofz

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_classes.Atom import Atom


PROJECT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "Plots"
    / "Broadening plots"
    / "NaI_saturation_effect_doppler_0_5.gif"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate the saturation of a Na I line as column density increases."
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
        default=192,
        help="Number of logarithmically spaced column-density frames.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="GIF playback rate.",
    )
    parser.add_argument(
        "--doppler-widths",
        type=float,
        nargs=2,
        default=[0.0, 5.0],
        metavar=("LEFT", "RIGHT"),
        help="Doppler broadening parameters for the left and right panels in km/s.",
    )
    parser.add_argument(
        "--min-column",
        type=float,
        default=1.0e5,
        help="Minimum column density in cm^-2.",
    )
    parser.add_argument(
        "--max-column",
        type=float,
        default=1.0e18,
        help="Maximum column density in cm^-2.",
    )
    parser.add_argument(
        "--velocity-limit",
        type=float,
        default=25.0,
        help="Symmetric velocity-axis limit in km/s.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.frames < 2:
        raise ValueError("--frames must be at least 2.")
    if args.fps < 1:
        raise ValueError("--fps must be at least 1.")
    if any(doppler_width < 0.0 for doppler_width in args.doppler_widths):
        raise ValueError("--doppler-widths values cannot be negative.")
    if args.min_column <= 0.0 or args.max_column <= args.min_column:
        raise ValueError("--max-column must be greater than --min-column, and both must be positive.")
    if args.velocity_limit <= 0.0:
        raise ValueError("--velocity-limit must be greater than zero.")
    if args.output.suffix.lower() != ".gif":
        raise ValueError("--output must use the .gif extension.")


def normalized_intensity(column_density: u.Quantity, cross_section: u.Quantity) -> np.ndarray:
    optical_depth = (column_density * cross_section).to_value(u.dimensionless_unscaled)
    return np.exp(-optical_depth)


def cross_section_on_grid(
    sodium: Atom,
    line_index: int,
    velocity: u.Quantity,
    doppler_width: u.Quantity,
) -> u.Quantity:
    velocity_value = velocity.to_value(u.km / u.s)
    lorentz_fwhm = (sodium.lam0 * sodium.A_ul / (2.0 * np.pi)).to(u.km / u.s)
    gamma = 0.5 * lorentz_fwhm[line_index, 0].to_value(u.km / u.s)
    integrated_cross_section = sodium.sig_0[line_index, 0]

    doppler_width_value = doppler_width.to_value(u.km / u.s)
    if np.isclose(doppler_width_value, 0.0):
        profile_value = gamma / (np.pi * (velocity_value**2 + gamma**2))
    else:
        gaussian_sigma = doppler_width_value / np.sqrt(2.0)
        z = (velocity_value + 1j * gamma) / (gaussian_sigma * np.sqrt(2.0))
        profile_value = np.real(wofz(z)) / (gaussian_sigma * np.sqrt(2.0 * np.pi))

    profile = profile_value * u.s / u.km
    return (profile * integrated_cross_section).to(u.cm**2)


def main() -> None:
    args = parse_args()
    validate_args(args)

    sodium = Atom("Na I", 5800 * u.AA, 6000 * u.AA)
    line_index = 0
    velocity = np.linspace(-args.velocity_limit, args.velocity_limit, 4001) * u.km / u.s
    doppler_widths = [value * u.km / u.s for value in args.doppler_widths]
    cross_sections = [
        cross_section_on_grid(sodium, line_index, velocity, doppler_width)
        for doppler_width in doppler_widths
    ]
    column_densities = (
        np.logspace(
            np.log10(args.min_column),
            np.log10(args.max_column),
            args.frames,
        )
        * u.cm**-2
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    lines = []
    fig.subplots_adjust(top=0.78, bottom=0.17, wspace=0.08)
    fig.suptitle("Self shielding effect", fontsize=17, y=0.98)
    column_text = fig.text(
        0.5,
        0.88,
        "",
        fontsize=17,
        ha="center",
        va="top",
    )
    density_colormap = LinearSegmentedColormap.from_list(
        "density_sweep",
        ["#2166ac", "#1a9850", "#fee08b", "#d73027"],
    )

    for panel_index, (axis, doppler_width) in enumerate(zip(axes, doppler_widths)):
        line, = axis.plot([], [], color="#1f77b4", linewidth=2.2)
        lines.append(line)

        axis.set_xlim(-args.velocity_limit, args.velocity_limit)
        axis.set_ylim(-0.02, 1.04)
        axis.set_title(
            rf"$\Delta \mathrm{{v}}_\mathrm{{D}} = "
            rf"{doppler_width.to_value(u.km / u.s):g}\,\mathrm{{km\,s^{{-1}}}}$",
            fontsize=16,
        )
        axis.set_xlabel(r"Relative velocity [$\mathrm{km\,s^{-1}}$]", fontsize=16)
        axis.tick_params(axis="both", labelsize=13)
        axis.xaxis.set_major_locator(MaxNLocator(7))
        axis.grid(alpha=0.25)

        if panel_index == 0:
            axis.set_ylabel("Relative intensity", fontsize=16)

    def update(frame_index: int):
        column_density = column_densities[frame_index]
        column_text_value = (
            rf"$N_{{\rm col}} = 10^{{{np.log10(column_density.to_value(u.cm**-2)):.2f}}}"
            rf"\,\mathrm{{cm^{{-2}}}}$"
        )
        frame_color = density_colormap(frame_index / (len(column_densities) - 1))
        artists = []
        for line, cross_section in zip(lines, cross_sections):
            intensity = normalized_intensity(column_density, cross_section)
            line.set_data(velocity.to_value(u.km / u.s), intensity)
            line.set_color(frame_color)
            artists.append(line)
        column_text.set_text(column_text_value)
        column_text.set_color(frame_color)
        artists.append(column_text)
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(column_densities),
        interval=1000 / args.fps,
        blit=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved saturation animation to: {args.output}")


if __name__ == "__main__":
    main()
