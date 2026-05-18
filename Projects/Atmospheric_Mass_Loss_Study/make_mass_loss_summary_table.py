from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import astropy.constants as const

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from Templates.Planets.planet_templates_updated import PLANET_TEMPLATES_UPDATED
from Templates.Systems.real_mass_loss_reference_systems import (
    REAL_MASS_LOSS_REFERENCE_SYSTEMS,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "Tables" / "Mass_loss_rate_advanced"
OUTPUT_CSV = RESULTS_DIR / "mass_loss_summary_all_cases.csv"

FAMILY_FILES = {
    "solar_system_fixed": "solar_system_fixed.txt",
    "distance_sweep": "distance_sweep.txt",
    "real_reference_systems": "real_reference_systems.txt",
    "p0_sweep": "p0_sweep.txt",
    "mu_sweep": "mu_sweep.txt",
    "surface_gravity_sweep": "surface_gravity_sweep.txt",
}

FAMILY_LABELS = {
    "solar_system_fixed": "Solar system analogues",
    "distance_sweep": "Distance sweep",
    "real_reference_systems": "Real reference systems",
    "p0_sweep": "P0 sweep",
    "mu_sweep": "mu sweep",
    "surface_gravity_sweep": "Surface gravity sweep",
}

FAMILY_ORDER = [
    "solar_system_fixed",
    "real_reference_systems",
    "distance_sweep",
    "p0_sweep",
    "mu_sweep",
    "surface_gravity_sweep",
]

TABLE_BASENAMES = {
    "solar_system_fixed": "solar_system_fixed_mass_loss_summary.tex",
    "real_reference_systems": "real_reference_systems_mass_loss_summary.tex",
    "distance_sweep": "distance_sweep_mass_loss_summary.tex",
    "p0_sweep": "p0_sweep_mass_loss_summary.tex",
    "mu_sweep": "mu_sweep_mass_loss_summary.tex",
    "surface_gravity_sweep": "surface_gravity_sweep_mass_loss_summary.tex",
}
REAL_NONZERO_GS_THRESHOLD = 1.0e-30
REAL_NONZERO_OUTPUT_TEX = RESULTS_DIR / "hd209458b_mass_loss_summary.tex"
MERCURY_NONZERO_OUTPUT_TEX = RESULTS_DIR / "mercury_like_mass_loss_summary.tex"
M_NEPTUNE_IN_MEARTH = 17.147


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def tex_number(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return r"\ensuremath{0}"
    exponent = int(math.floor(math.log10(abs(value))))
    if -2 <= exponent <= 3:
        text = f"{value:.{digits}f}"
        return rf"\ensuremath{{{text.rstrip('0').rstrip('.')}}}"
    mantissa = value / (10 ** exponent)
    mantissa_text = f"{mantissa:.{digits}f}".rstrip("0").rstrip(".")
    return rf"\ensuremath{{{mantissa_text}\times10^{{{exponent}}}}}"


def plain_number(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    if -2 <= exponent <= 3:
        text = f"{value:.{digits}f}"
        return text.rstrip("0").rstrip(".")
    return f"{value:.{digits}e}"


def tex_sigfigs(value: float, sigfigs: int = 2) -> str:
    if value == 0.0:
        return r"\ensuremath{0}"
    text = f"{value:.{sigfigs}g}"
    if "e" not in text and "E" not in text:
        return rf"\ensuremath{{{text}}}"
    mantissa, exponent = text.lower().split("e")
    return rf"\ensuremath{{{mantissa}\times10^{{{int(exponent)}}}}}"


def finite_float(value: str | float | None, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def latex_species_name(species: str) -> str:
    parts = species.split()
    ion_charge_labels = {
        "I": "0",
        "II": "+",
        "III": "2+",
        "IV": "3+",
        "V": "4+",
    }
    if len(parts) == 2 and parts[1] in ion_charge_labels:
        element = latex_escape(parts[0])
        charge_label = ion_charge_labels[parts[1]]
        return rf"\ensuremath{{\mathrm{{{element}}}^{{{charge_label}}}}}"

    escaped = latex_escape(species)
    return re.sub(r"([A-Za-z])(\d+)", r"\1$_{\2}$", escaped)


def real_system_category(row: dict[str, str | float]) -> str:
    return str(REAL_MASS_LOSS_REFERENCE_SYSTEMS[str(row["planet_key"])]["category"])


def real_mass_display_tex(row: dict[str, str | float]) -> str:
    mass_mearth = float(row["planet_mass_Mearth"])
    category = real_system_category(row)
    if category == "rocky":
        return rf"\ensuremath{{{int(round(mass_mearth))}\,M_\oplus}}"
    if category in {"sub_neptune", "neptune"}:
        mass_mnep = mass_mearth / M_NEPTUNE_IN_MEARTH
        return rf"\ensuremath{{{mass_mnep:.1f}\,M_{{\rm Nep}}}}"
    mass_mjup = mass_mearth / const.M_jup.to_value(const.M_earth)
    return rf"\ensuremath{{{mass_mjup:.1f}\,M_{{\rm J}}}}"


def read_family_rows(family: str) -> list[dict[str, str]]:
    path = RESULTS_DIR / FAMILY_FILES[family]
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_total_rows() -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []
    for family in FAMILY_ORDER:
        rows = read_family_rows(family)
        all_rows.extend(row for row in rows if row["species"] == "TOTAL_INCLUDED_SPECIES")
    return all_rows


def planet_mass_mearth(row: dict[str, str]) -> float:
    family = row["test_family"]
    planet_key = row["planet"]
    if family == "real_reference_systems":
        return REAL_MASS_LOSS_REFERENCE_SYSTEMS[planet_key]["planet"]["mass"].to_value(const.M_earth)

    mass_mearth = PLANET_TEMPLATES_UPDATED[planet_key]["mass"].to_value(const.M_earth)
    if family == "surface_gravity_sweep" and row.get("mass_scale"):
        mass_mearth *= float(row["mass_scale"])
    return mass_mearth


def parameter_text(row: dict[str, str]) -> str:
    family = row["test_family"]
    if family in {"solar_system_fixed", "real_reference_systems", "distance_sweep"}:
        return f"a={plain_number(float(row['distance_AU']))} AU"
    if family == "p0_sweep":
        return f"P0={plain_number(float(row['P0_bar']))} bar"
    if family == "mu_sweep":
        return f"mu={plain_number(float(row['mu_amu']))}"
    if family == "surface_gravity_sweep":
        g_value = plain_number(float(row["surface_gravity_m_s2"]))
        scale_value = plain_number(float(row["mass_scale"]))
        return f"g={g_value} m s^-2; xM={scale_value}"
    return ""


def parameter_tex(row: dict[str, str]) -> str:
    family = row.get("test_family", row.get("family_key", ""))
    if family in {"solar_system_fixed", "real_reference_systems", "distance_sweep"}:
        return f"$a={plain_number(float(row['distance_AU']))}$ AU"
    if family == "p0_sweep":
        return f"$P_0={plain_number(float(row['P0_bar']))}$ bar"
    if family == "mu_sweep":
        return f"$\\mu={plain_number(float(row['mu_amu']))}$"
    if family == "surface_gravity_sweep":
        g_value = plain_number(float(row["surface_gravity_m_s2"]))
        scale_value = plain_number(float(row["mass_scale"]))
        return f"$g={g_value}$ m s$^{{-2}}$; $x_M={scale_value}$"
    return ""


def family_row_label_header(family_key: str) -> str:
    if family_key == "solar_system_fixed":
        return "Planet"
    if family_key == "real_reference_systems":
        return "System"
    if family_key == "distance_sweep":
        return "$a$ [AU]"
    if family_key == "p0_sweep":
        return "$P_0$ [bar]"
    if family_key == "mu_sweep":
        return "$\\mu$ [amu]"
    if family_key == "surface_gravity_sweep":
        return "$g$ [m s$^{-2}$]"
    return "Case"


def family_row_label_tex(row: dict[str, str | float]) -> str:
    family = str(row["family_key"])
    if family == "solar_system_fixed":
        return latex_escape(str(row["planet_label"]))
    if family == "real_reference_systems":
        return latex_escape(str(row["planet_label"]))
    if family == "distance_sweep":
        return plain_number(float(str(row["distance_AU"])))
    if family == "p0_sweep":
        return tex_number(float(str(row["P0_bar"])))
    if family == "mu_sweep":
        return plain_number(float(str(row["mu_amu"])))
    if family == "surface_gravity_sweep":
        return plain_number(float(str(row["surface_gravity_m_s2"])))
    return latex_escape(str(row["planet_label"]))


def family_caption(family_key: str) -> str:
    if family_key == "solar_system_fixed":
        return "Mass-loss summary for the solar-system analogue cases."
    if family_key == "real_reference_systems":
        return "Mass-loss summary for the real reference systems."
    if family_key == "distance_sweep":
        return "Mass-loss summary for the distance sweep of the inflated hot Jupiter around the F8 star."
    if family_key == "p0_sweep":
        return "Mass-loss summary for the $P_0$ sweep of the inflated hot Jupiter around the F8 star at 0.05 AU."
    if family_key == "mu_sweep":
        return "Mass-loss summary for the $\\mu$ sweep of the inflated hot Jupiter around the F8 star at 0.05 AU."
    if family_key == "surface_gravity_sweep":
        return "Mass-loss summary for the surface-gravity sweep of the super-Earth rocky planet around the F8 star at 0.1 AU."
    return "Mass-loss summary."


def build_summary_rows() -> list[dict[str, str | float]]:
    summary: list[dict[str, str | float]] = []
    for row in read_total_rows():
        mass_mearth = planet_mass_mearth(row)
        mass_lost_1myr = float(row["mass_lost_Mearth_1Myr"])
        relative_loss_1myr = mass_lost_1myr / mass_mearth if mass_mearth else 0.0
        summary.append(
            {
                "family_key": row["test_family"],
                "family": FAMILY_LABELS[row["test_family"]],
                "planet_key": row["planet"],
                "planet_label": row["planet_label"],
                "parameters": parameter_text(row),
                "distance_AU": row.get("distance_AU", ""),
                "P0_bar": row.get("P0_bar", ""),
                "mu_amu": row.get("mu_amu", ""),
                "surface_gravity_m_s2": row.get("surface_gravity_m_s2", ""),
                "mass_scale": row.get("mass_scale", ""),
                "planet_mass_Mearth": mass_mearth,
                "total_shell_mass_g": float(row["total_shell_mass_g"]),
                "mass_lost_g_1Myr": float(row["mass_lost_g_1Myr"]),
                "mass_loss_rate_g_s": float(row["mass_loss_rate_g_s"]),
                "mass_lost_Mearth_1Myr": mass_lost_1myr,
                "relative_loss_1Myr": relative_loss_1myr,
            }
        )
    return summary


def write_csv(rows: list[dict[str, str | float]]) -> None:
    fieldnames = [
        "family_key",
        "family",
        "planet_key",
        "planet_label",
        "parameters",
        "distance_AU",
        "P0_bar",
        "mu_amu",
        "surface_gravity_m_s2",
        "mass_scale",
        "planet_mass_Mearth",
        "total_shell_mass_g",
        "mass_lost_g_1Myr",
        "mass_loss_rate_g_s",
        "mass_lost_Mearth_1Myr",
        "relative_loss_1Myr",
    ]
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def real_species_fraction_entries() -> list[tuple[str, str]]:
    rows = read_family_rows("real_reference_systems")
    totals: dict[str, dict[str, str]] = {}
    species_rows_by_planet: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        planet_key = row["planet"]
        species = row["species"]
        if species == "TOTAL_INCLUDED_SPECIES":
            totals[planet_key] = row
        elif species:
            species_rows_by_planet.setdefault(planet_key, []).append(row)

    ordered_planets = [
        planet_key
        for planet_key in REAL_MASS_LOSS_REFERENCE_SYSTEMS
        if planet_key in totals or planet_key in species_rows_by_planet
    ]
    ordered_planets.extend(
        sorted(planet_key for planet_key in species_rows_by_planet if planet_key not in ordered_planets)
    )

    entries: list[tuple[str, str]] = []
    for planet_key in ordered_planets:
        total_mdot = finite_float(totals.get(planet_key, {}).get("mass_loss_rate_g_s"))
        has_nonzero_total = total_mdot > REAL_NONZERO_GS_THRESHOLD
        if not has_nonzero_total:
            continue

        species_rows = list(species_rows_by_planet.get(planet_key, []))
        composition = REAL_MASS_LOSS_REFERENCE_SYSTEMS.get(planet_key, {}).get("planet", {}).get("composition", {})
        species_order = {species: index for index, species in enumerate(composition)}
        species_rows.sort(key=lambda row: species_order.get(row["species"], len(species_order)))

        if planet_key in totals:
            planet_label = totals[planet_key]["planet_label"]
        elif species_rows:
            planet_label = species_rows[0]["planet_label"]
        else:
            planet_label = planet_key

        species_rows_by_name = {row["species"]: row for row in species_rows}
        ordered_species = list(composition)
        ordered_species.extend(row["species"] for row in species_rows if row["species"] not in species_order)

        species_contributions: list[tuple[str, float, float, int]] = []
        for species in ordered_species:
            row = species_rows_by_name.get(species)
            mass_loss_rate = finite_float(row.get("mass_loss_rate_g_s")) if row else 0.0
            fraction = mass_loss_rate / total_mdot if has_nonzero_total else 0.0
            sort_mass_loss_rate = mass_loss_rate if has_nonzero_total else 0.0
            species_contributions.append(
                (species, fraction, sort_mass_loss_rate, species_order.get(species, len(species_order)))
            )

        species_contributions.sort(key=lambda item: (-item[2], item[3]))

        parts: list[str] = []
        for species, fraction, _, _ in species_contributions:
            parts.append(f"{latex_species_name(species)}: {tex_sigfigs(fraction, sigfigs=2)}")

        entries.append((latex_escape(planet_label), ", ".join(parts) if parts else r"\ensuremath{0}"))

    return entries


def real_species_fraction_table_tex() -> list[str]:
    entries = real_species_fraction_entries()
    if not entries:
        return []

    lines = [
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        (
            r"\caption{Species-level contributions to the real-reference-system "
            r"mass-loss rates. Each entry is the fraction "
            r"$\dot{M}_i/\dot{M}_{\rm tot}$ for that system. Only systems with "
            r"nonzero modeled escape are listed; composition species with no "
            r"modeled contribution are retained as 0.}"
        ),
        r"\label{tab:real_reference_species_mass_loss_fractions}",
        r"\begin{tabular}{@{}p{2.4cm}p{10.7cm}@{}}",
        r"\toprule",
        r"System & Species fractions of total $\dot{M}$ \\",
        r"\midrule",
    ]

    for planet_label, fractions in entries:
        lines.append(f"{planet_label} & {fractions} " + r"\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return lines


def write_family_tex(family_key: str, rows: list[dict[str, str | float]]) -> None:
    output_tex = RESULTS_DIR / TABLE_BASENAMES[family_key]
    row_header = family_row_label_header(family_key)
    if family_key == "real_reference_systems":
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{1.05}",
            (
                r"\caption{Mass-loss summary for the real reference systems. "
                r"The third column shows relative planet mass lost in a mega-year. "
                r"$M_{\rm shell}$ is the modeled atmospheric mass between the exobase and "
                r"the Hill radius. The final column gives the amount of such shell "
                r"atmospheres lost in a mega-year.}"
            ),
            rf"\label{{tab:mass_loss_summary_{family_key}}}",
            r"\begin{tabular}{@{}p{2.0cm}p{1.3cm}p{1.5cm}p{1.6cm}p{1.6cm}p{1.8cm}@{}}",
            r"\toprule",
            (
                r"System & $M_{\rm p}$ & $\dot{M}$ [g s$^{-1}$] & "
                r"$M_p$ Myr$^{-1}$ & $M_{\rm shell}$ [g] & $M_{atm}$ Myr$^{-1}$ \\"
            ),
            r"\midrule",
        ]
        for row in rows:
            shell_mass = float(row["total_shell_mass_g"])
            atmospheres_lost = float(row["mass_lost_g_1Myr"]) / shell_mass if shell_mass > 0.0 else 0.0
            lines.append(
                " & ".join(
                    [
                        family_row_label_tex(row),
                        real_mass_display_tex(row),
                        tex_sigfigs(float(row["mass_loss_rate_g_s"]), sigfigs=2),
                        tex_sigfigs(float(row["relative_loss_1Myr"]), sigfigs=2),
                        tex_sigfigs(shell_mass, sigfigs=2),
                        tex_sigfigs(atmospheres_lost, sigfigs=2),
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        lines.extend(real_species_fraction_table_tex())
        output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        rf"\caption{{{family_caption(family_key)} Relative loss is defined as $\Delta M(1\,{{\rm Myr}})/M_{{\rm p}}$.}}",
        rf"\label{{tab:mass_loss_summary_{family_key}}}",
        r"\begin{tabular}{@{}p{2.6cm}p{1.6cm}p{2.1cm}p{2.1cm}p{1.9cm}@{}}",
        r"\toprule",
        rf"{row_header} & $M_{{\rm p}}$ [$M_\oplus$] & $\dot{{M}}$ [g s$^{{-1}}$] & $\Delta M(1\,{{\rm Myr}})$ [$M_\oplus$] & $\Delta M / M_{{\rm p}}$ \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            " & ".join(
                [
                    family_row_label_tex(row),
                    tex_number(float(row["planet_mass_Mearth"])),
                    tex_number(float(row["mass_loss_rate_g_s"])),
                    tex_number(float(row["mass_lost_Mearth_1Myr"])),
                    tex_number(float(row["relative_loss_1Myr"])),
                ]
            )
            + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_tables(rows: list[dict[str, str | float]]) -> None:
    rows_by_family: dict[str, list[dict[str, str | float]]] = {family: [] for family in FAMILY_ORDER}
    for row in rows:
        rows_by_family[str(row["family_key"])].append(row)
    for family_key in FAMILY_ORDER:
        family_rows = rows_by_family[family_key]
        if family_key == "real_reference_systems":
            family_rows = [
                row for row in family_rows if float(row["mass_loss_rate_g_s"]) > REAL_NONZERO_GS_THRESHOLD
            ]
        write_family_tex(family_key, family_rows)


def write_real_nonzero_table(rows: list[dict[str, str | float]]) -> None:
    real_rows = [
        row
        for row in rows
        if str(row["family_key"]) == "real_reference_systems"
        and float(row["mass_loss_rate_g_s"]) > REAL_NONZERO_GS_THRESHOLD
    ]
    real_rows.sort(key=lambda row: float(row["mass_loss_rate_g_s"]), reverse=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Mass-loss summary for the real reference systems with nonzero modeled escape. The relative loss is defined as $\Delta M(1\,\mathrm{Myr})/M_{\rm p}$. The ``atmospheres lost'' column is defined as $\Delta M(1\,\mathrm{Myr})/M_{\rm shell}$, where $M_{\rm shell}$ is the modeled exobase-to-Hill shell mass.}",
        r"\label{tab:real_reference_nonzero_mass_loss_summary}",
        r"\begin{tabular}{l l c c c c}",
        r"\toprule",
        r"Name & Type & $M_{\rm p}$ & $\dot{M}$ [g s$^{-1}$] & $\Delta M(1\,\mathrm{Myr})/M_{\rm p}$ & Atmospheres lost in 1 Myr \\",
        r"\midrule",
    ]

    for row in real_rows:
        system_def = REAL_MASS_LOSS_REFERENCE_SYSTEMS[str(row["planet_key"])]
        system_type = latex_escape(
            {
                "rocky": "Rocky",
                "sub_neptune": "Sub-Neptune",
                "neptune": "Neptune",
                "gas_giant": "Gas giant",
            }.get(str(system_def.get("category", "")), str(system_def.get("category", "")))
        )
        shell_mass = float(row["total_shell_mass_g"])
        atmospheres_lost = float(row["mass_lost_g_1Myr"]) / shell_mass if shell_mass > 0.0 else 0.0
        lines.append(
            " & ".join(
                [
                    latex_escape(str(row["planet_label"])).replace(" ", "~"),
                    system_type,
                    real_mass_display_tex(row),
                    tex_sigfigs(float(row["mass_loss_rate_g_s"]), sigfigs=2),
                    tex_sigfigs(float(row["relative_loss_1Myr"]), sigfigs=2),
                    tex_sigfigs(atmospheres_lost, sigfigs=2),
                ]
            )
            + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    REAL_NONZERO_OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mercury_nonzero_table(rows: list[dict[str, str | float]]) -> None:
    mercury_rows = [
        row
        for row in rows
        if str(row["family_key"]) == "solar_system_fixed"
        and str(row["planet_key"]) == "mercury_like"
        and float(row["mass_loss_rate_g_s"]) > REAL_NONZERO_GS_THRESHOLD
    ]
    if not mercury_rows:
        return

    row = mercury_rows[0]
    shell_mass = float(row["total_shell_mass_g"])
    atmospheres_lost = float(row["mass_lost_g_1Myr"]) / shell_mass if shell_mass > 0.0 else 0.0
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        (
            r"\caption{Mass-loss summary for the Mercury-like solar-system analogue. "
            r"The third column shows relative planet mass lost in a mega-year. "
            r"$M_{\rm shell}$ is the modeled atmospheric mass between the exobase and "
            r"the Hill radius. The final column gives the amount of such shell "
            r"atmospheres lost in a mega-year.}"
        ),
        r"\label{tab:mass_loss_summary_mercury_like}",
        r"\begin{tabular}{@{}p{2.4cm}p{1.35cm}p{1.55cm}p{1.65cm}p{1.65cm}p{1.9cm}@{}}",
        r"\toprule",
        r"System & $M_{\rm p}$ & $\dot{M}$ [g s$^{-1}$] & $M_p$ Myr$^{-1}$ & $M_{\rm shell}$ [g] & $M_{atm}$ Myr$^{-1}$ \\",
        r"\midrule",
        " & ".join(
            [
                latex_escape(str(row["planet_label"])),
                rf"\ensuremath{{{float(row['planet_mass_Mearth']):.3f}\,M_\oplus}}",
                tex_sigfigs(float(row["mass_loss_rate_g_s"]), sigfigs=2),
                tex_sigfigs(float(row["relative_loss_1Myr"]), sigfigs=2),
                tex_sigfigs(shell_mass, sigfigs=2),
                tex_sigfigs(atmospheres_lost, sigfigs=2),
            ]
        )
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    MERCURY_NONZERO_OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_summary_rows()
    write_csv(rows)
    write_tex_tables(rows)
    write_real_nonzero_table(rows)
    write_mercury_nonzero_table(rows)
    print(f"Saved {OUTPUT_CSV}")
    for family_key in FAMILY_ORDER:
        print(f"Saved {RESULTS_DIR / TABLE_BASENAMES[family_key]}")
    print(f"Saved {REAL_NONZERO_OUTPUT_TEX}")
    print(f"Saved {MERCURY_NONZERO_OUTPUT_TEX}")


if __name__ == "__main__":
    main()
