import pathlib
import re
import sys

import astropy.constants as const

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_func.Templates.Systems.real_mass_loss_reference_systems import (
    REAL_MASS_LOSS_REFERENCE_SYSTEMS,
)


OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "Tables"
OUTPUT_TEX = OUTPUT_DIR / "real_mass_loss_reference_systems_table.tex"
SYSTEM_COL_WIDTH = "2.0cm"
TYPE_COL_WIDTH = "2.0cm"
STELLAR_COL_WIDTH = "2.6cm"
PLANET_COL_WIDTH = "2.95cm"
COMPOSITION_COL_WIDTH = "3.35cm"


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
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def latexify_formula(text: str) -> str:
    escaped = latex_escape(text)
    return re.sub(r"([A-Za-z])(\d+)", r"\1$_{\2}$", escaped)


def latexify_species(species: str) -> str:
    if " " in species:
        return latex_escape(species)
    return latexify_formula(species)


def latex_url(text: str) -> str:
    return str(text).replace("%", r"\%")


def format_plain_number(value: float) -> str:
    text = f"{value:.3f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def format_compact_number(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) <= 1.0e-3:
        text = f"{value:.1e}"
        mantissa, exponent = text.split("e")
        exponent = int(exponent)
        mantissa = mantissa.rstrip("0").rstrip(".")
        if mantissa == "1":
            return rf"10^{{{exponent}}}"
        return rf"{mantissa}\times10^{{{exponent}}}"
    if abs(value) < 0.1:
        text = f"{value:.4f}"
        return text.rstrip("0").rstrip(".")
    return format_plain_number(value)


def format_mass(system_def: dict) -> str:
    category = str(system_def.get("category", "")).strip().lower()
    planet = system_def["planet"]
    if category == "gas_giant":
        value = planet["mass"].to_value(const.M_jup)
        return f"{format_plain_number(value)} $M_\\mathrm{{J}}$"
    value = planet["mass"].to_value(const.M_earth)
    return f"{format_plain_number(value)} $M_\\oplus$"


def format_radius(system_def: dict) -> str:
    category = str(system_def.get("category", "")).strip().lower()
    planet = system_def["planet"]
    if category == "gas_giant":
        value = planet["radius"].to_value(const.R_jup)
        return f"{format_plain_number(value)} $R_\\mathrm{{J}}$"
    value = planet["radius"].to_value(const.R_earth)
    return f"{format_plain_number(value)} $R_\\oplus$"


def format_planet_type(system_def: dict) -> str:
    category = str(system_def.get("category", "")).strip().lower()
    mapping = {
        "rocky": "Rocky",
        "sub_neptune": "Sub-Neptune",
        "neptune": "Neptune",
        "gas_giant": "Gas giant",
    }
    return mapping.get(category, latex_escape(str(system_def.get("category", ""))))


def stellar_parameter_lines(system_def: dict) -> list[str]:
    star = system_def["star"]
    return [
        rf"$T_{{\rm eff}}={format_plain_number(float(star['teff_K']))}$ K",
        rf"$R_\star={format_plain_number(star['radius'].to_value(const.R_sun))}$ $R_\odot$",
        rf"$M_\star={format_plain_number(star['mass'].to_value(const.M_sun))}$ $M_\odot$",
        rf"$v\sin i={format_plain_number(star['vsini'].to_value(star['vsini'].unit))}$ km s$^{{-1}}$",
        rf"$\epsilon={format_plain_number(star['epsilon'].value)}$",
    ]


def planet_parameter_lines(system_def: dict) -> list[str]:
    category = str(system_def.get("category", "")).strip().lower()
    planet = system_def["planet"]
    if category == "gas_giant":
        radius_text = rf"$R_{{\rm p}}={format_plain_number(planet['radius'].to_value(const.R_jup))}$ $R_{{\rm J}}$"
        mass_text = rf"$M_{{\rm p}}={format_plain_number(planet['mass'].to_value(const.M_jup))}$ $M_{{\rm J}}$"
    else:
        radius_text = rf"$R_{{\rm p}}={format_plain_number(planet['radius'].to_value(const.R_earth))}$ $R_\oplus$"
        mass_text = rf"$M_{{\rm p}}={format_plain_number(planet['mass'].to_value(const.M_earth))}$ $M_\oplus$"
    return [
        radius_text,
        mass_text,
        rf"$a={format_compact_number(float(system_def['distance_au']))}$ AU",
        rf"$T={format_plain_number(planet['T'].to_value(planet['T'].unit))}$ K",
        rf"$\mu={format_plain_number(planet['mu'].value)}$",
        rf"$P_0={format_compact_number(planet['P0'].to_value(planet['P0'].unit))}$ bar",
    ]


def composition_lines(system_def: dict) -> list[str]:
    return [
        f"{latexify_species(species)} = {format_plain_number(float(fraction))}"
        for species, fraction in system_def["planet"]["composition"].items()
    ]


def multiline_text(lines: list[str]) -> str:
    return r" \\ ".join(lines)


def parbox_cell(width: str, content: str) -> str:
    return rf"\parbox[t]{{{width}}}{{\raggedright\vspace{{0pt}} {content}}}"


def ordered_system_items():
    return [
        ("gj1132_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["gj1132_b"]),
        ("gj1214_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["gj1214_b"]),
        ("gj436_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["gj436_b"]),
        ("hd209458_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["hd209458_b"]),
    ]


def build_source_indices():
    mapping = {}
    ordered_urls = []
    for _, system_def in ordered_system_items():
        url = str(system_def.get("planet_source_url", "")).strip()
        if url and url not in mapping:
            mapping[url] = len(mapping) + 1
            ordered_urls.append(url)
    return mapping, ordered_urls


def build_table_tex() -> str:
    source_indices, ordered_urls = build_source_indices()

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\caption{Real reference systems adopted in the mass-loss study. Stellar parameters, adopted planet/orbit/atmosphere parameters, and compositions are listed for each system.}",
        r"\label{tab:real_mass_loss_reference_systems}",
        rf"\begin{{tabular}}{{@{{}}p{{{SYSTEM_COL_WIDTH}}}p{{{TYPE_COL_WIDTH}}}p{{{STELLAR_COL_WIDTH}}}p{{{PLANET_COL_WIDTH}}}p{{{COMPOSITION_COL_WIDTH}}}@{{}}}}",
        r"\toprule",
        r"System & Type & Stellar~parameters & Planet~parameters & Composition \\",
        r"\midrule",
    ]

    for _, system_def in ordered_system_items():
        name = system_def["planet"]["label"]
        url = str(system_def.get("planet_source_url", "")).strip()
        source_marker = ""
        if url:
            source_marker = rf"\textsuperscript{{{source_indices[url]}}}"
        system_cell = parbox_cell(
            SYSTEM_COL_WIDTH,
            f"{latex_escape(name).replace(' ', '~')}{source_marker}",
        )
        type_cell = parbox_cell(TYPE_COL_WIDTH, format_planet_type(system_def))
        stellar_cell = parbox_cell(
            STELLAR_COL_WIDTH,
            multiline_text(stellar_parameter_lines(system_def)),
        )
        planet_cell = parbox_cell(
            PLANET_COL_WIDTH,
            multiline_text(planet_parameter_lines(system_def)),
        )
        composition_cell = parbox_cell(
            COMPOSITION_COL_WIDTH,
            multiline_text(composition_lines(system_def)),
        )
        lines.append(
            f"{system_cell} & "
            f"{type_cell} & "
            f"{stellar_cell} & "
            f"{planet_cell} & "
            f"{composition_cell} \\\\"
        )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    if ordered_urls:
        source_parts = [
            rf"\textsuperscript{{{source_indices[url]}}}\url{{{latex_url(url)}}}"
            for url in ordered_urls
        ]
        lines.append(r"\vspace{0.3em}")
        lines.append(r"\parbox{0.98\linewidth}{\scriptsize " + f"Sources: {'; '.join(source_parts)}." + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(build_table_tex(), encoding="utf-8")
    print(f"Saved {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
