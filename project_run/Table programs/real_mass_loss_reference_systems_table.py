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
COMPOSITION_COL_WIDTH = "4.2cm"

CATEGORY_BLOCKS = [
    ("Rocky reference systems", ["gj1132_b", "55cnc_e"]),
    ("Sub-Neptunes / Neptunes", ["gj1214_b", "hd56414_b", "gj436_b"]),
    (
        "Hot and ultra-hot Jupiters",
        ["51peg_b", "hd209458_b", "wasp174_b", "wasp193_b", "kelt9_b"],
    ),
]


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


def format_mu(value: float) -> str:
    return f"{float(value):.2f}"


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
    if category == "rocky":
        return "Rocky"
    if category == "sub_neptune":
        return "Sub-Neptune"
    if category == "neptune":
        return "Neptune"
    if category == "gas_giant":
        if str(system_def.get("exobase_template_key", "")) == "ultra_hot_jupiter":
            return "Ultra-hot Jupiter"
        return "Hot Jupiter"
    return latex_escape(str(system_def.get("category", "")))


def format_fraction(value: float) -> str:
    value = float(value)
    if value < 1e-3 and value > 0:
        text = f"{value:.0e}"
        mantissa, exponent = text.split("e")
        exponent = int(exponent)
        if mantissa == "1":
            return rf"$10^{{{exponent}}}$"
        return rf"${mantissa}\times10^{{{exponent}}}$"
    if value < 0.01 and value > 0:
        text = f"{value:.3f}"
    else:
        text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def ordered_system_items():
    return [
        ("gj1132_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["gj1132_b"]),
        ("55cnc_e", REAL_MASS_LOSS_REFERENCE_SYSTEMS["55cnc_e"]),
        ("gj1214_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["gj1214_b"]),
        ("hd56414_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["hd56414_b"]),
        ("gj436_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["gj436_b"]),
        ("51peg_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["51peg_b"]),
        ("hd209458_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["hd209458_b"]),
        ("wasp174_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["wasp174_b"]),
        ("wasp193_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["wasp193_b"]),
        ("kelt9_b", REAL_MASS_LOSS_REFERENCE_SYSTEMS["kelt9_b"]),
    ]


def iter_grouped_systems():
    for heading, keys in CATEGORY_BLOCKS:
        group = [(key, REAL_MASS_LOSS_REFERENCE_SYSTEMS[key]) for key in keys]
        yield heading, group


def build_source_indices():
    mapping = {}
    ordered_urls = []
    for _, system_def in ordered_system_items():
        url = str(system_def.get("planet_source_url", "")).strip()
        if url and url not in mapping:
            mapping[url] = len(mapping) + 1
            ordered_urls.append(url)
    return mapping, ordered_urls


def composition_text(system_def: dict) -> str:
    return ", ".join(
        f"{latexify_species(species)}: {format_fraction(float(fraction))}"
        for species, fraction in system_def["planet"]["composition"].items()
    )


def build_table_tex() -> str:
    source_indices, ordered_urls = build_source_indices()

    lines = [
        r"\begin{landscape}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{longtable}{llcccccccccccp{" + COMPOSITION_COL_WIDTH + r"}}",
        r"\caption{Real reference systems adopted in the mass-loss study. The table lists the adopted stellar and planet parameters together with the atmospheric composition used in the calculations.}\label{tab:real_mass_loss_reference_systems} \\",
        r"\toprule",
        r"Planet & Type & $T_{\rm eff}$ [K] & $R_\star$ & $M_\star$ & $v\sin i$ & $\epsilon$ & $R_{\rm p}$ & $M_{\rm p}$ & $a$ [AU] & $T$ [K] & $\mu$ & $P_0$ [bar] & Composition \\",
        r"\midrule",
        r"\endfirsthead",
        "",
        r"\toprule",
        r"Planet & Type & $T_{\rm eff}$ [K] & $R_\star$ & $M_\star$ & $v\sin i$ & $\epsilon$ & $R_{\rm p}$ & $M_{\rm p}$ & $a$ [AU] & $T$ [K] & $\mu$ & $P_0$ [bar] & Composition \\",
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule",
        r"\multicolumn{14}{r}{Continued on next page} \\",
        r"\midrule",
        r"\endfoot",
        "",
        r"\bottomrule",
        r"\endlastfoot",
        "",
    ]

    for heading, group in iter_grouped_systems():
        lines.extend(
            [
                rf"\multicolumn{{14}}{{l}}{{\textbf{{{heading}}}}} \\",
                r"\midrule",
            ]
        )
        for _, system_def in group:
            name = system_def["planet"]["label"]
            url = str(system_def.get("planet_source_url", "")).strip()
            source_marker = ""
            if url:
                source_marker = rf"\textsuperscript{{{source_indices[url]}}}"
            star = system_def["star"]
            planet = system_def["planet"]
            composition_cell = composition_text(system_def)
            lines.append(
                " & ".join(
                    [
                        latex_escape(name).replace(" ", "~") + source_marker,
                        format_planet_type(system_def),
                        format_plain_number(float(star["teff_K"])),
                        format_plain_number(star["radius"].to_value(const.R_sun)) + r" $R_\odot$",
                        format_plain_number(star["mass"].to_value(const.M_sun)) + r" $M_\odot$",
                        format_plain_number(star["vsini"].to_value(star["vsini"].unit)),
                        format_plain_number(star["epsilon"].value),
                        format_radius(system_def),
                        format_mass(system_def),
                        f"${format_compact_number(float(system_def['distance_au']))}$",
                        format_plain_number(planet["T"].to_value(planet["T"].unit)),
                        format_mu(planet["mu"].value),
                        f"${format_compact_number(planet['P0'].to_value(planet['P0'].unit))}$",
                        composition_cell,
                    ]
                )
                + r" \\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = ""
    else:
        lines.append("")
    lines.append(r"\end{longtable}")
    lines.append("")
    if ordered_urls:
        source_parts = [
            rf"\textsuperscript{{{source_indices[url]}}}\url{{{latex_url(url)}}}"
            for url in ordered_urls
        ]
        lines.append(r"\vspace{0.3em}")
        lines.append(r"\parbox{0.98\linewidth}{\scriptsize " + f"Sources: {'; '.join(source_parts)}." + "}")
    lines.append(r"\end{landscape}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(build_table_tex(), encoding="utf-8")
    print(f"Saved {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
