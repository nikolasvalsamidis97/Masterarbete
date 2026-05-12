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
FULL_WIDTH_NOTE_COL = r"@{}p{\dimexpr\linewidth-2\tabcolsep\relax}@{}"

SOURCE_REFERENCES = [
    ("xue2024jwst", r"\cite{xue2024jwst}"),
    ("berta2015rocky", r"\cite{berta2015rocky}"),
    ("swain2021detection", r"\cite{swain2021detection}"),
    ("bourrier201855", r"\cite{bourrier201855}"),
    ("moutou2026characterising", r"\cite{moutou2026characterising}"),
    ("hu2024secondary", r"\cite{hu2024secondary}"),
    ("mahajan2024using", r"\cite{mahajan2024using}"),
    ("charbonneau2009super", r"\cite{charbonneau2009super}"),
    ("kempton2023reflective", r"\cite{kempton2023reflective}"),
    ("giacalone2022hd", r"\cite{giacalone2022hd}"),
    ("otegi2020revisited", r"\cite{otegi2020revisited}"),
    ("maxted2022analysis", r"\cite{maxted2022analysis}"),
    ("bourrier2018orbital", r"\cite{bourrier2018orbital}"),
    ("maciejewski2015gj", r"\cite{maciejewski2015gj}"),
    ("turner2016ground", r"\cite{turner2016ground}"),
    ("stevenson2010possible", r"\cite{stevenson2010possible}"),
    ("knutson2014featureless", r"\cite{knutson2014featureless}"),
    ("ehrenreich2015giant", r"\cite{ehrenreich2015giant}"),
    ("rosenthal2021california", r"\cite{rosenthal2021california}"),
    ("valenti2005spectroscopic", r"\cite{valenti2005spectroscopic}"),
    ("van2009directly", r"\cite{van2009directly}"),
    ("martins2015evidence", r"\cite{martins2015evidence}"),
    ("brogi2013detection", r"\cite{brogi2013detection}"),
    ("birkby2017discovery", r"\cite{birkby2017discovery}"),
    ("bonomo2017gaps", r"\cite{bonomo2017gaps}"),
    ("stassun2017accurate", r"\cite{stassun2017accurate}"),
    ("barstow2017consistent", r"\cite{barstow2017consistent}"),
    ("charbonneau2002detection", r"\cite{charbonneau2002detection}"),
    ("vidal2003extended", r"\cite{vidal2003extended}"),
    ("macdonald2017hd209458b", r"\cite{macdonald2017hd209458b}"),
    ("mancini2020highly", r"\cite{mancini2020highly}"),
    ("yee2025super", r"\cite{yee2025super}"),
    ("barkaoui2024extended", r"\cite{barkaoui2024extended}"),
    ("gaudi2017giant", r"\cite{gaudi2017giant}"),
    ("yan2018extended", r"\cite{yan2018extended}"),
    ("hoeijmakers2018atomic", r"\cite{hoeijmakers2018atomic}"),
    ("hoeijmakers2019spectral", r"\cite{hoeijmakers2019spectral}"),
]

SOURCE_INDEX = {key: index + 1 for index, (key, _) in enumerate(SOURCE_REFERENCES)}

SYSTEM_SOURCE_KEYS = {
    "gj1132_b": ["xue2024jwst", "berta2015rocky", "swain2021detection"],
    "55cnc_e": ["bourrier201855", "moutou2026characterising", "hu2024secondary"],
    "gj1214_b": ["mahajan2024using", "charbonneau2009super", "kempton2023reflective"],
    "hd56414_b": ["giacalone2022hd", "otegi2020revisited"],
    "gj436_b": [
        "maxted2022analysis",
        "bourrier2018orbital",
        "maciejewski2015gj",
        "turner2016ground",
        "stevenson2010possible",
        "knutson2014featureless",
        "ehrenreich2015giant",
    ],
    "51peg_b": [
        "rosenthal2021california",
        "valenti2005spectroscopic",
        "van2009directly",
        "martins2015evidence",
        "brogi2013detection",
        "birkby2017discovery",
    ],
    "hd209458_b": [
        "rosenthal2021california",
        "bonomo2017gaps",
        "stassun2017accurate",
        "barstow2017consistent",
        "charbonneau2002detection",
        "vidal2003extended",
        "macdonald2017hd209458b",
    ],
    "wasp174_b": ["mancini2020highly"],
    "wasp193_b": ["yee2025super", "barkaoui2024extended"],
    "kelt9_b": ["gaudi2017giant", "yan2018extended", "hoeijmakers2018atomic", "hoeijmakers2019spectral"],
}

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


def iter_grouped_systems():
    for heading, keys in CATEGORY_BLOCKS:
        group = [(key, REAL_MASS_LOSS_REFERENCE_SYSTEMS[key]) for key in keys]
        yield heading, group


def composition_text(system_def: dict) -> str:
    return ", ".join(
        f"{latexify_species(species)}: {format_fraction(float(fraction))}"
        for species, fraction in system_def["planet"]["composition"].items()
    )


def source_marker_for_system(system_key: str) -> str:
    source_keys = SYSTEM_SOURCE_KEYS.get(system_key, [])
    if not source_keys:
        return ""
    marker = ",".join(str(SOURCE_INDEX[source_key]) for source_key in source_keys)
    return rf"\textsuperscript{{{marker}}}"


def source_footnote_parts() -> list[str]:
    return [
        rf"\textsuperscript{{{index}}}{reference_text}"
        for index, (_, reference_text) in enumerate(SOURCE_REFERENCES, start=1)
    ]


def build_table_tex() -> str:
    source_note = "Sources: " + "; ".join(source_footnote_parts()) + "."
    composition_note = (
        r"Atmospheric compositions are normalized, literature-informed proxy mixtures for the "
        r"mass-loss calculations; they are not direct observational abundance measurements. "
        r"Molecules are included only when present in the local molecule template library."
    )
    lines = [
        r"\begin{landscape}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{longtable}{llcccccccccccp{" + COMPOSITION_COL_WIDTH + r"}c}",
        r"\caption{Real reference systems adopted in the mass-loss study. The table lists the adopted stellar and planet parameters together with the atmospheric composition used in the calculations. The final column gives literature sources for the stellar and planetary parameters and for the atmospheric-composition basis; $\epsilon$, $\mu$, $P_0$, and composition are adopted model quantities.}\label{tab:real_mass_loss_reference_systems} \\",
        r"\toprule",
        r"Planet & Type & $T_{\rm eff}$ [K] & $R_\star$ & $M_\star$ & $v\sin i$ & $\epsilon$ & $R_{\rm p}$ & $M_{\rm p}$ & $a$ [AU] & $T$ [K] & $\mu$ & $P_0$ [bar] & Composition & Sources \\",
        r"\midrule",
        r"\endfirsthead",
        "",
        r"\toprule",
        r"Planet & Type & $T_{\rm eff}$ [K] & $R_\star$ & $M_\star$ & $v\sin i$ & $\epsilon$ & $R_{\rm p}$ & $M_{\rm p}$ & $a$ [AU] & $T$ [K] & $\mu$ & $P_0$ [bar] & Composition & Sources \\",
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule",
        r"\multicolumn{15}{r}{Continued on next page} \\",
        r"\midrule",
        r"\endfoot",
        "",
        r"\bottomrule",
        rf"\multicolumn{{15}}{{{FULL_WIDTH_NOTE_COL}}}{{\tiny {source_note}}} \\[0.2em]",
        rf"\multicolumn{{15}}{{{FULL_WIDTH_NOTE_COL}}}{{\tiny {composition_note}}} \\",
        r"\endlastfoot",
        "",
    ]

    for heading, group in iter_grouped_systems():
        lines.extend(
            [
                rf"\multicolumn{{15}}{{l}}{{\textbf{{{heading}}}}} \\",
                r"\midrule",
            ]
        )
        for system_key, system_def in group:
            name = system_def["planet"]["label"]
            star = system_def["star"]
            planet = system_def["planet"]
            composition_cell = composition_text(system_def)
            lines.append(
                " & ".join(
                    [
                        latex_escape(name).replace(" ", "~"),
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
                        source_marker_for_system(system_key),
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
    lines.append(r"\end{landscape}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(build_table_tex(), encoding="utf-8")
    print(f"Saved {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
