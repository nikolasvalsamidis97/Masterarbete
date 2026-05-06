import csv
import pathlib
import re
import sys

import astropy.constants as const
import astropy.units as u

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from project_func.Templates.Planets.planet_templates_updated import PLANET_TEMPLATES_UPDATED


OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "Tables"
PROPERTIES_CSV = OUTPUT_DIR / "planet_template_properties_table.csv"
PROPERTIES_TEX = OUTPUT_DIR / "planet_template_properties_table.tex"
COMPOSITIONS_CSV = OUTPUT_DIR / "planet_template_compositions_table.csv"
COMPOSITIONS_TEX = OUTPUT_DIR / "planet_template_compositions_table.tex"

CATEGORY_BLOCKS = [
    (
        "Rocky / terrestrial templates",
        [
            "mercury_like",
            "earth_like",
            "mars_like",
            "super_earth_rocky",
            "lava_world",
            "volatile_super_earth",
            "alkali_exosphere_rocky",
            "metal_rich_secondary",
        ],
    ),
    (
        "Sub-Neptunes / mini-Neptunes / Neptunes",
        [
            "mini_neptune_cool",
            "mini_neptune_warm",
            "sub_neptune",
            "warm_neptune",
            "hot_neptune",
            "super_puff",
        ],
    ),
    (
        "Gas giants",
        [
            "cold_jupiter",
            "warm_jupiter",
            "hot_jupiter",
            "inflated_hot_jupiter",
            "ultra_hot_jupiter",
        ],
    ),
]


def latex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def latexify_formula(text: str) -> str:
    escaped = latex_escape(text)
    escaped = re.sub(r"([A-Za-z])(\d+)", r"\1$_{\2}$", escaped)
    return escaped


def latexify_species(species: str) -> str:
    if " " in species:
        return latex_escape(species)
    return latexify_formula(species)


def latexify_note(text: str) -> str:
    escaped = latex_escape(text)
    escaped = re.sub(r"([A-Za-z])(\d+)", r"\1$_{\2}$", escaped)
    return escaped


def format_plain_number(value: float) -> str:
    text = f"{value:.3f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def format_scientific_latex(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(f"{value:e}".split("e")[1])
    mantissa = value / (10 ** exponent)
    mantissa_rounded = round(mantissa, 1)
    if abs(mantissa_rounded - 1.0) < 1e-12:
        return rf"$10^{{{exponent}}}$"
    mantissa_text = f"{mantissa_rounded:.1f}".rstrip("0").rstrip(".")
    return rf"${mantissa_text}\times10^{{{exponent}}}$"


def format_fraction(value: float, latex: bool) -> str:
    if value < 1e-3 and value > 0:
        return format_scientific_latex(value) if latex else f"{value:.0e}"
    if value < 0.01 and value > 0:
        text = f"{value:.3f}"
    elif value < 0.1:
        text = f"{value:.2f}"
    else:
        text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def format_pressure_bar(value: float) -> str:
    if value < 0.01 and value > 0:
        return format_scientific_latex(value)
    return format_plain_number(value)


def radius_string(template: dict) -> str:
    if template["category"] == "gas_giant":
        value = template["radius"].to_value(const.R_jup)
        return f"{format_plain_number(value)} $R_\\mathrm{{J}}$"
    value = template["radius"].to_value(const.R_earth)
    return f"{format_plain_number(value)} $R_\\oplus$"


def mass_string(template: dict) -> str:
    if template["category"] == "gas_giant":
        value = template["mass"].to_value(const.M_jup)
        return f"{format_plain_number(value)} $M_\\mathrm{{J}}$"
    value = template["mass"].to_value(const.M_earth)
    return f"{format_plain_number(value)} $M_\\oplus$"


def iter_grouped_templates():
    for heading, template_names in CATEGORY_BLOCKS:
        group = [
            (name, PLANET_TEMPLATES_UPDATED[name])
            for name in template_names
            if name in PLANET_TEMPLATES_UPDATED
        ]
        if group:
            yield heading, group


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_properties_rows() -> list[dict[str, str]]:
    rows = []
    for _, group in iter_grouped_templates():
        for key, template in group:
            rows.append(
                {
                    "template": key,
                    "category": template["category"],
                    "radius": radius_string(template),
                    "mass": mass_string(template),
                    "T_K": format_plain_number(template["T"].to_value(u.K)),
                    "mu": format_plain_number(template["mu"].value),
                    "P0_bar": format_plain_number(template["P0"].to_value(u.bar)),
                    "notes": template["notes"],
                }
            )
    return rows


def build_composition_rows() -> list[dict[str, str]]:
    rows = []
    for _, group in iter_grouped_templates():
        for key, template in group:
            rows.append(
                {
                    "template": key,
                    "category": template["category"],
                    "composition": ", ".join(
                        f"{species}: {format_fraction(float(fraction), latex=False)}"
                        for species, fraction in template["composition"].items()
                    ),
                }
            )
    return rows


def write_compositions_tex(path: pathlib.Path) -> None:
    lines = [
        r"\begin{longtable}{lll}",
        r"\caption{Adopted atmospheric compositions for the planet templates. Abundances are given as mixing ratios.}",
        r"\label{tab:planet_templates_compositions} \\",
        r"\toprule",
        r"Template & Category & Composition \\",
        r"\midrule",
        r"\endfirsthead",
        "",
        r"\toprule",
        r"Template & Category & Composition \\",
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule",
        r"\multicolumn{3}{r}{Continued on next page} \\",
        r"\midrule",
        r"\endfoot",
        "",
        r"\bottomrule",
        r"\endlastfoot",
        "",
    ]

    for heading, group in iter_grouped_templates():
        lines.extend([
            rf"\multicolumn{{3}}{{l}}{{\textbf{{{heading}}}}} \\",
            r"\midrule",
        ])
        for key, template in group:
            composition = ", ".join(
                f"{latexify_species(species)}: {format_fraction(float(fraction), latex=True)}"
                for species, fraction in template["composition"].items()
            )
            lines.append(
                f"{latex_escape(key)} & {latex_escape(template['category'])} & {composition} \\\\"
            )
        lines.append(r"\midrule")
        lines.append("")

    lines.append(r"\end{longtable}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_properties_tex(path: pathlib.Path) -> None:
    lines = [
        r"\begin{longtable}{llcccccl}",
        r"\caption{Planet templates used in this work, grouped by atmospheric or structural category. The table lists the adopted radius, mass, atmospheric temperature, mean molecular weight, reference pressure, and a short description for each template.}",
        r"\label{tab:planet_templates_properties} \\",
        r"\toprule",
        r"Template & Category & $R$ & $M$ & $T$ [K] & $\mu$ & $P_0$ [bar] & Notes \\",
        r"\midrule",
        r"\endfirsthead",
        "",
        r"\toprule",
        r"Template & Category & $R$ & $M$ & $T$ [K] & $\mu$ & $P_0$ [bar] & Notes \\",
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule",
        r"\multicolumn{8}{r}{Continued on next page} \\",
        r"\midrule",
        r"\endfoot",
        "",
        r"\bottomrule",
        r"\endlastfoot",
        "",
    ]

    for heading, group in iter_grouped_templates():
        lines.extend([
            rf"\multicolumn{{8}}{{l}}{{\textbf{{{heading}}}}} \\",
            r"\midrule",
        ])
        for key, template in group:
            lines.append(
                f"{latex_escape(key)} & {latex_escape(template['category'])} & "
                f"{radius_string(template)} & {mass_string(template)} & "
                f"{format_plain_number(template['T'].to_value(u.K))} & "
                f"{format_plain_number(template['mu'].value)} & "
                f"{format_pressure_bar(template['P0'].to_value(u.bar))} & "
                f"{latexify_note(template['notes'])} \\\\"
            )
        lines.append(r"\midrule")
        lines.append("")

    lines.append(r"\end{longtable}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    properties_rows = build_properties_rows()
    composition_rows = build_composition_rows()

    write_csv(PROPERTIES_CSV, ["template", "category", "radius", "mass", "T_K", "mu", "P0_bar", "notes"], properties_rows)
    write_csv(COMPOSITIONS_CSV, ["template", "category", "composition"], composition_rows)
    write_properties_tex(PROPERTIES_TEX)
    write_compositions_tex(COMPOSITIONS_TEX)

    print(f"Saved {PROPERTIES_CSV}")
    print(f"Saved {PROPERTIES_TEX}")
    print(f"Saved {COMPOSITIONS_CSV}")
    print(f"Saved {COMPOSITIONS_TEX}")


if __name__ == "__main__":
    main()
