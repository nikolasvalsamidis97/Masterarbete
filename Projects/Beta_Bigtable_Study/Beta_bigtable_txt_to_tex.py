import math
import pathlib
import re
from typing import Callable, Iterable, List

import numpy as np
import pandas as pd


OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "results" / "Tables"
RAW_ATOMS_FILE = OUTPUT_DIR / "beta_bigtable_atoms.txt"
RAW_MOLECULES_FILE = OUTPUT_DIR / "beta_bigtable_molecules.txt"
OUTPUT_ATOMS_TEX = OUTPUT_DIR / "beta_bigtable_atoms_to_log.tex"
OUTPUT_MOLECULES_TEX = OUTPUT_DIR / "beta_bigtable_molecules_to_log.tex"
EXCLUDED_TEFFS_K = {19000}
MIN_DISPLAY_LOG_BETA_SIGMA = 0.01


def latex_species_name(species: str) -> str:
    parts = str(species).split()
    if len(parts) == 2:
        element, stage = parts
        if stage == "I":
            return element
        if stage == "II":
            return rf"{element}$^+$"
        if stage == "III":
            return rf"{element}$^{{++}}$"
        return species.replace(" ", "~")

    text = str(species).replace("_", r"\_")
    return re.sub(r"(\d+)", lambda match: rf"$_{{{match.group(1)}}}$", text)


def safe_log10(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return np.nan
    return float(np.log10(value))


def symmetric_log10_error(value: float, err: float) -> float:
    if not np.isfinite(value) or not np.isfinite(err):
        return np.nan
    if value <= 0.0 or err < 0.0:
        return np.nan
    lower = value - err
    upper = value + err
    if lower <= 0.0 or upper <= 0.0:
        return np.nan
    return float(0.5 * abs(np.log10(upper / lower)))


def format_fixed_or_dash(value: float, decimals: int) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.{decimals}f}"


def format_tex_integer(value: int | float) -> str:
    rounded = int(round(float(value)))
    if abs(rounded) >= 10000:
        return f"{rounded:,}".replace(",", r"\,")
    return str(rounded)


def format_compact_or_dash(value: float, decimals: int = 2) -> str:
    if not np.isfinite(value):
        return "-"
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def decimals_from_error(err: float, min_decimals: int = 2, max_decimals: int = 8) -> int:
    if not np.isfinite(err) or err <= 0.0:
        return min_decimals
    decimals = int(math.ceil(-math.log10(abs(err))))
    return max(min_decimals, min(max_decimals, decimals))


def format_value_and_error(value: float, err: float) -> tuple[str, str]:
    if not np.isfinite(value):
        return "-", "-"
    if not np.isfinite(err):
        return format_fixed_or_dash(value, 2), "-"

    decimals = decimals_from_error(err)
    return format_fixed_or_dash(value, decimals), format_fixed_or_dash(err, decimals)


def build_atom_row_values(row: pd.Series) -> List[str]:
    log_beta = safe_log10(float(row["beta"]))
    log_beta_err = symmetric_log10_error(float(row["beta"]), float(row["beta_err"]))
    log_n_half = safe_log10(float(row["n_half_beta_cm2"]))
    beta_text = format_fixed_or_dash(log_beta, 2)
    beta_err_display = (
        max(log_beta_err, MIN_DISPLAY_LOG_BETA_SIGMA)
        if np.isfinite(log_beta_err)
        else np.nan
    )
    beta_err_text = format_fixed_or_dash(beta_err_display, 2)
    n_half_text = format_compact_or_dash(log_n_half, decimals=2)
    return [beta_text, beta_err_text, n_half_text]


def build_molecule_row_values(row: pd.Series) -> List[str]:
    log_beta = safe_log10(float(row["beta"]))
    log_n_half = safe_log10(float(row["n_half_beta_cm2"]))
    return [
        format_compact_or_dash(log_beta, decimals=2),
        format_compact_or_dash(log_n_half, decimals=2),
    ]


def build_col_spec(n_teff: int, n_subcols: int) -> str:
    return "l" + "".join(r"!{\vrule width 0.25pt}" + ("c" * n_subcols) for _ in range(n_teff))


def make_longtable_block(
    rows: List[str],
    teff_headers: List[int],
    species_header: str,
    subheaders: List[str],
    block_title: str,
    caption: str,
    label: str,
) -> str:
    n_subcols = len(subheaders)
    n_cols = 1 + n_subcols * len(teff_headers)
    col_spec = build_col_spec(len(teff_headers), n_subcols)
    top_header = " & ".join(
        [species_header]
        + [rf"\multicolumn{{{n_subcols}}}{{c}}{{{format_tex_integer(teff)} K}}" for teff in teff_headers]
    )
    sub_header = " & ".join([""] + subheaders * len(teff_headers))

    lines = [
        f"\\begin{{longtable}}{{{col_spec}}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}} \\\\",
        "\\toprule",
        f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{block_title}}}}} \\\\",
        "\\midrule",
        top_header + " \\\\",
        sub_header + " \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        top_header + " \\\\",
        sub_header + " \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        f"\\multicolumn{{{n_cols}}}{{r}}{{Continued on next page}} \\\\",
        "\\midrule",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
        *rows,
        "\\end{longtable}",
    ]
    return "\n".join(lines)


def unique_star_rows(df: pd.DataFrame) -> pd.DataFrame:
    star_rows = df[["star_key", "teff_k"]].drop_duplicates().copy()
    star_rows = star_rows[~star_rows["teff_k"].isin(EXCLUDED_TEFFS_K)]
    return star_rows.reset_index(drop=True)


def split_in_halves(items: Iterable[int]) -> List[List[int]]:
    item_list = list(items)
    mid = int(math.ceil(len(item_list) / 2))
    return [item_list[:mid], item_list[mid:]]


def load_raw_table(raw_file: pathlib.Path) -> pd.DataFrame:
    if not raw_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(raw_file, sep="\t")
    if df.empty:
        return df

    required_columns = {"species", "b_label", "teff_k", "beta", "n_half_beta_cm2"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Raw table {raw_file.name} is missing required columns: {sorted(missing)}")

    df = df[df["teff_k"].notna()].copy()
    df["teff_k"] = df["teff_k"].astype(int)
    df = df[~df["teff_k"].isin(EXCLUDED_TEFFS_K)].copy()
    return df


def build_category_tex(
    df: pd.DataFrame,
    species_header: str,
    subheaders: List[str],
    row_builder: Callable[[pd.Series], List[str]],
    block_title_builder: Callable[[str, int], str],
    caption_builder: Callable[[str, int], str],
    label_prefix: str,
) -> str:
    if df.empty:
        return ""

    star_rows = unique_star_rows(df)
    teff_order = star_rows["teff_k"].tolist()
    teff_chunks = [chunk for chunk in split_in_halves(teff_order) if chunk]
    species_order = list(dict.fromkeys(df["species"].tolist()))
    b_order = list(dict.fromkeys(df["b_label"].tolist()))
    n_subcols = len(subheaders)

    blocks: List[str] = []
    for b_index, b_label in enumerate(b_order):
        subset_b = df[df["b_label"] == b_label]
        b_blocks: List[str] = []
        for part_index, teff_chunk in enumerate(teff_chunks, start=1):
            rows = []
            for species in species_order:
                subset_species = subset_b[subset_b["species"] == species]
                row_values: List[str] = []
                for teff_k in teff_chunk:
                    row = subset_species[subset_species["teff_k"] == teff_k]
                    if row.empty:
                        row_values.extend(["-"] * n_subcols)
                    else:
                        row_values.extend(row_builder(row.iloc[0]))

                if all(value == "-" for value in row_values):
                    continue

                rows.append("{} & {} \\\\".format(latex_species_name(species), " & ".join(row_values)))

            if not rows:
                continue

            b_blocks.append(
                "\n".join(
                    [
                        "",
                        make_longtable_block(
                            rows,
                            teff_chunk,
                            species_header=species_header,
                            subheaders=subheaders,
                            block_title=block_title_builder(b_label, part_index),
                            caption=caption_builder(b_label, part_index),
                            label=f"{label_prefix}_b{b_label}_part{part_index}",
                        ),
                        "",
                    ]
                )
            )

        if b_blocks:
            blocks.append("\n".join(b_blocks))
        if b_index < len(b_order) - 1 and b_blocks:
            blocks.append("\n\\clearpage\n")

    return "\n".join(blocks)


def build_atoms_tex(df: pd.DataFrame) -> str:
    return build_category_tex(
        df,
        species_header="Ion",
        subheaders=[
            r"$\log_{10}(\beta)$",
            r"$\sigma_{\log_{10}(\beta)}$",
            r"$\log_{10}(N_{\beta/2})$",
        ],
        row_builder=build_atom_row_values,
        block_title_builder=lambda b_label, part_index: (
            f"Atoms: $b = {b_label}$ km s$^{{-1}}$ "
            f"($\\beta$ at $N_{{\\rm col}}=0$; Part {part_index})"
        ),
        caption_builder=lambda b_label, part_index: (
            "Atomic $\\log_{10}(\\beta)$-values at $N_{\\rm col}=0$, "
            "$\\log_{10}(\\beta)$-errors, and the column density where "
            "$\\beta = \\beta(N_{\\rm col}=0)/2$, written as "
            "$\\log_{10}(N_{\\beta/2})$, for selected stellar temperatures at "
            f"$b={b_label}\\ \\mathrm{{km\\,s^{{-1}}}}$ (Part {part_index})."
        ),
        label_prefix="tab: beta_bigtable_atoms",
    )


def build_molecules_tex(df: pd.DataFrame) -> str:
    return build_category_tex(
        df,
        species_header="Molecule",
        subheaders=[r"$\log\beta$", r"$\log N_{\beta/2}$"],
        row_builder=build_molecule_row_values,
        block_title_builder=lambda b_label, part_index: (
            f"Molecules: $b = {b_label}$ km s$^{{-1}}$ "
            f"($\\beta$ at $N_{{\\rm col}}=0$; Part {part_index})"
        ),
        caption_builder=lambda b_label, part_index: (
            "Molecular $\\log\\beta$-values at $N_{\\rm col}=0$ and the column "
            "density where $\\beta = \\beta(N_{\\rm col}=0)/2$, written as "
            "$\\log N_{\\beta/2}$, for selected stellar temperatures at "
            f"$b={b_label}\\ \\mathrm{{km\\,s^{{-1}}}}$ (Part {part_index})."
        ),
        label_prefix="tab: beta_bigtable_molecules",
    )


def main() -> None:
    atoms_df = load_raw_table(RAW_ATOMS_FILE)
    if not atoms_df.empty:
        atoms_tex = build_atoms_tex(atoms_df)
        OUTPUT_ATOMS_TEX.write_text(atoms_tex, encoding="utf-8")
        print(f"Read raw atom table: {RAW_ATOMS_FILE}")
        print(f"Saved atom TeX table: {OUTPUT_ATOMS_TEX}")
    else:
        print(f"Skipping atom TeX build: no usable raw table at {RAW_ATOMS_FILE}")

    molecules_df = load_raw_table(RAW_MOLECULES_FILE)
    if not molecules_df.empty:
        molecules_tex = build_molecules_tex(molecules_df)
        OUTPUT_MOLECULES_TEX.write_text(molecules_tex, encoding="utf-8")
        print(f"Read raw molecule table: {RAW_MOLECULES_FILE}")
        print(f"Saved molecule TeX table: {OUTPUT_MOLECULES_TEX}")
    else:
        print(f"Skipping molecule TeX build: no usable raw table at {RAW_MOLECULES_FILE}")


if __name__ == "__main__":
    main()
