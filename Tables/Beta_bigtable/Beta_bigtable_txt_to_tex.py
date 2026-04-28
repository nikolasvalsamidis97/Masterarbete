import math
import pathlib
from typing import Iterable, List

import numpy as np
import pandas as pd


RAW_FILE = pathlib.Path(__file__).resolve().parent / "beta_bigtable_atoms.txt"
OUTPUT_TEX = pathlib.Path(__file__).resolve().parent / "beta_bigtable_atoms_to_log.tex"
EXCLUDED_TEFFS_K = {19000}


def broadening_label(b_kms: float) -> str:
    if float(b_kms) == 0.0:
        return "0"
    if float(b_kms).is_integer():
        return str(int(float(b_kms)))
    return f"{float(b_kms):g}"


def latex_species_name(species: str) -> str:
    parts = str(species).split()
    if len(parts) != 2:
        return str(species).replace("_", r"\_")

    element, stage = parts
    if stage == "I":
        return element
    if stage == "II":
        return rf"{element}$^+$"
    if stage == "III":
        return rf"{element}$^{{++}}$"
    return species.replace(" ", "~")


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


def build_row_values_tau0(row: pd.Series) -> List[str]:
    log_beta_tau0 = safe_log10(float(row["beta_tau0"]))
    log_err_tau0 = symmetric_log10_error(float(row["beta_tau0"]), float(row["beta_err_tau0"]))
    beta_tau0_text, err_tau0_text = format_value_and_error(log_beta_tau0, log_err_tau0)
    return [beta_tau0_text, err_tau0_text, "0"]


def build_row_values_tau1(row: pd.Series) -> List[str]:
    log_beta_tau1 = safe_log10(float(row["beta_tau1"]))
    log_n_tau1 = safe_log10(float(row["n_tau1_cm2"]))
    log_err_tau1 = symmetric_log10_error(float(row["beta_tau1"]), float(row["beta_err_tau1"]))
    beta_tau1_text, err_tau1_text = format_value_and_error(log_beta_tau1, log_err_tau1)
    n_tau1_text = format_compact_or_dash(log_n_tau1, decimals=2)
    return [beta_tau1_text, err_tau1_text, n_tau1_text]


def build_col_spec(n_teff: int) -> str:
    return "l" + "".join(r"!{\vrule width 0.25pt}ccc" for _ in range(n_teff))


def make_longtable_block(
    rows: List[str],
    teff_headers: List[int],
    block_title: str,
    caption: str,
    label: str,
) -> str:
    n_cols = 1 + 3 * len(teff_headers)
    col_spec = build_col_spec(len(teff_headers))
    top_header = " & ".join(
        ["Ion"] + [rf"\multicolumn{{3}}{{c}}{{{teff} K}}" for teff in teff_headers]
    )
    sub_header = " & ".join(
        [""] + [r"$\log\beta$ & $\sigma_{\log\beta}$ & $N_{\rm col}$" for _ in teff_headers]
    )

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


def build_tex(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    star_rows = unique_star_rows(df)
    teff_order = star_rows["teff_k"].tolist()
    teff_chunks = [chunk for chunk in split_in_halves(teff_order) if chunk]
    species_order = list(dict.fromkeys(df["species"].tolist()))
    b_order = list(dict.fromkeys(df["b_label"].tolist()))

    blocks: List[str] = []
    for b_index, b_label in enumerate(b_order):
        subset_b = df[df["b_label"] == b_label]
        b_blocks: List[str] = []
        for part_index, teff_chunk in enumerate(teff_chunks, start=1):
            rows = []
            for species in species_order:
                subset_species = subset_b[subset_b["species"] == species]
                row_tau0 = []
                row_tau1 = []
                for teff_k in teff_chunk:
                    row = subset_species[subset_species["teff_k"] == teff_k]
                    if row.empty:
                        row_tau0.extend(["-", "-", "-"])
                        row_tau1.extend(["-", "-", "-"])
                    else:
                        row_tau0.extend(build_row_values_tau0(row.iloc[0]))
                        row_tau1.extend(build_row_values_tau1(row.iloc[0]))
                rows.append("{} & {} \\\\".format(latex_species_name(species), " & ".join(row_tau0)))
                rows.append("{} & {} \\\\".format("", " & ".join(row_tau1)))
                rows.append(r"\specialrule{0.25pt}{0pt}{0pt}")

            block_title = (
                f"Atoms: $b = {b_label}$ km s$^{{-1}}$ "
                f"(first row: $N_{{\\tau=0}}$, second row: $\\log N_{{\\tau=1}}$; Part {part_index})"
            )
            caption = (
                "Atomic $\\log\\beta$-values, logarithmic $\\beta$-errors, and "
                f"species-wise column densities for selected stellar temperatures at "
                f"$b={b_label}\\ \\mathrm{{km\\,s^{{-1}}}}$. "
                "The logarithmic uncertainty is computed as "
                "$\\sigma_{\\log\\beta} = \\frac{1}{2}\\left|\\log_{10}\\left(\\frac{\\beta + \\Delta\\beta}{\\beta - \\Delta\\beta}\\right)\\right|$. "
                f"For each species, the first row gives the $\\tau=0$ values and the second row gives the "
                f"$\\tau=1$ values with $N_{{\\rm col}}$ written as $\\log N_{{\\rm col}}$ (Part {part_index})."
            )
            label = f"tab: beta_bigtable_atoms_b{b_label}_part{part_index}"
            b_blocks.append(
                "\n".join(
                    [
                        "",
                        make_longtable_block(
                            rows,
                            teff_chunk,
                            block_title=block_title,
                            caption=caption,
                            label=label,
                        ),
                        "",
                    ]
                )
            )
        blocks.append("\n".join(b_blocks))
        if b_index < len(b_order) - 1:
            blocks.append("\n\\clearpage\n")

    return "\n".join(blocks)


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw atom table not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE, sep="\t")
    df = df[df["teff_k"].notna()].copy()
    df["teff_k"] = df["teff_k"].astype(int)
    df = df[~df["teff_k"].isin(EXCLUDED_TEFFS_K)].copy()

    tex = build_tex(df)
    OUTPUT_TEX.write_text(tex, encoding="utf-8")
    print(f"Read raw table: {RAW_FILE}")
    print(f"Saved TeX table: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
