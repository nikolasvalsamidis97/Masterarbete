import csv
import pathlib

import matplotlib.pyplot as plt

from make_four_example_plot import (
    TABLES_BASE_DIR,
    EXOBASE_TABLE,
    OUTPUT_DIR,
    PLANET_ORDER,
    PLOT_OVERRIDES,
    apply_shared_figure_layout,
    load_exobase_heights,
    plot_example_panel,
)
from project_utils.r_beta1_table_sources import find_species_rbeta1_table


SUMMARY_CSV = OUTPUT_DIR.parent / "summary" / "r_beta1_summary_by_planet_species.csv"
OUTPUT_PDF = OUTPUT_DIR / "four_example_strongest_responders.pdf"
OUTPUT_PDF_TEMPLATE = OUTPUT_DIR / "four_example_{rank_label}_strongest_responders.pdf"
RANK_LABELS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
}


def load_summary_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing summary CSV: {path}. Run the r_beta1 summary generator first."
        )

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def strongest_responder_species(planet_key: str, rows: list[dict[str, str]], rank: int = 1) -> str:
    candidates = []
    for row in rows:
        if row.get("planet") != planet_key:
            continue

        coolest_teff = row.get("coolest_teff_below_exobase_K", "")
        distance_at_coolest = row.get("largest_distance_at_coolest_teff_AU", "")
        min_rbeta = row.get("min_r_beta1_over_Rp", "")
        if (
            not coolest_teff
            or coolest_teff.lower() == "nan"
            or not distance_at_coolest
            or distance_at_coolest.lower() == "nan"
            or not min_rbeta
            or min_rbeta.lower() == "nan"
        ):
            continue

        candidates.append(
            (
                float(coolest_teff),
                -float(distance_at_coolest),
                float(min_rbeta),
                row["species"],
            )
        )

    if not candidates:
        raise ValueError(f"No finite strongest-responder candidate found for planet={planet_key}")

    candidates.sort()
    if rank < 1 or rank > len(candidates):
        raise ValueError(
            f"Requested strongest-responder rank {rank} for {planet_key}, "
            f"but only {len(candidates)} ranked species are available."
        )
    return candidates[rank - 1][3]


def selected_table_paths(rows: list[dict[str, str]], rank: int) -> list[pathlib.Path]:
    table_paths: list[pathlib.Path] = []
    for planet_key in PLANET_ORDER:
        species = strongest_responder_species(planet_key, rows, rank=rank)
        table_path = find_species_rbeta1_table(TABLES_BASE_DIR, planet_key, species)
        if table_path is None:
            raise FileNotFoundError(f"Missing selected strongest-responder table: {table_path}")
        table_paths.append(table_path)
    return table_paths


def write_rank_plot(rank: int, summary_rows: list[dict[str, str]], exobase_heights: dict[tuple[str, str], float]) -> None:
    rank_label = RANK_LABELS.get(rank, f"rank{rank}")
    output_pdf = OUTPUT_PDF_TEMPLATE.with_name(
        OUTPUT_PDF_TEMPLATE.name.format(rank_label=rank_label)
    )
    table_paths = selected_table_paths(summary_rows, rank=rank)
    plt.rcParams.update(PLOT_OVERRIDES["font_sizes"])
    panel_width, panel_height = PLOT_OVERRIDES["figsize"]
    fig, axes = plt.subplots(2, 2, figsize=(2 * panel_width, 2 * panel_height), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    legend_handles = []
    legend_labels = []
    x_limits_list = []
    y_limits_list = []

    for ax, table_path in zip(axes_flat, table_paths):
        handles, labels, x_limits, y_limits = plot_example_panel(ax, table_path, exobase_heights)
        if handles and not legend_handles:
            legend_handles = handles
            legend_labels = labels
        if x_limits is not None:
            x_limits_list.append(x_limits)
        if y_limits is not None:
            y_limits_list.append(y_limits)
    if PLOT_OVERRIDES.get("tight_layout", True):
        apply_shared_figure_layout(
            fig,
            axes,
            legend_handles,
            legend_labels,
            x_limits_list,
            y_limits_list,
        )
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    if rank == 1:
        fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Selected {rank_label}-strongest responder species:")
    for table_path in table_paths:
        import plot_by_txt_file as plot_txt

        metadata, _, _ = plot_txt.parse_header_and_table(table_path)
        print(f"  {metadata.get('planet', table_path.parent.name)} -> {metadata.get('species', table_path.stem)}")
    print(f"Saved plot: {output_pdf}")
    if rank == 1:
        print(f"Saved plot: {OUTPUT_PDF}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = load_summary_rows(SUMMARY_CSV)
    exobase_heights = load_exobase_heights(EXOBASE_TABLE)

    write_rank_plot(1, summary_rows, exobase_heights)
    for rank in [2, 3, 4]:
        write_rank_plot(rank, summary_rows, exobase_heights)


if __name__ == "__main__":
    main()
