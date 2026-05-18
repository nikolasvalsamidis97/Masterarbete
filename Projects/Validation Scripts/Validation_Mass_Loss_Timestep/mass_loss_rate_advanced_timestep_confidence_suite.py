import csv
from dataclasses import dataclass
import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

import mass_loss_rate_advanced_timestep_convergence as base


@dataclass(frozen=True)
class BenchmarkCase:
    planet_key: str
    star_key: str
    distance_au: float
    species: str


BENCHMARKS = [
    BenchmarkCase("mercury_like", "G1", 0.387, "K I"),
    BenchmarkCase("super_earth_rocky", "G1", 0.08, "Na I"),
    BenchmarkCase("super_puff", "F8", 0.2514, "H I"),
    BenchmarkCase("hot_neptune", "M1", 0.0291, "H I"),
    BenchmarkCase("inflated_hot_jupiter", "A0", 0.05, "Na I"),
]

OUTPUT_DIR = base.OUTPUT_DIR


def run_benchmark(case: BenchmarkCase, exobase_rows: dict) -> list[dict]:
    base.SYSTEM = base.advanced.AdvancedSystem(
        test_family="convergence",
        planet_key=case.planet_key,
        star_key=case.star_key,
        distance_au=case.distance_au,
    )
    base.advanced.SELECTED_ATOMIC_SPECIES = [case.species]
    results = [base.run_case(step_case, exobase_rows) for step_case in base.TIMESTEP_CASES]

    slug = (
        f"{base.advanced.base_mass_loss.safe_name(case.planet_key)}_"
        f"{base.advanced.base_mass_loss.safe_name(case.star_key)}_"
        f"{case.distance_au:g}AU_timestep_convergence"
    )
    base.save_csv(results, OUTPUT_DIR / f"{slug}.csv")
    base.save_summary(results, OUTPUT_DIR / f"{slug}.txt")
    return results


def summarize_case(case: BenchmarkCase, results: list[dict]) -> dict:
    reference_total = results[-1]["total_row"]
    reference_mdot = float(reference_total["mass_loss_rate_g_s"]) if reference_total is not None else np.nan
    rows_by_label = {result["label"]: result for result in results}

    def rel(label: str) -> float:
        total_row = rows_by_label[label]["total_row"]
        mdot = float(total_row["mass_loss_rate_g_s"]) if total_row is not None else np.nan
        if not np.isfinite(reference_mdot) or reference_mdot == 0.0 or not np.isfinite(mdot):
            return np.nan
        return (mdot - reference_mdot) / reference_mdot

    return {
        "planet_key": case.planet_key,
        "star_key": case.star_key,
        "distance_au": case.distance_au,
        "species": case.species,
        "quadruple_rel": rel("quadruple_step"),
        "double_rel": rel("double_step"),
        "baseline_rel": rel("baseline"),
        "half_rel": rel("half_step"),
        "reference_mdot_g_s": reference_mdot,
        "quadruple_step_ceiling_hit": any(
            int(row.get("max_steps_any_cell", 0)) >= base.advanced.MAX_STEPS
            for row in rows_by_label["quadruple_step"]["species_rows"]
        ),
    }


def save_suite_summary(summary_rows: list[dict]) -> None:
    csv_path = OUTPUT_DIR / "coarse_timestep_confidence_suite.csv"
    txt_path = OUTPUT_DIR / "coarse_timestep_confidence_suite.txt"

    fieldnames = [
        "planet_key",
        "star_key",
        "distance_au",
        "species",
        "quadruple_rel",
        "double_rel",
        "baseline_rel",
        "half_rel",
        "reference_mdot_g_s",
        "quadruple_step_ceiling_hit",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    quadruple_abs = [
        abs(float(row["quadruple_rel"]))
        for row in summary_rows
        if np.isfinite(row["quadruple_rel"])
    ]
    double_abs = [
        abs(float(row["double_rel"]))
        for row in summary_rows
        if np.isfinite(row["double_rel"])
    ]

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Advanced Mass-Loss Coarse-Timestep Confidence Suite\n")
        f.write("==================================================\n")
        f.write(
            "All cases use the reduced convergence grid from the timestep benchmark "
            "script and compare each coarse timestep against the quarter-step "
            "reference (DT_FRACTION=0.02, DT_MAX_S=100 s).\n\n"
        )
        for row in summary_rows:
            f.write(
                f"{row['planet_key']} / {row['star_key']} / {row['distance_au']} AU / {row['species']}: "
                f"quadruple_rel={float(row['quadruple_rel']):.6e}, "
                f"double_rel={float(row['double_rel']):.6e}, "
                f"baseline_rel={float(row['baseline_rel']):.6e}, "
                f"half_rel={float(row['half_rel']):.6e}, "
                f"step_ceiling_hit={row['quadruple_step_ceiling_hit']}\n"
            )

        if quadruple_abs:
            f.write("\nSuite maxima\n")
            f.write("-----------\n")
            f.write(f"max_abs_quadruple_rel={max(quadruple_abs):.6e}\n")
            f.write(f"median_abs_quadruple_rel={np.median(quadruple_abs):.6e}\n")
        if double_abs:
            f.write(f"max_abs_double_rel={max(double_abs):.6e}\n")
            f.write(f"median_abs_double_rel={np.median(double_abs):.6e}\n")

    print(f"Saved suite CSV to {csv_path}")
    print(f"Saved suite summary to {txt_path}")


def main() -> None:
    exobase_rows = base.advanced.base_mass_loss.load_exobase_table(base.advanced.EXOBASE_TABLE)
    summary_rows = []
    for case in BENCHMARKS:
        results = run_benchmark(case, exobase_rows)
        summary_rows.append(summarize_case(case, results))
    save_suite_summary(summary_rows)


if __name__ == "__main__":
    main()
