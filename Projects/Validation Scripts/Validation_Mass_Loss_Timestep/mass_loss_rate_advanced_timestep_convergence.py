import csv
import importlib.util
import pathlib
import sys
import time

import numpy as np
import astropy.units as u

sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))


ADVANCED_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "Atmospheric_Mass_Loss_Study"
    / "Mass_loss_rate_advanced.py"
)
ADVANCED_SPEC = importlib.util.spec_from_file_location(
    "mass_loss_rate_advanced_module",
    ADVANCED_SCRIPT_PATH,
)
if ADVANCED_SPEC is None or ADVANCED_SPEC.loader is None:
    raise ImportError(f"Could not load advanced mass-loss script from {ADVANCED_SCRIPT_PATH}")
advanced = importlib.util.module_from_spec(ADVANCED_SPEC)
ADVANCED_SPEC.loader.exec_module(advanced)


SYSTEM = advanced.AdvancedSystem(
    test_family="convergence",
    planet_key="mercury_like",
    star_key="G1",
    distance_au=0.387,
)

TIMESTEP_CASES = [
    {"label": "quadruple_step", "dt_fraction": 0.32, "dt_max_s": 8.0e3},
    {"label": "double_step", "dt_fraction": 0.16, "dt_max_s": 4.0e3},
    {"label": "baseline", "dt_fraction": 0.08, "dt_max_s": 2.0e3},
    {"label": "half_step", "dt_fraction": 0.04, "dt_max_s": 5.0e2},
    {"label": "quarter_step", "dt_fraction": 0.02, "dt_max_s": 1.0e2},
]

# Keep the spatial problem moderate so the timestep study is practical to run.
advanced.N_RHO = 10
advanced.N_X = 30
advanced.COLUMN_STEPS = 40
advanced.SELECTED_ATOMIC_SPECIES = ["K I"]

OUTPUT_DIR = (
    pathlib.Path(__file__).resolve().parent
    / "results"
    / "Tables"
    / "convergence"
)


def run_case(case: dict, exobase_rows: dict) -> dict:
    advanced.DT_FRACTION = float(case["dt_fraction"])
    advanced.DT_MAX_S = float(case["dt_max_s"])
    advanced.configure_base_module()

    planet_case = advanced.base_mass_loss.get_planet_template(SYSTEM.planet_key)
    species_list = advanced.selected_atomic_species(planet_case)
    planet = advanced.base_mass_loss.build_planet(planet_case)
    star = advanced.base_mass_loss.get_star(SYSTEM.star_key)
    system = advanced.base_mass_loss.PlanetarySystem(planet, star, SYSTEM.distance_au * u.AU)

    species_rows = []
    skipped = []
    case_start = time.perf_counter()
    print(
        f"Running {case['label']}: planet={SYSTEM.planet_key}, star={SYSTEM.star_key}, "
        f"distance={SYSTEM.distance_au:g} AU, DT_FRACTION={case['dt_fraction']}, DT_MAX_S={case['dt_max_s']}"
    )
    for species in species_list:
        try:
            row = advanced.mass_loss_for_species_advanced(
                SYSTEM.test_family,
                SYSTEM.planet_key,
                SYSTEM.star_key,
                SYSTEM.distance_au,
                species,
                planet_case,
                system,
                exobase_rows,
            )
        except Exception as exc:
            skipped.append((species, f"{type(exc).__name__}: {exc}"))
            continue
        species_rows.append(row)
        print(
            f"  {species}: Mdot={float(row['mass_loss_rate_g_s']):.3e} g/s, "
            f"max_steps_any_cell={int(row['max_steps_any_cell'])}"
        )

    total_row = advanced.total_row_from_species_rows(species_rows)
    if total_row is not None:
        print(
            f"  TOTAL: Mdot={float(total_row['mass_loss_rate_g_s']):.3e} g/s, "
            f"elapsed={time.perf_counter() - case_start:.1f} s"
        )
    return {
        "label": case["label"],
        "dt_fraction": float(case["dt_fraction"]),
        "dt_max_s": float(case["dt_max_s"]),
        "elapsed_s": time.perf_counter() - case_start,
        "species_rows": species_rows,
        "total_row": total_row,
        "skipped": skipped,
    }


def save_csv(results: list[dict], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_label",
        "dt_fraction",
        "dt_max_s",
        "species",
        "mass_loss_rate_g_s",
        "escaping_shell_mass_g",
        "mean_escape_time_s",
        "min_escape_time_s",
        "n_escape_cells",
        "max_steps_any_cell",
        "elapsed_s",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            rows = list(result["species_rows"])
            if result["total_row"] is not None:
                rows.append(result["total_row"])
            for row in rows:
                writer.writerow(
                    {
                        "case_label": result["label"],
                        "dt_fraction": result["dt_fraction"],
                        "dt_max_s": result["dt_max_s"],
                        "species": row.get("species", ""),
                        "mass_loss_rate_g_s": row.get("mass_loss_rate_g_s", np.nan),
                        "escaping_shell_mass_g": row.get("escaping_shell_mass_g", np.nan),
                        "mean_escape_time_s": row.get("mean_escape_time_s", np.nan),
                        "min_escape_time_s": row.get("min_escape_time_s", np.nan),
                        "n_escape_cells": row.get("n_escape_cells", ""),
                        "max_steps_any_cell": row.get("max_steps_any_cell", ""),
                        "elapsed_s": result["elapsed_s"],
                    }
                )


def save_summary(results: list[dict], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_total = results[-1]["total_row"]
    reference_mdot = (
        float(reference_total["mass_loss_rate_g_s"])
        if reference_total is not None
        else np.nan
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write("Advanced Mass-Loss Timestep Convergence\n")
        f.write("=====================================\n")
        f.write(f"planet: {SYSTEM.planet_key}\n")
        f.write(f"star: {SYSTEM.star_key}\n")
        f.write(
            f"stellar_teff_K: {advanced.base_mass_loss.infer_teff_from_star_template(SYSTEM.star_key)}\n"
        )
        f.write(f"distance_AU: {SYSTEM.distance_au}\n")
        f.write(f"method: advanced_trajectory_recomputed_acceleration\n")
        f.write(
            f"grid: N_RHO={advanced.N_RHO}, N_X={advanced.N_X}, "
            f"COLUMN_STEPS={advanced.COLUMN_STEPS}, RHO_GRID_POWER={advanced.RHO_GRID_POWER}\n"
        )
        f.write(
            "note: Reduced spatial grid and a single representative escaping species "
            "were used so the saved test isolates timestep sensitivity with practical runtime.\n"
        )
        f.write(f"species_subset: {advanced.SELECTED_ATOMIC_SPECIES}\n")
        f.write(
            f"reference_case: {results[-1]['label']} "
            f"(DT_FRACTION={results[-1]['dt_fraction']}, DT_MAX_S={results[-1]['dt_max_s']})\n"
        )
        f.write("\nTotals\n")
        f.write("------\n")

        for result in results:
            total_row = result["total_row"]
            mdot = float(total_row["mass_loss_rate_g_s"]) if total_row is not None else np.nan
            rel = np.nan
            if np.isfinite(reference_mdot) and reference_mdot != 0.0 and np.isfinite(mdot):
                rel = (mdot - reference_mdot) / reference_mdot
            hit_step_ceiling = any(
                int(row.get("max_steps_any_cell", 0)) >= advanced.MAX_STEPS
                for row in result["species_rows"]
            )
            f.write(
                f"{result['label']}: DT_FRACTION={result['dt_fraction']}, "
                f"DT_MAX_S={result['dt_max_s']}, "
                f"Mdot={mdot:.12e} g/s, "
                f"rel_to_reference={rel:.6e}, "
                f"elapsed={result['elapsed_s']:.1f} s, "
                f"step_ceiling_hit={hit_step_ceiling}\n"
            )

        f.write("\nPer-Species Totals\n")
        f.write("-----------------\n")
        for result in results:
            f.write(f"{result['label']}:\n")
            for row in result["species_rows"]:
                f.write(
                    f"  {row['species']}: "
                    f"Mdot={float(row['mass_loss_rate_g_s']):.12e} g/s, "
                    f"mean_t={float(row['mean_escape_time_s']):.12e} s, "
                    f"n_escape_cells={int(row['n_escape_cells'])}, "
                    f"max_steps_any_cell={int(row['max_steps_any_cell'])}\n"
                )
            if result["skipped"]:
                f.write("  skipped:\n")
                for species, reason in result["skipped"]:
                    f.write(f"    {species}: {reason}\n")


def main() -> None:
    exobase_rows = advanced.base_mass_loss.load_exobase_table(advanced.EXOBASE_TABLE)
    results = [run_case(case, exobase_rows) for case in TIMESTEP_CASES]

    slug = (
        f"{advanced.base_mass_loss.safe_name(SYSTEM.planet_key)}_"
        f"{advanced.base_mass_loss.safe_name(SYSTEM.star_key)}_"
        f"{SYSTEM.distance_au:g}AU_timestep_convergence"
    )
    csv_path = OUTPUT_DIR / f"{slug}.csv"
    txt_path = OUTPUT_DIR / f"{slug}.txt"
    save_csv(results, csv_path)
    save_summary(results, txt_path)
    print(f"Saved timestep convergence CSV to {csv_path}")
    print(f"Saved timestep convergence summary to {txt_path}")


if __name__ == "__main__":
    main()
