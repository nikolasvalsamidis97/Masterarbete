from __future__ import annotations

import pathlib


EXOBASE_TABLE_NAME = "exobase_table_planets.csv"


def canonical_exobase_table_path(repo_root: pathlib.Path) -> pathlib.Path:
    return (
        repo_root
        / "Projects"
        / "Atmospheric_Exobase_Calculation"
        / "results"
        / "Tables"
        / "Exobase"
        / EXOBASE_TABLE_NAME
    )


def legacy_exobase_table_path(repo_root: pathlib.Path) -> pathlib.Path:
    return (
        repo_root
        / "Projects"
        / "Atmospheric_Exobase_Calculation"
        / "results"
        / "Plots"
        / "Exobase"
        / EXOBASE_TABLE_NAME
    )


def resolve_exobase_table_path(repo_root: pathlib.Path) -> pathlib.Path:
    canonical = canonical_exobase_table_path(repo_root)
    if canonical.exists():
        return canonical

    legacy = legacy_exobase_table_path(repo_root)
    if legacy.exists():
        return legacy

    return canonical
