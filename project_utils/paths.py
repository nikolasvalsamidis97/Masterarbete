from __future__ import annotations

from pathlib import Path


def repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a file or directory inside it."""
    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "project_classes").is_dir() and (candidate / "Templates").is_dir():
            return candidate

    raise RuntimeError(f"Could not find repository root from {current}")


def project_dir(start: str | Path) -> Path:
    """Return the Projects/<name> directory containing a script."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if candidate.parent.name == "Projects":
            return candidate

    raise RuntimeError(f"Could not find project directory from {current}")


def project_results_dir(start: str | Path) -> Path:
    return project_dir(start) / "results"


def project_tables_dir(start: str | Path) -> Path:
    return project_results_dir(start) / "Tables"


def project_plots_dir(start: str | Path) -> Path:
    return project_results_dir(start) / "Plots"


def project_runtime_cache_dir(start: str | Path, name: str) -> Path:
    return project_tables_dir(start) / "runtime_cache" / name


def weight_cache_dir() -> Path:
    return repo_root(Path(__file__)) / ".cache" / "weight_cache"
