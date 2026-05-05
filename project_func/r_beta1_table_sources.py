from __future__ import annotations

import pathlib


DEFAULT_RBETA1_ROOT_NAMES = (
    "r_at_beta1",
    "r_at_beta1_atoms",
    "r_at_beta1_molecules",
)


def safe_name(value: str) -> str:
    return str(value).replace(" ", "").replace("/", "_")


def existing_rbeta1_roots(
    base_dir: pathlib.Path,
    root_names: tuple[str, ...] = DEFAULT_RBETA1_ROOT_NAMES,
) -> list[pathlib.Path]:
    return [base_dir / root_name for root_name in root_names if (base_dir / root_name).exists()]


def _preferred_path(candidates: list[tuple[pathlib.Path, int]]) -> pathlib.Path:
    return max(
        candidates,
        key=lambda item: (item[0].stat().st_mtime_ns, -item[1]),
    )[0]


def discover_rbeta1_table_files(
    base_dir: pathlib.Path,
    root_names: tuple[str, ...] = DEFAULT_RBETA1_ROOT_NAMES,
) -> list[pathlib.Path]:
    roots = existing_rbeta1_roots(base_dir, root_names=root_names)
    if not roots:
        return []

    grouped_candidates: dict[pathlib.Path, list[tuple[pathlib.Path, int]]] = {}
    for priority_index, root in enumerate(roots):
        for table_path in root.glob("*_r_beta1/*.txt"):
            relative_key = table_path.relative_to(root)
            grouped_candidates.setdefault(relative_key, []).append((table_path, priority_index))

    chosen_paths = [_preferred_path(candidates) for candidates in grouped_candidates.values()]
    return sorted(chosen_paths, key=lambda path: path.as_posix())


def resolve_rbeta1_table_file(
    base_dir: pathlib.Path,
    relative_path: str | pathlib.Path,
    root_names: tuple[str, ...] = DEFAULT_RBETA1_ROOT_NAMES,
) -> pathlib.Path | None:
    relative_path = pathlib.Path(relative_path)
    candidates: list[tuple[pathlib.Path, int]] = []
    for priority_index, root in enumerate(existing_rbeta1_roots(base_dir, root_names=root_names)):
        candidate = root / relative_path
        if candidate.exists():
            candidates.append((candidate, priority_index))
    if not candidates:
        return None
    return _preferred_path(candidates)


def find_species_rbeta1_table(
    base_dir: pathlib.Path,
    planet_key: str,
    species: str,
    root_names: tuple[str, ...] = DEFAULT_RBETA1_ROOT_NAMES,
) -> pathlib.Path | None:
    relative_path = pathlib.Path(f"{planet_key}_r_beta1") / f"{safe_name(species)}_r_beta1.txt"
    return resolve_rbeta1_table_file(base_dir, relative_path, root_names=root_names)
