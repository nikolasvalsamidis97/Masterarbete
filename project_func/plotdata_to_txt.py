

from __future__ import annotations

import pathlib
from typing import Iterable, Mapping, Sequence

import numpy as np


"""
Generic helper for saving computed plot data to a self-describing text table.

Purpose
-------
This file should only store DATA information, not plotting style choices.
The idea is that an expensive compute script writes one text file, and a
separate plotting script later reads that file and decides how to plot it.

Supported structure
-------------------
- One x-axis column
- One or more y-series columns
- Minimal metadata header describing what the data represent

Recommended header fields
-------------------------
Required:
- dataset_name
- x_label
- x_unit
- y_label
- y_unit

Strongly recommended when multiple y-series exist:
- series_label
- series_unit
- series_values

This is especially important for cases like "r_beta1 / R_p vs Teff" with one
curve per orbital distance.
"""


DEFAULT_FLOAT_FORMAT = ".10g"



def _as_float_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr



def _as_2d_float_array(values: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {arr.shape}")
    return arr



def _format_value(value: float, float_format: str) -> str:
    if not np.isfinite(value):
        return "nan"
    return format(float(value), float_format)



def _default_series_column_name(y_label: str, series_value: str | float) -> str:
    clean_y = str(y_label).strip().replace(" ", "_")
    clean_series = str(series_value).strip().replace(" ", "_")
    return f"{clean_y}__{clean_series}"



def save_plotdata_txt(
    output_path: str | pathlib.Path,
    *,
    dataset_name: str,
    x_label: str,
    x_unit: str,
    y_label: str,
    y_unit: str,
    x_values: Sequence[float] | np.ndarray,
    y_matrix: Sequence[Sequence[float]] | np.ndarray,
    series_values: Sequence[str | float] | None = None,
    series_label: str | None = None,
    series_unit: str | None = None,
    column_names: Sequence[str] | None = None,
    extra_metadata: Mapping[str, str] | None = None,
    float_format: str = DEFAULT_FLOAT_FORMAT,
) -> pathlib.Path:
    """
    Save plot data to a text file with a compact, data-focused header.

    Parameters
    ----------
    output_path
        Destination .txt file.
    dataset_name
        Name of the saved dataset, e.g. "HI_r_beta1".
    x_label, x_unit
        Description and unit of the x-axis data.
    y_label, y_unit
        Description and unit of the y-axis data.
    x_values
        One-dimensional x grid.
    y_matrix
        Two-dimensional array with shape (n_x, n_series).
    series_values
        Optional identifiers for each y-series, e.g. distances [0.1, 0.5, 1.0].
    series_label, series_unit
        Metadata describing what the series dimension means, e.g. "distance"
        and "AU".
    column_names
        Optional explicit names for the y columns. If omitted, names are built
        from y_label and series_values.
    extra_metadata
        Optional additional metadata lines to write into the header.
    float_format
        Format string used for finite numeric values.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_arr = _as_float_array(x_values, "x_values")
    y_arr = _as_2d_float_array(y_matrix, "y_matrix")

    if y_arr.shape[0] != x_arr.size:
        raise ValueError(
            f"x_values length ({x_arr.size}) must match y_matrix first dimension ({y_arr.shape[0]})"
        )

    n_series = y_arr.shape[1]

    if series_values is None:
        series_values = [f"series_{i}" for i in range(n_series)]
    if len(series_values) != n_series:
        raise ValueError(
            f"series_values length ({len(series_values)}) must match number of y-series ({n_series})"
        )

    if column_names is None:
        column_names = [_default_series_column_name(y_label, value) for value in series_values]
    if len(column_names) != n_series:
        raise ValueError(
            f"column_names length ({len(column_names)}) must match number of y-series ({n_series})"
        )

    header_lines = [
        f"# dataset_name: {dataset_name}",
        f"# x_label: {x_label}",
        f"# x_unit: {x_unit}",
        f"# y_label: {y_label}",
        f"# y_unit: {y_unit}",
    ]

    if series_label is not None:
        header_lines.append(f"# series_label: {series_label}")
    if series_unit is not None:
        header_lines.append(f"# series_unit: {series_unit}")
    if series_values is not None:
        header_lines.append("# series_values: " + ", ".join(str(v) for v in series_values))

    if extra_metadata:
        for key, value in extra_metadata.items():
            header_lines.append(f"# {key}: {value}")

    x_column_name = f"x__{x_label.strip().replace(' ', '_')}"
    all_column_names = [x_column_name] + list(column_names)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write("#\n")
        f.write("\t".join(all_column_names) + "\n")

        for i in range(x_arr.size):
            row = [_format_value(x_arr[i], float_format)]
            row.extend(_format_value(y_arr[i, j], float_format) for j in range(n_series))
            f.write("\t".join(row) + "\n")

    return output_path



def example_rbeta_header() -> dict[str, str]:
    """Small example of the minimal metadata needed for the r_beta1 tables."""
    return {
        "dataset_name": "HI_r_beta1",
        "x_label": "Stellar Teff",
        "x_unit": "K",
        "y_label": "r_beta1 / R_p",
        "y_unit": "dimensionless",
        "series_label": "distance",
        "series_unit": "AU",
    }