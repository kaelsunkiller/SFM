"""Shared helpers for source-data preparation scripts.

Reference: Methods §Statistical analysis.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SourceDataArgs:
    """Common CLI arguments for source-data scripts.

    Parameters
    ----------
    data_root : pathlib.Path
        Root directory containing manuscript analysis artifacts.
    output : pathlib.Path
        Output spreadsheet path.
    """

    data_root: Path
    output: Path


def parse_common_args(default_output: str) -> SourceDataArgs:
    """Parse standard source-data CLI arguments.

    Parameters
    ----------
    default_output : str
        Default output filename.

    Returns
    -------
    SourceDataArgs
        Parsed argument bundle.
    """

    parser = argparse.ArgumentParser(description="Prepare source-data table.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.getenv("SFM_ANALYSIS_ROOT", ".")),
        help="Root directory for manuscript analysis artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(default_output),
        help="Output spreadsheet path.",
    )
    ns = parser.parse_args()
    return SourceDataArgs(data_root=ns.data_root, output=ns.output)


def configure_paths(data_root: Path | None) -> None:
    """Re-bind ``generate_source_data`` module-level path globals.

    The ``generate_source_data`` module reads ``SFM_ANALYSIS_ROOT`` at
    import time; when one of the per-figure shims is invoked with an
    explicit ``--data-root``, this helper updates both the environment
    variable and the already-imported module globals so subsequent
    builder calls resolve to the requested artifact tree.

    Parameters
    ----------
    data_root : pathlib.Path or None
        Override path. ``None`` is a no-op so the env-var default
        applies.

    Returns
    -------
    None
        Updates module globals as a side effect.
    """

    if data_root is None:
        return

    os.environ["SFM_ANALYSIS_ROOT"] = str(data_root)

    from . import generate_source_data as gsd

    root = Path(data_root)
    gsd.ROOT = root
    gsd.P0 = root / "data_for_figures" / "p0"
    gsd.PLOT_V2 = root / "Plot_v2"
    gsd.RR = root / "revision_records"
    gsd.BIOMARKER_DIR = root / "data_for_figures" / "biomarker"


def write_single_sheet(output: Path, sheet_name: str, table: pd.DataFrame) -> Path:
    """Write a single DataFrame into a one-sheet Excel workbook.

    Provided for any external caller that wants the lightweight
    one-sheet behaviour of the original public stub. The figure-level
    builders in ``generate_source_data`` write multi-sheet workbooks
    directly via ``pandas.ExcelWriter`` and do not call this helper.

    Parameters
    ----------
    output : pathlib.Path
        Target workbook path.
    sheet_name : str
        Sheet name.
    table : pandas.DataFrame
        Table to write.

    Returns
    -------
    pathlib.Path
        Output workbook path.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name=sheet_name, index=False)
    return output
