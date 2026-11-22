"""Read GEO series-matrix files and collapse probes to gene symbols.

The parser has no network dependency: download the ``*_series_matrix.txt.gz`` and
the platform (GPL) annotation table yourself and pass the local paths.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _open(path: Path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" \
        else open(path, "rt", encoding="utf-8", errors="replace")


def _split(line: str) -> list[str]:
    return [v.strip().strip('"') for v in line.rstrip("\n").split("\t")]


def _clean_key(key: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", key.strip().lower()).strip("_")
    return key or "field"


def read_series_matrix(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (expression: probes x samples, metadata: samples x fields).

    ``!Sample_characteristics_ch1`` entries of the form ``"key: value"`` become
    their own metadata columns (e.g. ``group``, ``age``, ``bmi``).
    """
    meta: dict[str, list[str]] = {}
    char_rows: list[list[str]] = []
    table: list[list[str]] = []
    in_table = False
    with _open(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                table.append(_split(line))
            elif line.startswith("!Sample_characteristics"):
                char_rows.append(_split(line)[1:])
            elif line.startswith("!Sample_"):
                fields = _split(line)
                key = _clean_key(fields[0].replace("!Sample_", ""))
                if key not in meta:  # keep first occurrence of repeated keys
                    meta[key] = fields[1:]
    if not table:
        raise ValueError(f"No expression table found in {path}")
    header, rows = table[0], table[1:]
    expr = pd.DataFrame(rows, columns=header).set_index(header[0])
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr.index = expr.index.astype(str)

    samples = header[1:]
    md = pd.DataFrame({k: v for k, v in meta.items() if len(v) == len(samples)}, index=samples)
    for i, row in enumerate(char_rows):
        for sample, value in zip(samples, row):
            if ":" in value:
                k, v = value.split(":", 1)
                col = _clean_key(k)
            else:
                col, v = f"characteristic_{i}", value
            md.loc[sample, col] = v.strip()
    md.index.name = "sample_id"
    return expr, md


def collapse_probes(expr: pd.DataFrame, annotation: pd.DataFrame, id_col: str,
                    symbol_col: str) -> pd.DataFrame:
    """Map probes to gene symbols; keep the highest-mean probe per gene.

    Probes without a symbol, or mapping to several genes (``///``), are dropped.
    """
    ann = annotation[[id_col, symbol_col]].dropna().astype(str)
    ann = ann[(ann[symbol_col].str.strip() != "") & ~ann[symbol_col].str.contains("///")]
    mapping = ann.set_index(id_col)[symbol_col].str.strip()
    keep = expr.index.intersection(mapping.index)
    if len(keep) == 0:
        raise ValueError("No probe IDs overlap between expression and annotation.")
    sub = expr.loc[keep].copy()
    sub["_gene"] = mapping.loc[keep].values
    sub["_mean"] = sub.drop(columns="_gene").mean(axis=1)
    sub = sub.sort_values("_mean", ascending=False).drop_duplicates("_gene")
    out = sub.drop(columns="_mean").set_index("_gene")
    out.index.name = "gene"
    return out.dropna(how="all")


def read_platform_table(path: str | Path) -> pd.DataFrame:
    """Read a GPL annotation table (full ``GPLxxx.txt`` or ``*.annot``), skipping '#' lines."""
    with _open(path) as fh:
        lines = [l for l in fh if not l.startswith(("#", "!", "^"))]
    from io import StringIO
    return pd.read_csv(StringIO("".join(lines)), sep="\t", dtype=str, low_memory=False)


def log2_if_needed(expr: pd.DataFrame) -> pd.DataFrame:
    """Apply log2(x+1) when values look linear-scale. Rank features make this cosmetic."""
    vals = expr.to_numpy(dtype=float)
    if np.nanmax(vals) > 100:
        return np.log2(expr.clip(lower=0) + 1)
    return expr
