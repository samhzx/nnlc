"""Utilities shared by the low-memory CSV processing paths.

The normal tools keep their DataFrame APIs for compatibility.  Streaming
callers use these helpers to process bounded chunks and to write output
incrementally without changing the CSV schema.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator

import numpy as np
import pandas as pd


DEFAULT_CHUNK_ROWS = 100_000


def require_distinct_paths(input_path: str, output_path: str) -> None:
    """Reject an in-place streaming write before it can truncate its input."""
    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    if input_abs == output_abs:
        raise ValueError("streaming input and output paths must be different")
    try:
        if os.path.exists(output_abs) and os.path.samefile(input_abs, output_abs):
            raise ValueError("streaming input and output paths refer to the same file")
    except FileNotFoundError:
        pass


def iter_csv_chunks(path: str, *, chunksize: int = DEFAULT_CHUNK_ROWS,
                    usecols=None) -> Iterator[pd.DataFrame]:
    """Yield CSV chunks with a bounded row count.

    ``low_memory=False`` keeps dtype inference consistent between chunks.  A
    positive chunk size is required so an accidental zero cannot cause an
    unbounded read.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    yield from pd.read_csv(path, chunksize=chunksize, usecols=usecols,
                           low_memory=False)


def open_csv_writer(path: str, columns: list[str]):
    """Open a UTF-8 CSV writer and emit its header."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(columns)
    return handle, writer


def sample_csv(path: str, *, max_rows: int = 100_000,
               chunksize: int = DEFAULT_CHUNK_ROWS,
               random_seed: int = 45) -> pd.DataFrame:
    """Return a deterministic, bounded uniform-priority sample of a CSV."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    priority_column = "_nnlc_sample_priority"
    input_columns = set(pd.read_csv(path, nrows=0).columns)
    while priority_column in input_columns:
        priority_column = "_" + priority_column
    rng = np.random.default_rng(random_seed)
    reservoir = None
    for chunk in iter_csv_chunks(path, chunksize=chunksize):
        chunk = chunk.copy()
        chunk[priority_column] = rng.random(len(chunk))
        if reservoir is not None:
            chunk = pd.concat((reservoir, chunk), ignore_index=True)
        if len(chunk) > max_rows:
            chunk = chunk.nsmallest(max_rows, priority_column)
        reservoir = chunk
    if reservoir is None or reservoir.empty:
        raise pd.errors.EmptyDataError("input CSV contains no rows")
    return reservoir.drop(columns=[priority_column]).reset_index(drop=True)
