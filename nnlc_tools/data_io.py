"""Shared data loading for NNLC tools."""

import os

import pandas as pd


class DataLoadError(ValueError):
    """Raised when an input data file or rlog directory cannot be read."""


_READ_ERRORS = (OSError, ValueError, ImportError)


def load_data(input_path):
    """Load lateral data from CSV, Parquet, or directory of rlogs.

    Returns a DataFrame, or None if no data found.
    """
    if os.path.isfile(input_path):
        try:
            if input_path.lower().endswith(".parquet"):
                return pd.read_parquet(input_path)
            return pd.read_csv(input_path)
        except _READ_ERRORS as exc:
            raise DataLoadError(f"无法读取数据文件 {input_path}: {exc}") from exc

    if os.path.isdir(input_path):
        import tempfile
        from nnlc_tools.extract_lateral_data import (
            extract_rlogs_to_csv, find_rlogs,
        )

        rlog_files = find_rlogs(input_path)
        if not rlog_files:
            return None
        temp_file = tempfile.NamedTemporaryFile(prefix="nnlc_data_", suffix=".csv", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        try:
            stats = extract_rlogs_to_csv(
                rlog_files, temp_path, show_progress=False,
            )
            if stats["rows_written"] == 0:
                return None
            return pd.read_csv(temp_path)
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            raise DataLoadError(f"无法读取 rlog 数据目录 {input_path}: {exc}") from exc
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    return None
