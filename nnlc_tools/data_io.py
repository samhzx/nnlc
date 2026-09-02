"""Shared data loading for NNLC tools."""

import os

import pandas as pd


def load_data(input_path):
    """Load lateral data from CSV, Parquet, or directory of rlogs.

    Returns a DataFrame, or None if no data found.
    """
    if os.path.isfile(input_path):
        if input_path.endswith(".parquet"):
            return pd.read_parquet(input_path)
        return pd.read_csv(input_path)

    if os.path.isdir(input_path):
        import tempfile
        from nnlc_tools.extract_lateral_data import (
            find_rlogs, extract_segment, _StreamingCsvWriter, extract_route_id,
        )

        rlog_files = find_rlogs(input_path)
        if not rlog_files:
            return None
        temp_file = tempfile.NamedTemporaryFile(prefix="nnlc_data_", suffix=".csv", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        stream = _StreamingCsvWriter(temp_path)
        try:
            for path in rlog_files:
                stream.route_id = extract_route_id(path)
                extract_segment(path, row_callback=stream.accept)
                stream.finish_segment()
            stream.finish()
            if stream.rows_written == 0:
                return None
            return pd.read_csv(temp_path)
        finally:
            stream.close()
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    return None
