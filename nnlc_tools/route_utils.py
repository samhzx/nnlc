"""Utilities for identifying routes from openpilot rlog paths."""

import os
import re


_RLOG_FILENAMES = {"rlog", "rlog.zst", "rlog.bz2"}
_TIMESTAMP_ROUTE_PATTERN = re.compile(
    r"^(?:[0-9a-fA-F]+\|)?\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}$"
)
_SUFFIXED_SEGMENT_PATTERN = re.compile(r"^(?P<route>.+)--(?P<segment>\d+)$")


def extract_route_id(path):
    """Return the logical route ID represented by an rlog file path.

    Device exports commonly use ``route--segment/rlog.zst`` while standard
    openpilot layouts use ``route/segment/rlog.zst``. The segment number is a
    minute-long part of a route and must not become part of the route ID.
    """
    parts = [part for part in os.fspath(path).replace("\\", "/").split("/") if part]
    rlog_index = next(
        (index for index in range(len(parts) - 1, -1, -1)
         if parts[index].lower() in _RLOG_FILENAMES),
        None,
    )
    if rlog_index is None or rlog_index == 0:
        return "unknown"

    parent_parts = parts[:rlog_index]
    immediate_parent = parent_parts[-1]

    # A conventional route timestamp already ends in digits, so recognize the
    # immediate parent before applying the generic ``route--segment`` rule.
    # Do not scan arbitrary ancestors: a date archive directory may itself
    # look like a route timestamp but must not override the segment directory.
    if _TIMESTAMP_ROUTE_PATTERN.fullmatch(immediate_parent):
        return immediate_parent

    suffixed_segment = _SUFFIXED_SEGMENT_PATTERN.fullmatch(immediate_parent)
    if suffixed_segment:
        return suffixed_segment.group("route")

    if immediate_parent.isdigit() and len(parent_parts) >= 2:
        return parent_parts[-2]

    return immediate_parent
