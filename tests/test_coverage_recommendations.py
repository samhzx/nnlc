"""Coverage plots should also say which roads still need collecting."""

import numpy as np
import pandas as pd

from nnlc_tools.visualize_coverage import (
    LAT_BINS,
    SPEED_BINS,
    analyze_coverage_gaps,
    format_coverage_gap_report,
    plot_coverage,
    write_coverage_recommendations,
)


def empty_counts():
    return np.zeros((SPEED_BINS.size - 1, LAT_BINS.size - 1), dtype=np.int64)


def fill_region(counts, speed_lo, speed_hi, lat_lo, lat_hi, value):
    speed_centers = 0.5 * (SPEED_BINS[:-1] + SPEED_BINS[1:])
    lat_centers = 0.5 * (LAT_BINS[:-1] + LAT_BINS[1:])
    mask = (
        (speed_centers[:, None] >= speed_lo)
        & (speed_centers[:, None] < speed_hi)
        & (lat_centers[None, :] >= lat_lo)
        & (lat_centers[None, :] < lat_hi)
    )
    counts[mask] = value
    return counts


def test_city_only_data_asks_for_highway_and_fast_curves():
    counts = empty_counts()
    fill_region(counts, 6, 12, -0.4, 0.4, 80)
    analysis = analyze_coverage_gaps(counts)
    ids = [item["id"] for item in analysis["findings"]]
    assert "highway_overall" in ids
    assert "mid_speed_medium" in ids
    report = format_coverage_gap_report(analysis)
    assert "缺高速公路工况" in report
    assert "20-30 m/s" in report


def test_one_sided_sharp_turns_are_called_out():
    counts = empty_counts()
    fill_region(counts, 10, 30, 1.1, 2.4, 40)
    analysis = analyze_coverage_gaps(counts)
    ids = [item["id"] for item in analysis["findings"]]
    assert "left_sharp" in ids
    assert "turn_imbalance" in ids
    report = format_coverage_gap_report(analysis)
    assert "左急弯" in report


def test_balanced_coverage_has_no_priority_gaps():
    counts = empty_counts()
    fill_region(counts, 6, 34, -2.4, 2.4, 80)
    analysis = analyze_coverage_gaps(counts, roll_stats={"mild": 4000, "strong": 800})
    assert analysis["findings"] == []
    report = format_coverage_gap_report(analysis)
    assert "覆盖较好" in report


def test_missing_roll_is_reported():
    counts = empty_counts()
    fill_region(counts, 6, 34, -2.4, 2.4, 80)
    analysis = analyze_coverage_gaps(counts, roll_stats={"mild": 10, "strong": 0})
    assert any(item["id"] == "roll" for item in analysis["findings"])


def test_advice_file_is_written_next_to_the_plot(tmp_path):
    counts = empty_counts()
    fill_region(counts, 6, 12, -0.3, 0.3, 60)
    image = tmp_path / "coverage.png"
    path, analysis = write_coverage_recommendations(str(image), counts)
    assert path == str(tmp_path / "coverage_gaps.txt")
    text = (tmp_path / "coverage_gaps.txt").read_text(encoding="utf-8")
    assert "建议优先补采" in text
    assert analysis["total_samples"] > 0


def test_plot_coverage_writes_advice_with_the_image(tmp_path):
    rows = []
    for speed in range(6, 12):
        for lat in (-0.2, 0.0, 0.2):
            rows.extend(
                {
                    "v_ego": float(speed),
                    "desired_lateral_accel": lat,
                    "active": True,
                    "standstill": False,
                    "roll": 0.0,
                }
                for _ in range(40)
            )
    df = pd.DataFrame(rows)
    image = tmp_path / "coverage.png"
    plot_coverage(df, str(image))
    assert image.is_file()
    advice = (tmp_path / "coverage_gaps.txt").read_text(encoding="utf-8")
    assert "NNLC 路况覆盖建议" in advice
    assert "高速公路" in advice or "20-30 m/s" in advice
