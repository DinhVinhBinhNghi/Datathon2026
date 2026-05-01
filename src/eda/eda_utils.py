# -*- coding: utf-8 -*-
"""Shared helpers for Datathon 2026 EDA figures."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COL = {
    "navy": "#1F3A5F",
    "blue": "#2F6B9A",
    "teal": "#2A9D8F",
    "green": "#2A9D8F",
    "orange": "#E76F51",
    "red": "#C0392B",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "panel": "#F9FAFB",
    "dark": "#111827",
    "bg": "#FFFFFF",
    "guide": "#4F86F7",
    "grid": "#E5E7EB",
}

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_file(data_dir: str | Path, filename: str, extra_dirs: Optional[Iterable[str | Path]] = None) -> Optional[Path]:
    data_dir = Path(data_dir)
    candidates = [
        data_dir / filename,
        data_dir / "raw" / filename,
        data_dir / "interim" / filename,
        data_dir / "processed" / filename,
        data_dir / "marts" / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / "data" / "raw" / filename,
        Path.cwd() / "data" / "interim" / filename,
        Path.cwd() / "data" / "processed" / filename,
        Path.cwd() / "data" / "marts" / filename,
    ]
    if extra_dirs:
        for d in extra_dirs:
            d = Path(d)
            candidates.extend([d / filename, d / "raw" / filename, d / "interim" / filename, d / "processed" / filename])
    for p in candidates:
        if p.exists():
            return p
    return None


def read_csv_smart(data_dir: str | Path, filename: str, parse_dates=None, low_memory=False) -> pd.DataFrame:
    path = find_file(data_dir, filename)
    if path is None:
        raise FileNotFoundError(f"Không tìm thấy {filename}. Hãy kiểm tra --data-dir.")
    return pd.read_csv(path, parse_dates=parse_dates, low_memory=low_memory)


def first_valid(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


def style_axis(ax, xy: bool = False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(axis="both", colors=COL["gray"], labelsize=9)
    ax.grid(True if xy else False, axis="both" if xy else "y", color=COL["grid"], lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def add_kpi_box(ax, title, value, subtitle, y, color, height=0.17):
    box = FancyBboxPatch(
        (0.03, y), 0.94, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.8,
        edgecolor="#E5E7EB",
        facecolor="#F9FAFB"
    )
    ax.add_patch(box)
    ax.text(0.07, y + height * 0.72, title, fontsize=9.2, color=COL["gray"])
    ax.text(0.07, y + height * 0.42, value, fontsize=15.2, color=color, weight="bold")
    ax.text(0.07, y + height * 0.18, subtitle, fontsize=8.2, color=COL["gray"])


def money_million_vi(x, pos=None):
    return f"{x/1e6:.0f}M"


def pct_fmt_fraction(x, pos=None):
    return f"{x*100:.0f}%"


def pct_fmt_percent(x, pos=None):
    return f"{x:.0f}%"


def add_period_shading_year_axis(ax):
    ax.axvline(2018.5, color=COL["gray"], lw=1.2, ls="--", alpha=0.8)
    ax.axvspan(2013, 2018.5, color="#D8E8F5", alpha=0.25, lw=0)
    ax.axvspan(2018.5, 2022, color="#E5E7EB", alpha=0.25, lw=0)
