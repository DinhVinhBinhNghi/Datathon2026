from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.utils.constants import RAW_FILES, DATE_COLUMNS
from src.utils.helpers import normalize_columns


def read_csv(path: str | Path, parse_dates: list[str] | None = None, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=parse_dates, **kwargs)
    return normalize_columns(df)


def load_raw_data(data_dir: str | Path = "data/raw", required: list[str] | None = None) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    required = required or RAW_FILES
    data = {}
    missing = []
    for name in required:
        path = data_dir / name
        if not path.exists():
            missing.append(name)
            continue
        data[name.replace(".csv", "")] = read_csv(path, parse_dates=DATE_COLUMNS.get(name))
    if missing:
        raise FileNotFoundError("Missing raw files: " + ", ".join(missing))
    return data


def load_table(name: str, data_dir: str | Path) -> pd.DataFrame:
    filename = name if name.endswith(".csv") else f"{name}.csv"
    return read_csv(Path(data_dir) / filename, parse_dates=DATE_COLUMNS.get(filename))
