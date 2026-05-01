from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.utils.helpers import ensure_dir


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    return path


def save_many(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> list[Path]:
    out_dir = ensure_dir(out_dir)
    paths = []
    for name, df in tables.items():
        filename = name if name.endswith(".csv") else f"{name}.csv"
        paths.append(save_csv(df, out_dir / filename))
    return paths
