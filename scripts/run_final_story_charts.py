# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eda.final_story_charts import run_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", help="Thư mục chứa CSV raw/interim/processed.")
    parser.add_argument("--out-dir", default="outputs/figures/main", help="Thư mục lưu hình.")
    args = parser.parse_args()
    run_all(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
