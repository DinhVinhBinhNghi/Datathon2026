# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eda.report_patch_charts import run_patch_a1_a3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", help="Thư mục chứa CSV")
    parser.add_argument("--out-dir", default="outputs/figures/main", help="Thư mục lưu hình")
    args = parser.parse_args()

    run_patch_a1_a3(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()