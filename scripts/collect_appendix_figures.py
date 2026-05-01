# -*- coding: utf-8 -*-
"""Copy selected exploratory figures into appendix folder.

Usage:
python scripts/collect_appendix_figures.py --source reports/figures --out outputs/figures/appendix_unused_for_main_report
"""
from pathlib import Path
import argparse
import shutil

DEFAULT_KEYWORDS = [
    "audit", "inventory", "promo", "promotion", "customer", "rfm", "margin", "stockout", "appendix"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="reports/figures", help="Folder chứa hình EDA cũ.")
    parser.add_argument("--out", default="outputs/figures/appendix_unused_for_main_report", help="Folder appendix.")
    parser.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS)
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"Không thấy source: {src}")
        return

    copied = 0
    for p in src.rglob("*"):
        if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".pdf", ".svg"]:
            continue
        name = p.name.lower()
        if any(k.lower() in name for k in args.keywords):
            target = out / p.name
            shutil.copy2(p, target)
            copied += 1
            print(f"copied: {p} -> {target}")

    print(f"Done. Copied {copied} appendix figures.")


if __name__ == "__main__":
    main()
