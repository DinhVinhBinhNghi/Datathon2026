from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.load_data import load_raw_data
from src.io.load_data import load_raw_data
from src.validation.key_checks import duplicate_key_report, relationship_report
from src.validation.business_rules import null_report
from src.validation.date_checks import date_range_report
from src.utils.constants import DATE_COLUMNS
from src.io.save_data import save_csv


def main():
    tables = load_raw_data("data/raw")
    save_csv(duplicate_key_report(tables), "reports/tables/audit_duplicate_keys.csv")
    save_csv(relationship_report(tables), "reports/tables/audit_fk_coverage.csv")
    save_csv(null_report(tables), "reports/tables/audit_nulls.csv")
    save_csv(date_range_report(tables, DATE_COLUMNS), "reports/tables/audit_date_ranges.csv")


if __name__ == "__main__":
    main()
