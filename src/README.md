# src package

Các module trong `src/` được thiết kế để notebook 01–04 không chỉ là code thử nghiệm mà có thể tái sử dụng trong repo.

## Nhóm module chính

- `io/`: đọc và lưu CSV.
- `validation/`: kiểm tra khóa, foreign key, null, ngày tháng và business rules.
- `joins/`: build fact/mart chính cho EDA và modeling.
- `features/`: tạo calendar, lag/rolling, promo, traffic, inventory features.
- `eda/`: bảng summary cho storyline EDA.
- `visualization/`: plot functions lưu figure vào `reports/figures/`.
- `modeling/`: split theo thời gian, metrics, baseline, training, inference.

## Cách chạy nhanh

```bash
python scripts/run_data_audit.py
python scripts/run_build_marts.py
python scripts/run_make_submission.py
```
