# Update May 01 — A3/A4/A6

Gói này là bản cập nhật nhẹ cho repo sau khi nhóm sửa đổi report:

- A3: thay bubble chart bằng bản ngắn 2 biểu đồ + panel KPI bên phải.
- A4: thêm conversion crisis — traffic tăng nhưng conversion/revenue per session giảm.
- A6: thêm ba kịch bản phục hồi H1/2024.

## Cách dùng nhanh

Copy folder này vào root repo `C:\datathon`, rồi dùng các cell trong:

```text
notebooks/colab_cells/
```

để ghi đè/chạy tiếp trong Colab. Sau khi xuất hình PNG, copy các hình vào:

```text
outputs/figures/main/
```

Tên hình nên dùng:

```text
A3_promo_refined_vi.png
A4_conversion_crisis_vi.png
A6_recovery_scenarios_H1_2024.png
```

## Nếu muốn chạy từ VS Code

File `scripts/run_update_charts_a3_a4_a6.py` là wrapper để bạn gắn các hàm vào pipeline sau. Vì notebook hiện đang là nguồn chính của các cell mới, cách an toàn nhất là chạy lại trong Colab trước, rồi sync hình và notebook lên GitHub.
