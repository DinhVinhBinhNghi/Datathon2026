import os

# Danh sách các thư mục cần tạo
folders = [
    "config",
    "data/raw", "data/interim", "data/processed", "data/marts", "data/submissions",
    "notebooks",
    "src/io", "src/validation", "src/transforms", "src/joins", "src/features", 
    "src/eda", "src/visualization", "src/modeling", "src/interpret", "src/utils",
    "scripts",
    "reports/figures", "reports/tables",
    "tests"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    # Tạo file .gitkeep để Git có thể track được các folder trống
    with open(os.path.join(folder, ".gitkeep"), "w") as f:
        pass
    
    # Tạo file __init__.py cho các folder trong src để biến chúng thành module
    if folder.startswith("src"):
        with open(os.path.join(folder, "__init__.py"), "w") as f:
            pass

print("✅ Đã dựng xong khung xương dự án!")