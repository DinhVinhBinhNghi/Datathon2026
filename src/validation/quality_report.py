def classify_nulls(df):
    """Gắn nhãn cho các cột dựa trên tỷ lệ Null và tính chất nghiệp vụ"""
    # Ví dụ logic tự động gợi ý nhãn
    report = {}
    for col in df.columns:
        null_pct = df[col].isnull().mean() * 100
        if null_pct > 80:
            label = "Structural (Likely)"
        elif null_pct == 0:
            label = "Mandatory/Clean"
        else:
            label = "Check Logic"
        report[col] = label
    return report