# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from src.eda.final_story_charts import (
    COL,
    ensure_dir,
    load_sales,
    load_item_level,
    style_axis,
)


# ============================================================
# Helpers
# ============================================================
def style_axis_xy_local(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(axis="both", colors=COL["gray"], labelsize=9)
    ax.grid(True, color=COL["grid"], lw=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def add_kpi_box_local(ax, title, value, subtitle, y, color, height=0.18):
    box = FancyBboxPatch(
        (0.03, y), 0.94, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.8,
        edgecolor="#E5E7EB",
        facecolor="#F9FAFB"
    )
    ax.add_patch(box)
    ax.text(0.07, y + height * 0.72, title, fontsize=9.0, color=COL["gray"])
    ax.text(0.07, y + height * 0.42, value, fontsize=14.8, color=color, weight="bold")
    ax.text(0.07, y + height * 0.18, subtitle, fontsize=8.0, color=COL["gray"])


def money_million_vi(x, pos=None):
    return f"{x/1e6:.0f}M"


def pct_fmt_vi(x, pos=None):
    return f"{x*100:.0f}%"


def safe_pct_change(after, before):
    if before is None or pd.isna(before) or before == 0:
        return np.nan
    return (after / before - 1) * 100


# ============================================================
# A1 gốc — KHÔNG overlay
# Save đè vào tên file report đang dùng để khỏi sửa LaTeX
# ============================================================
def plot_a1_original(data_dir: str | Path, out_dir: str | Path) -> Dict[str, float]:
    sales = load_sales(data_dir)
    sales = sales[sales["date"].dt.year.between(2013, 2022)].copy()

    monthly = (
        sales.assign(ym=sales["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("ym", as_index=False)
        .agg(Revenue=("Revenue", "sum"), COGS=("COGS", "sum"))
    )
    monthly["year"] = monthly["ym"].dt.year
    monthly["month"] = monthly["ym"].dt.month
    monthly["gross_margin"] = (monthly["Revenue"] - monthly["COGS"]) / monthly["Revenue"].replace(0, np.nan)

    pre = monthly[monthly["year"].between(2013, 2018)].copy()
    post = monthly[monthly["year"].between(2019, 2022)].copy()

    avg_monthly_pre = pre["Revenue"].mean()
    avg_monthly_post = post["Revenue"].mean()
    revenue_shift = safe_pct_change(avg_monthly_post, avg_monthly_pre)

    gm_pre = (pre["Revenue"].sum() - pre["COGS"].sum()) / pre["Revenue"].sum()
    gm_post = (post["Revenue"].sum() - post["COGS"].sum()) / post["Revenue"].sum()
    gm_change_pp = (gm_post - gm_pre) * 100

    peak_pre = pre["Revenue"].max()
    peak_post = post["Revenue"].max()
    peak_change = safe_pct_change(peak_post, peak_pre)

    august = monthly[monthly["month"] == 8].copy()
    august_odd_negative = august[(august["year"] % 2 == 1) & (august["gross_margin"] < 0)].copy()

    fig = plt.figure(figsize=(15.8, 8.4), facecolor="white")
    gs = GridSpec(2, 2, width_ratios=[5.5, 1.55], height_ratios=[1.05, 1.0], figure=fig, wspace=0.18, hspace=0.22)

    ax_rev = fig.add_subplot(gs[0, 0])
    ax_gm = fig.add_subplot(gs[1, 0], sharex=ax_rev)
    ax_kpi = fig.add_subplot(gs[:, 1])
    ax_kpi.axis("off")

    # ---------------- Revenue panel ----------------
    ax_rev.axvspan(pd.Timestamp("2013-01-01"), pd.Timestamp("2018-12-31"), color="#D8E8F5", alpha=0.24, lw=0)
    ax_rev.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"), color="#E5E7EB", alpha=0.25, lw=0)
    ax_rev.axvline(pd.Timestamp("2019-01-01"), color=COL["gray"], lw=1.2, ls="--", alpha=0.8)

    ax_rev.plot(monthly["ym"], monthly["Revenue"], color=COL["navy"], lw=2.7)

    pre_peak = pre.loc[pre["Revenue"].idxmax()]
    post_peak = post.loc[post["Revenue"].idxmax()]

    ax_rev.scatter(pre_peak["ym"], pre_peak["Revenue"], s=70, color=COL["blue"], edgecolor="white", zorder=5)
    ax_rev.scatter(post_peak["ym"], post_peak["Revenue"], s=70, color=COL["orange"], edgecolor="white", zorder=5)

    ax_rev.annotate(
        f"Đỉnh trước 2019\n{pre_peak['Revenue']/1e6:.0f}M",
        xy=(pre_peak["ym"], pre_peak["Revenue"]),
        xytext=(-30, 18),
        textcoords="offset points",
        fontsize=8.8,
        color=COL["blue"],
        weight="bold",
        arrowprops=dict(arrowstyle="-", color=COL["blue"], lw=1.0),
    )

    ax_rev.annotate(
        f"Đỉnh sau 2018\n{post_peak['Revenue']/1e6:.0f}M",
        xy=(post_peak["ym"], post_peak["Revenue"]),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=8.8,
        color=COL["orange"],
        weight="bold",
        arrowprops=dict(arrowstyle="-", color=COL["orange"], lw=1.0),
    )

    ax_rev.text(pd.Timestamp("2015-10-01"), ax_rev.get_ylim()[1] * 0.94,
                "2013–2018\nmặt bằng cao", ha="center", va="top",
                fontsize=9.5, color=COL["blue"], weight="bold")

    ax_rev.text(pd.Timestamp("2020-12-01"), ax_rev.get_ylim()[1] * 0.94,
                "2019–2022\nmặt bằng thấp", ha="center", va="top",
                fontsize=9.5, color=COL["gray"], weight="bold")

    ax_rev.set_title("A1. Doanh thu chuyển sang mặt bằng thấp hơn sau 2018", loc="left",
                     fontsize=14.8, color=COL["dark"], pad=10)
    ax_rev.set_ylabel("Doanh thu tháng", fontsize=10.2, color=COL["navy"])
    ax_rev.yaxis.set_major_formatter(mticker.FuncFormatter(money_million_vi))
    style_axis_xy_local(ax_rev)
    ax_rev.tick_params(axis="x", labelbottom=False)

    # ---------------- Margin panel ----------------
    ax_gm.axvspan(pd.Timestamp("2013-01-01"), pd.Timestamp("2018-12-31"), color="#D8E8F5", alpha=0.24, lw=0)
    ax_gm.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"), color="#E5E7EB", alpha=0.25, lw=0)
    ax_gm.axvline(pd.Timestamp("2019-01-01"), color=COL["gray"], lw=1.2, ls="--", alpha=0.8)

    ax_gm.plot(monthly["ym"], monthly["gross_margin"], color=COL["green"], lw=2.2)
    ax_gm.axhline(0, color=COL["dark"], lw=0.9, alpha=0.6)

    negative = monthly[monthly["gross_margin"] < 0].copy()
    ax_gm.scatter(negative["ym"], negative["gross_margin"], s=45, color=COL["red"], edgecolor="white", linewidth=0.9, zorder=5)

    if len(august_odd_negative) > 0:
        ax_gm.scatter(
            august_odd_negative["ym"],
            august_odd_negative["gross_margin"],
            s=110,
            facecolors="none",
            edgecolors=COL["red"],
            linewidth=1.8,
            zorder=6
        )
        worst_aug = august_odd_negative.loc[august_odd_negative["gross_margin"].idxmin()]
        ax_gm.annotate(
            "Tháng 8 năm lẻ\nâm biên lặp lại",
            xy=(worst_aug["ym"], worst_aug["gross_margin"]),
            xytext=(-95, 26),
            textcoords="offset points",
            fontsize=8.6,
            color=COL["red"],
            weight="bold",
            arrowprops=dict(arrowstyle="->", color=COL["red"], lw=1.0),
        )

    ax_gm.set_title("Biên lợi nhuận gộp không hồi phục tương ứng với doanh thu", loc="left",
                    fontsize=12.6, color=COL["dark"], pad=8)
    ax_gm.set_ylabel("Biên lợi nhuận gộp", fontsize=10.2, color=COL["green"])
    ax_gm.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt_vi))
    ax_gm.set_xlabel("Thời gian", fontsize=10.0, color=COL["gray"])
    style_axis_xy_local(ax_gm)

    years = pd.date_range("2013-01-01", "2023-01-01", freq="YS")
    ax_gm.set_xticks(years)
    ax_gm.set_xticklabels([d.year for d in years], fontsize=9)

    # ---------------- KPI panel ----------------
    ax_kpi.text(0.03, 0.97, "Phát hiện chính", fontsize=14.0, weight="bold", color=COL["dark"], va="top")

    add_kpi_box_local(
        ax_kpi,
        "Doanh thu TB/tháng",
        f"{revenue_shift:.1f}%",
        f"{avg_monthly_pre/1e6:.0f}M → {avg_monthly_post/1e6:.0f}M",
        y=0.72,
        color=COL["red"]
    )

    add_kpi_box_local(
        ax_kpi,
        "Đỉnh doanh thu tháng",
        f"{peak_change:.1f}%",
        f"{peak_pre/1e6:.0f}M → {peak_post/1e6:.0f}M",
        y=0.49,
        color=COL["orange"]
    )

    add_kpi_box_local(
        ax_kpi,
        "Biên LN gộp",
        f"{gm_change_pp:.1f} điểm %",
        f"{gm_pre*100:.1f}% → {gm_post*100:.1f}%",
        y=0.26,
        color=COL["gray"]
    )

    add_kpi_box_local(
        ax_kpi,
        "Tháng 8 năm lẻ âm biên",
        f"{len(august_odd_negative)} lần",
        "Mẫu bất thường cần điều tra",
        y=0.03,
        color=COL["red"]
    )

    fig.suptitle(
        "Doanh thu giảm mang tính cấu trúc, không chỉ là dao động mùa vụ",
        x=0.04, y=0.985, ha="left",
        fontsize=18.0, weight="bold", color=COL["dark"]
    )
    fig.text(
        0.04, 0.947,
        "Tổng hợp theo tháng từ sales.csv. Hình này là bản A1 gốc, tách riêng doanh thu và biên lợi nhuận.",
        fontsize=9.2, color=COL["gray"]
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = ensure_dir(out_dir) / "A1_overlay_revenue_margin_vi.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "a1_revenue_shift_pct": float(revenue_shift),
        "a1_peak_change_pct": float(peak_change),
        "a1_gm_change_pp": float(gm_change_pp),
        "a1_negative_august_count": int(len(august_odd_negative)),
    }


# ============================================================
# A3 mới — 2 biểu đồ + KPI panel
# Save đè đúng tên file report đang dùng
# ============================================================
def plot_a3_refined(data_dir: str | Path, out_dir: str | Path) -> Dict[str, float]:
    """
    A3 đúng bản report mong muốn:
    - 2 biểu đồ: promo revenue/margin + 90-day repeat
    - panel KPI bên phải
    - ghi đè outputs/figures/main/A3_promo_refined_vi.png
    """
    items = load_item_level(data_dir)

    items["discount_amount"] = items["discount_amount"].fillna(0)
    items["net_item_revenue"] = items["net_item_revenue"].clip(lower=0)
    items["gross_merch_value"] = items["gross_merch_value"].clip(lower=0)

    valid_status = ["delivered", "returned", "shipped"]
    df = items[
        items["order_status"].isin(valid_status)
        & items["order_date"].dt.year.between(2019, 2022)
    ].copy()

    order_level = (
        df.groupby(["order_id", "customer_id", "order_date"], as_index=False)
        .agg(
            revenue=("net_item_revenue", "sum"),
            gross_merch_value=("gross_merch_value", "sum"),
            gross_profit=("gross_profit", "sum"),
            discount_amount=("discount_amount", "sum"),
            promo_used=("promo_used_flag", "max"),
        )
    )

    order_level["gross_margin"] = (
        order_level["gross_profit"] / order_level["revenue"].replace(0, np.nan)
    )

    promo_summary = (
        order_level.groupby("promo_used", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            revenue=("revenue", "sum"),
            gross_merch_value=("gross_merch_value", "sum"),
            gross_profit=("gross_profit", "sum"),
            discount_amount=("discount_amount", "sum"),
        )
    )

    promo_summary["revenue_share"] = promo_summary["revenue"] / promo_summary["revenue"].sum()
    promo_summary["gross_margin"] = promo_summary["gross_profit"] / promo_summary["revenue"].replace(0, np.nan)
    promo_summary["discount_rate"] = promo_summary["discount_amount"] / promo_summary["gross_merch_value"].replace(0, np.nan)
    promo_summary = promo_summary.sort_values("promo_used").reset_index(drop=True)

    # 90-day repeat
    order_level = order_level.sort_values(["customer_id", "order_date", "order_id"]).copy()
    order_level["next_order_date"] = order_level.groupby("customer_id")["order_date"].shift(-1)
    order_level["next_gap_days"] = (order_level["next_order_date"] - order_level["order_date"]).dt.days

    max_date = order_level["order_date"].max()
    order_level["eligible_90d"] = order_level["order_date"] <= (max_date - pd.Timedelta(days=90))
    order_level["repurchase_90d"] = (
        order_level["eligible_90d"]
        & order_level["next_gap_days"].between(1, 90)
    ).astype(int)

    repeat_summary = (
        order_level[order_level["eligible_90d"]]
        .groupby("promo_used", as_index=False)
        .agg(
            eligible_orders=("order_id", "nunique"),
            repurchase_90d=("repurchase_90d", "sum")
        )
    )

    repeat_summary["repurchase_90d_rate"] = (
        repeat_summary["repurchase_90d"] / repeat_summary["eligible_orders"]
    )
    repeat_summary = repeat_summary.sort_values("promo_used").reset_index(drop=True)

    nonpromo = promo_summary[promo_summary["promo_used"] == 0].iloc[0]
    promo = promo_summary[promo_summary["promo_used"] == 1].iloc[0]

    nonpromo_repeat = repeat_summary[repeat_summary["promo_used"] == 0].iloc[0]
    promo_repeat = repeat_summary[repeat_summary["promo_used"] == 1].iloc[0]

    nonpromo_revenue_share = nonpromo["revenue_share"] * 100
    promo_revenue_share = promo["revenue_share"] * 100

    nonpromo_gm = nonpromo["gross_margin"] * 100
    promo_gm = promo["gross_margin"] * 100
    gm_gap = promo_gm - nonpromo_gm

    nonpromo_repeat_rate = nonpromo_repeat["repurchase_90d_rate"] * 100
    promo_repeat_rate = promo_repeat["repurchase_90d_rate"] * 100
    repeat_gap = promo_repeat_rate - nonpromo_repeat_rate

    # ============================================================
    # Plot
    # ============================================================
    fig = plt.figure(figsize=(14.2, 4.85), facecolor="white")
    gs = GridSpec(
        1, 3,
        width_ratios=[1.25, 1.0, 0.82],
        figure=fig,
        wspace=0.34
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    axk = fig.add_subplot(gs[0, 2])
    axk.axis("off")

    labels = ["Đơn không\nkhuyến mãi", "Đơn khuyến\nmãi"]
    x = np.arange(2)
    bar_w = 0.30

    # ---------------- Chart A ----------------
    revenue_vals = [nonpromo_revenue_share, promo_revenue_share]
    gm_vals = [nonpromo_gm, promo_gm]

    ax1b = ax1.twinx()

    bars_rev = ax1.bar(
        x - bar_w / 2,
        revenue_vals,
        width=bar_w,
        color=["#9CA3AF", "#F4A261"],
        alpha=0.92,
        label="% doanh thu"
    )

    bars_gm = ax1b.bar(
        x + bar_w / 2,
        gm_vals,
        width=bar_w,
        color=["#1F3A5F", "#E63946"],
        alpha=0.95,
        label="Biên LN gộp"
    )

    ax1.set_title(
        "A. KM mua doanh thu bằng cách bào sạch điểm % biên LN",
        loc="left",
        fontsize=9.8,
        color=COL["dark"],
        weight="bold"
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.2)
    ax1.set_ylabel("% doanh thu hậu 2018", fontsize=8.3, color=COL["gray"])
    ax1.set_ylim(0, max(revenue_vals) * 1.28)
    ax1.tick_params(axis="both", labelsize=8)
    style_axis(ax1)

    ax1b.set_ylabel("Biên LN gộp (%)", fontsize=8.3, color=COL["gray"])
    ax1b.set_ylim(0, max(gm_vals) * 1.58 if max(gm_vals) > 0 else 5)
    ax1b.tick_params(axis="y", colors=COL["gray"], labelsize=8)
    ax1b.spines["top"].set_visible(False)
    ax1b.spines["right"].set_color("#D1D5DB")

    for rect in bars_rev:
        h = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            h + max(revenue_vals) * 0.025,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=COL["dark"],
            weight="bold"
        )

    for rect in bars_gm:
        h = rect.get_height()
        ax1b.text(
            rect.get_x() + rect.get_width() / 2,
            h + max(gm_vals) * 0.045,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=COL["dark"],
            weight="bold"
        )

    ax1.legend(
        [bars_rev[0], bars_gm[0]],
        ["% doanh thu", "Biên LN gộp"],
        frameon=False,
        fontsize=7.8,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.98)
    )

    # ---------------- Chart B ----------------
    repeat_vals = [nonpromo_repeat_rate, promo_repeat_rate]

    bars_repeat = ax2.bar(
        x,
        repeat_vals,
        width=0.48,
        color=["#1F3A5F", "#E63946"],
        alpha=0.92
    )

    ax2.set_title(
        "B. KM KHÔNG kéo được khách quay lại luận điểm “đầu tư” sụp",
        loc="left",
        fontsize=9.8,
        color=COL["dark"],
        weight="bold"
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.2)
    ax2.set_ylabel("% khách quay lại trong 90 ngày", fontsize=8.3, color=COL["gray"])
    ax2.set_ylim(0, max(repeat_vals) * 1.35)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:.0f}%"))
    ax2.tick_params(axis="both", labelsize=8)
    style_axis(ax2)

    for rect in bars_repeat:
        h = rect.get_height()
        ax2.text(
            rect.get_x() + rect.get_width() / 2,
            h + max(repeat_vals) * 0.035,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=COL["dark"],
            weight="bold"
        )

    ax2.text(
        0.50,
        0.92,
        f"Gap: {repeat_gap:.1f} điểm %\n(promo còn thấp hơn)",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color="#E63946",
        weight="bold"
    )

    # ---------------- KPI panel ----------------
    axk.text(
        0.03, 0.98,
        "Chẩn đoán khuyến mãi",
        fontsize=12.8,
        weight="bold",
        color=COL["dark"],
        va="top"
    )

    add_kpi_box_local(
        axk,
        "Doanh thu từ đơn KM",
        f"{promo_revenue_share:.1f}%",
        "Tỷ trọng doanh thu sau 2018",
        y=0.70,
        color="#F4A261",
        height=0.18
    )

    add_kpi_box_local(
        axk,
        "Chênh lệch biên LN",
        f"{gm_gap:.1f} điểm %",
        "GM đơn KM − đơn không KM",
        y=0.47,
        color="#E63946",
        height=0.18
    )

    add_kpi_box_local(
        axk,
        "Chênh lệch mua lại 90 ngày",
        f"{repeat_gap:.1f} điểm %",
        "Mua lại đơn KM − không KM",
        y=0.24,
        color="#E63946",
        height=0.18
    )

    fig.suptitle(
        "A3. KHUYẾN MÃI: bán doanh thu, KHÔNG xây loyalty — cần tái thiết kế",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=13.0,
        weight="bold",
        color=COL["dark"]
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = ensure_dir(out_dir) / "A3_promo_refined_vi.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "a3_promo_revenue_share_pct": float(promo_revenue_share),
        "a3_promo_margin_gap_pp": float(gm_gap),
        "a3_promo_90d_repeat_gap_pp": float(repeat_gap),
    }


# ============================================================
# Runner
# ============================================================
def run_patch_a1_a3(data_dir: str | Path, out_dir: str | Path) -> Dict[str, float]:
    out_dir = ensure_dir(out_dir)
    summary = {}

    summary.update(plot_a1_original(data_dir, out_dir))
    summary.update(plot_a3_refined(data_dir, out_dir))

    summary_path = Path(out_dir) / "patch_a1_a3_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Đã ghi đè A1 và A3 vào: {out_dir}")
    print(f"Đã lưu metrics: {summary_path}")
    return summary