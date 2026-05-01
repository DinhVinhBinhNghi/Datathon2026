import os
import numpy as np
import pandas as pd # Added pandas import for robust loading
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
# Ensure sales_df is loaded. If not already defined, load it from DATA_PATH.
# This makes the cell robust to kernel restarts or out-of-order execution.
if 'sales_df' not in locals() and 'sales_df' not in globals():
    print("`sales_df` was not found in memory. Attempting to load from file...")
    # Assuming DATA_PATH is defined in a previous cell (e.g., R20p0I-dehQd)
    if 'DATA_PATH' not in locals() and 'DATA_PATH' not in globals():
        raise NameError("`DATA_PATH` is not defined. Please ensure data loading cells (like R20p0I-dehQd and Cv-L1gz5KJmY) have been executed.")
    try:
        sales_df = pd.read_csv(os.path.join(DATA_PATH, 'sales.csv'), parse_dates=["Date"])
        print("`sales_df` loaded successfully from file.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find 'sales.csv' at {os.path.join(DATA_PATH, 'sales.csv')}. Please check DATA_PATH and file existence.")
    except Exception as e:
        raise Exception(f"Error loading sales.csv: {e}")

sales = sales_df.copy()

# Ensure the 'Date' column is in datetime format
sales["Date"] = pd.to_datetime(sales["Date"])

# Basic validation
required_cols = {"Date", "Revenue", "COGS"}
missing_cols = required_cols - set(sales.columns)
if missing_cols:
    raise ValueError(f"sales.csv thiếu cột: {missing_cols}")

sales = sales.sort_values("Date").copy()
sales["year"] = sales["Date"].dt.year
sales["month"] = sales["Date"].dt.month
sales["ym"] = sales["Date"].dt.to_period("M").dt.to_timestamp()

# Daily gross margin
sales["gross_margin"] = (sales["Revenue"] - sales["COGS"]) / sales["Revenue"]

# ------------------------------------------------------------
# 2. Monthly aggregation
# ------------------------------------------------------------
monthly = (
    sales.groupby("ym", as_index=False)
    .agg(
        Revenue=("Revenue", "sum"),
        COGS=("COGS", "sum"),
        n_days=("Date", "count")
    )
)

monthly["gross_margin"] = (monthly["Revenue"] - monthly["COGS"]) / monthly["Revenue"]
monthly["year"] = monthly["ym"].dt.year
monthly["month"] = monthly["ym"].dt.month
monthly["period"] = np.where(monthly["year"] <= 2018, "2013–2018", "2019–2022")

# Keep full-year comparison only
pre = monthly[(monthly["year"] >= 2013) & (monthly["year"] <= 2018)].copy()
post = monthly[(monthly["year"] >= 2019) & (monthly["year"] <= 2022)].copy()

def summarize_period(df):
    total_rev = df["Revenue"].sum()
    total_cogs = df["COGS"].sum()
    total_days = df["n_days"].sum()

    peak_idx = df["Revenue"].idxmax()
    min_gm_idx = df["gross_margin"].idxmin()

    return pd.Series({
        "Total Revenue": total_rev,
        "Avg Daily Revenue": total_rev / total_days,
        "Avg Monthly Revenue": df["Revenue"].mean(),
        "Peak Month Revenue": df.loc[peak_idx, "Revenue"],
        "Peak Month": df.loc[peak_idx, "ym"].strftime("%Y-%m"),
        "Weighted Gross Margin": (total_rev - total_cogs) / total_rev,
        "Worst Gross Margin": df.loc[min_gm_idx, "gross_margin"],
        "Worst GM Month": df.loc[min_gm_idx, "ym"].strftime("%Y-%m")
    })

summary = pd.DataFrame({
    "2013–2018": summarize_period(pre),
    "2019–2022": summarize_period(post)
})

# Month-8 anomaly check
august = monthly[monthly["month"] == 8].copy()
august["is_odd_year"] = august["year"] % 2 == 1
august_anomaly = august[august["gross_margin"] < 0].copy()

negative_gm_months = monthly[monthly["gross_margin"] < 0].copy()

print("===== A1 PERIOD SUMMARY =====")
display(summary)

print("\n===== AUGUST GROSS MARGIN CHECK =====")
display(
    august[["ym", "Revenue", "COGS", "gross_margin"]]
    .assign(
        Revenue=lambda d: d["Revenue"].round(0),
        COGS=lambda d: d["COGS"].round(0),
        gross_margin=lambda d: (d["gross_margin"] * 100).round(2)
    )
)

print("\n===== NEGATIVE GROSS MARGIN MONTHS =====")
display(
    negative_gm_months[["ym", "Revenue", "COGS", "gross_margin"]]
    .assign(
        Revenue=lambda d: d["Revenue"].round(0),
        COGS=lambda d: d["COGS"].round(0),
        gross_margin=lambda d: (d["gross_margin"] * 100).round(2)
    )
)

# ------------------------------------------------------------
# 3. Pretty plotting helpers
# ------------------------------------------------------------
COL = {
    "navy": "#1F3A5F",
    "blue": "#2F6B9A",
    "light_blue": "#D8E8F5",
    "orange": "#E76F51",
    "red": "#C0392B",
    "green": "#2A9D8F",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
    "bg": "#FFFFFF"
}

def money_million(x, pos=None):
    return f"{x/1e6:.0f}M"

def pct_fmt(x, pos=None):
    return f"{x*100:.0f}%"

def add_period_shading(ax):
    # 2013–2018
    ax.axvspan(
        pd.Timestamp("2013-01-01"),
        pd.Timestamp("2018-12-31"),
        color=COL["light_blue"],
        alpha=0.28,
        lw=0
    )
    # 2019–2022
    ax.axvspan(
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2022-12-31"),
        color=COL["light_gray"],
        alpha=0.33,
        lw=0
    )
    # Split line
    ax.axvline(
        pd.Timestamp("2019-01-01"),
        color=COL["gray"],
        lw=1.4,
        ls="--",
        alpha=0.9
    )

def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(axis="both", colors=COL["gray"], labelsize=9)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)

def add_kpi_box(ax, title, value, subtitle, y, color):
    box = FancyBboxPatch(
        (0.03, y), 0.94, 0.145,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.8,
        edgecolor="#E5E7EB",
        facecolor="#F9FAFB"
    )
    ax.add_patch(box)
    ax.text(0.07, y + 0.105, title, fontsize=9.3, color=COL["gray"], weight="medium")
    ax.text(0.07, y + 0.058, value, fontsize=15.5, color=color, weight="bold")
    ax.text(0.07, y + 0.025, subtitle, fontsize=8.2, color=COL["gray"])

# ------------------------------------------------------------
# 4. Create figure
# ------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"

fig = plt.figure(figsize=(15.5, 8.2), facecolor=COL["bg"])
gs = GridSpec(
    2, 3,
    width_ratios=[4.8, 4.8, 2.45],
    height_ratios=[1.15, 1],
    figure=fig,
    wspace=0.34,
    hspace=0.32
)

ax_rev = fig.add_subplot(gs[0, :2])
ax_gm = fig.add_subplot(gs[1, :2], sharex=ax_rev)
ax_kpi = fig.add_subplot(gs[:, 2])
ax_kpi.axis("off")

# ---------------- Revenue panel ----------------
add_period_shading(ax_rev)

ax_rev.plot(
    monthly["ym"],
    monthly["Revenue"],
    color=COL["navy"],
    lw=2.4,
    zorder=3
)

# Mark peak pre/post
for df, label, color in [
    (pre, "Peak trước 2019", COL["blue"]),
    (post, "Peak sau 2018", COL["orange"])
]:
    peak_row = df.loc[df["Revenue"].idxmax()]
    ax_rev.scatter(
        peak_row["ym"],
        peak_row["Revenue"],
        s=80,
        color=color,
        edgecolor="white",
        linewidth=1.4,
        zorder=5
    )
    ax_rev.annotate(
        f"{label}\n{peak_row['Revenue']/1e6:.0f}M",
        xy=(peak_row["ym"], peak_row["Revenue"]),
        xytext=(8, 18),
        textcoords="offset points",
        fontsize=8.7,
        color=color,
        weight="bold",
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0)
    )

ax_rev.set_title(
    "A1. Revenue shifted to a lower regime after 2018",
    loc="left",
    fontsize=15,
    color=COL["dark"],
    pad=12
)
ax_rev.set_ylabel("Monthly revenue", fontsize=10, color=COL["gray"])
ax_rev.yaxis.set_major_formatter(mticker.FuncFormatter(money_million))
style_axis(ax_rev)

# Period labels
ax_rev.text(
    pd.Timestamp("2015-12-01"),
    ax_rev.get_ylim()[1] * 0.93,
    "2013–2018\nhigher revenue base",
    ha="center",
    va="top",
    fontsize=9.5,
    color=COL["blue"],
    weight="bold"
)
ax_rev.text(
    pd.Timestamp("2020-12-01"),
    ax_rev.get_ylim()[1] * 0.93,

    "2019–2022\nlower revenue base",
    ha="center",
    va="top",
    fontsize=9.5,
    color=COL["gray"],
    weight="bold"
)

# ---------------- Gross margin panel ----------------
add_period_shading(ax_gm)

ax_gm.plot(
    monthly["ym"],
    monthly["gross_margin"],
    color=COL["green"],
    lw=2.0,
    zorder=3
)

ax_gm.axhline(
    0,
    color=COL["dark"],
    lw=1,
    alpha=0.75
)

# Highlight negative GM months
ax_gm.scatter(
    negative_gm_months["ym"],
    negative_gm_months["gross_margin"],
    s=64,
    color=COL["red"],
    edgecolor="white",
    linewidth=1.2,
    zorder=6,
    label="Negative gross margin"
)

# Highlight August odd-year anomaly
august_odd_negative = august[(august["is_odd_year"]) & (august["gross_margin"] < 0)]
ax_gm.scatter(
    august_odd_negative["ym"],
    august_odd_negative["gross_margin"],
    s=120,
    facecolors="none",
    edgecolors=COL["red"],
    linewidth=2.0,
    zorder=7
)

if len(august_odd_negative) > 0:
    worst_aug = august_odd_negative.loc[august_odd_negative["gross_margin"].idxmin()]
    ax_gm.annotate(
        "August odd-year\nmargin collapse",
        xy=(worst_aug["ym"], worst_aug["gross_margin"]),
        xytext=(-105, 28),
        textcoords="offset points",
        fontsize=8.8,
        color=COL["red"],
        weight="bold",
        arrowprops=dict(arrowstyle="->", color=COL["red"], lw=1.1)
    )

ax_gm.set_title(
    "Gross margin anomaly repeats in odd-year Augusts",
    loc="left",
    fontsize=13,
    color=COL["dark"],
    pad=10
)
ax_gm.set_ylabel("Gross margin", fontsize=10, color=COL["gray"])
ax_gm.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
style_axis(ax_gm)

# X-axis formatting
years = pd.date_range("2013-01-01", "2023-01-01", freq="YS")
ax_gm.set_xticks(years)
ax_gm.set_xticklabels([d.year for d in years], fontsize=9)

# ---------------- KPI side panel ----------------
ax_kpi.text(
    0.03, 0.965,
    "Key audit findings",
    fontsize=14.5,
    weight="bold",
    color=COL["dark"],
    va="top"
)

# Calculations for KPI boxes
avg_daily_pre = summary.loc["Avg Daily Revenue", "2013–2018"]
avg_daily_post = summary.loc["Avg Daily Revenue", "2019–2022"]
avg_daily_change = (avg_daily_post / avg_daily_pre - 1) * 100

peak_pre = summary.loc["Peak Month Revenue", "2013–2018"]
peak_post = summary.loc["Peak Month Revenue", "2019–2022"]
peak_change = (peak_post / peak_pre - 1) * 100

gm_pre = summary.loc["Weighted Gross Margin", "2013–2018"]
gm_post = summary.loc["Weighted Gross Margin", "2019–2022"]
gm_change_pp = (gm_post - gm_pre) * 100

neg_aug_count = len(august_odd_negative)

add_kpi_box(
    ax_kpi,
    "Avg daily revenue",
    f"{avg_daily_change:.1f}%",
    f"{avg_daily_pre/1e6:.2f}M → {avg_daily_post/1e6:.2f}M",
    y=0.76,
    color=COL["red"]
)

add_kpi_box(
    ax_kpi,
    "Peak monthly revenue",
    f"{peak_change:.1f}%",
    f"{peak_pre/1e6:.0f}M → {peak_post/1e6:.0f}M",
    y=0.57,
    color=COL["orange"]
)

add_kpi_box(
    ax_kpi,
    "Weighted gross margin",
    f"{gm_change_pp:.1f} pp",
    f"{gm_pre*100:.1f}% → {gm_post*100:.1f}%",
    y=0.38,
    color=COL["gray"]
)

add_kpi_box(
    ax_kpi,
    "Negative August pattern",
    f"{neg_aug_count} times",
    "Odd-year Augusts show negative GM",
    y=0.19,
    color=COL["red"]
)

ax_kpi.text(
    0.03, 0.08,
    "Interpretation:\nThe issue is not simple seasonality.\nAfter 2018, peak revenue weakened,\nwhile recurring August margin shocks\nneed data/business audit.",
    fontsize=9.2,
    color=COL["dark"],
    linespacing=1.35,
    va="top"
)

# Main title
fig.suptitle(
    "Revenue decline is structural, not just seasonal",
    x=0.04,
    y=0.985,
    ha="left",
    fontsize=18.5,
    weight="bold",
    color=COL["dark"]
)

fig.text(
    0.04,
    0.945,
    "Monthly aggregation from sales.csv. Gross margin = (Revenue − COGS) / Revenue.",
    fontsize=9.5,
    color=COL["gray"]
)

plt.tight_layout(rect=[0, 0, 1, 0.93])

# ------------------------------------------------------------
# 5. Save figure
# ------------------------------------------------------------
OUT_PATH = "A1_revenue_regime_shift_margin_anomaly.png"
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

print(f"Saved figure to: {OUT_PATH}")