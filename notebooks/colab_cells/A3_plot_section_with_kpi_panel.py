# A3 plot section with KPI panel
# Use this to replace the plotting section of A3 after promo_summary/repeat_summary numbers are computed.
# It expects these variables to exist:
# nonpromo_revenue_share, promo_revenue_share, nonpromo_gm, promo_gm, gm_gap,
# nonpromo_repeat_rate, promo_repeat_rate, repeat_gap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

COL = {
    "navy": "#1F3A5F",
    "orange": "#F4A261",
    "red": "#E63946",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
}
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "bold"

def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(axis="both", colors=COL["gray"], labelsize=8.8)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8, alpha=0.75)
    ax.set_axisbelow(True)

def add_kpi_box(ax, title, value, subtitle, y, color):
    box = FancyBboxPatch((0.03, y), 0.94, 0.18, boxstyle="round,pad=0.018,rounding_size=0.025", linewidth=0.8, edgecolor="#E5E7EB", facecolor="#F9FAFB")
    ax.add_patch(box)
    ax.text(0.08, y + 0.124, title, fontsize=8.8, color=COL["gray"])
    ax.text(0.08, y + 0.073, value, fontsize=14.8, color=color, weight="bold")
    ax.text(0.08, y + 0.032, subtitle, fontsize=7.8, color=COL["gray"])

fig = plt.figure(figsize=(14.2, 4.85), facecolor="white")
gs = GridSpec(1, 3, width_ratios=[1.25, 1.0, 0.82], figure=fig, wspace=0.34)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
axk = fig.add_subplot(gs[0, 2])
axk.axis("off")

labels = ["Đơn không\nkhuyến mãi", "Đơn khuyến\nmãi"]
x = np.arange(2)
bar_w = 0.30
revenue_vals = [nonpromo_revenue_share, promo_revenue_share]
gm_vals = [nonpromo_gm, promo_gm]

ax1b = ax1.twinx()
bars_rev = ax1.bar(x - bar_w/2, revenue_vals, width=bar_w, color=["#9CA3AF", COL["orange"]], alpha=0.92)
bars_gm = ax1b.bar(x + bar_w/2, gm_vals, width=bar_w, color=[COL["navy"], COL["red"]], alpha=0.95)
ax1.set_title(f"A. KM mua {promo_revenue_share:.1f}% doanh thu\nbằng cách bào sạch {abs(gm_gap):.1f} điểm % biên LN", loc="left", fontsize=10.8, color=COL["dark"])
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("% doanh thu sau 2018", fontsize=9, color=COL["gray"])
ax1.set_ylim(0, max(revenue_vals) * 1.25)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:.0f}"))
ax1b.set_ylabel("Biên LN gộp (%)", fontsize=9, color=COL["gray"])
ax1b.set_ylim(0, max(gm_vals) * 1.55 if max(gm_vals) > 0 else 5)
ax1b.tick_params(axis="y", colors=COL["gray"], labelsize=8)
ax1b.spines["top"].set_visible(False)
ax1b.spines["right"].set_color("#D1D5DB")
style_axis(ax1)

for rect in bars_rev:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2, h + max(revenue_vals)*0.025, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color=COL["dark"], weight="bold")
for rect in bars_gm:
    h = rect.get_height()
    ax1b.text(rect.get_x() + rect.get_width()/2, h + max(gm_vals)*0.04, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color=rect.get_facecolor(), weight="bold")
ax1.text(0.50, 0.92, "■ % doanh thu", transform=ax1.transAxes, fontsize=8.4, color="#9CA3AF")
ax1.text(0.50, 0.84, "■ Biên LN gộp", transform=ax1.transAxes, fontsize=8.4, color=COL["navy"])

repeat_vals = [nonpromo_repeat_rate, promo_repeat_rate]
bars_repeat = ax2.bar(x, repeat_vals, width=0.48, color=[COL["navy"], COL["red"]], alpha=0.92)
ax2.set_title("B. KM KHÔNG kéo được khách quay lại\nluận điểm “đầu tư” sụp", loc="left", fontsize=10.8, color=COL["dark"])
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("% khách quay lại trong 90 ngày", fontsize=9, color=COL["gray"])
ax2.set_ylim(0, max(repeat_vals) * 1.35)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:.0f}%"))
style_axis(ax2)
for rect in bars_repeat:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2, h + max(repeat_vals)*0.035, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.7, color=COL["dark"], weight="bold")
ax2.text(0.50, 0.92, f"Gap: {repeat_gap:.1f} điểm %\n(promo còn thấp hơn)", transform=ax2.transAxes, ha="center", va="top", fontsize=9.0, color=COL["red"], weight="bold")

axk.text(0.03, 0.98, "Chẩn đoán khuyến mãi", fontsize=14.0, weight="bold", color=COL["dark"], va="top")
add_kpi_box(axk, "Doanh thu từ đơn KM", f"{promo_revenue_share:.1f}%", "Tỷ trọng doanh thu sau 2018", y=0.70, color=COL["orange"])
add_kpi_box(axk, "Chênh lệch biên LN", f"{gm_gap:.1f} điểm %", "GM đơn KM − đơn không KM", y=0.47, color=COL["red"])
add_kpi_box(axk, "Chênh lệch mua lại 90 ngày", f"{repeat_gap:.1f} điểm %", "Mua lại đơn KM − không KM", y=0.24, color=COL["red"])

fig.suptitle("A3. KHUYẾN MÃI: bán doanh thu, KHÔNG xây loyalty — cần tái thiết kế", x=0.02, y=1.02, ha="left", fontsize=13.0, weight="bold", color=COL["dark"])
fig.text(0.02, 0.945, "Đơn hợp lệ gồm delivered, returned và shipped. So sánh giai đoạn 2019–2022.", fontsize=8.8, color=COL["gray"], ha="left")
plt.tight_layout(rect=[0, 0, 1, 0.92])
OUT_PATH = "A3_promo_refined_vi.png"
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
print(f"Saved figure to: {OUT_PATH}")
