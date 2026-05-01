from __future__ import annotations
import matplotlib.pyplot as plt
from src.visualization.plot_style import set_plot_style, save_current_figure


def plot_category_revenue(category_summary, save_path: str | None = None):
    set_plot_style()
    df = category_summary.sort_values("revenue")
    fig, ax = plt.subplots()
    ax.barh(df["category"], df["revenue"])
    ax.set_title("Revenue by Product Category")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Category")
    if save_path:
        save_current_figure(save_path)
    return fig, ax


def plot_sku_pareto(pareto, save_path: str | None = None):
    set_plot_style()
    fig, ax = plt.subplots()
    ax.plot(pareto["sku_rank"], pareto["cum_revenue_share"])
    ax.axhline(0.8, linestyle="--")
    ax.set_title("SKU Pareto Curve")
    ax.set_xlabel("SKU rank by revenue")
    ax.set_ylabel("Cumulative revenue share")
    if save_path:
        save_current_figure(save_path)
    return fig, ax
