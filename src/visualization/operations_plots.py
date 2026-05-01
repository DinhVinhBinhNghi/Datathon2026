from __future__ import annotations
import matplotlib.pyplot as plt
from src.visualization.plot_style import set_plot_style, save_current_figure


def plot_action_matrix(action_matrix, save_path: str | None = None):
    set_plot_style()
    fig, ax = plt.subplots()
    x = action_matrix.get("avg_stockout_days", action_matrix.get("stockout_months"))
    y = action_matrix["revenue"]
    ax.scatter(x, y, alpha=0.45)
    ax.set_title("SKU Action Matrix: Revenue vs Stockout Pressure")
    ax.set_xlabel("Average stockout days")
    ax.set_ylabel("Revenue")
    if save_path:
        save_current_figure(save_path)
    return fig, ax


def plot_return_reasons(return_summary, save_path: str | None = None, top_n: int = 10):
    set_plot_style()
    df = return_summary.groupby("return_reason", as_index=False)["return_records"].sum().sort_values("return_records", ascending=True).tail(top_n)
    fig, ax = plt.subplots()
    ax.barh(df["return_reason"], df["return_records"])
    ax.set_title("Top Return Reasons")
    ax.set_xlabel("Return records")
    ax.set_ylabel("Return reason")
    if save_path:
        save_current_figure(save_path)
    return fig, ax
