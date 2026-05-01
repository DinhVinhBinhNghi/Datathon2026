from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization.plot_style import set_plot_style, save_current_figure


def plot_monthly_revenue(monthly: pd.DataFrame, save_path: str | None = None):
    set_plot_style()
    df = monthly.copy()
    df["year_month"] = pd.to_datetime(df["year_month"])
    fig, ax = plt.subplots()
    ax.plot(df["year_month"], df["revenue"])
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    if save_path:
        save_current_figure(save_path)
    return fig, ax


def plot_seasonality(daily: pd.DataFrame, save_path: str | None = None):
    set_plot_style()
    df = daily.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["month"] = df["Date"].dt.month
    season = df.groupby("month", as_index=False)["Revenue"].mean()
    fig, ax = plt.subplots()
    ax.bar(season["month"], season["Revenue"])
    ax.set_title("Average Daily Revenue by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average daily revenue")
    if save_path:
        save_current_figure(save_path)
    return fig, ax
