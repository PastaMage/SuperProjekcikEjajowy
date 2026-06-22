import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import pandas as pd


PLATFORM_LABELS = ["TikTok", "Instagram", "Both"]
PLATFORM_COLORS = ["#378ADD", "#D4537E", "#1D9E75"]

HIST_COLORS = ["#378ADD", "#D4537E", "#1D9E75", "#BA7517", "#533AB7", "#D85A30", "#639922", "#993556", "#185FA5"]

DEPRESSION_COLORS = {"No Depression": "#1D9E75", "Depression": "#D85A30"}

DEPRESSION_COL = "depression_label"
DEPRESSION_LABELS = {0: "No Depression", 1: "Depression"}

NUMERIC_COLS = [
    "age",
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level",
    "addiction_level",
]

STYLE = {
    "bg":          "#F8F7F4",
    "panel":       "#FFFFFF",
    "text_dark":   "#2C2C2A",
    "text_muted":  "#888780",
    "grid":        "#D3D1C7",
    "mean_line":   "#2C2C2A",
    "median_line": "#888780",
    "spine_color": "#D3D1C7",
}

def _apply_base_style(fig: Figure) -> None:
    fig.patch.set_facecolor(STYLE["bg"])


def _style_ax(ax: Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(STYLE["panel"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["spine_color"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=STYLE["text_muted"], labelsize=9)
    ax.xaxis.label.set_color(STYLE["text_muted"])
    ax.yaxis.label.set_color(STYLE["text_muted"])
    ax.grid(axis="y", color=STYLE["grid"], linewidth=0.7, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=STYLE["text_dark"], pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)

def plot_platform_usage(df: pd.DataFrame) -> None:
    counts = df["platform_usage"].value_counts().reindex(PLATFORM_LABELS)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _apply_base_style(fig)

    bars = ax.bar(PLATFORM_LABELS, counts.to_numpy(), color=PLATFORM_COLORS, width=0.55, zorder=3)

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.02,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color=STYLE["text_dark"],
        )

    _style_ax(ax, title="Platform Usage Distribution", ylabel="Count")
    ax.set_xticks(range(len(PLATFORM_LABELS)))
    ax.set_xticklabels(PLATFORM_LABELS, fontsize=11, color=STYLE["text_dark"])
    ax.set_ylim(0, counts.max() * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.5)
    plt.show()


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    _apply_base_style(fig)
    fig.suptitle(
        "Distribution of Key Numeric Features",
        fontsize=14, fontweight="bold",
        color=STYLE["text_dark"], y=1.01,
    )

    for ax, col, color in zip(axes.flatten(), NUMERIC_COLS, HIST_COLORS):
        data = df[col].dropna()
        mean_val   = data.mean()
        median_val = data.median()

        ax.hist(data, bins=20, color=color, edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
        ax.axvline(mean_val,   color=STYLE["mean_line"],   ls="--", lw=1.4, zorder=4)
        ax.axvline(median_val, color=STYLE["median_line"], ls=":",  lw=1.4, zorder=4)

        legend_handles = [
            mpatches.Patch(color=STYLE["mean_line"],   label=f"Mean: {mean_val:.1f}"),
            mpatches.Patch(color=STYLE["median_line"], label=f"Median: {median_val:.1f}"),
        ]
        ax.legend(handles=legend_handles, fontsize=7.5, frameon=False,
                  labelcolor=STYLE["text_muted"])

        _style_ax(ax, title=col.replace("_", " ").title(), xlabel="Value", ylabel="Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    plt.show()


def plot_depression_balance(df: pd.DataFrame) -> None:
    counts = (
        df[DEPRESSION_COL]
        .value_counts()
        .rename(index=DEPRESSION_LABELS)
        .reindex(["Depression", "No Depression"])
    )
    colors = [DEPRESSION_COLORS[label] for label in counts.index]
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 2.8))
    _apply_base_style(fig)

    bars = ax.barh(counts.index, counts.to_numpy(), color=colors, height=0.45, zorder=3)

    for bar, (label, count) in zip(bars, counts.items()):
        pct = count / total * 100
        ax.text(
            count + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({pct:.1f}%)",
            va="center", fontsize=10, fontweight="bold",
            color=STYLE["text_dark"],
        )

    _style_ax(ax, title="Depression Label Distribution", xlabel="Count")
    ax.grid(axis="x", color=STYLE["grid"], linewidth=0.7, linestyle="-", zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.tick_params(axis="y", labelsize=11, colors=STYLE["text_dark"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.5)
    plt.show()

    print(f"\nValue counts:\n{counts}")
    print(f"\nColumn dtypes:\n{df.dtypes}")