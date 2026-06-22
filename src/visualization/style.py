from cycler import cycler
from matplotlib.axes import Axes
from matplotlib import pyplot as plt

BRAND_BURGUNDY = "#7a003c"
TEXT_GRAY = "#3a3a3a"
GRID_GRAY = "#d9d9d9"
SPINE_GRAY = "#b8b8b8"

CONTRAST_COLORS = (
    BRAND_BURGUNDY,
    "#0072B2",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#332288",
)


def set_presentation_style() -> None:
    """Apply the project plotting palette for white-background slides/reports."""
    plt.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=CONTRAST_COLORS),
            "axes.edgecolor": SPINE_GRAY,
            "axes.labelcolor": TEXT_GRAY,
            "axes.titlecolor": BRAND_BURGUNDY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": TEXT_GRAY,
            "xtick.color": TEXT_GRAY,
            "ytick.color": TEXT_GRAY,
            "grid.color": GRID_GRAY,
            "legend.frameon": False,
        }
    )


def style_axis(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent title, label, grid, and spine styling to one axes."""
    ax.set_title(title, fontsize=13, color=BRAND_BURGUNDY, fontweight="semibold")
    ax.set_xlabel(xlabel, color=TEXT_GRAY)
    ax.set_ylabel(ylabel, color=TEXT_GRAY)
    ax.tick_params(colors=TEXT_GRAY)
    ax.grid(True, alpha=0.35, color=GRID_GRAY)

    for spine in ax.spines.values():
        spine.set_color(SPINE_GRAY)
