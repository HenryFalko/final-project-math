"""Shared plotting style for all notebooks (keeps the report figures consistent)."""
import os

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# Palette: blue = from scratch, orange = scikit-learn, grey = reference
C_SCRATCH = "#2a6f97"
C_SKLEARN = "#e07a1f"
C_CLIMATE = "#2a6f97"
C_GEO = "#8d6e63"
C_NEUTRAL = "#7a7a7a"
C_ACCENT = "#c0392b"


def setup():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    })


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name))
