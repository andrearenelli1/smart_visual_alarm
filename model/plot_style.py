"""
Matplotlib style - Computer Modern Roman, screen-friendly sizes.
Import this module BEFORE any plt.* call.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

IEEE_RC = {
    # Font
    "font.family":                 "serif",
    "font.serif":                  ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "font.size":                   12,
    "axes.titlesize":              13,
    "axes.labelsize":              12,
    "xtick.labelsize":             11,
    "ytick.labelsize":             11,
    "legend.fontsize":             11,
    "figure.titlesize":            14,
    # Math (handles Greek letters and special chars)
    "mathtext.fontset":            "cm",
    "axes.formatter.use_mathtext": True,
    # Layout
    "axes.linewidth":              0.8,
    "grid.linewidth":              0.5,
    "lines.linewidth":             1.8,
    "patch.linewidth":             0.8,
    "xtick.major.width":           0.8,
    "ytick.major.width":           0.8,
    # Grid
    "axes.grid":                   True,
    "grid.alpha":                  0.4,
    "grid.linestyle":              "--",
    # Figure
    "figure.dpi":                  100,
    "savefig.dpi":                 150,
    "savefig.bbox":                "tight",
    "savefig.pad_inches":          0.1,
}


def apply():
    """Apply style globally."""
    mpl.rcParams.update(IEEE_RC)


# Apply automatically on import
apply()


# ── Figure sizes (inches) ─────────────────────────────────────────────────────
# Designed for screen readability; 150 dpi → approx pixel sizes shown below
SINGLE = (7,   5)      # ~1050 x  750 px
DOUBLE = (14,  5.5)    # ~2100 x  825 px  (two side-by-side panels)
SQUARE = (7,   6.5)    # ~1050 x  975 px
WIDE   = (14,  7)      # ~2100 x 1050 px  (2x2 grid)
TALL   = (7,  10)      # ~1050 x 1500 px
