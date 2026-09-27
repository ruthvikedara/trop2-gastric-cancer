"""Re-exports genolib.style."""

from genolib.style import *          # noqa: F401,F403
from genolib.style import (          # noqa: F401  (explicit: names `*` skips / IDE clarity)
    FIREBRICK3,
    GOLDENROD3,
    DODGERBLUE3,
    CHARCOAL,
    MID_GRAY,
    PAL,
    TICK_FS,
    LABEL_FS,
    TITLE_FS,
    LEGEND_FS,
    ANNOT_FS,
    CELL_FS,
    HEATMAP_YTICK_FS,
    HEATMAP_XTICK_FS,
    apply_pub_style,
    clean_ax,
    savefig,
    fmt_p,
    stars,
    axis_label,
    _tint,
)
