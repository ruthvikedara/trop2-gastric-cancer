"""Figure style: palette and plotting helpers."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path

# core palette (R color anchors)
FIREBRICK3  = '#CD2626'
GOLDENROD3  = '#CD9B1D'
DODGERBLUE3 = '#1874CD'
CHARCOAL    = '#1A1A1A'
MID_GRAY    = '#8A8A8A'

def _tint(hex_color, alpha):
    """Return an RGBA string (lighter by alpha blending over white)."""
    r, g, b = to_rgba(hex_color)[:3]
    r = r * alpha + 1 * (1 - alpha)
    g = g * alpha + 1 * (1 - alpha)
    b = b * alpha + 1 * (1 - alpha)
    return f'#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}'

PAL = {
    # direction coding
    'tumor':     FIREBRICK3,    # tumor / high / increased
    'normal':    DODGERBLUE3,   # normal / low / decreased
    'high':      FIREBRICK3,
    'low':       DODGERBLUE3,
    'increased': FIREBRICK3,
    'decreased': DODGERBLUE3,
    # accent
    'gold':      GOLDENROD3,
    'b7h4':      GOLDENROD3,    # B7-H4 / combo
    'combo':     GOLDENROD3,
    'stroma':    GOLDENROD3,    # confound / stroma
    # gold / black clinical style
    'accent_high': GOLDENROD3,  # TROP2-high / tumor
    'accent_low':  CHARCOAL,    # TROP2-low / normal
    'accent_mid':  MID_GRAY,    # mid / reference
    # neutral
    'gray':      '#CCCCCC',     # n.s. / neutral
    # MSI
    'msi':       GOLDENROD3,    # MSI-H (the "special" category)
    'mss':       DODGERBLUE3,
    # epithelial vs mesenchymal
    'ep':        DODGERBLUE3,
    'mp':        FIREBRICK3,
    # molecular subtypes (4 needed, tints + accents)
    'EBV':       GOLDENROD3,
    'MSI':       '#A8D5A8',     # keep a green for MSI (standard convention)
    'GS':        _tint(FIREBRICK3, 0.55),
    'CIN':       DODGERBLUE3,
    'MSS/TP53+': DODGERBLUE3,
    'MSS/TP53-': _tint(DODGERBLUE3, 0.55),
    'MSS/EMT':   FIREBRICK3,
    'EP':        DODGERBLUE3,
    'MP':        FIREBRICK3,
    # Lauren
    'Intestinal': '#A8D5A8',
    'Diffuse':    GOLDENROD3,
    'Mixed':      '#CCCCCC',
    # scRNA dataset colors
    'Kumar':     DODGERBLUE3,
    'Sathe':     FIREBRICK3,
    # TCGA normal sub-shade
    'tcga_norm': _tint(DODGERBLUE3, 0.55),
}

# Manuscript-readable defaults: bias UP so panels
# stay legible after multi-panel assembly / talk projection. Prefer these over
# matplotlib defaults; do not shrink below the floor without a reason.
TICK_FS = 9
LABEL_FS = 11
TITLE_FS = 12
LEGEND_FS = 9
ANNOT_FS = 9
CELL_FS = 8          # heatmap cell annotations
HEATMAP_YTICK_FS = 10
HEATMAP_XTICK_FS = 9


def apply_pub_style():
    """Set rcParams once for a script. Call at top of main()."""
    plt.rcParams.update({
        'font.size': TICK_FS,
        'axes.titlesize': TITLE_FS,
        'axes.labelsize': LABEL_FS,
        'xtick.labelsize': TICK_FS,
        'ytick.labelsize': TICK_FS,
        'legend.fontsize': LEGEND_FS,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,   # editable text in Illustrator
        'ps.fonttype': 42,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def clean_ax(ax, keep_left=True, keep_bottom=True):
    """Remove top/right spines for a clean look."""
    spines = {'top': False, 'right': False,
              'left': keep_left, 'bottom': keep_bottom}
    for sp, vis in spines.items():
        ax.spines[sp].set_visible(vis)


def savefig(fig, out_path, dpi=300, extra_artists=None, pad_inches=0.05):
    """Save a figure to PDF + PNG in the given location (no suffix needed).

    extra_artists: optional list of artists (e.g. legends anchored OUTSIDE the
    axes) to include in the 'tight' bounding box so they aren't clipped.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kw = {'bbox_extra_artists': extra_artists} if extra_artists else {}
    for ext in ('.pdf', '.png'):
        fig.savefig(out.with_suffix(ext), dpi=dpi, bbox_inches='tight',
                    pad_inches=pad_inches, **kw)
    plt.close(fig)
    print(f'  saved {out.name}.{{pdf,png}}')


def fmt_p(p):
    """Lab-standard p-value formatter for in-figure text."""
    import numpy as np
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'p = n/a'
    if p < 0.001:
        return f'p = {p:.2e}'
    return f'p = {p:.3f}'


def stars(p):
    """Deprecated: lab plotting guide prefers exact p-values over significance stars."""
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def axis_label(quantity, unit=None, transform=None):
    """Build an axis label that always states units / transforms.

    Examples:
        axis_label('TROP2 expression', transform='log2')
        → 'TROP2 expression (log2)'
        axis_label('Spearman r with TROP2', unit='median across cohorts')
        → 'Spearman r with TROP2 (median across cohorts)'
        axis_label('Expression', unit='transcriptome percentile', transform=None)
        → 'Expression (transcriptome percentile)'
    """
    bits = []
    if transform:
        bits.append(str(transform))
    if unit:
        bits.append(str(unit))
    if not bits:
        return quantity
    return f'{quantity} ({", ".join(bits)})'
