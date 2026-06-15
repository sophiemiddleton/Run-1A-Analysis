"""
Style configuration file for consistent plotting and visualization settings.
Import and use: from styles import COLORS, FONTS, PLOT_STYLE
"""

# Color palette
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17becf',
    'background': '#ffffff',
    'text': '#000000',
    'grid': '#cccccc',
}

# Font settings
FONTS = {
    'title': {
        'size': 16,
        'weight': 'bold',
        'family': 'sans-serif',
    },
    'label': {
        'size': 12,
        'weight': 'normal',
        'family': 'sans-serif',
    },
    'tick': {
        'size': 10,
        'weight': 'normal',
        'family': 'sans-serif',
    },
    'legend': {
        'size': 10,
        'weight': 'normal',
        'family': 'sans-serif',
    },
}

# Plot styling
PLOT_STYLE = {
    'figure_size': (10, 6),
    'dpi': 100,
    'line_width': 2,
    'marker_size': 8,
    'alpha': 0.7,
    'grid': True,
    'grid_style': '--',
    'grid_alpha': 0.3,
}

# Matplotlib rcParams
MATPLOTLIB_RC = {
    'figure.figsize': PLOT_STYLE['figure_size'],
    'figure.dpi': PLOT_STYLE['dpi'],
    'font.size': FONTS['label']['size'],
    'lines.linewidth': PLOT_STYLE['line_width'],
    'lines.markersize': PLOT_STYLE['marker_size'],
    'axes.labelsize': FONTS['label']['size'],
    'axes.titlesize': FONTS['title']['size'],
    'xtick.labelsize': FONTS['tick']['size'],
    'ytick.labelsize': FONTS['tick']['size'],
    'legend.fontsize': FONTS['legend']['size'],
    'axes.grid': PLOT_STYLE['grid'],
    'grid.linestyle': PLOT_STYLE['grid_style'],
    'grid.alpha': PLOT_STYLE['grid_alpha'],
    'grid.color': COLORS['grid'],
}
