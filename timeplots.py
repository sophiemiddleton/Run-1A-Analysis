import uproot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# --- Font and Plot Configuration ---
preferred_serifs = ['DejaVu Serif', 'Times New Roman', 'Times', 'Palatino']
available_fonts = {f.name for f in mfm.fontManager.ttflist}
chosen_serif = next((f for f in preferred_serifs if f in available_fonts), 'DejaVu Serif')

mpl.rcParams.update({
  'font.family': 'serif',
  'font.serif': [chosen_serif],
  'font.size': 11,
  'axes.linewidth': 1.2,
  'xtick.direction': 'in',
  'ytick.direction': 'in',
  'xtick.major.size': 6,
  'ytick.major.size': 6,
  'xtick.minor.size': 3,
  'ytick.minor.size': 3,
  'xtick.top': True,
  'ytick.right': True,
  'figure.dpi': 150,
})

# --- Load Data ---
combined_file = uproot.open("/exp/mu2e/app/home/mu2epro/ensembles/RPC_timestudy/nts.owner.FilterPlots.version.sequencer.root")
features = combined_file["PionFilter/GenAna"]
df = features.arrays(library='pd')

# Broad limit for the visible plot range (100 to 510)
plot_mask = (df["globaltime"] >= 100) & (df["globaltime"] <= 510)
times_plot = df["globaltime"][plot_mask].to_numpy()
weights_plot = df["weight"][plot_mask].to_numpy()

# --- Calculate Weighted Bin Contents and Errors ---
# 42 edges = 41 bins from 100 to 510 -> exactly 10 ns bin width
bin_edges = np.linspace(100, 510, 42) 
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
bin_width = bin_edges[1] - bin_edges[0]

# Calculate weighted counts per bin (sum of weights)
counts, _ = np.histogram(times_plot, bins=bin_edges, weights=weights_plot)

# Calculate sum of weights squared for proper error bars (matching ROOT Sumw2)
variance, _ = np.histogram(times_plot, bins=bin_edges, weights=weights_plot**2)
errors = np.sqrt(variance)

# --- Non-linear Chi2 Fit via curve_fit (t: 250 to 450) ---
# Filter global bin arrays down to your strict fit window
fit_bins_mask = (bin_centers >= 250) & (bin_centers <= 450)

x_fit = bin_centers[fit_bins_mask]
y_fit = counts[fit_bins_mask]
y_err = errors[fit_bins_mask]

# Exclude empty bins or bins with zero error to keep the fit stable
valid_fit = (y_fit > 0) & (y_err > 0)
x_fit = x_fit[valid_fit]
y_fit = y_fit[valid_fit]
y_err = y_err[valid_fit]

# Define the explicit exponential function description
def exp_func(x, N0, lam):
    return N0 * np.exp(lam * x)

# Perform the direct Chi2 regression on the raw counts
# p0 provides stable initial guesses [N0, lambda]
popt, pcov = curve_fit(
    exp_func, x_fit, y_fit, 
    sigma=y_err, absolute_sigma=True, 
    p0=[y_fit[0] * 2, -0.045]
)

n0_val = popt[0]
lam_val = popt[1]
lam_err = np.sqrt(pcov[1][1]) # Extract uncertainty on the slope parameter

# --- Plotting ---
fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
ax1.set_yscale('log')

# 1. Plot raw weighted data points as crosses (E1 style)
valid = counts > 0
ax1.errorbar(
    bin_centers[valid], counts[valid], 
    yerr=errors[valid], xerr=bin_width/2, 
    fmt='none', ecolor='darkblue', elinewidth=1, capsize=0, label="pions"
)

# 2. Plot the smooth fit function across the domain
time_plot = np.linspace(250, 450, 1000)
fitcurve_curve = exp_func(time_plot, n0_val, lam_val)

ax1.plot(time_plot, fitcurve_curve, color="red", linestyle="-", linewidth=2.0, label="Chi2 fit")

# --- ROOT-Style Stats Box ---
total_entries = len(times_plot)
mean_val = np.average(times_plot, weights=weights_plot)
std_val = np.sqrt(np.average((times_plot - mean_val)**2, weights=weights_plot))

stats_text = (
    f"PionGlobalTimeWeight\n"
    f"Entries         {total_entries}\n"
    f"Mean            {mean_val:.1f}\n"
    f"Std Dev         {std_val:.2f}\n"
    f"Slope    {lam_val:.5f} ± {lam_err:.5f}"
)

ax1.text(
    0.98, 0.96, stats_text, 
    transform=ax1.transAxes, fontsize=9, fontfamily='monospace',
    va='top', ha='right', 
    bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
)

# --- Labels & Tuning ---
ax1.set_title("Stopped pion time", fontsize=14, pad=12)
ax1.set_xlabel("time (ns)", ha='right', x=1.0)
ax1.set_ylabel("Pions (A.U.)", ha='right', y=1.0)
ax1.set_xlim(100, 510)
ax1.set_ylim(1e-4, 1000) 

plt.savefig("stopped_pion_time.png", bbox_inches='tight')
plt.show()
