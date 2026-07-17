import uproot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
# --- Font and Plot Configuration ---
preferred_serifs = ['DejaVu Serif', 'Times New Roman', 'Times', 'Palatino']
available_fonts = {f.name for f in mfm.fontManager.ttflist}
chosen_serif = next((f for f in preferred_serifs if f in available_fonts), 'DejaVu Serif')

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': [chosen_serif],
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 9,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
})

# --- Load Data ---
combined_file = uproot.open("/exp/mu2e/app/home/mu2epro/ensembles/RPC_timestudy/nts.owner.FilterPlots_physical_100.version.sequencer.root")
combined_file_nonphys250 = uproot.open("/exp/mu2e/app/home/mu2epro/ensembles/RPC_timestudy/nts.owner.FilterPlots-250.version.sequencer.root")
combined_file_nonphys350 = uproot.open("/exp/mu2e/app/home/mu2epro/ensembles/RPC_timestudy/nts.owner.FilterPlots-350.version.sequencer.root")

# 1. Load Non-physical Data (350 - All and Select)
features_nonphys350 = combined_file_nonphys350["PionFilter/GenAna_all"]
df_nonphys350 = features_nonphys350.arrays(["endglobaltime", "startglobaltime", "parentstarttime", "weight"], library='pd')

features_nonphys350_select = combined_file_nonphys350["PionFilter/GenAna_select"]
df_nonphys350_select = features_nonphys350_select.arrays(["parentstarttime_select", "endglobaltime_select"], library='pd')

# 2. Load Non-physical Data (250 - All and Select)
features_nonphys250 = combined_file_nonphys250["PionFilter/GenAna_all"]
df_nonphys250 = features_nonphys250.arrays(["endglobaltime", "startglobaltime", "parentstarttime", "weight"], library='pd')

features_nonphys250_select = combined_file_nonphys250["PionFilter/GenAna_select"]
df_nonphys250_select = features_nonphys250_select.arrays(["parentstarttime_select", "endglobaltime_select"], library='pd')

# 3. Load Physical Main Data (For the other 3 panels)
features = combined_file["PionFilter/GenAna_all"]
df = features.arrays(["endglobaltime", "startglobaltime", "parentstarttime", "weight"], library='pd')
df["delta_time"] = df["endglobaltime"] - df["startglobaltime"]

# --- Target Masks ---
# Non-physical 350 masks
plot_mask_nonphys350 = (df_nonphys350["endglobaltime"] >= 100) & (df_nonphys350["endglobaltime"] <= 550)
parent_nonphys350_all = df_nonphys350["parentstarttime"][plot_mask_nonphys350].to_numpy()

select_mask_nonphys350 = (df_nonphys350_select["endglobaltime_select"] >= 100) & (df_nonphys350_select["endglobaltime_select"] <= 550)
parent_nonphys350_sel = df_nonphys350_select["parentstarttime_select"][select_mask_nonphys350].to_numpy()

# Non-physical 250 masks
plot_mask_nonphys250 = (df_nonphys250["endglobaltime"] >= 100) & (df_nonphys250["endglobaltime"] <= 550)
parent_nonphys250_all = df_nonphys250["parentstarttime"][plot_mask_nonphys250].to_numpy()

select_mask_nonphys250 = (df_nonphys250_select["endglobaltime_select"] >= 100) & (df_nonphys250_select["endglobaltime_select"] <= 550)
parent_nonphys250_sel = df_nonphys250_select["parentstarttime_select"][select_mask_nonphys250].to_numpy()

# Physical masks (for panels 2, 3, 4)
plot_mask = (df["endglobaltime"] >= 100) & (df["endglobaltime"] <= 550)
times_plot = df["endglobaltime"][plot_mask].to_numpy()
start_plot = df["startglobaltime"][plot_mask].to_numpy()
delta_plot = df["delta_time"][plot_mask].to_numpy()

# --- CUSTOM BINNING & WIDTHS ---
bin_edges_abs = np.arange(100, 555, 5) 
bin_centers_abs = (bin_edges_abs[:-1] + bin_edges_abs[1:]) / 2.0
w_abs = 5.0

bin_edges_prot = np.arange(-100, 105, 5)
bin_centers_prot = (bin_edges_prot[:-1] + bin_edges_prot[1:]) / 2.0

bin_edges_delta = np.arange(0, 50.5, 0.5)
bin_centers_delta = (bin_edges_delta[:-1] + bin_edges_delta[1:]) / 2.0
w_delta = 0.5

# --- Compute RAW Counts and Errors for Plot 1 (No Scaling) ---
# 350 Lines
counts_p350_all, _ = np.histogram(parent_nonphys350_all, bins=bin_edges_prot)
errors_p350_all = np.sqrt(counts_p350_all)

counts_p350_sel, _ = np.histogram(parent_nonphys350_sel, bins=bin_edges_prot)
errors_p350_sel = np.sqrt(counts_p350_sel)

# 250 Lines
counts_p250_all, _ = np.histogram(parent_nonphys250_all, bins=bin_edges_prot)
errors_p250_all = np.sqrt(counts_p250_all)

counts_p250_sel, _ = np.histogram(parent_nonphys250_sel, bins=bin_edges_prot)
errors_p250_sel = np.sqrt(counts_p250_sel)

# --- Compute Internal Normalizations for Plots 2, 3, 4 ---
counts_start, _ = np.histogram(start_plot, bins=bin_edges_abs)
int_start = np.sum(counts_start) * w_abs
norm_start = counts_start / int_start
err_start = np.sqrt(counts_start) / int_start

counts_end, _ = np.histogram(times_plot, bins=bin_edges_abs)
int_end = np.sum(counts_end) * w_abs
norm_end = counts_end / int_end
err_end = np.sqrt(counts_end) / int_end

counts_delta, _ = np.histogram(delta_plot, bins=bin_edges_delta)
int_delta = np.sum(counts_delta) * w_delta
norm_delta = counts_delta / int_delta
err_delta = np.sqrt(counts_delta) / int_delta

# --- Non-linear Fit on Normalized End Time ---
fit_bins_mask = (bin_centers_abs >= 270) & (bin_centers_abs <= 450)
x_fit = bin_centers_abs[fit_bins_mask]
y_fit = norm_end[fit_bins_mask]
y_err = err_end[fit_bins_mask]

valid_fit = (y_fit > 0) & (y_err > 0)
x_fit = x_fit[valid_fit]
y_fit = y_fit[valid_fit]
y_err = y_err[valid_fit]

def exp_func(x, N0, lam):
    return N0 * np.exp(lam * x)

popt, pcov = curve_fit(exp_func, x_fit, y_fit, sigma=y_err, absolute_sigma=True, p0=[y_fit[0] * 2, -0.045])
n0_val, lam_val = popt[0], popt[1]
lam_err = np.sqrt(pcov[1][1])

# --- Plotting Subplots ---
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))

# --- PLOT 1: Proton Start Time (Raw Non-Phys Master and Subsets) ---
# 1a. Non-Phys 350 Master (Raw Counts)
ax1.hist(parent_nonphys350_all, bins=bin_edges_prot, histtype='step', color='black', linewidth=1.5, label="All Protons")
valid_1a = counts_p350_all > 0
ax1.errorbar(bin_centers_prot[valid_1a], counts_p350_all[valid_1a], yerr=errors_p350_all[valid_1a], fmt='none', ecolor='black', elinewidth=1, capsize=0)

# 1b. Non-Phys 350 Selected Subset
ax1.hist(parent_nonphys350_sel, bins=bin_edges_prot, histtype='step', color='crimson', linestyle='--', linewidth=1.5, label=r"$t_{ST}$ > 350")
valid_1b = counts_p350_sel > 0
ax1.errorbar(bin_centers_prot[valid_1b], counts_p350_sel[valid_1b], yerr=errors_p350_sel[valid_1b], fmt='none', ecolor='crimson', elinewidth=1, capsize=0)

# 2b. Non-Phys 250 Selected Subset
ax1.hist(parent_nonphys250_sel, bins=bin_edges_prot, histtype='step', color='green', linestyle='-.', linewidth=1.2, label=r"$t_{ST}$ > 250")
valid_2b = counts_p250_sel > 0
ax1.errorbar(bin_centers_prot[valid_2b], counts_p250_sel[valid_2b], yerr=errors_p250_sel[valid_2b], fmt='none', ecolor='green', elinewidth=1, capsize=0)

# Combined Statistics Box
stats_text_1 = (f"All Entries: {len(parent_nonphys350_all)}\n"
                f"t > 250 Sel Entries: {len(parent_nonphys250_sel)}\n"
                                f"t > 350 Sel Entries: {len(parent_nonphys350_sel)}")
ax1.text(0.05, 0.95, stats_text_1, transform=ax1.transAxes, fontsize=20, va='top', ha='left', bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
#ax1.set_yscale('log')
ax1.text(0.0, 1.02, "Mu2e Simulation", 
                fontsize=20, fontweight='bold', ha='left', va='bottom', transform=ax1.transAxes, zorder=100)
#ax1.set_title("Proton Bunch Hit Time ($t_{0}$)", fontsize=13, pad=12)
ax1.set_xlabel(r"Parent Proton Arrival Time $t_{0}$ [ns]",fontsize=mpl.rcParams.get('axes.titlesize', 24))
ax1.set_ylabel("Protons / 5 ns", fontsize=mpl.rcParams.get('axes.titlesize', 24))
ax1.set_xlim(-100, 100)
ax1.set_ylim(1, 8000)
# Matching minor/major tick configurations exactly
ax1.minorticks_on()
ax1.tick_params(direction='in', which='both', top=True, right=True, labelsize=18)
ax1.yaxis.set_minor_formatter(ticker.NullFormatter()) 


ax1.legend(loc='upper right', frameon=False, fontsize=16)
plt.tight_layout()
plt.savefig("stopped_pion_protons.pdf", bbox_inches='tight')
plt.show()


# --- PLOT 2: Pion Production Time ---
fig, ax2= plt.subplots(1,1)
ax2.set_yscale('log')
ax2.hist(start_plot, bins=bin_edges_abs, density=True, histtype='step', color='darkred', linewidth=1.5)
valid_2 = norm_start > 0
ax2.errorbar(bin_centers_abs[valid_2], norm_start[valid_2], yerr=err_start[valid_2], fmt='none', ecolor='darkred', elinewidth=1, capsize=0)

stats_text_2 = f"Entries: {len(start_plot)}\nMean: {np.mean(start_plot):.1f}\nStd Dev: {np.std(start_plot):.2f}"
ax2.text(0.05, 0.95, stats_text_2, transform=ax2.transAxes, fontsize=9, va='top', ha='left', bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
ax2.set_title("Pion Production Time ($t_{start}$)", fontsize=13, pad=12)
ax2.set_xlabel("time (ns)", ha='right', x=1.0)
ax2.set_ylabel("Probability Density / 5 ns", ha='right', y=1.0)
ax2.set_xlim(130, 460)
ax2.set_ylim(1e-6, 1e-1)
plt.tight_layout()
plt.savefig("stopped_pion_productiontime.png", bbox_inches='tight')
plt.show()

# --- PLOT 3: Stopped Pion Time ---
fig, (ax3) = plt.subplots(1, 1, figsize=(10, 8))
ax3.set_yscale('log')
ax3.hist(times_plot, bins=bin_edges_abs, density=True, histtype='step', color='darkblue', linewidth=1.5)
valid_3 = norm_end > 0
ax3.errorbar(bin_centers_abs[valid_3], norm_end[valid_3], yerr=err_end[valid_3], fmt='none', ecolor='darkblue', elinewidth=1, capsize=0)

time_plot_range = np.linspace(350, 400, 1000)
##ax3.plot(time_plot_range, exp_func(time_plot_range, n0_val, lam_val), color="red", linestyle="-", linewidth=2.0, label="Chi2 fit")

stats_text_3 = f"Entries: {len(times_plot)}\nMean: {np.mean(times_plot):.1f}\nStd Dev: {np.std(times_plot):.2f}\nSlope: {lam_val:.5f} ± {lam_err:.5f}"
#ax3.text(0.55, 0.95, stats_text_3, transform=ax3.transAxes, fontsize=16, va='top', ha='left', bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))

ax3.set_xlim(130, 460)
ax3.set_ylim(1e-6, 0.013)
#plt.tight_layout()

ax3.text(0.0, 1.02, "Mu2e Simulation", 
                fontsize=20, fontweight='bold', ha='left', va='bottom', transform=ax3.transAxes, zorder=100)
#ax1.set_title("Proton Bunch Hit Time ($t_{0}$)", fontsize=13, pad=12)
ax3.set_xlabel(r"Stopped Pion Time ($t_{ST}$) [ns]", fontsize=mpl.rcParams.get('axes.titlesize', 24))
ax3.set_ylabel("Probability Density / 5 ns", fontsize=mpl.rcParams.get('axes.titlesize', 24))

# Matching minor/major tick configurations exactly
ax3.minorticks_on()
ax3.tick_params(direction='in', which='both', top=True, right=True, labelsize=18)
ax3.yaxis.set_minor_formatter(ticker.NullFormatter()) 

ax3.legend(loc='upper right', frameon=False, fontsize=16)
plt.tight_layout()
plt.savefig("stopped_pion_times.pdf", bbox_inches='tight')
plt.show()

# --- PLOT 4: Pion Propagation Duration ---
fig, ax4 = plt.subplots(1,1)
ax4.set_yscale('log')
ax4.hist(delta_plot, bins=bin_edges_delta, density=True, histtype='step', color='darkgreen', linewidth=1.5)
valid_4 = norm_delta > 0
ax4.errorbar(bin_centers_delta[valid_4], norm_delta[valid_4], yerr=err_delta[valid_4], fmt='none', ecolor='darkgreen', elinewidth=1, capsize=0)

stats_text_4 = f"Entries: {len(delta_plot)}\nMean: {np.mean(delta_plot):.1f}\nStd Dev: {np.std(delta_plot):.2f}"
ax4.text(0.05, 0.95, stats_text_4, transform=ax4.transAxes, fontsize=9, va='top', ha='left', bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
ax4.set_title("Pion Propagation Duration ($t_{end} - t_{start}$)", fontsize=13, pad=12)
ax4.set_xlabel("$\Delta$ time (ns)", ha='right', x=1.0)
ax4.set_ylabel("Probability Density / 0.5 ns", ha='right', y=1.0)
ax4.set_xlim(10, 50)
ax4.set_ylim(1e-5, 5e-1)

plt.tight_layout()
plt.savefig("stopped_pion_proptime.png", bbox_inches='tight')
plt.show()