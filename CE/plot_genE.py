#!/usr/bin/env python
import uproot
import matplotlib.pyplot as plt
import numpy as np
# Publication-style matplotlib defaults: choose an available serif font
import matplotlib.font_manager as mfm
import matplotlib as mpl
preferred_serifs = ['DejaVu Serif', 'Times New Roman', 'Times', 'Palatino']
available_fonts = {f.name for f in mfm.fontManager.ttflist}
chosen_serif = next((f for f in preferred_serifs if f in available_fonts), 'DejaVu Serif')

mpl.rcParams.update({
  'font.family': 'serif',
  'font.serif': [chosen_serif],
  'font.size': 10,
  'axes.titlesize': 12,
  'axes.labelsize': 10,
  'xtick.labelsize': 9,
  'ytick.labelsize': 9,
  'legend.fontsize': 9,
  'axes.titleweight': 'bold',
  'axes.labelweight': 'normal',
  'axes.linewidth': 1.0,
  'grid.linewidth': 0.5,
  'figure.dpi': 150,
})

# Read ROOT file paths from CE_gen.txt
with open('CE_gen.txt', 'r') as f:
    files = [line.strip() for line in f if line.strip()]

print(f"Reading files:\n{files[0]}\n{files[1]}")

# Function to extract genE values from a ROOT file
def get_genE_values(filepath):
    root_file = uproot.open(filepath)
    tree = root_file["generate/GenAna"]
    genE_values = tree["genE"].array(library="np")
    return genE_values

# Extract data from both files
genE_CeEnd = get_genE_values(files[0])
genE_CeLL = get_genE_values(files[1])

print(f"CeEnd: {len(genE_CeEnd)} events")
print(f"CeLL: {len(genE_CeLL)} events")

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histograms on same axis

ax.hist(genE_CeLL, bins=25, range=(80, 110),  label='Szafron', color= "#ff8000", histtype="bar")
ax.hist(genE_CeEnd, bins=25, range=(80, 110),  label='Endpoint = 104.97 MeV', color='black', histtype="step", linestyle='dashed', linewidth=2.0)
fig.text(0.1, 0.98, "Mu2e Simulation", fontsize=14, fontweight='bold', fontstyle='italic',
           ha='left', va='top', transform=fig.transFigure, zorder=100)
ax.set_xlabel(r'E$_e$ [MeV]', fontsize=14)
ax.set_ylabel('Events / 1.2 MeV', fontsize=14)
ax.legend(fontsize=12)
ax.set_yscale('log')
#ax.grid(True, alpha=0.3)

# Calculate fraction of CeLL sample not in the 104.97 bin
bin_width = (110 - 80) / 25  # 1.2 MeV per bin
bin_center = 104.97
bin_lower = bin_center - bin_width / 2
bin_upper = bin_center + bin_width / 2

within_bin = np.sum((genE_CeLL >= bin_lower) & (genE_CeLL <= bin_upper))
outside_bin = len(genE_CeLL) - within_bin
fraction_outside = outside_bin / len(genE_CeLL)

print(f"\nCeLL Sample Analysis:")
print(f"Bin around 104.97 MeV: {bin_lower:.2f} - {bin_upper:.2f} MeV")
print(f"Events within bin: {within_bin}")
print(f"Events outside bin: {outside_bin}")
print(f"Fraction outside bin: {fraction_outside:.4f} ({fraction_outside*100:.2f}%)")

plt.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.1)
plt.savefig('genE_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as genE_comparison.png")
plt.show()
