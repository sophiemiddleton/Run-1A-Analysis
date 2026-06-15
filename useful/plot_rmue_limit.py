#!/usr/bin/env python3
"""Simple script to plot Rmue limit vs days from CSV file."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Read CSV
df = pd.read_csv('rmue_v_time.csv')

# Calculate improvement over SINDRUM-II
sindrum_limit = 7e-13
df['improvement_factor'] = sindrum_limit / df['limitRmue']

# Create plot with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Rmue Limit vs Days
ax1.plot(df['Days'], df['limitRmue'], 'o-', linewidth=2.5, markersize=8, 
        color='steelblue', markerfacecolor='lightblue', markeredgecolor='steelblue', markeredgewidth=1.5)

ax1.set_xlabel('Days', fontsize=12, fontweight='bold')
ax1.set_ylabel(r'$R_{\mu e}$ Limit', fontsize=12, fontweight='bold')
ax1.set_title(r'$R_{\mu e}$ Limit vs Data Collection Time', fontsize=13, fontweight='bold')

# Add major and minor ticks
ax1.minorticks_on()
ax1.tick_params(which='major', length=8, width=1.2)
ax1.tick_params(which='minor', length=4, width=0.8)

# Format y-axis in scientific notation
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Add "Preliminary" label
ax1.text(0.95, 0.95, 'Mu2e Simulation (Preliminary)', transform=ax1.transAxes,
        fontsize=16, fontweight='bold', style='italic',
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Improvement Factor vs Days
ax2.plot(df['Days'], df['improvement_factor'], 's-', linewidth=2.5, markersize=8, 
        color='darkgreen', markerfacecolor='lightgreen', markeredgecolor='darkgreen', markeredgewidth=1.5)

ax2.set_xlabel('Days', fontsize=12, fontweight='bold')
ax2.set_ylabel('Improvement Factor (vs SINDRUM-II)', fontsize=12, fontweight='bold')
ax2.set_title(r'Improvement over SINDRUM-II Limit ($7 \times 10^{-13}$)', fontsize=13, fontweight='bold')

# Add major and minor ticks
ax2.minorticks_on()
ax2.tick_params(which='major', length=8, width=1.2)
ax2.tick_params(which='minor', length=4, width=0.8)

# Add "Preliminary" label
ax2.text(0.15, 0.95, 'Mu2e Simulation (Preliminary)', transform=ax2.transAxes,
        fontsize=16, fontweight='bold', style='italic',
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('rmue_limit_vs_days.png', dpi=150, bbox_inches='tight')
print("Saved plot to: rmue_limit_vs_days.png")
plt.close()

# Print summary
print("\nRmue Limit Summary:")
print(f"  Day 5:  {df.loc[0, 'limitRmue']:.2e} (improvement: {df.loc[0, 'improvement_factor']:.2f}x)")
print(f"  Day 60: {df.loc[len(df)-1, 'limitRmue']:.2e} (improvement: {df.loc[len(df)-1, 'improvement_factor']:.2f}x)")
print(f"  Limit improvement (5→60 days): {df.loc[0, 'limitRmue'] / df.loc[len(df)-1, 'limitRmue']:.2f}x")
