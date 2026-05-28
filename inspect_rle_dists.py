#!/usr/bin/env python3
"""Inspect the actual distributions of resolution and loss from RLE calibration"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from scipy import stats

# Load skimmed data
skimmed_path = "RLE/common/skimmed_flat_mom_MDC2025an.pkl"
with open(skimmed_path, 'rb') as f:
    skimmed_data = pkl.load(f)

# Extract distributions
res_data = skimmed_data['entrance']['reco'] - skimmed_data['entrance']['mc']
loss_data = skimmed_data['entrance']['mc'] - skimmed_data['entrance']['gen']

print(f"Resolution data: {len(res_data)} events")
print(f"  Mean: {np.mean(res_data):.6f}")
print(f"  Std:  {np.std(res_data):.6f}")
print(f"  Skewness: {stats.skew(res_data):.6f}")
print(f"  Kurtosis: {stats.kurtosis(res_data):.6f}")
print(f"  Min: {np.min(res_data):.6f}")
print(f"  Max: {np.max(res_data):.6f}")
print(f"  Q1: {np.percentile(res_data, 25):.6f}")
print(f"  Median: {np.percentile(res_data, 50):.6f}")
print(f"  Q3: {np.percentile(res_data, 75):.6f}")

print(f"\nLoss data: {len(loss_data)} events")
print(f"  Mean: {np.mean(loss_data):.6f}")
print(f"  Std:  {np.std(loss_data):.6f}")
print(f"  Skewness: {stats.skew(loss_data):.6f}")
print(f"  Kurtosis: {stats.kurtosis(loss_data):.6f}")
print(f"  Min: {np.min(loss_data):.6f}")
print(f"  Max: {np.max(loss_data):.6f}")
print(f"  Q1: {np.percentile(loss_data, 25):.6f}")
print(f"  Median: {np.percentile(loss_data, 50):.6f}")
print(f"  Q3: {np.percentile(loss_data, 75):.6f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Resolution histogram with Gaussian fit
ax = axes[0, 0]
counts, bins, _ = ax.hist(res_data, bins=100, density=True, alpha=0.7, label='Data')
mu, sigma = np.mean(res_data), np.std(res_data)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Gaussian fit')
ax.set_xlabel('Resolution (MeV)')
ax.set_ylabel('Density')
ax.set_title('Resolution Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Resolution Q-Q plot
ax = axes[0, 1]
stats.probplot(res_data, dist="norm", plot=ax)
ax.set_title('Resolution Q-Q Plot (Gaussian test)')
ax.grid(True, alpha=0.3)

# Loss histogram with Gaussian fit
ax = axes[1, 0]
counts, bins, _ = ax.hist(loss_data, bins=100, density=True, alpha=0.7, label='Data')
mu, sigma = np.mean(loss_data), np.std(loss_data)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Gaussian fit')
ax.set_xlabel('Loss (MeV)')
ax.set_ylabel('Density')
ax.set_title('Loss Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Loss Q-Q plot
ax = axes[1, 1]
stats.probplot(loss_data, dist="norm", plot=ax)
ax.set_title('Loss Q-Q Plot (Gaussian test)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('RLE/common/rle_distributions_inspect.png', dpi=150, bbox_inches='tight')
print(f"\nSaved plot to RLE/common/rle_distributions_inspect.png")

# Normality tests
print("\n" + "="*60)
print("NORMALITY TESTS (higher p-value = more Gaussian)")
print("="*60)

# Shapiro-Wilk test (best for sample sizes < 5000)
if len(res_data) < 5000:
    stat, p = stats.shapiro(res_data)
    print(f"Resolution Shapiro-Wilk test: p-value = {p:.6e} {'✓ GAUSSIAN' if p > 0.05 else '✗ NOT GAUSSIAN'}")
    stat, p = stats.shapiro(loss_data)
    print(f"Loss Shapiro-Wilk test:       p-value = {p:.6e} {'✓ GAUSSIAN' if p > 0.05 else '✗ NOT GAUSSIAN'}")

# Kolmogorov-Smirnov test
stat, p = stats.kstest(res_data, 'norm', args=(np.mean(res_data), np.std(res_data)))
print(f"Resolution K-S test:          p-value = {p:.6e} {'✓ GAUSSIAN' if p > 0.05 else '✗ NOT GAUSSIAN'}")
stat, p = stats.kstest(loss_data, 'norm', args=(np.mean(loss_data), np.std(loss_data)))
print(f"Loss K-S test:                p-value = {p:.6e} {'✓ GAUSSIAN' if p > 0.05 else '✗ NOT GAUSSIAN'}")

# Anderson-Darling test
result = stats.anderson(res_data, dist='norm')
print(f"Resolution Anderson-Darling:  statistic = {result.statistic:.6f}, critical = {result.critical_values}")
result = stats.anderson(loss_data, dist='norm')
print(f"Loss Anderson-Darling:        statistic = {result.statistic:.6f}, critical = {result.critical_values}")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("="*60)
if np.abs(stats.skew(res_data)) > 0.5 or np.abs(stats.skew(loss_data)) > 0.5:
    print("⚠️  Distributions show significant skewness!")
    print("    Consider using histogram-based PDFs instead of Gaussians")
if np.abs(stats.kurtosis(res_data)) > 1 or np.abs(stats.kurtosis(loss_data)) > 1:
    print("⚠️  Distributions show significant kurtosis (heavy/light tails)!")
    print("    Consider using histogram-based PDFs instead of Gaussians")
