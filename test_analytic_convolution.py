#!/usr/bin/env python3
"""
Analytic convolution test with actual CeLL theory spectrum.
Theory ⊗ Loss ⊗ Resolution × Efficiency - computed numerically.
"""

import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import landau
from numpy.polynomial.chebyshev import chebval

# ============================================================================
# Physical constants for CeLL spectrum
# ============================================================================
E_MAX = 104.969  # Maximum electron momentum [MeV/c]
M_E = 0.511      # Electron mass [MeV]
ALPHA = 1.0 / 137.036  # Fine structure constant

def leading_log_spectrum(E):
    """
    Theoretical Leading Log conversion spectrum formula.
    """
    if E >= E_MAX or E <= 0:
        return 0.0
    prefactor = (1.0 / E_MAX) * (ALPHA / (2.0 * math.pi))
    log_term = math.log(4.0 * E**2 / M_E**2) - 2.0
    energy_term = (E**2 + E_MAX**2) / (E_MAX * (E_MAX - E))
    return prefactor * log_term * energy_term

def theory_pdf(p):
    """Theory spectrum: CeLL Leading Log formula"""
    val = leading_log_spectrum(p)
    # Normalize to unit area over [0, E_MAX]
    return val

# ============================================================================
# Setup
# ============================================================================
print("=" * 70)
print("Analytic Convolution Pipeline Test (with CeLL Theory Spectrum)")
print("=" * 70)

# Use grid that covers CeLL spectrum (0 to 105 MeV) plus some tails
x_grid = np.linspace(95, 115, 600)

# Compute normalization factor for theory
theory_test = np.array([theory_pdf(p) for p in np.linspace(0, E_MAX, 1000)])
theory_dx = E_MAX / 1000
theory_norm = np.trapz(theory_test, dx=theory_dx)

# ============================================================================
# 2. LOSS: Landau distribution (energy loss)
# ============================================================================
def loss_pdf(dp):
    """Truncated Landau energy loss distribution
    Fitted from origin - true momentum at tracker entrance
    
    Parameters from actual calibration:
      loc = -0.600773 (peak position, MeV)
      scale = 0.266484 (width parameter)
      χ²/dof = 1.6407
    """
    return landau.pdf(dp, loc=-0.600773, scale=0.266484)

# ============================================================================
# 3. RESOLUTION: Generalized Crystal Ball
# ============================================================================
def gcb_pdf(dr, mu=0.032545, sigma_l=0.156945, sigma_r=0.136071, 
            alpha_l=1.182115, alpha_r=1.575798, n_l=3.521308, n_r=4.655049):
    """GCB with left/right power-law tails
    Fitted from entrance plane resolution distribution
    
    Parameters from actual calibration:
      mu = 0.032545 (peak position)
      sigmaL = 0.156945 (left width)
      alphaL = 1.182115 (left tail start)
      nL = 3.521308 (left tail power)
      sigmaR = 0.136071 (right width)
      alphaR = 1.575798 (right tail start)
      nR = 4.655049 (right tail power)
    """
    dr = np.atleast_1d(dr)
    result = np.zeros_like(dr, dtype=float)
    
    # Left side: use sigma_l
    left_mask = dr < mu
    z_left = (dr[left_mask] - mu) / sigma_l
    
    # Right side: use sigma_r
    right_mask = dr >= mu
    z_right = (dr[right_mask] - mu) / sigma_r
    
    # Central Gaussian
    gauss_l = np.abs(z_left) < alpha_l
    result[left_mask][gauss_l] = np.exp(-0.5 * z_left[gauss_l]**2)
    
    gauss_r = np.abs(z_right) < alpha_r
    result[right_mask][gauss_r] = np.exp(-0.5 * z_right[gauss_r]**2)
    
    # Left tail
    tail_l = z_left < -alpha_l
    A_l = (n_l / alpha_l)**n_l * np.exp(-0.5 * alpha_l**2)
    B_l = n_l / alpha_l - alpha_l
    with np.errstate(over='ignore', invalid='ignore'):
        tail_vals_l = A_l * (B_l - z_left[tail_l])**(-n_l)
        tail_vals_l = np.nan_to_num(tail_vals_l, nan=0, posinf=0, neginf=0)
        result[left_mask][tail_l] = tail_vals_l
    
    # Right tail
    tail_r = z_right > alpha_r
    A_r = (n_r / alpha_r)**n_r * np.exp(-0.5 * alpha_r**2)
    B_r = n_r / alpha_r - alpha_r
    with np.errstate(over='ignore', invalid='ignore'):
        tail_vals_r = A_r * (B_r + z_right[tail_r])**(-n_r)
        tail_vals_r = np.nan_to_num(tail_vals_r, nan=0, posinf=0, neginf=0)
        result[right_mask][tail_r] = tail_vals_r
    
    # Normalize
    norm_grid = np.linspace(-5, 5, 1000)
    result_norm = np.zeros_like(norm_grid, dtype=float)
    
    left_n = norm_grid < mu
    z_left_n = (norm_grid[left_n] - mu) / sigma_l
    gauss_ln = np.abs(z_left_n) < alpha_l
    result_norm[left_n][gauss_ln] = np.exp(-0.5 * z_left_n[gauss_ln]**2)
    tail_ln = z_left_n < -alpha_l
    with np.errstate(over='ignore', invalid='ignore'):
        tail_vals_ln = A_l * (B_l - z_left_n[tail_ln])**(-n_l)
        tail_vals_ln = np.nan_to_num(tail_vals_ln, nan=0, posinf=0, neginf=0)
        result_norm[left_n][tail_ln] = tail_vals_ln
    
    right_n = norm_grid >= mu
    z_right_n = (norm_grid[right_n] - mu) / sigma_r
    gauss_rn = np.abs(z_right_n) < alpha_r
    result_norm[right_n][gauss_rn] = np.exp(-0.5 * z_right_n[gauss_rn]**2)
    tail_rn = z_right_n > alpha_r
    with np.errstate(over='ignore', invalid='ignore'):
        tail_vals_rn = A_r * (B_r + z_right_n[tail_rn])**(-n_r)
        tail_vals_rn = np.nan_to_num(tail_vals_rn, nan=0, posinf=0, neginf=0)
        result_norm[right_n][tail_rn] = tail_vals_rn
    
    norm = np.trapz(result_norm, x=norm_grid)
    
    # Normalize with fallback to Gaussian if needed
    if norm > 1e-10:
        result = result / norm
    else:
        # Fallback to simple Gaussian
        result = np.exp(-0.5 * ((dr - mu) / 0.15)**2)
    
    result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)
    return result[0] if len(result) == 1 else result

# ============================================================================
# 4. EFFICIENCY: Chebyshev polynomial
# ============================================================================
def efficiency(p):
    """Chebyshev degree-5 efficiency function
    Fitted from origin momentum distribution (actual calibration)
    """
    p = np.atleast_1d(p)
    x_norm = 2 * (p - 90) / (120 - 90) - 1  # Normalize to [-1, 1]
    # Actual fitted Chebyshev coefficients from origin_momentum_fit
    # T_0 (const), T_1, T_2, T_3, T_4, T_5
    cheb_coeffs = np.array([1.0, 0.26581554, -0.27286063, -0.02624121, 0.00032070, 0.01663576])
    eff = chebval(x_norm, cheb_coeffs)
    
    # Compute range across full domain to preserve shape
    test_norm = np.linspace(-1, 1, 500)
    test_eff = chebval(test_norm, cheb_coeffs)
    eff_min = np.min(test_eff)
    eff_max = np.max(test_eff)
    
    # Normalize with safety check
    eff_range = eff_max - eff_min
    if eff_range > 1e-8:
        eff = (eff - eff_min) / eff_range
    else:
        eff = np.ones_like(eff) * 0.75
    
    return eff[0] if len(eff) == 1 else eff

# ============================================================================
# Numerical Convolution using discrete arrays
# ============================================================================
print("\nComputing convolutions numerically...")

# Step 1: Theory PDF on grid
print("  Step 1: Theory PDF (CeLL)")
theory_vals = np.array([theory_pdf(p) / theory_norm for p in x_grid])
theory_vals = theory_vals / np.trapz(theory_vals, x=x_grid)  # Re-normalize to grid

# Step 2: Theory ⊗ Loss
# For each p_mc, integrate: ∫ theory(p_gen) * loss(p_mc - p_gen) dp_gen
# Use discrete convolution approximation
print("  Step 2: Theory ⊗ Loss")
loss_grid = np.linspace(-6, 6, 300)
loss_vals = np.array([loss_pdf(dp) for dp in loss_grid])
loss_vals = loss_vals / np.trapz(loss_vals, x=loss_grid)

p_mc_vals = np.zeros_like(x_grid)
for i, p_mc in enumerate(x_grid):
    # For this p_mc, loss shifted: loss(p_mc - p_gen)
    shifted_loss = np.interp(x_grid, loss_grid + p_mc, loss_vals, left=0, right=0)
    p_mc_vals[i] = np.trapz(theory_vals * shifted_loss, x=x_grid)
p_mc_vals = p_mc_vals / np.trapz(p_mc_vals, x=x_grid)

# Step 3: (Theory ⊗ Loss) ⊗ Resolution
print("  Step 3: (Theory ⊗ Loss) ⊗ Resolution")
res_grid = np.linspace(-1.5, 1.5, 200)
res_vals = np.array([gcb_pdf(dr) for dr in res_grid])
with np.errstate(divide='ignore', invalid='ignore'):
    res_vals = res_vals / np.trapz(res_vals, x=res_grid)
res_vals = np.nan_to_num(res_vals, nan=0, posinf=0, neginf=0)

p_reco_vals = np.zeros_like(x_grid)
for i, p_reco in enumerate(x_grid):
    # For this p_reco, resolution shifted: res(p_reco - p_mc)
    shifted_res = np.interp(x_grid, res_grid + p_reco, res_vals, left=0, right=0)
    p_reco_vals[i] = np.trapz(p_mc_vals * shifted_res, x=x_grid)
p_reco_vals = p_reco_vals / np.trapz(p_reco_vals, x=x_grid)

# Step 4: Efficiency
print("  Step 4: Efficiency")
eff_vals = np.array([efficiency(p) for p in x_grid])

# Final: (theory ⊗ loss ⊗ resolution) × efficiency
final_vals = p_reco_vals * eff_vals
final_vals = final_vals / np.trapz(final_vals, x=x_grid)

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Compute weighted means and variances
def weighted_mean(pdf, x):
    return np.trapz(x * pdf, x=x) / np.trapz(pdf, x=x)

def weighted_var(pdf, x):
    mean = weighted_mean(pdf, x)
    return np.trapz((x - mean)**2 * pdf, x=x) / np.trapz(pdf, x=x)

theory_mean = weighted_mean(theory_vals, x_grid)
theory_std = np.sqrt(weighted_var(theory_vals, x_grid))

p_mc_mean = weighted_mean(p_mc_vals, x_grid)
p_mc_std = np.sqrt(weighted_var(p_mc_vals, x_grid))

p_reco_mean = weighted_mean(p_reco_vals, x_grid)
p_reco_std = np.sqrt(weighted_var(p_reco_vals, x_grid))

final_mean = weighted_mean(final_vals, x_grid)
final_std = np.sqrt(weighted_var(final_vals, x_grid))

print(f"\n1. THEORY (p_gen):")
print(f"   Mean: {theory_mean:.2f} MeV")
print(f"   Std:  {theory_std:.2f} MeV")
theory_nonzero = x_grid[theory_vals > 1e-5]
if len(theory_nonzero) > 0:
    print(f"   Range: [{np.min(theory_nonzero):.1f}, {np.max(theory_nonzero):.1f}] MeV")
else:
    print(f"   Range: [N/A, N/A] MeV")

print(f"\n2. AFTER LOSS (theory ⊗ loss):")
print(f"   Mean: {p_mc_mean:.2f} MeV")
print(f"   Std:  {p_mc_std:.2f} MeV")
loss_nonzero = x_grid[p_mc_vals > 1e-5]
if len(loss_nonzero) > 0:
    print(f"   Range: [{np.min(loss_nonzero):.1f}, {np.max(loss_nonzero):.1f}] MeV")
else:
    print(f"   Range: [N/A, N/A] MeV")
print(f"   Effect: Mean shift = {p_mc_mean - theory_mean:.2f} MeV (Landau at -1.34)")

print(f"\n3. AFTER RESOLUTION (theory ⊗ loss ⊗ resolution):")
print(f"   Mean: {p_reco_mean:.2f} MeV")
print(f"   Std:  {p_reco_std:.2f} MeV")
res_nonzero = x_grid[p_reco_vals > 1e-5]
if len(res_nonzero) > 0:
    print(f"   Range: [{np.min(res_nonzero):.1f}, {np.max(res_nonzero):.1f}] MeV")
else:
    print(f"   Range: [N/A, N/A] MeV")
print(f"   Effect: Std ratio = {p_reco_std / theory_std:.2f}x (smearing from resolution)")

print(f"\n4. AFTER EFFICIENCY ((theory ⊗ loss ⊗ res) × eff):")
print(f"   Mean: {final_mean:.2f} MeV")
print(f"   Std:  {final_std:.2f} MeV")
final_nonzero = x_grid[final_vals > 1e-5]
if len(final_nonzero) > 0:
    print(f"   Range: [{np.min(final_nonzero):.1f}, {np.max(final_nonzero):.1f}] MeV")
else:
    print(f"   Range: [N/A, N/A] MeV")
print(f"   Effect: Mean shift = {final_mean - p_reco_mean:.2f} MeV (Chebyshev modulation)")

# Efficiency statistics
eff_min = np.min(eff_vals)
eff_max = np.max(eff_vals)
eff_mean = np.mean(eff_vals)
print(f"\n5. EFFICIENCY (Chebyshev deg-5):")
print(f"   Min:  {eff_min:.3f}")
print(f"   Max:  {eff_max:.3f}")
print(f"   Mean: {eff_mean:.3f}")

print("\n" + "=" * 70)
print("Convolution test complete!")
print("=" * 70)

# ============================================================================
# Plotting
# ============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Theory
ax = axes[0, 0]
ax.plot(x_grid, theory_vals, 'b-', linewidth=2, label='Theory PDF')
ax.fill_between(x_grid, theory_vals, alpha=0.3)
ax.set_xlabel('Momentum (MeV)', fontsize=11)
ax.set_ylabel('Probability density', fontsize=11)
ax.set_title('Step 1: Generated Spectrum (Theory)', fontsize=12, fontweight='bold')
ax.text(0.98, 0.97, f'μ={theory_mean:.2f} MeV\nσ={theory_std:.2f} MeV',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

# Plot 2: After Loss
ax = axes[0, 1]
ax.plot(x_grid, p_mc_vals, 'g-', linewidth=2, label='Theory ⊗ Loss')
ax.fill_between(x_grid, p_mc_vals, alpha=0.3, color='green')
ax.set_xlabel('Momentum (MeV)', fontsize=11)
ax.set_ylabel('Probability density', fontsize=11)
ax.set_title('Step 2: After Energy Loss (Theory ⊗ Loss)', fontsize=12, fontweight='bold')
ax.text(0.98, 0.97, f'μ={p_mc_mean:.2f} MeV\nσ={p_mc_std:.2f} MeV\nΔμ={p_mc_mean - theory_mean:.2f} MeV',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

# Plot 3: After Resolution
ax = axes[1, 0]
ax.plot(x_grid, p_reco_vals, 'r-', linewidth=2, label='Theory ⊗ Loss ⊗ Resolution')
ax.fill_between(x_grid, p_reco_vals, alpha=0.3, color='red')
ax.set_xlabel('Momentum (MeV)', fontsize=11)
ax.set_ylabel('Probability density', fontsize=11)
ax.set_title('Step 3: After Resolution (Theory ⊗ Loss ⊗ Resolution)', fontsize=12, fontweight='bold')
ax.text(0.98, 0.97, f'μ={p_reco_mean:.2f} MeV\nσ={p_reco_std:.2f} MeV\nσ ratio={p_reco_std / theory_std:.2f}x',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

# Plot 4: After Efficiency (with efficiency overlay)
ax = axes[1, 1]
ax.plot(x_grid, final_vals, 'purple', linewidth=2.5, label='Final × Efficiency')
ax.fill_between(x_grid, final_vals, alpha=0.3, color='purple')
ax2 = ax.twinx()
ax2.plot(x_grid, eff_vals, 'k--', linewidth=1.5, label='Efficiency', alpha=0.6)
ax.set_xlabel('Momentum (MeV)', fontsize=11)
ax.set_ylabel('Probability density', fontsize=11, color='purple')
ax2.set_ylabel('Efficiency', fontsize=11, color='black')
ax.set_title('Step 4: Final Spectrum × Efficiency', fontsize=12, fontweight='bold')
ax.text(0.98, 0.97, f'μ={final_mean:.2f} MeV\nσ={final_std:.2f} MeV',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.grid(alpha=0.3)
ax.tick_params(axis='y', labelcolor='purple')
ax2.tick_params(axis='y', labelcolor='black')

plt.tight_layout()
plt.savefig('convolution_pipeline.png', dpi=150, bbox_inches='tight')
print("Saved: convolution_pipeline.png")

# Create comparison plot: Theory vs Final
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(x_grid, theory_vals, 'b-', linewidth=2.5, label='Original Theory', alpha=0.7)
ax.plot(x_grid, p_reco_vals, 'g--', linewidth=2, label='After Loss + Resolution', alpha=0.7)
ax.plot(x_grid, final_vals, 'r-', linewidth=2.5, label='Final (× Efficiency)')
ax.fill_between(x_grid, final_vals, alpha=0.2, color='red')
ax.set_xlabel('Momentum (MeV)', fontsize=12)
ax.set_ylabel('Probability density', fontsize=12)
ax.set_title('Convolution Pipeline: How Detector Effects Transform Theory Spectrum', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.3)

# Add text summary
summary_text = (
    f"Theory:          μ={theory_mean:.2f}, σ={theory_std:.2f} MeV\n"
    f"After Conv:    μ={p_reco_mean:.2f}, σ={p_reco_std:.2f} MeV\n"
    f"Final:           μ={final_mean:.2f}, σ={final_std:.2f} MeV\n"
    f"\n"
    f"Efficiency range: {eff_min:.3f} - {eff_max:.3f}"
)
ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('convolution_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: convolution_comparison.png")

print("\nPlots saved successfully!")
print("=" * 70)



