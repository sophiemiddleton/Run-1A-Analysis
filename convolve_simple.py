#!/usr/bin/env python3
"""
Simplified convolution test - implementing from archive pattern exactly
"""
import numpy as np
import zfit
from RLE.rle_functions import load_calibration
from spectrum import TheorySpectrum

# Config
momentum_range = (95, 110)
lo, hi = momentum_range

# Load calibration
calib = load_calibration('RLE/common/calibration.json')
print(f"DSCB params: {calib['dscb']}")

# 1. Create theory PDF on NATIVE MOMENTUM observable
obs_mom = zfit.Space('x', limits=momentum_range)
theory_spectrum = TheorySpectrum(mom_range=momentum_range, binwidth=0.1, verbosity=0)
theory_pdf = theory_spectrum.get_pdf(obs=obs_mom, name='CE_theory')
print(f"✓ Theory PDF created on obs_mom [{lo}, {hi}]")

# 2. Create resolution model on NATIVE DIFFERENCE observable
res_range = (-2, 2)
lo_res, hi_res = res_range
obs_res = zfit.Space('x', limits=res_range)

res_pdf = zfit.pdf.DoubleCB(
    mu=zfit.Parameter('res_mu', calib['dscb']['mu'], floating=False),
    sigma=zfit.Parameter('res_sigma', calib['dscb']['sigma'], floating=False),
    alphal=zfit.Parameter('res_alphal', calib['dscb']['alphaL'], floating=False),
    alphar=zfit.Parameter('res_alphar', calib['dscb']['alphaR'], floating=False),
    nl=zfit.Parameter('res_nl', calib['dscb']['nL'], floating=False),
    nr=zfit.Parameter('res_nr', calib['dscb']['nR'], floating=False),
    obs=obs_res
)
print(f"✓ Resolution PDF created on obs_res [{lo_res}, {hi_res}]")

# 3. Compute blended observable spaces (from archive pattern)
# obs_conv = [lo_theory - hi_res, hi_theory - lo_res]
# obs_full = [lo_theory + lo_res, hi_theory + hi_res]
lo_conv = lo - hi_res
hi_conv = hi - lo_res
obs_conv = zfit.Space('x', limits=(lo_conv, hi_conv))

lo_full = lo + lo_res
hi_full = hi + hi_res
obs_full = zfit.Space('x', limits=(lo_full, hi_full))

print(f"Theory obs: [{lo}, {hi}]")
print(f"Res obs: [{lo_res}, {hi_res}]")
print(f"Convolution obs: [{lo_conv}, {hi_conv}]")
print(f"Full obs: [{lo_full}, {hi_full}]")

# 4. Create FFTConvPDFV1
print("\nCreating FFTConvPDFV1...")
try:
    nbins_true = int((hi - lo) / 0.1)
    convolved_pdf = zfit.pdf.FFTConvPDFV1(
        func=theory_pdf,      # on obs_mom [95, 110]
        kernel=res_pdf,       # on obs_res [-2, 2]
        n=nbins_true,         # FFT bins for theory range
        obs=obs_conv,         # output on [95-2, 110-(-2)] = [93, 112]
        norm=obs_full         # normalization on [95+(-2), 110+2] = [93, 112]
    )
    print("✓ FFTConvPDFV1 created successfully")
except Exception as e:
    print(f"✗ FFTConvPDFV1 creation FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. Evaluate on grid
print("\nEvaluating convolution...")
x_conv = np.linspace(lo_conv, hi_conv, 200).reshape(-1, 1)
try:
    convolved_vals = convolved_pdf.pdf(x_conv)
    print("✓ FFTConvPDFV1 evaluation successful")
    print(f"  Shape: {convolved_vals.shape}")
    print(f"  Min: {convolved_vals.numpy().min():.6e}, Max: {convolved_vals.numpy().max():.6e}")
    print(f"  Has NaN: {np.isnan(convolved_vals.numpy()).any()}")
except Exception as e:
    print(f"✗ Evaluation FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 6. Plot results
print("\nCreating plots...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Theory PDF
x_theory = np.linspace(lo, hi, 200).reshape(-1, 1)
theory_vals = theory_pdf.pdf(x_theory).numpy()
axes[0].plot(x_theory.flatten(), theory_vals, 'b-', linewidth=2)
axes[0].set_xlabel('Momentum (MeV)', fontsize=11)
axes[0].set_ylabel('PDF', fontsize=11)
axes[0].set_title('CE Theory Spectrum', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

# Resolution kernel
x_res = np.linspace(lo_res, hi_res, 200).reshape(-1, 1)
res_vals = res_pdf.pdf(x_res).numpy()
axes[1].plot(x_res.flatten(), res_vals, 'r-', linewidth=2)
axes[1].set_xlabel('Momentum Difference (MeV)', fontsize=11)
axes[1].set_ylabel('PDF', fontsize=11)
axes[1].set_title('DSCB Resolution Kernel', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

# Convolved result
axes[2].plot(x_conv.flatten(), convolved_vals.numpy(), 'g-', linewidth=2)
axes[2].set_xlabel('Momentum (MeV)', fontsize=11)
axes[2].set_ylabel('Convolved PDF', fontsize=11)
axes[2].set_title('Theory ⊗ Resolution', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('convolution_test.pdf', dpi=150)
print("✓ Plot saved to convolution_test.pdf")
plt.close()

print("\n✓ SUCCESS!")

