import numpy as np
import scipy.stats as stats

# ---------------------------------------------------------
# 1. Experimental Inputs & Configuration
# ---------------------------------------------------------
N_STOPPED_MU = 3.4e15
EFFICIENCY = 0.1025
EFF_ERROR = 0.05 * EFFICIENCY    # Assume a 5% relative systematic uncertainty
BACKGROUND = 26.0
B_ERROR = 2.6                     # Assume a 10% systematic uncertainty on background

N_TOYS = 100_000                  # Number of toy MC pseudo-experiments per point
TARGET_CLS = 0.10                 # 90% Confidence Level Upper Limit -> CLs = 0.10

# Calculate Single Event Sensitivity (SES)
ses = 1.0 / (N_STOPPED_MU * EFFICIENCY)
print(f"Single Event Sensitivity (SES): {ses:.3e}")

# ---------------------------------------------------------
# 2. Define the Test Statistic
# ---------------------------------------------------------
# We use the profile likelihood ratio or a simple log-likelihood ratio (LLR).
# For standard counting experiments, the number of observed events N itself 
# can serve as an optimal test statistic. Higher N favors S+B, lower N favors B-only.
def compute_cls(s_hypo, b_nominal, b_err, eff_nominal, eff_err, n_toys):
    # Generate nuisance parameters for B-only hypothesis (H0)
    # Ensure background doesn't fluctuate below zero
    b_toys_h0 = np.maximum(0, np.random.normal(b_nominal, b_err, n_toys))
    n_obs_h0 = np.random.poisson(b_toys_h0)
    
    # Generate nuisance parameters for Signal + Background hypothesis (H1)
    b_toys_h1 = np.maximum(0, np.random.normal(b_nominal, b_err, n_toys))
    eff_toys_h1 = np.maximum(0, np.random.normal(eff_nominal, eff_err, n_toys))
    
    # Scale the hypothesized signal by the fluctuated efficiency relative to nominal
    s_scaled = s_hypo * (eff_toys_h1 / eff_nominal)
    n_obs_h1 = np.random.poisson(s_scaled + b_toys_h1)
    
    # In an Asimov projection, the "actual observed" data is exactly the nominal background
    n_asimov = b_nominal
    
    # Calculate CL_(s+b) = P(N <= N_asimov | H1)
    cl_sb = np.sum(n_obs_h1 <= n_asimov) / n_toys
    
    # Calculate CL_b = P(N <= N_asimov | H0)
    cl_b = np.sum(n_obs_h0 <= n_asimov) / n_toys
    
    # Guard against division by zero if cl_b is extremely small
    if cl_b == 0:
        return 1.0
        
    cls = cl_sb / cl_b
    return cls

# ---------------------------------------------------------
# 3. Scan Signal Hypotheses to Find the 90% CL Upper Limit
# ---------------------------------------------------------
print("\nScanning signal hypotheses using Toy Monte Carlo...")
print(f"{'Signal (events)':<20}{'CLs Value':<15}")
print("-" * 35)

# Coarse scan to bracket the limit, followed by a linear interpolation
s_scan = np.linspace(5.0, 12.0, 15)
cls_results = []

for s in s_scan:
    cls_val = compute_cls(s, BACKGROUND, B_ERROR, EFFICIENCY, EFF_ERROR, N_TOYS)
    cls_results.append(cls_val)
    print(f"{s:<20.2f}{cls_val:<15.4f}")

# Interpolate to find exact S where CLs = 0.10
s_90 = np.interp(TARGET_CLS, cls_results[::-1], s_scan[::-1])

# ---------------------------------------------------------
# 4. Translate to Physics Limit (R_mue)
# ---------------------------------------------------------
r_mue_limit = s_90 * ses

print("-" * 35)
print(f"\nRESULTS FROM TOY STUDY:")
print(f"90% CL Upper Limit on Signal Events (S_90): {s_90:.2f} events")
print(f"Projected 90% CL Upper Limit R_mue:       {r_mue_limit:.3e}")