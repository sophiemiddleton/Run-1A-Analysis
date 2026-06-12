"""
RLE convolution functions - apply fitted parameters to transform momentum distributions

Uses zfit models for PDFs (Chebyshev, GCB, Landau) to enable fitting CE spectra
with full RLE convolution baked in.
"""

import json
import numpy as np
import zfit
from RLE.landau_pdf import trunc_landau
import warnings


def load_calibration(json_path):
    """Load calibration parameters from JSON
    
    Args:
        json_path (str): Path to calibration.json
        
    Returns:
        dict: Parameters with keys 'chebyshev', 'gcb', 'landau'
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def get_loss_model(obs, params):
    """Get zfit truncated Landau loss distribution with fixed parameters
    
    Args:
        obs (zfit.Space): Observable space
        params (dict): {'loc', 'scale'}
        
    Returns:
        zfit.pdf: Truncated Landau PDF
    """
    loc = zfit.Parameter(f'landau_loc', params['loc'], floating=False)
    scale = zfit.Parameter(f'landau_scale', params['scale'], floating=False)
    return trunc_landau(obs=obs, loc=loc, scale=scale)


def get_resolution_model(obs, params):
    """Get zfit Generalized Crystal Ball resolution distribution with fixed parameters
    
    Args:
        obs (zfit.Space): Observable space
        params (dict): GCB parameters
        
    Returns:
        zfit.pdf: CrystalBall PDF
    """
    mu = zfit.Parameter('gcb_mu', params['mu'], floating=False)
    sigma = zfit.Parameter('gcb_sigma', (params['sigmaL'] + params['sigmaR']) / 2, floating=False)
    alpha = zfit.Parameter('gcb_alpha', (params['alphaL'] + params['alphaR']) / 2, floating=False)
    n = zfit.Parameter('gcb_n', (params['nL'] + params['nR']) / 2, floating=False)
    
    return zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=alpha, n=n)


def get_efficiency_model(obs, params):
    """Get zfit Chebyshev efficiency function with fixed parameters
    
    Args:
        obs (zfit.Space): Observable space (must be on [95, 110])
        params (dict): {'coeffs': [c0, c1, c2, c3, c4, c5]}
        
    Returns:
        zfit.pdf: Chebyshev PDF
    """
    coeffs_list = params['coeffs']
    # Skip c0 (implicit in zfit Chebyshev), create fixed parameters for c1-c5
    coeffs = [zfit.Parameter(f'cheb_c{i}', coeffs_list[i+1], floating=False) 
              for i in range(len(coeffs_list)-1)]
    return zfit.pdf.Chebyshev(obs=obs, coeffs=coeffs)


def evaluate_pdf(model, x_vals):
    """Evaluate a zfit model on an array of x values
    
    Args:
        model (zfit.pdf): zfit PDF model
        x_vals (array): Values to evaluate on
        
    Returns:
        array: PDF values
    """
    x_reshaped = x_vals.reshape(-1, 1)
    return zfit.run(model.pdf(x_reshaped)).flatten()


def convolve_numerical(f, x_grid, kernel_grid, kernel_vals):
    """Discrete convolution of function f with kernel on x_grid
    
    Args:
        f (array): Function values
        x_grid (array): Momentum grid where f is defined
        kernel_grid (array): Grid for kernel
        kernel_vals (array): Kernel values on grid (should be normalized)
        
    Returns:
        array: Convolved function values
    """
    result = np.zeros_like(f)
    dx_kernel = kernel_grid[1] - kernel_grid[0] if len(kernel_grid) > 1 else 0.01
    
    # For each output grid point
    for i, x in enumerate(x_grid):
        # Shift kernel to this position: integral of f(x-k) * kernel(k) dk
        shifted_kernel_x = x - kernel_grid
        
        # Interpolate input function at shifted kernel positions
        shifted_f = np.interp(shifted_kernel_x, x_grid, f, left=0, right=0)
        
        # Convolve
        result[i] = np.trapz(shifted_f * kernel_vals, dx=dx_kernel)
    
    return result


def apply_rle_convolution(x_grid, theory_vals, calibration_path):
    """Apply full RLE convolution: theory → loss → resolution → efficiency
    
    Args:
        x_grid (array): Momentum grid [MeV] on [95, 110]
        theory_vals (array): Theory spectrum values on x_grid
        calibration_path (str): Path to calibration.json
        
    Returns:
        dict: {
            'x_grid': momentum grid,
            'theory': original theory,
            'after_loss': theory ⊗ loss,
            'after_resolution': theory ⊗ loss ⊗ resolution,
            'final': (theory ⊗ loss ⊗ resolution) × efficiency,
            'efficiency': efficiency values,
            'models': {'loss_model', 'resolution_model', 'efficiency_model'}
        }
    """
    calib = load_calibration(calibration_path)
    
    # Setup observable spaces
    p_obs = zfit.Space('p', limits=(95, 110))
    dp_obs = zfit.Space('dp', limits=(-5, 2))
    dr_obs = zfit.Space('dr', limits=(-1, 1))
    
    # Create models with fixed calibrated parameters
    loss_model = get_loss_model(dp_obs, calib['landau'])
    resolution_model = get_resolution_model(dr_obs, calib['gcb'])
    efficiency_model = get_efficiency_model(p_obs, calib['chebyshev'])
    
    # Step 1: Convolve with loss (Landau)
    loss_grid = np.linspace(-5, 2, 300)
    loss_kernel = evaluate_pdf(loss_model, loss_grid)
    loss_kernel = loss_kernel / np.trapz(loss_kernel, dx=loss_grid[1]-loss_grid[0])
    
    after_loss = convolve_numerical(theory_vals, x_grid, loss_grid, loss_kernel)
    
    # Step 2: Convolve with resolution (GCB)
    res_grid = np.linspace(-1, 1, 200)
    res_kernel = evaluate_pdf(resolution_model, res_grid)
    res_kernel = res_kernel / np.trapz(res_kernel, dx=res_grid[1]-res_grid[0])
    
    after_resolution = convolve_numerical(after_loss, x_grid, res_grid, res_kernel)
    
    # Step 3: Multiply by efficiency (Chebyshev)
    eff = evaluate_pdf(efficiency_model, x_grid)
    
    # Normalize efficiency to [0, 1]
    eff_min = np.min(eff)
    eff_max = np.max(eff)
    if eff_max - eff_min > 1e-8:
        eff = (eff - eff_min) / (eff_max - eff_min)
    
    final = after_resolution * eff
    
    return {
        'x_grid': x_grid,
        'theory': theory_vals,
        'after_loss': after_loss,
        'after_resolution': after_resolution,
        'final': final,
        'efficiency': eff,
        'models': {
            'loss': loss_model,
            'resolution': resolution_model,
            'efficiency': efficiency_model
        }
    }

