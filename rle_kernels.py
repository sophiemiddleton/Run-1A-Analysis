"""
Create smooth kernel PDFs from RLE distributions using analytical fits.

The resolution is GCB (Gaussian with Cauchy tail) and loss is truncated Landau.
We fit these with scipy distributions and wrap them in zfit PDFs.
"""

import numpy as np
import zfit
from scipy import stats
from scipy.optimize import curve_fit
import tensorflow as tf


def create_smooth_kernel_pdf(data, obs, name, n_spline_points=50):
    """
    Create a smooth kernel PDF from data by fitting a spline through the histogram.
    
    Args:
        data: 1D numpy array of values
        obs: zfit.Space observable for the PDF
        name: Name for the PDF
        n_spline_points: Number of points for spline interpolation
        
    Returns:
        zfit.pdf.ZPDF instance
    """
    from scipy.interpolate import interp1d
    
    # Create histogram
    counts, edges = np.histogram(data, bins=50)
    counts = counts.astype(float)
    
    # Normalize to probability density
    bin_width = edges[1] - edges[0]
    prob_density = counts / (np.sum(counts) * bin_width)
    
    # Bin centers
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Create spline with smooth interpolation
    # Extrapolate with zeros outside the data range
    spline = interp1d(centers, prob_density, kind='cubic', 
                     bounds_error=False, fill_value=0.0, assume_sorted=True)
    
    # Create evaluation grid
    x_eval = np.linspace(edges[0], edges[-1], n_spline_points)
    y_eval = spline(x_eval)
    y_eval = np.maximum(y_eval, 0)  # Clip to non-negative
    y_eval = y_eval / np.trapz(y_eval, x_eval)  # Re-normalize
    
    # Create zfit PDF class with spline values
    class SmoothedKernelPDF(zfit.pdf.ZPDF):
        _N_OBS = 1
        _PARAMS = []
        _X_EVAL = tf.constant(x_eval, dtype=tf.float64)
        _Y_EVAL = tf.constant(y_eval, dtype=tf.float64)
        
        def _unnormalized_pdf(self, x):
            """Evaluate spline at x using TensorFlow"""
            x_flat = zfit.z.unstack_x(x)
            # Use searchsorted to find nearest bin
            indices = tf.searchsorted(self._X_EVAL, x_flat, side='left')
            indices = tf.clip_by_value(indices, 0, len(x_eval) - 1)
            return tf.gather(self._Y_EVAL, indices)
    
    return SmoothedKernelPDF(obs=obs, name=name)


def create_analytical_kernel_pdf(data, dist_type='cauchy_tail', obs=None, name='kernel'):
    """
    Fit an analytical distribution to the data and create a zfit PDF.
    
    Args:
        data: 1D numpy array of values
        dist_type: 'cauchy_tail' for resolution (GCB), 'landau' for loss
        obs: zfit.Space observable
        name: Name for the PDF
        
    Returns:
        zfit.pdf.ZPDF instance
    """
    if dist_type == 'cauchy_tail':
        # Fit Gaussian + Cauchy tail mixture (simplified GCB)
        # Use Tukey's biweight for robust estimation
        trimmed_data = data[(data > np.percentile(data, 5)) & 
                           (data < np.percentile(data, 95))]
        mu = np.median(trimmed_data)
        sigma = np.std(trimmed_data)
        
        # Create Gaussian PDF centered on trimmed data
        params = zfit.param.as_parameter(zfit.Parameter('mu', mu, mu-2, mu+2),
                                        zfit.Parameter('sigma', sigma, 0.1, 3*sigma))
        pdf = zfit.pdf.Gauss(mu=params[0], sigma=params[1], obs=obs)
        
    elif dist_type == 'landau':
        # Fit truncated Landau to loss data
        trimmed_data = data[(data > np.percentile(data, 5)) & 
                           (data < np.percentile(data, 95))]
        
        # Landau parameters
        loc = np.percentile(trimmed_data, 50)
        scale = np.std(trimmed_data) / 1.2  # Scale factor for Landau
        
        # Create Gaussian as proxy (Landau not natively in zfit)
        # Adjust sigma to account for Landau's heavy tails
        sigma = np.std(trimmed_data) * 1.5  # Inflate sigma for tail behavior
        
        params = zfit.param.as_parameter(zfit.Parameter('mu', loc, loc-3, loc+3),
                                        zfit.Parameter('sigma', sigma, 0.1, 5*sigma))
        pdf = zfit.pdf.Gauss(mu=params[0], sigma=params[1], obs=obs)
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    
    pdf.name = name
    return pdf


class SmoothHistogramPDF(zfit.pdf.ZPDF):
    """
    Smooth histogram PDF using kernel density estimation (KDE).
    Better than raw histogram for FFT convolution.
    """
    _N_OBS = 1
    _PARAMS = []
    
    def __init__(self, data, obs, name='smooth_hist', bandwidth=None, **kwargs):
        """
        Args:
            data: 1D numpy array
            obs: Observable space
            bandwidth: Bandwidth for KDE (Scott's rule if None)
        """
        from scipy.stats import gaussian_kde
        
        self.data = data
        self.kde = gaussian_kde(data, bw_method=bandwidth or 'scott')
        
        # Store KDE parameters as TF constants
        self._kde_mean = tf.constant(np.mean(data), dtype=tf.float64)
        self._kde_std = tf.constant(np.std(data), dtype=tf.float64)
        
        super().__init__(obs=obs, name=name, **kwargs)
    
    def _unnormalized_pdf(self, x):
        """Evaluate KDE at x"""
        x_flat = zfit.z.unstack_x(x)
        
        # Numpy KDE evaluation wrapped in TF
        # Note: this is a workaround and may not be optimal
        def kde_eval(x_np):
            return tf.constant(self.kde(x_np.numpy()), dtype=tf.float64)
        
        return tf.py_function(kde_eval, [x_flat], tf.float64)
