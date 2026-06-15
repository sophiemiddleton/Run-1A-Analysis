#!/usr/bin/env python3
"""
DIO (Decay In Orbit) background spectrum model for momentum space.

Uses a 5th-8th order polynomial parameterization fitted to DIO tail data:
    spectrum = a5 * delta^5 + a6 * delta^6 + a7 * delta^7 + a8 * delta^8
where delta = M_MU - x - x^2 / (2 * m_Al)

Reference: Mu2e physics backgrounds
"""

import numpy as np
import tensorflow as tf
import zfit
from helper import make_HistogramPDF


# Physical constants
M_MU = 105.658  # Muon mass [MeV]
M_AL = 25133.0  # Aluminum-27 mass [MeV]


class DIOPoly58(zfit.pdf.ZPDF):
    """5th to 8th order polynomial for DIO background (momentum space)."""
    _N_OBS = 1
    _PARAMS = ['a5', 'a6', 'a7', 'a8']

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        a5 = self.params['a5']
        a6 = self.params['a6']
        a7 = self.params['a7']
        a8 = self.params['a8']
        
        # Kinematic endpoint function
        # delta = (M_MU - x - x^2 / (2 * m_Al))
        delta = M_MU - x - x**2 / (2.0 * M_AL)
        
        # Ensure delta is positive (physical region)
        delta = tf.nn.relu(delta)
        
        # Polynomial: a5*delta^5 + a6*delta^6 + a7*delta^7 + a8*delta^8
        result = (a5 * tf.pow(delta, 5.0) + 
                 a6 * tf.pow(delta, 6.0) + 
                 a7 * tf.pow(delta, 7.0) + 
                 a8 * tf.pow(delta, 8.0))
        
        return result


class TheoryDIOSpectrum:
    """
    DIO background spectrum generator using polynomial parameterization.
    
    The DIO spectrum is parameterized with coefficients a5-a8 fitted to
    simulation data. This class can generate either the PDF directly or
    convert to a histogram-based PDF for integration with convolution.
    """
    
    def __init__(self, mom_range=(95, 110), binwidth=0.1, verbosity=1):
        """
        Initialize DIO spectrum generator.
        
        Args:
            mom_range: tuple (lo, hi) - momentum range in MeV
            binwidth: bin width for histogram generation (default 0.1 MeV)
            verbosity: print level (0=silent, 1=normal, 2=verbose)
        """
        self.mom_range = mom_range
        self.binwidth = binwidth
        self.verbosity = verbosity
        self.spectrum_values = None
        self.spectrum_edges = None
        
    def get_pdf(self, obs, params=None, name='DIO_theory'):
        """
        Create DIO spectrum PDF using polynomial model.
        
        Args:
            obs: zfit observable space
            params: dict with 'a5', 'a6', 'a7', 'a8' parameters
                   If None, uses default fitted values
            name: name for the PDF
            
        Returns:
            DIOPoly58 PDF instance
        """
        # Default parameters from DIO calibration
        if params is None:
            params = {
                'a5': (8.97879e-17, 1e-17, 1e-16),
                'a6': (1.17169e-17, 1e-18, 1e-16),
                'a7': (-1.06599e-19, -1e-18, -1e-19),
                'a8': (8.14251e-20, 1e-20, 1e-19)
            }
        
        # Create zfit parameters
        a5_param = zfit.Parameter('a5', params['a5'][0], floating=False)
        a6_param = zfit.Parameter('a6', params['a6'][0], floating=False)
        a7_param = zfit.Parameter('a7', params['a7'][0], floating=False)
        a8_param = zfit.Parameter('a8', params['a8'][0], floating=False)
        
        # Create PDF
        dio_pdf = DIOPoly58(
            obs=obs,
            a5=a5_param,
            a6=a6_param,
            a7=a7_param,
            a8=a8_param,
            name=name
        )
        
        if self.verbosity > 0:
            print(f"[DIO Spectrum]")
            #print(f"  Observable: {obs.names}")
            print(f"  a5 = {params['a5'][0]:.6e}")
            print(f"  a6 = {params['a6'][0]:.6e}")
            print(f"  a7 = {params['a7'][0]:.6e}")
            print(f"  a8 = {params['a8'][0]:.6e}")
        
        return dio_pdf
    
    def get_histogram_pdf(self, obs, params=None, name='DIO_theory'):
        """
        Create DIO spectrum as histogram PDF for convolution.
        
        Generates bins of the DIO polynomial spectrum and wraps in HistogramPDF.
        
        Args:
            obs: zfit observable space
            params: dict with coefficients (default: fitted values)
            name: name for the PDF
            
        Returns:
            HistogramPDF instance
        """
        lo, hi = self.mom_range
        
        # Create evaluation grid
        n_bins = int((hi - lo) / self.binwidth)
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        
        # Default parameters
        if params is None:
            params = {
                'a5': 8.97879e-17,
                'a6': 1.17169e-17,
                'a7': -1.06599e-19,
                'a8': 8.14251e-20
            }
        
        # Evaluate spectrum at bin centers
        a5, a6, a7, a8 = params['a5'], params['a6'], params['a7'], params['a8']
        delta = M_MU - centers - centers**2 / (2.0 * M_AL)
        delta = np.maximum(delta, 0)  # Ensure positive
        
        spectrum_vals = (a5 * delta**5 + 
                        a6 * delta**6 + 
                        a7 * delta**7 + 
                        a8 * delta**8)
        
        # Normalize to unit area
        bin_width = edges[1] - edges[0]
        integral = np.sum(spectrum_vals) * bin_width
        if integral > 0:
            spectrum_vals = spectrum_vals / integral
        
        if self.verbosity > 0:
            print(f"[DIO Histogram Spectrum]")
            print(f"  Bins: {n_bins}, Range: [{lo}, {hi}]")
            print(f"  Spectrum min/max: {spectrum_vals.min():.6e} / {spectrum_vals.max():.6e}")
            print(f"  Integral: {integral:.6e}")
        
        # Create HistogramPDF
        HistogramPDFClass = make_HistogramPDF(spectrum_vals, edges)
        dio_hist_pdf = HistogramPDFClass(obs=obs, name=name)
        
        return dio_hist_pdf
    
    def plot_spectrum(self, filename='DIO_spectrum.pdf', include_poly=True):
        """
        Plot the DIO spectrum.
        
        Args:
            filename: output PDF filename
            include_poly: also plot individual polynomial terms
        """
        import matplotlib.pyplot as plt
        
        lo, hi = self.mom_range
        x = np.linspace(lo, hi, 500)
        
        # Evaluate spectrum
        delta = M_MU - x - x**2 / (2.0 * M_AL)
        delta = np.maximum(delta, 0)
        
        a5, a6, a7, a8 = 8.97879e-17, 1.17169e-17, -1.06599e-19, 8.14251e-20
        spectrum = a5 * delta**5 + a6 * delta**6 + a7 * delta**7 + a8 * delta**8
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, spectrum, 'b-', linewidth=2.5, label='DIO Spectrum')
        
        if include_poly:
            ax.plot(x, a5 * delta**5, '--', alpha=0.6, label='a5·δ⁵')
            ax.plot(x, a6 * delta**6, '--', alpha=0.6, label='a6·δ⁶')
            ax.plot(x, a7 * delta**7, '--', alpha=0.6, label='a7·δ⁷')
            ax.plot(x, a8 * delta**8, '--', alpha=0.6, label='a8·δ⁸')
        
        ax.set_xlabel('Momentum [MeV/c]', fontsize=12)
        ax.set_ylabel('Spectrum (a.u.)', fontsize=12)
        ax.set_title('DIO Background Spectrum', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Saved to {filename}")
        plt.close()


if __name__ == '__main__':
    # Example usage
    print("DIO Spectrum Module")
    print("=" * 50)
    
    # Create spectrum generator
    dio_spec = TheoryDIOSpectrum(mom_range=(95, 110), binwidth=0.1, verbosity=2)
    
    # Create observable
    obs_dio = zfit.Space('x', limits=(95, 110))
    
    # Get polynomial PDF
    print("\n1. Creating polynomial PDF...")
    dio_pdf = dio_spec.get_pdf(obs=obs_dio)
    
    # Get histogram PDF
    print("\n2. Creating histogram PDF...")
    dio_hist_pdf = dio_spec.get_histogram_pdf(obs=obs_dio)
    
    # Plot
    print("\n3. Plotting spectrum...")
    dio_spec.plot_spectrum(filename='DIO_spectrum_test.pdf')
    
    print("\n✓ DIO spectrum module ready")
