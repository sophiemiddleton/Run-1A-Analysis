import numpy as np
import math
import zfit

# Physical constants for CeLL spectrum (Mu2e specific)
E_MAX = 104.969  # Maximum electron momentum [MeV/c]
M_E = 0.511      # Electron mass [MeV]
ALPHA = 1.0 / 137.036  # Fine structure constant

def LeadingLog(E):
    """
    Theoretical Leading Log conversion spectrum formula.
    Standard implementation for single-value evaluation.
    """
    prefactor = (1.0 / E_MAX) * (ALPHA / (2.0 * math.pi))
    log_term  = math.log(4.0 * E**2 / M_E**2) - 2.0
    energy_term = (E**2 + E_MAX**2) / (E_MAX * (E_MAX - E))
    
    val = prefactor * log_term * energy_term
    return max(0.0, val)


def binned_spectrum_CeLL(binwidth: float = 0.1):
    """
    Calculates the binned Conversion Electron spectrum data (values, edges).
    """
    # 1. Determine binning grid
    nbins = int(math.ceil(E_MAX / binwidth))
    upedge = binwidth * nbins
    
    edges = np.linspace(0.0, upedge, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    
    # 2. Vectorized calculation (Replaces np.vectorize for speed)
    # Clip centers slightly away from E_MAX to avoid division by zero
    safe_centers = np.clip(centers, 1e-6, E_MAX - 1e-6)
    
    prefactor = (1.0 / E_MAX) * (ALPHA / (2.0 * np.pi))
    log_vals  = np.log(4.0 * safe_centers**2 / M_E**2) - 2.0
    en_vals   = (safe_centers**2 + E_MAX**2) / (E_MAX * (E_MAX - safe_centers))
    
    values = prefactor * log_vals * en_vals
    
    # 3. Apply physical boundaries
    values = np.where(centers < E_MAX, values, 0.0)
    values = np.maximum(0.0, values)
    
    # 4. Normalization and Endpoint handling
    # Handle the 'delta-like' behavior at the endpoint bin
    integral_excluding_last = np.sum(values[:-1] * binwidth)
    
    if integral_excluding_last < 1.0:
        values[-1] = (1.0 - integral_excluding_last) / binwidth
    else:
        # Fallback to standard unit normalization if integral already exceeds 1
        values /= np.sum(values * binwidth)
        
    return (values, edges)


class TheorySpectrum:
    """
    Builds a theory spectrum PDF for Mu2e signal analysis.
    
    Wraps the binned CeLL spectrum and creates a zfit PDF for use in
    convolution with resolution and loss distributions.
    """
    
    def __init__(self, mom_range=(90, 120), binwidth=0.1, verbosity=0):
        """
        Initialize theory spectrum builder.
        
        Args:
            mom_range (tuple): (min, max) momentum range in MeV/c
            binwidth (float): Bin width for spectrum calculation in MeV/c
            verbosity (int): Verbosity level for logging
        """
        self.mom_range = mom_range
        self.binwidth = binwidth
        self.verbosity = verbosity
        
        # Compute spectrum
        self.spectrum_values, self.spectrum_edges = binned_spectrum_CeLL(binwidth)
        
        if verbosity > 0:
            print(f"[TheorySpectrum] Spectrum computed: {len(self.spectrum_values)} bins")
            print(f"[TheorySpectrum] Range: [{self.spectrum_edges[0]:.2f}, {self.spectrum_edges[-1]:.2f}] MeV")
    
    def get_pdf(self, obs=None, name="theory_spectrum"):
        """
        Create a zfit PDF from the binned spectrum.
        
        Uses histogram interpolation to create a smooth PDF.
        
        Args:
            obs (zfit.Space): Observable space (momentum). If None, creates from mom_range.
            name (str): Name for the PDF
            
        Returns:
            zfit.pdf.BasePDF: A zfit PDF object that can be used in convolution
        """
        # Create observable if not provided
        if obs is None:
            obs = zfit.Space('mom', limits=self.mom_range)
        
        # Import helper function for robust histogram PDF creation
        from helper import make_HistogramPDF
        
        # Create histogram PDF from binned spectrum
        spectrum_counts = self.spectrum_values * self.binwidth
        
        # Use the helper function to create a proper zfit histogram PDF
        HistogramPDFClass = make_HistogramPDF(spectrum_counts, self.spectrum_edges)
        pdf = HistogramPDFClass(obs=obs, name=name)
        
        return pdf
    
    def plot_spectrum(self, output_file=None):
        """
        Plot the theory spectrum.
        
        Args:
            output_file (str): Optional path to save plot
        """
        import matplotlib.pyplot as plt
        
        bin_centers = (self.spectrum_edges[:-1] + self.spectrum_edges[1:]) / 2.0
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.step(bin_centers, self.spectrum_values, where='mid', color='blue', linewidth=2)
        ax.fill_between(bin_centers, self.spectrum_values, step='mid', alpha=0.3, color='blue')
        
        ax.set_xlabel("Momentum [MeV/c]")
        ax.set_ylabel("Probability Density")
        ax.set_title("CeLL Theory Spectrum (Leading Log)")
        ax.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            if self.verbosity > 0:
                print(f"[TheorySpectrum] Spectrum plot saved to {output_file}")
        
        return fig, ax

