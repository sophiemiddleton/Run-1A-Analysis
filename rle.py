"""
Resolution, Loss, Efficiency (RLE) calibration module for Mu2e analysis

Functions for generating calibration parameters from flat electron sample.
Can be called directly from process.py after data is loaded.
"""

import sys
import gc
import numpy as np
import awkward as ak
import pickle as pkl
import matplotlib.pyplot as plt
from collections import OrderedDict
import warnings
import os


try:
    import zfit
    import hist as hist
    import mplhep
except ImportError as e:
    warnings.warn(f"Optional fitting packages not available: {e}")

from pyutils.pyvector import Vector
from pyutils.pyselect import Select
from pyutils.pylogger import Logger


class RLE:
    """
    Resolution, Loss, Efficiency calculator for tracker calibration
    
    Functions that process already-loaded data (from process.py) to generate
    calibration parameters for CE/DIO signal fitting.
    """
    
    def __init__(self, verbosity=1):
        """
        Initialize RLE calculator
        
        Args:
            verbosity (int): Logging verbosity level
        """
        self.verbosity = verbosity
        self.logger = Logger(print_prefix="[RLE]", verbosity=self.verbosity)
        self.vector = Vector()
        self.selector = Select()
        
        # Fitting parameters
        self.acbtype = "gcb"
        self.landau_loss = True
        self.conv_resloss = True
        self.binwidth_eval = 0.1
        self.p_bins = [95., 97., 99., 101., 103., 105.]
        self.planes = ['entrance', 'middle', 'exit']
        
        self.logger.log("Initialized RLE calculator", "info")
    
    def generate_efficiency(self, data, output_dir="./common"):
        """
        Generate efficiency plot from origin momentum distribution
        
        Uses generated momentum distribution normalized by event count.
        This represents the shape of accepted events across momentum range.
        
        Args:
            data (dict): Processed data with 'trkmc', 'trkfit', 'trk' keys
            output_dir (str): Output directory for plots
            
        Returns:
            tuple: (efficiency array, momentum bin edges)
        """
        self.logger.log("Generating efficiency from generated momentum distribution", "info")
        
        # Select flat e- gen particles at simulation level (rank==0)
        flat_e_sim = ((data['trkmc']["trkmcsim"]["startCode"] == 173) & 
                      (data['trkmc']["trkmcsim"]["rank"] == 0) & 
                      (data['trkmc']["trkmcsim"]["nhits"] > 0))
        
        # Reduce to track level: select tracks that have at least one flat electron sim
        flat_e_trk = ak.any(flat_e_sim, axis=-1)
        
        # Reduce to event level: keep events with at least one flat electron track
        flat_e_evt = ak.any(flat_e_trk, axis=-1)
        
        data_flat = {
            'trkmc': data['trkmc'][flat_e_evt],
            'trkfit': data['trkfit'][flat_e_evt],
            'trk': data['trk'][flat_e_evt],
        }
        
        # Extract origin momentum from all flat electrons
        trkmcsim = data_flat['trkmc']["trkmcsim"]
        origin_per_track = trkmcsim[(trkmcsim["rank"] == 0) & (trkmcsim["nhits"] > 0)]
        origin_per_track = ak.firsts(origin_per_track, axis=-1)
        origin_mom = self.vector.get_mag(origin_per_track, 'mom')
        
        # Convert to numpy and clean
        origin_mom_array = np.array(ak.flatten(origin_mom, axis=None))
        origin_mom_array = origin_mom_array[~np.isnan(origin_mom_array)]
        
        self.logger.log(f"Total generated flat electrons: {len(origin_mom_array)}", "info")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Fit range and binning
        fit_range = (90, 120)
        n_bins = 63
        
        # Create histogram of generated momenta
        h_gen, bin_edges = np.histogram(origin_mom_array, bins=n_bins, range=fit_range)
        
        # Normalize by per-bin expected count assuming 1,500,000 events uniformly over 70-120 MeV
        # (100 files × 15,000 events per file)
        # Only histogram range 79.8-105, so scale expected accordingly
        full_range = (70, 120)  # Assumed uniform generation range
        hist_range = fit_range
        total_events_assumed = 1500000  # 100 files × 15,000 per file
        fraction_in_hist_range = (hist_range[1] - hist_range[0]) / (full_range[1] - full_range[0])
        expected_in_hist_range = total_events_assumed * fraction_in_hist_range
        per_bin_expected = expected_in_hist_range / n_bins
        h_eff = h_gen / per_bin_expected
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot
        ax.stairs(h_eff, bin_edges, label='Momentum Distribution (normalized)', linewidth=2)
        ax.errorbar(bin_centers, h_eff, 
                   yerr=np.sqrt(h_gen) / per_bin_expected, fmt='o', markersize=4,
                   capsize=3, elinewidth=1, color='black', alpha=0.5)
        
        # Add total efficiency text box
        total_eff = np.sum(h_gen) / 1500000
        ax.text(0.98, 0.97, f'Total Efficiency: {total_eff:.4f}', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel("Generated Momentum [MeV/c]")
        ax.set_ylabel("Normalized Counts")
        ax.set_title("Flat Electron Generated Momentum Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        os.makedirs(output_dir, exist_ok=True)
        plot_file = f"{output_dir}/flat_efficiency.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        self.logger.log(f"Efficiency plot saved to {plot_file}", "info")
        plt.close()
        
        # Save efficiency
        eff_file = f"{output_dir}/efficiency.pkl"
        with open(eff_file, 'wb') as f:
            pkl.dump([h_eff, bin_edges], f)
        self.logger.log(f"Saved efficiency.pkl: {h_eff.shape}", "success")
        self.logger.log(f"  Min efficiency: {np.min(h_eff):.6f}", "info")
        self.logger.log(f"  Max efficiency: {np.max(h_eff):.6f}", "info")
        
        return h_eff, bin_edges
        
    
    def fit_origin_momentum_chebyshev(self, data, output_dir="./common"):
        """
        Fit Chebyshev polynomial to origin momentum distribution
        
        Creates origin_momentum_fit.png showing the fitted Chebyshev polynomial
        
        Args:
            data (dict): Processed data with 'trkmc' key
            output_dir (str): Output directory for plot
        """
        self.logger.log("Fitting Chebyshev polynomial to origin momentum", "info")
        
        # Select flat e- gen particles at simulation level (rank==0)
        flat_e_sim = ((data['trkmc']["trkmcsim"]["startCode"] == 173) & 
                      (data['trkmc']["trkmcsim"]["rank"] == 0) & 
                      (data['trkmc']["trkmcsim"]["nhits"] > 0))
        
        # Reduce to track level: select tracks that have at least one flat electron sim
        flat_e_trk = ak.any(flat_e_sim, axis=-1)
        
        # Reduce to event level: keep events with at least one flat electron track
        flat_e_evt = ak.any(flat_e_trk, axis=-1)
        
        data_flat = {
            'trkmc': data['trkmc'][flat_e_evt],
        }
        
        # Extract origin momentum from all flat electrons
        trkmcsim = data_flat['trkmc']["trkmcsim"]
        origin_per_track = trkmcsim[(trkmcsim["rank"] == 0) & (trkmcsim["nhits"] > 0)]
        origin_per_track = ak.firsts(origin_per_track, axis=-1)
        origin_mom = self.vector.get_mag(origin_per_track, 'mom')
        
        # Convert to numpy and clean
        origin_mom_array = np.array(ak.flatten(origin_mom, axis=None))
        origin_mom_array = origin_mom_array[~np.isnan(origin_mom_array)]
        
        self.logger.log(f"Fitting {len(origin_mom_array)} events to Chebyshev polynomial", "info")
        
        # Fit range and setup
        fit_range = (90, 120)
        n_bins = 100
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Data histogram
        data_hist, data_bins, _ = ax1.hist(origin_mom_array, bins=n_bins, range=fit_range, 
                                           label='Data', histtype='step', color='blue', linewidth=2)
        data_bin_center = (data_bins[:-1] + data_bins[1:]) / 2
        ax1.errorbar(data_bin_center, data_hist, yerr=np.sqrt(data_hist), fmt='o', color='blue', 
                    capsize=2, markersize=4, alpha=0.6)
        
        # Fit with zfit
        try:
            obs_mom = zfit.Space('x', limits=fit_range)
            mom_zfit = zfit.Data.from_numpy(array=origin_mom_array, obs=obs_mom)
            
            # Chebyshev polynomial parameters
            N_flat = zfit.Parameter('N_flat', 10000, 100, 500000)
            c1 = zfit.Parameter("c1", 0.1, -2, 2)
            c2 = zfit.Parameter("c2", 0.1, -2, 2)
            c3 = zfit.Parameter("c3", 0.1, -2, 2)
            c4 = zfit.Parameter("c4", 0.1, -2, 2)
            c5 = zfit.Parameter("c5", 0.1, -2, 2)
            
            coeffs = [c1, c2, c3, c4, c5]
            poly_model = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N_flat)
            
            # Fit
            nll = zfit.loss.ExtendedUnbinnedNLL(model=poly_model, data=mom_zfit)
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()
            
            # Print parameters
            print(f"\n{'='*70}", flush=True)
            print(f"CHEBYSHEV POLYNOMIAL FIT TO ORIGIN MOMENTUM", flush=True)
            print(f"{'='*70}", flush=True)
            print(f"Fit range: [{fit_range[0]}, {fit_range[1]}] MeV", flush=True)
            print(f"N_events (yield): {result.params[N_flat]['value']:.0f} ± {result.params[N_flat]['hesse']['error']:.0f}", flush=True)
            print(f"\nChebyshev Polynomial Coefficients:", flush=True)
            for param in [c1, c2, c3, c4, c5]:
                param_name = param.name
                param_value = result.params[param]['value']
                param_error = result.params[param]['hesse']['error']
                print(f"  {param_name}: {param_value:+.8f} ± {param_error:.8f}", flush=True)
            print(f"{'='*70}\n", flush=True)
            
            # Plot fitted curve
            bin_width = (fit_range[1] - fit_range[0]) / n_bins
            mom_plot = np.linspace(fit_range[0], fit_range[1], 200).reshape(-1, 1)
            poly_model_curve = zfit.run(poly_model.pdf(mom_plot) * result.params[N_flat]['value'] * bin_width)
            ax1.plot(mom_plot.flatten(), poly_model_curve.flatten(), color='red', linestyle='--', 
                    linewidth=2, label='Chebyshev Fit')
            
            # Add text box with parameters
            param_text = (
                f"Chebyshev Polynomial Fit:\n"
                f"$N_{{events}} = {result.params[N_flat]['value']:.0f}$\n"
                f"$c_{{1}} = {result.params[c1]['value']:+.4f}$\n"
                f"$c_{{2}} = {result.params[c2]['value']:+.4f}$\n"
                f"$c_{{3}} = {result.params[c3]['value']:+.4f}$\n"
                f"$c_{{4}} = {result.params[c4]['value']:+.4f}$\n"
                f"$c_{{5}} = {result.params[c5]['value']:+.4f}$"
            )
            ax1.text(0.98, 0.97, param_text, transform=ax1.transAxes, fontsize=10, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Ratio plot
            fit_at_bin_center = zfit.run(poly_model.pdf(data_bin_center.reshape(-1, 1)) * result.params[N_flat]['value'] * bin_width)
            ratio = data_hist / fit_at_bin_center
            ax2.errorbar(data_bin_center, ratio, yerr=np.sqrt(data_hist) / fit_at_bin_center, 
                        fmt='o', color='blue', capsize=2, markersize=4)
            ax2.axhline(1, color='gray', linestyle='--')
            ax2.set_ylabel('Data / Fit')
            ax2.set_ylim(0.5, 1.5)
            ax2.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.log(f"Fit failed: {e}", "warning")
            ax1.text(0.5, 0.5, f"Fit Failed: {e}", transform=ax1.transAxes, 
                    fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        # Labels and legend
        ax1.set_ylabel('# of events per bin')
        ax1.set_title('Origin Momentum with Chebyshev Polynomial Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.set_xlabel('Generated Momentum [MeV/c]')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_file = f"{output_dir}/origin_momentum_fit.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        self.logger.log(f"Origin momentum fit plot saved to {plot_file}", "info")
        plt.close()
    
    def generate_skimmed_data(self, data, output_file="skimmed_flat_mom_MDC2025an.pkl"):
        """
        Extract and save skimmed momentum data at each tracker plane
        
        Args:
            data (dict): Processed data with 'trkmc', 'trkfit', 'trk' keys
            output_file (str): Output file path
            
        Returns:
            dict: Skimmed data with gen/mc/reco momenta per plane
        """
        self.logger.log("Generating skimmed momentum data", "info")
        
        # Select flat e- gen particles at simulation level
        flat_e_sim = ((data['trkmc']["trkmcsim"]["startCode"] == 173) & 
                      (data['trkmc']["trkmcsim"]["rank"] == 0) & 
                      (data['trkmc']["trkmcsim"]["nhits"] > 0))
        
        # Reduce to track level: select tracks that have at least one flat electron sim
        flat_e_trk = ak.any(flat_e_sim, axis=-1)
        
        # Reduce to event level: keep events with at least one flat electron track
        flat_e_evt = ak.any(flat_e_trk, axis=-1)
        
        # Filter data to events with flat electrons
        data_flat_fit = data['trkfit'][flat_e_evt]
        data_flat_mc = data['trkmc'][flat_e_evt]
        
        data_flat = {}
        
        for sid, plane in enumerate(self.planes):
            self.logger.log(f"  Processing {plane} plane (sid={sid})...", "info")
            
            # Select segments at the specified plane by sid
            # Create masks for reco and MC segments at this station
            at_plane_reco = (data_flat_fit["trksegs"]["sid"] == sid)
            at_plane_mc = (data_flat_fit["trksegsmc"]["sid"] == sid)
            
            good_track = (ak.sum(at_plane_reco, axis=2) >= 1)
            good_track = (good_track) & (ak.sum(at_plane_mc, axis=2) == 1)
            
            gc.collect()
            
            # Broadcast good_track mask to segment level by adding axis
            good_track_seg = good_track[:, :, None]
            
            # Extract momenta using ak.mask() to avoid array deletion issues
            reco_segs = ak.mask(data_flat_fit["trksegs"], (at_plane_reco) & good_track_seg)
            reco_mom = self.vector.get_mag(reco_segs, 'mom')
            reco_mom = ak.nan_to_none(reco_mom)
            reco_mom = ak.drop_none(reco_mom)
            reco_mom = np.array(ak.flatten(reco_mom, axis=None))
            
            mc_segs = ak.mask(data_flat_fit["trksegsmc"], (at_plane_mc) & good_track_seg)
            mc_mom = self.vector.get_mag(mc_segs, 'mom')
            mc_mom = ak.nan_to_none(mc_mom)
            mc_mom = ak.drop_none(mc_mom)
            mc_mom = np.array(ak.flatten(mc_mom, axis=None))
            
            # For gen_mom, only extract from tracks that have segments at this plane
            # Select the first MC truth entry (rank=0) for each track, only if track has segments
            gen_segs = ak.mask(data_flat_mc["trkmcsim"], good_track_seg)
            # Take only first MC truth per track (rank should already be filtered to 0)
            gen_mom_jagged = ak.firsts(gen_segs, axis=-1)
            gen_mom = self.vector.get_mag(gen_mom_jagged, 'mom')
            gen_mom = ak.nan_to_none(gen_mom)
            gen_mom = ak.drop_none(gen_mom)
            gen_mom = np.array(ak.flatten(gen_mom, axis=None))
            
            self.logger.log(f"    Reco: {len(reco_mom)}, MC: {len(mc_mom)}, Gen: {len(gen_mom)}", "debug")
            
            data_flat[plane] = {'reco': reco_mom, 'mc': mc_mom, 'gen': gen_mom}
            gc.collect()
        
        # Save skimmed data
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'wb') as f:
            pkl.dump(data_flat, f)
        
        self.logger.log(f"Saved skimmed data to {output_file}", "success")
        return data_flat
    
    def fit_resolution_loss(self, skimmed_data_path, output_dir="./common", fit_types=['res', 'loss']):
        """
        Fit resolution and loss distributions and save plots
        
        Args:
            skimmed_data_path (str): Path to skimmed_flat_mom_v2.pkl
            output_dir (str): Output directory for plots
            fit_types (list): Types of fits to perform ('res', 'loss', 'resloss')
            
        Returns:
            dict: Fit results
        """
        import zfit
        try:
            # Import custom truncated Landau
            from RLE.landau_pdf import trunc_landau
            self.logger.log("Custom trunc_landau imported successfully", "debug")
        except ImportError:
            try:
                from landau_pdf import trunc_landau
                self.logger.log("Custom trunc_landau imported successfully", "debug")
            except Exception as e:
                self.logger.log(f"ERROR: Failed to import trunc_landau - {type(e).__name__}: {e}", "error")
                import traceback
                self.logger.log(traceback.format_exc(), "error")
                return None
        
        self.logger.log("Starting resolution/loss fitting", "info")
        
        # Load skimmed data
        try:
            with open(skimmed_data_path, 'rb') as f:
                dict_flat = pkl.load(f)
            self.logger.log(f"Loaded skimmed data from {skimmed_data_path}", "debug")
        except Exception as e:
            self.logger.log(f"ERROR: Failed to load skimmed data - {e}", "error")
            return None
        
        # Create output subdirectory for fits
        fit_dir = f"{output_dir}/fits"
        os.makedirs(fit_dir, exist_ok=True)
        self.logger.log(f"Created fit output directory: {fit_dir}", "debug")
        
        results = {}
        fit_params_list = []  # Collect all fit parameters for CSV export
        
        for fit_type in fit_types:
            self.logger.log(f"  Fitting {fit_type}...", "info")
            
            fig_results = []
            
            for plane in self.planes:
                self.logger.log(f"    {plane} plane", "info")
                
                try:
                    if fit_type == 'res':
                        mom_in = dict_flat[plane]['mc']
                        mom_out = dict_flat[plane]['reco']
                        res_range = (-1, 1)
                        xlabel = r"$p_{reco} - p_{mc}$ (MeV)"
                        title_fit = "Resolution"
                    elif fit_type == 'loss':
                        mom_in = dict_flat[plane]['gen']
                        mom_out = dict_flat[plane]['mc']
                        res_range = (-5, 5)
                        xlabel = r"$p_{mc} - p_{gen}$ (MeV)"
                        title_fit = "Energy Loss"
                    else:  # resloss
                        mom_in = dict_flat[plane]['gen']
                        mom_out = dict_flat[plane]['reco']
                        res_range = (-6, 6)
                        xlabel = r"$p_{reco} - p_{gen}$ (MeV)"
                        title_fit = "Resolution + Loss"
                    
                    res_slice = mom_out - mom_in
                    self.logger.log(f"      Data shape: {len(res_slice)}, range: [{res_slice.min():.2f}, {res_slice.max():.2f}]", "debug")
                    
                    # Create figure with histogram and fit
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    
                    # Histogram
                    counts, bins, patches = ax.hist(res_slice, bins=50, range=res_range, 
                                                   label='Flat e⁻', alpha=0.7, color='orange')
                    
                    # Add error bars
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    ax.errorbar(bin_centers, counts, yerr=np.sqrt(counts), fmt='none', color='black', capsize=2, elinewidth=1)
                    
                    self.logger.log(f"      Histogram created", "debug")
                    
                    # Fit based on type
                    try:
                        obs = zfit.Space('x', res_range[0], res_range[1])
                        data = zfit.Data(data=res_slice, obs=obs)
                        self.logger.log(f"      zfit objects created", "debug")
                        
                        if fit_type == 'res':
                            # Resolution fit with Generalized Crystal Ball
                            self.logger.log(f"      Starting GCB fit...", "debug")
                            mu     = zfit.Parameter('mu', 0, res_range[0], res_range[1])
                            sigmaL = zfit.Parameter('sigmaL', 0.2, 0.01, 1.0)
                            sigmaR = zfit.Parameter('sigmaR', 0.2, 0.01, 1.0)
                            alphaL = zfit.Parameter('alphaL', 1.0, 0.1, 3.0)
                            alphaR = zfit.Parameter('alphaR', 1.0, 0.1, 3.0)
                            nL     = zfit.Parameter('nL', 2.0, 0.5, 12.0)
                            nR     = zfit.Parameter('nR', 2.0, 0.5, 12.0)
                            
                            pdf = zfit.pdf.GeneralizedCB(obs=obs, mu=mu, sigmal=sigmaL, sigmar=sigmaR, 
                                                         alphal=alphaL, alphar=alphaR, nl=nL, nr=nR)
                            fit_label = 'Generalized Crystal Ball Fit'
                        else:
                            # Energy loss fit with truncated Landau
                            self.logger.log(f"      Starting trunc_landau fit...", "debug")
                            loc = zfit.Parameter('loc', -1.0, res_range[0], res_range[1])
                            scale = zfit.Parameter('scale', 1.0, 0.1, 5.0)
                            pdf = trunc_landau(loc=loc, scale=scale, obs=obs)
                            fit_label = 'Truncated Landau Fit'
                        
                        self.logger.log(f"      PDF created", "debug")
                        
                        nll = zfit.loss.UnbinnedNLL(model=pdf, data=data)
                        minimizer = zfit.minimize.Minuit(verbosity=0)
                        self.logger.log(f"      Starting minimization...", "debug")
                        result = minimizer.minimize(nll)
                        self.logger.log(f"      Minimization complete", "debug")
                        
                        # Plot fitted curve
                        x_plot = np.linspace(res_range[0], res_range[1], 200)
                        y_plot = pdf.pdf(x_plot).numpy() * len(res_slice) * (res_range[1] - res_range[0]) / 50
                        ax.plot(x_plot, y_plot, 'r-', linewidth=2, label=fit_label)
                        
                        # Log fit parameters with better formatting
                        self.logger.log(f"      {fit_label} Parameters:", "info")
                        
                        chi2 = None
                        reduced_chi2 = None
                        
                        for param in result.params:
                            try:
                                # Try to get errors if available
                                errors = result.errors()
                                param_err = None
                                if param in errors:
                                    lower_err = abs(errors[param]['lower'])
                                    upper_err = abs(errors[param]['upper'])
                                    avg_err = (lower_err + upper_err) / 2
                                    param_err = avg_err
                                    self.logger.log(f"        {param.name:12s} = {float(param.value()):10.6f} ± {avg_err:.6f}", "info")
                                else:
                                    self.logger.log(f"        {param.name:12s} = {float(param.value()):10.6f}", "info")
                            except:
                                self.logger.log(f"        {param.name:12s} = {float(param.value()):10.6f}", "info")
                                param_err = None
                        
                        # Calculate chi-square
                        try:
                            bin_centers = (bins[:-1] + bins[1:]) / 2
                            bin_width = bins[1] - bins[0]
                            expected = pdf.pdf(bin_centers).numpy() * len(res_slice) * bin_width
                            chi2 = np.sum((counts - expected)**2 / np.maximum(expected, 1))
                            ndof = len(counts) - len(result.params)
                            reduced_chi2 = chi2 / max(ndof, 1)
                            self.logger.log(f"        χ² = {chi2:.2f}, χ²/dof = {reduced_chi2:.4f}", "info")
                        except:
                            pass
                        
                        # Collect fit results for CSV export
                        for param in result.params:
                            try:
                                errors = result.errors()
                                param_err = None
                                if param in errors:
                                    lower_err = abs(errors[param]['lower'])
                                    upper_err = abs(errors[param]['upper'])
                                    param_err = (lower_err + upper_err) / 2
                                
                                fit_params_list.append({
                                    'fit_type': fit_type,
                                    'plane': plane,
                                    'parameter': param.name,
                                    'value': float(param.value()),
                                    'error': param_err,
                                    'chi2': chi2,
                                    'reduced_chi2': reduced_chi2
                                })
                            except:
                                pass
                        
                    except Exception as e:
                        self.logger.log(f"      Fit failed: {str(e)}", "warn")
                    
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel('Events per bin')
                    ax.set_title(f'{title_fit} - {plane} plane')
                    ax.legend()
                    
                    plot_path = f"{fit_dir}/{fit_type}_{plane}.png"
                    fig.savefig(plot_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                    self.logger.log(f"      Plot saved to {plot_path}", "debug")
                    fig_results.append(plot_path)
                    
                except Exception as e:
                    self.logger.log(f"    ERROR processing {plane}: {e}", "error")
                    import traceback
                    self.logger.log(traceback.format_exc(), "debug")
            
            results[fit_type] = fig_results
        
        # Save fit parameters to CSV
        if fit_params_list:
            import csv
            csv_path = f"{output_dir}/fit_parameters.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['fit_type', 'plane', 'parameter', 'value', 'error', 'chi2', 'reduced_chi2']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in fit_params_list:
                    writer.writerow(row)
            self.logger.log(f"Saved fit parameters to {csv_path}", "success")
        
        self.logger.log(f"Saved fit plots to {fit_dir}/", "success")
        return results




def generate_rle_calibration(combined_data, output_dir="./common", run_fits=True):
    """
    Generate RLE calibration from already-processed data
    
    Call this after data has been loaded and analyzed in process.py
    
    Args:
        combined_data (dict): Combined processed data from AnaProcessor
        output_dir (str): Output directory
        run_fits (bool): Generate fit plots
        
    Returns:
        dict: Results including efficiency, skimmed data, and fit plots
    """
    rle = RLE(verbosity=1)
    
    rle.logger.log("="*60, "info")
    rle.logger.log("STARTING RLE CALIBRATION FROM PROCESS.PY", "info")
    rle.logger.log("="*60, "info")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate efficiency
    rle.logger.log("\nStep 1: Generating origin momentum fit...", "info")
    rle.generate_efficiency(combined_data, output_dir)
    rle.fit_origin_momentum_chebyshev(combined_data, output_dir)
    
    # Step 2: Generate skimmed data
    rle.logger.log("\nStep 2: Generating skimmed data...", "info")
    skimmed_path = f"{output_dir}/skimmed_flat_mom_MDC2025an.pkl"
    data_flat = rle.generate_skimmed_data(combined_data, skimmed_path)
    
    # Step 3: Fit resolution/loss (optional)
    fit_results = None
    if run_fits:
        rle.logger.log("\nStep 3: Fitting resolution and loss...", "info")
        try:
            fit_results = rle.fit_resolution_loss(skimmed_path, output_dir, ['res', 'loss'])
        except Exception as e:
            rle.logger.log(f"Fitting failed: {e}", "warning")
    
    rle.logger.log("="*60, "success")
    rle.logger.log("RLE CALIBRATION COMPLETE", "success")
    rle.logger.log("="*60, "success")
    rle.logger.log(f"\nOutput files saved to: {output_dir}/", "info")
    rle.logger.log(f"  - efficiency_MDC2025an.pkl", "info")
    rle.logger.log(f"  - flat_efficiency.png", "info")
    rle.logger.log(f"  - origin_momentum_fit.png (Chebyshev polynomial fit)", "info")
    rle.logger.log(f"  - skimmed_flat_mom_MDC2025an.pkl", "info")
    if fit_results:
        rle.logger.log(f"  - fits/*.png (resolution/loss plots)", "info")
    
    # Load efficiency that was just created
    efficiency_data = None
    eff_file = f"{output_dir}/efficiency.pkl"
    if os.path.exists(eff_file):
        try:
            with open(eff_file, 'rb') as f:
                efficiency_data = pkl.load(f)
        except Exception as e:
            rle.logger.log(f"Failed to load efficiency: {e}", "warn")
    
    return {
        'efficiency': efficiency_data,
        'skimmed_data': data_flat,
        'fit_plots': fit_results
    }
