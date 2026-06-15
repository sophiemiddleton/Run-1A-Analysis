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
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
import warnings
import os
import traceback


try:
    import zfit
    import hist as hist
    import mplhep
except ImportError as e:
    warnings.warn(f"Optional fitting packages not available: {e}")

try:
    import tensorflow as tf
except ImportError as e:
    warnings.warn(f"TensorFlow not available: {e}")

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
        
        # Global momentum range parameters
        self.fit_range = (95, 110)  # Range for efficiency and Chebyshev fits
        self.full_range = (70, 120)  # Assumed uniform generation range
        
        # Global convolution observable space parameters
        self.p_obs_range = (90, 120)  # Momentum space for convolution
        self.loss_range = (-5, 2)  # Loss kernel space: (min, max)
        self.res_range = (-1, 1)  # Resolution kernel space (symmetric)
        
        # Resolution and loss fit ranges
        self.res_fit_range = (-1, 1)  # Resolution fit range
        self.loss_fit_range = (-5, 5)  # Loss fit range
        self.resloss_fit_range = (-6, 6)  # Combined resolution+loss fit range
        
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
        fit_range = self.fit_range
        n_bins = 63
        
        # Create histogram of generated momenta
        h_gen, bin_edges = np.histogram(origin_mom_array, bins=n_bins, range=fit_range)
        
        # Normalize by per-bin expected count using actual number of files and correct fraction in fit range
        full_range = self.full_range  # Assumed uniform generation range
        hist_range = fit_range
        # Count number of files in file_lists/FlateMinus.txt
        flist_path = os.path.join(os.path.dirname(__file__), "file_lists/FlateMinus.txt")
        try:
            with open(flist_path, "r") as f:
                n_files = sum(1 for _ in f)
        except Exception as e:
            self.logger.log(f"Could not read file list: {e}", "error")
            n_files = 100  # fallback default
        total_events_assumed = 15000 * n_files
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
        total_eff = np.sum(h_gen) / expected_in_hist_range
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
            
        Returns:
            dict: Fitted coefficients {'c1', 'c2', 'c3', 'c4', 'c5'} or None if fit fails
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
        fit_range = self.fit_range
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
            result = None
        
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
        
        # Return fitted parameters with errors
        if result is not None:
            return {
                'c1': {'value': float(result.params[c1]['value']), 'error': float(result.params[c1]['hesse']['error'])},
                'c2': {'value': float(result.params[c2]['value']), 'error': float(result.params[c2]['hesse']['error'])},
                'c3': {'value': float(result.params[c3]['value']), 'error': float(result.params[c3]['hesse']['error'])},
                'c4': {'value': float(result.params[c4]['value']), 'error': float(result.params[c4]['hesse']['error'])},
                'c5': {'value': float(result.params[c5]['value']), 'error': float(result.params[c5]['hesse']['error'])},
            }
        else:
            return None
    
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
            dict: Fit results including fitted parameters with errors
                  {'res': {...}, 'loss': {...}}
                  Each fit type contains parameters averaged across planes
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
        param_results = {'res': {}, 'loss': {}}  # Store fitted parameters per fit type
        
        for fit_type in fit_types:
            self.logger.log(f"  Fitting {fit_type}...", "info")
            
            fig_results = []
            fit_type_params = {}  # Store all parameters for this fit type
            
            for plane in self.planes:
                self.logger.log(f"    {plane} plane", "info")
                
                try:
                    if fit_type == 'res':
                        mom_in = dict_flat[plane]['mc']
                        mom_out = dict_flat[plane]['reco']
                        res_range = self.res_fit_range
                        xlabel = r"$p_{reco} - p_{mc}$ (MeV)"
                        title_fit = "Resolution"
                    elif fit_type == 'loss':
                        mom_in = dict_flat[plane]['gen']
                        mom_out = dict_flat[plane]['mc']
                        res_range = self.loss_fit_range
                        xlabel = r"$p_{mc} - p_{gen}$ (MeV)"
                        title_fit = "Energy Loss"
                    else:  # resloss
                        mom_in = dict_flat[plane]['gen']
                        mom_out = dict_flat[plane]['reco']
                        res_range = self.resloss_fit_range
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
                            param_list = [mu, sigmaL, sigmaR, alphaL, alphaR, nL, nR]
                        else:
                            # Energy loss fit with truncated Landau
                            self.logger.log(f"      Starting trunc_landau fit...", "debug")
                            loc = zfit.Parameter('loc', -1.0, res_range[0], res_range[1])
                            scale = zfit.Parameter('scale', 1.0, 0.1, 5.0)
                            pdf = trunc_landau(loc=loc, scale=scale, obs=obs)
                            fit_label = 'Truncated Landau Fit'
                            param_list = [loc, scale]
                        
                        self.logger.log(f"      PDF created", "debug")
                        
                        nll = zfit.loss.UnbinnedNLL(model=pdf, data=data)
                        minimizer = zfit.minimize.Minuit(verbosity=0)
                        self.logger.log(f"      Starting minimization...", "debug")
                        result = minimizer.minimize(nll)
                        self.logger.log(f"      Minimization complete", "debug")
                        
                        # Compute Hesse errors (like the Chebyshev fit does)
                        try:
                            result.hesse()
                        except Exception as e:
                            self.logger.log(f"      WARNING: Failed to compute Hesse errors - {e}", "warn")
                        
                        # Plot fitted curve
                        x_plot = np.linspace(res_range[0], res_range[1], 200)
                        y_plot = pdf.pdf(x_plot).numpy() * len(res_slice) * (res_range[1] - res_range[0]) / 50
                        ax.plot(x_plot, y_plot, 'r-', linewidth=2, label=fit_label)
                        
                        # Print fit parameters with errors to stdout for user inspection
                        print(f"\n{'='*70}", flush=True)
                        print(f"{fit_label.upper()} - {plane.upper()} PLANE", flush=True)
                        print(f"{'='*70}", flush=True)
                        
                        # Log fit parameters with better formatting
                        self.logger.log(f"      {fit_label} Parameters:", "info")
                        
                        chi2 = None
                        reduced_chi2 = None
                        plane_params = {}
                        
                        # Extract errors from result.params[param]['hesse'] (like Chebyshev fit does)
                        for param in param_list:
                            param_value = float(param.value())
                            param_err = None
                            
                            # Try to extract error from result.params like the Chebyshev fit does
                            try:
                                if param in result.params:
                                    param_info = result.params[param]
                                    if isinstance(param_info, dict) and 'hesse' in param_info:
                                        param_err = param_info['hesse'].get('error', None)
                            except Exception as e:
                                self.logger.log(f"      Could not extract error for {param.name}: {e}", "debug")
                            
                            # Print with or without error
                            if param_err is not None:
                                self.logger.log(f"        {param.name:12s} = {param_value:10.6f} ± {param_err:.6f}", "info")
                                print(f"  {param.name:12s}: {param_value:+.6f} ± {param_err:.6f}", flush=True)
                            else:
                                self.logger.log(f"        {param.name:12s} = {param_value:10.6f} (no error)", "info")
                                print(f"  {param.name:12s}: {param_value:+.6f} (no error)", flush=True)
                            
                            plane_params[param.name] = {'value': param_value, 'error': param_err}
                            
                            # Also collect for CSV export
                            fit_params_list.append({
                                'fit_type': fit_type,
                                'plane': plane,
                                'parameter': param.name,
                                'value': param_value,
                                'error': param_err,
                                'chi2': None,  # Will be set below
                                'reduced_chi2': None
                            })

                        try:
                            bin_centers = (bins[:-1] + bins[1:]) / 2
                            bin_width = bins[1] - bins[0]
                            expected = pdf.pdf(bin_centers).numpy() * len(res_slice) * bin_width
                            chi2 = np.sum((counts - expected)**2 / np.maximum(expected, 1))
                            ndof = len(counts) - len(param_list)
                            reduced_chi2 = chi2 / max(ndof, 1)
                            self.logger.log(f"        χ² = {chi2:.2f}, χ²/dof = {reduced_chi2:.4f}", "info")
                            print(f"χ² = {chi2:.2f}, χ²/dof = {reduced_chi2:.4f}", flush=True)
                        except:
                            pass
                        
                        print(f"{'='*70}\n", flush=True)
                        
                        # Store plane results
                        fit_type_params[plane] = plane_params
                        
                        # Update chi2 values in fit_params_list for this plane
                        for item in fit_params_list[-len(param_list):]:
                            item['chi2'] = chi2
                            item['reduced_chi2'] = reduced_chi2
                        
                    except Exception as e:
                        self.logger.log(f"      Fit failed: {str(e)}", "warn")
                        print(f"Fit failed: {str(e)}", flush=True)
                    
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
                    print(f"ERROR processing {plane}: {e}", flush=True)
                    import traceback
                    self.logger.log(traceback.format_exc(), "debug")
            
            results[fit_type] = fig_results
            param_results[fit_type] = fit_type_params
        
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
        
        return {'plots': results, 'parameters': param_results}




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
    
    # Initialize calibration and errors dictionaries
    calibration = {}
    calibration_errors = {}
    
    # Step 1: Generate efficiency
    rle.logger.log("\nStep 1: Generating origin momentum fit...", "info")
    rle.generate_efficiency(combined_data, output_dir)
    chebyshev_params = rle.fit_origin_momentum_chebyshev(combined_data, output_dir)
    if chebyshev_params:
        calibration['chebyshev'] = {
            'coeffs': [1.0, chebyshev_params['c1']['value'], chebyshev_params['c2']['value'], 
                       chebyshev_params['c3']['value'], chebyshev_params['c4']['value'], chebyshev_params['c5']['value']]
        }
        calibration_errors['chebyshev'] = {
            'coeffs': [0.0, chebyshev_params['c1']['error'], chebyshev_params['c2']['error'], 
                       chebyshev_params['c3']['error'], chebyshev_params['c4']['error'], chebyshev_params['c5']['error']]
        }
    
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
            
            # Extract fitted parameters from fit results and build calibration dict
            if fit_results and 'parameters' in fit_results:
                params_by_type = fit_results['parameters']
                
                # Process loss fit results - use ENTRANCE plane only
                if 'loss' in params_by_type and 'entrance' in params_by_type['loss']:
                    print(f"\n{'='*70}", flush=True)
                    print(f"UPDATING CALIBRATION.JSON - LOSS PARAMETERS (ENTRANCE PLANE)", flush=True)
                    print(f"{'='*70}", flush=True)
                    entrance_loss_params = params_by_type['loss']['entrance']
                    calibration['landau'] = {}
                    calibration_errors['landau'] = {}
                    for param_name in ['loc', 'scale']:
                        if param_name in entrance_loss_params:
                            param_dict = entrance_loss_params[param_name]
                            val = param_dict['value']
                            err = param_dict['error']
                            calibration['landau'][param_name] = val
                            calibration_errors['landau'][param_name] = err
                            if err is not None:
                                print(f"  {param_name:12s}: {val:+.6f} ± {err:.6f}", flush=True)
                            else:
                                print(f"  {param_name:12s}: {val:+.6f}", flush=True)
                    print()
                
                # Process resolution fit results - use ENTRANCE plane only
                if 'res' in params_by_type and 'entrance' in params_by_type['res']:
                    print(f"{'='*70}", flush=True)
                    print(f"UPDATING CALIBRATION.JSON - RESOLUTION PARAMETERS (ENTRANCE PLANE)", flush=True)
                    print(f"{'='*70}", flush=True)
                    entrance_res_params = params_by_type['res']['entrance']
                    calibration['gcb'] = {}
                    calibration_errors['gcb'] = {}
                    for param_name in ['mu', 'sigmaL', 'sigmaR', 'alphaL', 'alphaR', 'nL', 'nR']:
                        if param_name in entrance_res_params:
                            param_dict = entrance_res_params[param_name]
                            val = param_dict['value']
                            err = param_dict['error']
                            calibration['gcb'][param_name] = val
                            calibration_errors['gcb'][param_name] = err
                            if err is not None:
                                print(f"  {param_name:12s}: {val:+.6f} ± {err:.6f}", flush=True)
                            else:
                                print(f"  {param_name:12s}: {val:+.6f}", flush=True)
                    print()
                
        except Exception as e:
            rle.logger.log(f"Fitting failed: {e}", "warning")
            print(f"Fitting failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
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
    
    # Export calibration parameters to JSON
    if calibration:
        json_path = f"{output_dir}/calibration.json"
        with open(json_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        rle.logger.log(f"  - calibration.json (fitted parameters)", "info")
        print(f"\n✓ Updated calibration.json with fitted parameters", flush=True)
    
    # Export calibration errors to JSON
    if calibration_errors:
        errors_path = f"{output_dir}/calibration_errors.json"
        with open(errors_path, 'w') as f:
            json.dump(calibration_errors, f, indent=2)
        rle.logger.log(f"  - calibration_errors.json (fit uncertainties)", "info")
        print(f"✓ Saved calibration_errors.json with fit uncertainties", flush=True)
    
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


# ============================================================================
# RLE Overlay and Fitting Functions (moved from process.py)
# ============================================================================

def plot_theory_with_rle(files, cuts, locations, signs, jobs=1, rle_calib_dir="RLE/common", 
                        mom_range=(95, 110), binwidth=0.1, output_file=None):
    """
    Plot reconstructed momentum data overlaid with theory convolved with RLE.
    
    Creates a theory spectrum from CeLL, loads resolution and loss distributions from
    RLE calibration, convolves them, and overlays on reco data histogram.
    
    Args:
        files: List of file list paths (.txt files)
        cuts: List of cut switches for each file
        locations: List of data locations (e.g., 'disk')
        signs: List of particle signs (e.g., 'minus', 'plus')
        jobs: Number of parallel jobs (default: 1)
        rle_calib_dir (str): Path to RLE calibration output directory
        mom_range (tuple): (min, max) momentum range for plotting
        binwidth (float): Bin width for theory spectrum
        output_file (str): Optional path to save plot
        
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    from pyutils.pylogger import Logger
    from helper import AnaProcessor, TheorySpectrum, Compare
    import pickle as pkl
    
    logger = Logger(print_prefix="[plot_theory_with_rle]", verbosity=1)
    
    try:
        # Extract reconstructed momentum from files using standard pipeline
        recomom = []
        for i, fil in enumerate(files):
            ana_processor = AnaProcessor(fil, jobs, signs[i], cuts[i], locations[i], "ensemble")
            results = ana_processor.execute()
            combine_result = results["combined_data"]
            
            selector = Select()
            
            # select only track front to fit to
            trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")
            
            # did the track intersect the ST?
            has_st = selector.has_ST(combine_result['trkfit'])
            
            # combined mask
            trkfit_ent = ak.mask(combine_result['trkfit']["trksegs"], trk_front)
            
            # make vector mag branch
            vector = Vector()
            mom_mag = vector.get_mag(trkfit_ent, 'mom')
            recomom.append(mom_mag)
        
        # Flatten reco data if list of arrays
        if isinstance(recomom, list):
            reco_flat = ak.flatten(ak.concatenate(recomom), axis=None)
        else:
            reco_flat = ak.flatten(recomom, axis=None)
        reco_np = np.array(reco_flat)
        logger.log(f"Loaded {len(reco_np)} reco events", "info")
        
        # Create theory spectrum
        logger.log("Creating theory spectrum...", "info")
        theory = TheorySpectrum(mom_range=mom_range, binwidth=binwidth, verbosity=1)
        theory_pdf = theory.get_pdf()
        
        # Create observable space for momentum (theory will be on this)
        obs_mom = zfit.Space('mom', limits=mom_range)
        
        # Load RLE calibration data
        logger.log("Loading RLE calibration...", "info")
        
        # Try to load skimmed data to get res and loss distributions
        skimmed_path = f"{rle_calib_dir}/skimmed_flat_mom_MDC2025an.pkl"
        try:
            with open(skimmed_path, 'rb') as f:
                skimmed_data = pkl.load(f)
            logger.log(f"Loaded skimmed data from {skimmed_path}", "debug")
            
            # Extract resolution and loss from entrance plane
            # IMPORTANT: These are NOT Gaussian! Resolution is GCB, Loss is truncated Landau
            res_data = skimmed_data['entrance']['reco'] - skimmed_data['entrance']['mc']
            loss_data = skimmed_data['entrance']['mc'] - skimmed_data['entrance']['gen']
            
            logger.log(f"Resolution distribution: {len(res_data)} events, mean={np.mean(res_data):.4f}, std={np.std(res_data):.4f}", "debug")
            logger.log(f"Loss distribution: {len(loss_data)} events, mean={np.mean(loss_data):.4f}, std={np.std(loss_data):.4f}", "debug")
            
            # Create histogram PDFs from actual distributions to preserve non-Gaussian shapes
            # CRITICAL: Create histograms on the ACTUAL observable bounds we'll use for convolution!
            logger.log("Creating histogram kernel PDFs from actual res/loss distributions...", "info")
            
            # Resolution kernel: trim to tighter percentile range to exclude tail
            # Use 10th-90th percentile instead of 1st-99th to focus on core distribution
            res_trimmed = res_data[(res_data > np.percentile(res_data, 10)) & 
                                  (res_data < np.percentile(res_data, 90))]
            
            # Use ±1.5σ bounds, symmetric around mean
            res_mean = np.mean(res_trimmed)
            res_std = np.std(res_trimmed)
            res_min_physical = max(res_mean - 1.5*res_std, -5.0)   # Cap at -5 MeV
            res_max_physical = min(res_mean + 1.5*res_std, 5.0)    # Cap at +5 MeV
            res_trimmed = res_trimmed[(res_trimmed >= res_min_physical) & 
                                      (res_trimmed <= res_max_physical)]
            
            res_nbins = 100
            res_counts, res_edges = np.histogram(res_trimmed, bins=res_nbins, 
                                                  range=(res_min_physical, res_max_physical))
            res_counts = res_counts / np.sum(res_counts)  # Normalize
            
            # Create histogram PDF for resolution on ACTUAL kernel observable space [-5, 5]
            from helper import make_HistogramPDF
            obs_res_kernel = zfit.Space('x', limits=(res_min_physical, res_max_physical))
            ResHistPDF = make_HistogramPDF(res_counts, res_edges)
            res_pdf = ResHistPDF(obs=obs_res_kernel)  # On [-5, 5] kernel space!
            
            logger.log(f"Resolution kernel: histogram from {len(res_trimmed)} events, range=[{res_min_physical:.4f}, {res_max_physical:.4f}]", "info")
            
            # Loss kernel: trim to tighter percentile range to exclude tail
            # Use 10th-90th percentile instead of 1st-99th to focus on core distribution
            loss_trimmed = loss_data[(loss_data > np.percentile(loss_data, 10)) & 
                                     (loss_data < np.percentile(loss_data, 90))]
            
            # Calculate loss mean and use tighter bounds around it (1.5σ)
            loss_mean = np.mean(loss_trimmed)
            loss_std = np.std(loss_trimmed)
            # Use ±1.5σ bounds for tighter kernel, symmetric around actual mean
            loss_min_physical = max(loss_mean - 1.5*loss_std, -15.0)   # Cap at -15 MeV
            loss_max_physical = min(loss_mean + 1.5*loss_std, 1.0)     # Cap at +1 MeV
            loss_trimmed = loss_trimmed[(loss_trimmed >= loss_min_physical) & 
                                        (loss_trimmed <= loss_max_physical)]
            
            loss_nbins = 100
            loss_counts, loss_edges = np.histogram(loss_trimmed, bins=loss_nbins,
                                                    range=(loss_min_physical, loss_max_physical))
            loss_counts = loss_counts / np.sum(loss_counts)  # Normalize
            
            # Create histogram PDF for loss on bounds centered at actual mean
            obs_loss_kernel = zfit.Space('x', limits=(loss_min_physical, loss_max_physical))
            LossHistPDF = make_HistogramPDF(loss_counts, loss_edges)
            loss_pdf = LossHistPDF(obs=obs_loss_kernel)  # Centered on actual distribution!
            
            logger.log(f"Loss kernel: histogram from {len(loss_trimmed)} events, range=[{loss_min_physical:.4f}, {loss_max_physical:.4f}]", "info")
            
        except FileNotFoundError:
            logger.log(f"Warning: Could not find {skimmed_path}", "warn")
            logger.log("Using default Gaussian PDFs as fallback", "warn")
            
            # Fallback: Use simple Gaussians
            res_pdf = zfit.pdf.Gauss(
                mu=zfit.Parameter('res_mu', 0.0, -0.5, 0.5),
                sigma=zfit.Parameter('res_sigma', 0.3, 0.01, 1.0),
                obs=obs_mom
            )
            loss_pdf = zfit.pdf.Gauss(
                mu=zfit.Parameter('loss_mu', -0.5, -2.0, 0.0),
                sigma=zfit.Parameter('loss_sigma', 0.5, 0.01, 2.0),
                obs=obs_mom
            )
        
        # Create comparison and plot
        comparison = Compare()
        title = "Reco Momentum with Theory ⊗ RLE Convolution"
        label = "CeLL (Leading Log) ⊗ RLE"
        
        if output_file is None:
            output_file = f"{rle_calib_dir}/theory_convolved_reco.png"
        
        fig, ax = comparison.convolve_with_rle(
            reco_data=reco_np,
            theory_pdf=theory_pdf,
            res_pdf=res_pdf,
            loss_pdf=loss_pdf,
            mom_range=mom_range,
            nbins=100,
            label=label,
            plot_title=title,
            output_file=output_file
        )
        
        logger.log(f"Theory convolution plot saved to {output_file}", "success")
        return fig, ax
        
    except Exception as e:
        logger.log(f"Error in theory convolution: {e}", "error")
        traceback.print_exc()
        return None, None


def apply_ce_rle_convolution(calibration_path="RLE/common/calibration.json", 
                             mom_range=(95, 110), binwidth=0.1, output_plot=None):
    """
    Apply RLE convolution to CeLL theory spectrum
    
    Uses the calibrated Chebyshev efficiency, GCB resolution, and Landau loss
    to convolve the CeLL leading-log spectrum.
    
    Args:
        calibration_path (str): Path to calibration.json from RLE
        mom_range (tuple): (min, max) momentum range
        binwidth (float): Bin width for momentum grid
        output_plot (str): Optional path to save convolution steps plot
        
    Returns:
        dict: {
            'x_grid': momentum grid,
            'theory': CeLL spectrum,
            'after_loss': after loss convolution,
            'after_resolution': after resolution convolution,
            'final': after efficiency scaling,
            'efficiency': efficiency values
        }
    """
    from spectrum import TheorySpectrum
    from RLE.rle_functions import apply_rle_convolution
    
    logger = Logger(print_prefix="[apply_ce_rle_convolution]", verbosity=1)
    
    try:
        logger.log("Creating CeLL theory spectrum...", "info")
        theory = TheorySpectrum(mom_range=mom_range, binwidth=binwidth, verbosity=0)
        
        # Create momentum grid
        x_grid = np.arange(mom_range[0], mom_range[1], binwidth)
        
        # Get theory PDF and evaluate on grid
        theory_pdf = theory.get_pdf()
        obs_mom = zfit.Space('p', limits=mom_range)
        theory_vals = zfit.run(theory_pdf.pdf(x_grid.reshape(-1, 1))).flatten()
        
        logger.log(f"Theory spectrum: {len(x_grid)} points from {mom_range[0]} to {mom_range[1]} MeV", "info")
        logger.log(f"Loading calibration from {calibration_path}...", "info")
        
        # Apply full RLE convolution
        result = apply_rle_convolution(x_grid, theory_vals, calibration_path)
        
        logger.log("RLE convolution complete:", "info")
        logger.log(f"  Theory integral: {np.trapz(result['theory'], x_grid):.4f}", "info")
        logger.log(f"  After loss integral: {np.trapz(result['after_loss'], x_grid):.4f}", "info")
        logger.log(f"  After resolution integral: {np.trapz(result['after_resolution'], x_grid):.4f}", "info")
        logger.log(f"  Final (× efficiency) integral: {np.trapz(result['final'], x_grid):.4f}", "info")
        logger.log(f"  Efficiency range: [{np.min(result['efficiency']):.3f}, {np.max(result['efficiency']):.3f}]", "info")
        
        # Optionally create plot
        if output_plot:
            logger.log(f"Creating convolution plot at {output_plot}...", "info")
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Step 1: Theory
            axes[0, 0].plot(result['x_grid'], result['theory'], 'b-', linewidth=2)
            axes[0, 0].set_ylabel('PDF')
            axes[0, 0].set_title('Step 1: CeLL Theory Spectrum')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Step 2: After loss
            axes[0, 1].plot(result['x_grid'], result['after_loss'], 'g-', linewidth=2)
            axes[0, 1].set_ylabel('PDF')
            axes[0, 1].set_title('Step 2: Theory ⊗ Landau Loss')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Step 3: After resolution
            axes[1, 0].plot(result['x_grid'], result['after_resolution'], 'r-', linewidth=2)
            axes[1, 0].set_ylabel('PDF')
            axes[1, 0].set_xlabel('Momentum [MeV]')
            axes[1, 0].set_title('Step 3: (Theory ⊗ Loss) ⊗ GCB Resolution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Step 4: Final with efficiency
            ax_final = axes[1, 1]
            ax_final.plot(result['x_grid'], result['final'], 'purple', linewidth=2, label='Final spectrum')
            ax_final_eff = ax_final.twinx()
            ax_final_eff.plot(result['x_grid'], result['efficiency'], 'orange', linewidth=1.5, linestyle='--', label='Efficiency')
            ax_final.set_ylabel('Final PDF (purple)', color='purple')
            ax_final_eff.set_ylabel('Efficiency (orange)', color='orange')
            ax_final.set_xlabel('Momentum [MeV]')
            ax_final.set_title('Step 4: Apply Chebyshev Efficiency')
            ax_final.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_plot, dpi=150, bbox_inches='tight')
            logger.log(f"Plot saved to {output_plot}", "success")
            plt.close()
        
        return result
        
    except Exception as e:
        logger.log(f"Error in RLE convolution: {e}", "error")
        traceback.print_exc()
        return None


def fit_convolved_spectrum_to_data(reco_momenta, spectrum_grid, spectrum_vals, mom_range=(95, 110), 
                                   binwidth=0.1, output_plot=None, rle_models=None, 
                                   theory_spectrum_func=None, calibration_path=None):
    """
    Fit a convolved spectrum to reconstructed data using zfit.
    
    Creates a histogram PDF with floating scale parameter and optionally floating RLE parameters.
    
    Args:
        reco_momenta (array): Reconstructed momentum values
        spectrum_grid (array): Momentum grid for the spectrum
        spectrum_vals (array): PDF values on the grid (density)
        mom_range (tuple): (min, max) momentum range
        binwidth (float): Bin width for histogram
        output_plot (str): Optional path to save diagnostic plot
        rle_models (dict): Optional dict with 'loss', 'resolution', 'efficiency' zfit models for joint RLE+scale fit
        theory_spectrum_func (callable): Optional function to recompute spectrum with different RLE params
        calibration_path (str): Optional path to calibration for accessing theory spectrum
        
    Returns:
        dict: {
            'fit_result': zfit.result.FitResult,
            'nll': negative log-likelihood value,
            'scale_factor': fitted normalization scale,
            'chi2_per_dof': approximate chi2 / dof,
            'n_dof': degrees of freedom,
            'n_events': number of events in reco,
            'success': True if fit converged,
            'rle_params': dict of fitted RLE parameters (if rle_models provided)
        }
    """
    logger = Logger(print_prefix="[fit_spectrum]", verbosity=1)
    
    try:
        # Filter reco to mom_range
        reco_filtered = reco_momenta[(reco_momenta >= mom_range[0]) & (reco_momenta <= mom_range[1])]
        n_events = len(reco_filtered)
        
        logger.log(f"Fitting spectrum to {n_events} reco events using zfit", "info")
        if rle_models:
            logger.log("  Including RLE parameters as floating in the fit", "info")
        else:
            logger.log("  Using pre-computed spectrum (RLE parameters fixed)", "info")
        
        # Create bin edges
        bin_edges = np.arange(mom_range[0], mom_range[1] + binwidth, binwidth)
        
        # Interpolate spectrum to bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        spectrum_at_centers = np.interp(bin_centers, spectrum_grid, spectrum_vals, left=0, right=0)
        
        # Normalize to probability
        spectrum_prob = spectrum_at_centers / np.sum(spectrum_at_centers) if np.sum(spectrum_at_centers) > 0 else spectrum_at_centers
        
        # Create zfit Space for fitting (momentum)
        obs = zfit.Space('p', limits=mom_range)
        
        # If RLE models provided, they're on 'x' space - need to create separate obs_x for convolution
        if rle_models:
            obs_x = zfit.Space('x', limits=mom_range)
        
        # Create a floating scale parameter
        scale_param = zfit.Parameter('spectrum_scale', 1.0, 0.1, 10.0)
        
        # Collect all floating parameters for the fit
        all_floating_params = [scale_param]
        
        # If RLE models provided, make their parameters floating
        rle_floating_params = []
        if rle_models:
            # Collect floating parameters from RLE models
            for model_key in ['loss', 'resolution']:
                if model_key in rle_models:
                    model = rle_models[model_key]
                    for param in model.get_params():
                        if param.floating:
                            all_floating_params.append(param)
                            rle_floating_params.append((model_key, param))
            
            logger.log(f"Floating RLE parameters in fit: {len(rle_floating_params)}", "info")
            for model_key, param in rle_floating_params:
                param_val = zfit.run(param)
                lower = param.lower if param.lower is not None else "unbounded"
                upper = param.upper if param.upper is not None else "unbounded"
                logger.log(f"  {param.name}: {param_val:.6f} (bounds: [{lower}, {upper}])", "debug")
        
        # Create a PDF with RLE convolution applied dynamically during fit
        logger.log("Creating PDF with dynamic RLE convolution...", "info")
        import tensorflow as tf
        from RLE.rle_functions import (
            convolve_numerical, evaluate_pdf, 
            get_loss_model, get_resolution_model, get_efficiency_model,
            load_calibration
        )
        
        if rle_models:
            # Use FFTConvPDFV1 with shared observable space for theory ⊗ loss ⊗ resolution
            logger.log("Creating FFTConvPDFV1-based convolution with floating RLE parameters", "info")
            
            # Create theory histogram PDF on the shared 'x' space (same as RLE models)
            spec_prob_norm = spectrum_prob / np.sum(spectrum_prob)
            prob_tensor_theory = tf.reshape(tf.constant(spec_prob_norm, dtype=tf.float64), [len(spec_prob_norm), 1])
            low_tensor_theory = tf.reshape(tf.constant(bin_edges[:-1], dtype=tf.float64), [len(spec_prob_norm), 1])
            high_tensor_theory = tf.reshape(tf.constant(bin_edges[1:], dtype=tf.float64), [len(spec_prob_norm), 1])
            
            def _theory_unnormalized(self, x):
                x_val = zfit.z.unstack_x(x)
                within_bounds = (x_val >= low_tensor_theory) & (x_val < high_tensor_theory)
                return tf.reduce_sum(tf.where(within_bounds, prob_tensor_theory, 0), axis=0)
            
            class_attrs_theory = {
                '_N_OBS': 1,
                '_PARAMS': [],
                '_unnormalized_pdf': _theory_unnormalized,
            }
            TheoryHistogramPDF = type('TheoryHistogramPDF', (zfit.pdf.ZPDF,), class_attrs_theory)
            theory_pdf_base = TheoryHistogramPDF(obs=obs_x, name='theory_base')
            
            # Get the RLE model PDFs from the dict (already on 'x' space with floating params)
            loss_pdf = rle_models['loss']
            res_pdf = rle_models['resolution']
            eff_pdf = rle_models['efficiency']
            
            # Create convolutions using FFTConvPDFV1
            from zfit.pdf import FFTConvPDFV1
            
            logger.log("Creating FFTConvPDFV1 for theory ⊗ loss convolution", "debug")
            conv_loss = FFTConvPDFV1(func=theory_pdf_base, kernel=loss_pdf, obs=obs_x, name='theory_loss_conv')
            
            logger.log("Creating FFTConvPDFV1 for (theory ⊗ loss) ⊗ resolution convolution", "debug")
            conv_loss_res = FFTConvPDFV1(func=conv_loss, kernel=res_pdf, obs=obs_x, name='theory_loss_res_conv')
            
            # Create a custom PDF that wraps the convolution result and applies the scale parameter
            def _scaled_convolved_unnormalized(self, x):
                scale = self.params[scale_param.name]
                # Evaluate the convolved PDF and scale it
                conv_val = conv_loss_res.pdf(x)
                return scale * conv_val
            
            class_attrs_scaled = {
                '_N_OBS': 1,
                '_PARAMS': [scale_param.name] + [p.name for _, p in rle_floating_params],
                '_unnormalized_pdf': _scaled_convolved_unnormalized,
            }
            
            ScaledConvolvedPDF = type('ScaledConvolvedPDF', (zfit.pdf.ZPDF,), class_attrs_scaled)
            theory_pdf = ScaledConvolvedPDF(
                obs=obs_x,
                spectrum_scale=scale_param,
                **{p.name: p for _, p in rle_floating_params},
                name='theory_scaled_convolved'
            )
            
            # Collect all floating parameters from RLE models
            param_names = [scale_param.name] + [p.name for _, p in rle_floating_params]
            all_floating_params = [scale_param] + [p for _, p in rle_floating_params]
            
        else:
            # No RLE models - use simple histogram PDF with just scale
            all_floating_params = [scale_param]
            
            prob_tensor = tf.reshape(tf.constant(spectrum_prob, dtype=tf.float64), [len(spectrum_prob), 1])
            low_tensor = tf.reshape(tf.constant(bin_edges[:-1], dtype=tf.float64), [len(spectrum_prob), 1])
            high_tensor = tf.reshape(tf.constant(bin_edges[1:], dtype=tf.float64), [len(spectrum_prob), 1])
            param_names = [scale_param.name]
            
            def _unnormalized_pdf_func(self, x):
                x_val = zfit.z.unstack_x(x)
                within_bounds = (x_val >= low_tensor) & (x_val < high_tensor)
                prob = tf.reduce_sum(tf.where(within_bounds, prob_tensor, 0), axis=0)
                scale = self.params[param_names[0]]
                return prob * scale
            
            class_attrs = {
                '_N_OBS': 1,
                '_PARAMS': param_names,
                '_unnormalized_pdf': _unnormalized_pdf_func,
            }
            
            ScaledHistogramPDF = type('ScaledHistogramPDF', (zfit.pdf.ZPDF,), class_attrs)
            theory_pdf = ScaledHistogramPDF(obs=obs, **{p.name: p for p in all_floating_params}, name='theory_spectrum')
        
        # Convert reco data to zfit dataset
        logger.log("Creating zfit dataset from reco data...", "info")
        # Use obs_x if RLE models provided (theory_pdf is on obs_x), otherwise use obs (theory_pdf is on obs)
        data_obs = obs_x if rle_models else obs
        reco_data = zfit.Data.from_numpy(obs=data_obs, array=reco_filtered.reshape(-1, 1))
        
        # Create unbinned likelihood
        logger.log("Setting up unbinned likelihood...", "info")
        loss = zfit.loss.UnbinnedNLL(model=theory_pdf, data=reco_data)
        
        # Minimize
        logger.log("Running zfit minimization over {} parameters...".format(len(all_floating_params)), "info")
        minimizer = zfit.minimize.Minuit(verbosity=0)
        fit_result = minimizer.minimize(loss)
        
        nll = fit_result.fmin
        success = fit_result.valid
        scale_fitted = zfit.run(scale_param)
        
        # Compute Hesse errors
        fit_result.hesse()
        
        # Estimate chi2/dof
        n_dof = max(len(reco_filtered) - len(all_floating_params), 1)
        chi2_per_dof = (2 * nll) / max(n_dof, 1)
        
        # Print the fit result directly
        logger.log("\n" + "="*80, "info")
        logger.log("DETAILED FIT RESULT", "info")
        logger.log("="*80, "info")
        print(fit_result)
        logger.log("="*80, "info")
        
        # Print detailed RLE parameter results if available
        if rle_floating_params:
            logger.log("\n" + "="*80, "info")
            logger.log("RLE FLOATING PARAMETERS - FIT RESULTS", "info")
            logger.log("="*80, "info")
            print("\nScale Parameter:")
            print(f"  spectrum_scale: {scale_fitted:.6f}")
            
            hesse_result = fit_result.hesse()
            if hesse_result:
                scale_err = hesse_result.get(scale_param.name, None)
                if scale_err is not None:
                    print(f"    Hesse error: ±{scale_err:.6f}")
            
            print("\nLoss Parameters (Landau):")
            for label, param in rle_floating_params:
                if 'landau' in label.lower() or 'loss' in label.lower():
                    param_val = zfit.run(param)
                    print(f"  {param.name}: {param_val:.6f}")
                    if hesse_result and param.name in hesse_result:
                        print(f"    Hesse error: ±{hesse_result[param.name]:.6f}")
                    print(f"    bounds: [{param.lower:.6f}, {param.upper:.6f}]")
                    print(f"    at_limit: {param_val >= param.upper - 1e-5 or param_val <= param.lower + 1e-5}")
            
            print("\nResolution Parameters (GCB):")
            for label, param in rle_floating_params:
                if 'resolution' in label.lower() or 'res' in label.lower() or 'gcb' in label.lower():
                    param_val = zfit.run(param)
                    print(f"  {param.name}: {param_val:.6f}")
                    if hesse_result and param.name in hesse_result:
                        print(f"    Hesse error: ±{hesse_result[param.name]:.6f}")
                    print(f"    bounds: [{param.lower:.6f}, {param.upper:.6f}]")
                    print(f"    at_limit: {param_val >= param.upper - 1e-5 or param_val <= param.lower + 1e-5}")
            logger.log("="*80, "info\n")
        
        # Collect RLE parameter results
        rle_results = {}
        if rle_floating_params:
            rle_results = {
                param.name: {
                    'value': zfit.run(param),
                    'error': fit_result.hesse()[param.name] if fit_result.hesse() and param.name in fit_result.hesse() else None
                }
                for _, param in rle_floating_params
            }
        
        # Create diagnostic plot if requested
        if output_plot:
            logger.log(f"Creating diagnostic plot at {output_plot}...", "info")
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Reco histogram with fitted spectrum overlay
            ax = axes[0]
            reco_counts, edges, _ = ax.hist(reco_filtered, bins=bin_edges, label=f'Reco Data ({n_events} events)',
                                             color='orange', alpha=0.7, edgecolor='none')
            
            # Overlay fitted spectrum (scaled) - histogram shows counts, so multiply normalized spectrum by n_events and scale
            fitted_spectrum = n_events * spectrum_prob * scale_fitted
            ax.plot(bin_centers, fitted_spectrum, 'r-', linewidth=2.5, label=f'Fitted Spectrum (scale={scale_fitted:.4f})')
            
            ax.set_xlabel('Momentum [MeV]', fontsize=12)
            ax.set_ylabel('Events / {:.2f} MeV'.format(binwidth), fontsize=12)
            ax.set_title('Reco Data vs Fitted Spectrum', fontsize=13, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Pull (residuals)
            ax = axes[1]
            pull = []
            pull_err = []
            for i, count in enumerate(reco_counts):
                expected = fitted_spectrum[i]
                error = np.sqrt(max(count, 1))  # Poisson error
                if error > 0:
                    pull.append((count - expected) / error)
                    pull_err.append(1.0)
                else:
                    pull.append(0)
                    pull_err.append(0)
            
            ax.errorbar(bin_centers, pull, yerr=pull_err, fmt='o-', color='darkblue', 
                       markersize=6, linewidth=1.5, label='Pull = (Data - Fit) / σ')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(y=-1, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_xlabel('Momentum [MeV]', fontsize=12)
            ax.set_ylabel('Pull', fontsize=12)
            ax.set_title('Residuals', fontsize=13, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_plot, dpi=150, bbox_inches='tight')
            logger.log(f"Plot saved to {output_plot}", "success")
            plt.close()
        
        return {
            'fit_result': fit_result,
            'nll': nll,
            'scale_factor': scale_fitted,
            'chi2_per_dof': chi2_per_dof,
            'n_dof': n_dof,
            'n_events': n_events,
            'success': success,
            'rle_params': rle_results
        }
        
    except Exception as e:
        logger.log(f"Fit failed: {e}", "error")
        traceback.print_exc()
        return None


def overlay_convolved_theory_on_reco_with_constraints(reco_momenta, calibration_path="RLE/common/calibration.json",
                                                      constraint_margin=0.15, mom_range=(95, 110), 
                                                      binwidth=0.1, output_plot=None, do_fit=False):
    """
    Overlay normalized convolved theory with floating constrained RLE parameters
    
    Generates floating parameter constraints from calibration.json with specified
    margin, applies RLE convolution, and overlays on reco histogram. Optionally fits
    the convolved spectrum to the reconstructed data.
    
    Args:
        reco_momenta (array): Reconstructed momentum values from data
        calibration_path (str): Path to calibration.json from RLE
        constraint_margin (float): Relative margin for constraints (default 0.15 = ±15%)
        mom_range (tuple): (min, max) momentum range
        binwidth (float): Bin width for momentum grid and histogram
        output_plot (str): Path to save overlay plot
        do_fit (bool): If True, fit the convolved spectrum to reco data using zfit
        
    Returns:
        dict: {
            'reco_momenta': filtered reco data,
            'convolution_result': result from apply_rle_convolution_with_constraints,
            'n_events': number of events in reco,
            'constraints_used': generated constraints,
            'constraint_margin': the margin used,
            'fit_result': zfit result dict (None if do_fit=False)
        }
    """
    logger = Logger(print_prefix="[overlay_with_constraints]", verbosity=1)
    
    try:
        # Print calibration summary first
        from RLE.rle_functions import print_calibration_summary
        print_calibration_summary(calibration_path)
        
        # Filter reco data to mom_range
        reco_filtered = reco_momenta[(reco_momenta >= mom_range[0]) & (reco_momenta <= mom_range[1])]
        n_events = len(reco_filtered)
        logger.log(f"Filtered reco data: {n_events} events in [{mom_range[0]}, {mom_range[1]}] MeV", "info")
        
        # Load calibration and define constraints
        logger.log(f"Generating floating parameter constraints from fit errors...", "info")
        with open(calibration_path, 'r') as f:
            calib = json.load(f)
        
        # Load calibration errors if available
        calibration_errors_path = calibration_path.replace('calibration.json', 'calibration_errors.json')
        calib_errors = None
        if os.path.exists(calibration_errors_path):
            try:
                with open(calibration_errors_path, 'r') as f:
                    calib_errors = json.load(f)
                logger.log(f"Loaded calibration errors from {calibration_errors_path}", "info")
            except Exception as e:
                logger.log(f"Warning: Could not load calibration errors - {e}", "warn")
        
        from RLE.rle_functions import define_loss_constraints, define_resolution_constraints
        loss_constraints = define_loss_constraints(calib['landau'], relative_margin=constraint_margin, 
                                                    error_params=calib_errors.get('landau') if calib_errors else None,
                                                    n_sigma=10)
        res_constraints = define_resolution_constraints(calib['gcb'], relative_margin=constraint_margin,
                                                        error_params=calib_errors.get('gcb') if calib_errors else None,
                                                        n_sigma=10)
        
        logger.log("Constraints generated successfully", "success")
        logger.log(f"  Loss: loc ∈ [{loss_constraints['loc'][0]:.6f}, {loss_constraints['loc'][1]:.6f}]", "debug")
        logger.log(f"  Loss: scale ∈ [{loss_constraints['scale'][0]:.6f}, {loss_constraints['scale'][1]:.6f}]", "debug")
        
        # Get convolved theory with constraints
        logger.log("Generating convolved theory spectrum with constrained parameters...", "info")
        from RLE.rle_functions import apply_rle_convolution_with_constraints
        from spectrum import TheorySpectrum
        
        # Create theory spectrum
        theory = TheorySpectrum(mom_range=mom_range, binwidth=binwidth, verbosity=0)
        x_grid = np.arange(mom_range[0], mom_range[1], binwidth)
        
        # Get theory PDF and evaluate on grid
        theory_pdf = theory.get_pdf()
        obs_mom = zfit.Space('p', limits=mom_range)
        theory_vals = zfit.run(theory_pdf.pdf(x_grid.reshape(-1, 1))).flatten()
        
        conv_result = apply_rle_convolution_with_constraints(
            x_grid, theory_vals, calibration_path,
            loss_constraints=loss_constraints,
            res_constraints=res_constraints
        )
        
        if conv_result is None:
            logger.log("Failed to generate convolution result", "error")
            return None
        
        x_grid = conv_result['x_grid']
        after_resolution = conv_result['after_resolution']
        
        # Debug: Compare convolution normalization with actual reco data
        integral_convolved = np.trapz(after_resolution, x_grid)
        eff = conv_result['efficiency']
        print(f"\n{'='*70}", flush=True)
        print(f"NORMALIZATION COMPARISON", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Actual reco events in [{mom_range[0]}, {mom_range[1]}]: {n_events}", flush=True)
        print(f"Convolved spectrum integral: {integral_convolved:.4f}", flush=True)
        print(f"Ratio (integral / reco_events): {integral_convolved / n_events:.4f}", flush=True)
        print(f"Efficiency max: {np.max(eff):.4f}, mean: {np.mean(eff):.4f}", flush=True)
        print(f"Efficiency sum: {np.sum(eff):.4f}", flush=True)
        print(f"{'='*70}", flush=True)
        
        # Normalize spectrum to integral=1 for the fit
        integral_before_eff = np.trapz(after_resolution, x_grid)
        if integral_before_eff > 0:
            theory_normalized_to_1 = after_resolution / integral_before_eff  # Normalize to integral=1 for fit
            theory_normalized_to_n = after_resolution * (n_events / integral_before_eff)  # Normalize to n_events for plot
        else:
            logger.log("Warning: after-resolution spectrum integral is 0 or negative", "warning")
            theory_normalized_to_1 = after_resolution
            theory_normalized_to_n = after_resolution
        # Debug: Integral of theory for plot (no binwidth, already normalized to n_events)
        integral_for_plot = np.trapz(theory_normalized_to_n, x_grid)
        print(f"Integral of theory for plot: {integral_for_plot:.4f}", flush=True)
        print(f"Expected total events (reco): {n_events}", flush=True)
        print(f"Ratio (theory_plot / reco): {integral_for_plot / n_events:.4f}", flush=True)
        
        logger.log(f"=== CONSTRAINED CONVOLUTION ===", "info")
        logger.log(f"Spectrum normalized to integral=1 (for fit) and to {n_events} events (for plot)", "info")
        
        # Create overlay plot
        if output_plot:
            logger.log(f"Creating overlay plot at {output_plot}...", "info")
            fig, ax = plt.subplots(figsize=(11, 8))
            
            # Reco histogram
            n_bins = int((mom_range[1] - mom_range[0]) / binwidth)
            counts, edges, patches = ax.hist(reco_filtered, bins=n_bins, range=mom_range,
                                              label=f'Reco Data ({n_events} events)',
                                              alpha=0.7, color='orange', edgecolor='none')
            
            print(f"\nReco histogram: {np.sum(counts)} events in {n_bins} bins", flush=True)
            print(f"After-resolution spectrum integral: {integral_before_eff:.1f}", flush=True)
            print(f"Scaling factor (reco_events / integral): {n_events / integral_before_eff:.4f}", flush=True)
            
            # Debug: Peak alignment
            bin_centers_reco = (edges[:-1] + edges[1:]) / 2
            peak_idx_reco = np.argmax(counts)
            peak_momentum_reco = bin_centers_reco[peak_idx_reco]
            peak_idx_theory = np.argmax(theory_normalized_to_n)
            peak_momentum_theory = x_grid[peak_idx_theory]
            momentum_shift = peak_momentum_theory - peak_momentum_reco
            print(f"\nPEAK ALIGNMENT:", flush=True)
            print(f"Reco peak momentum: {peak_momentum_reco:.4f} MeV (bin {peak_idx_reco})", flush=True)
            print(f"Theory peak momentum: {peak_momentum_theory:.4f} MeV (bin {peak_idx_theory})", flush=True)
            print(f"Momentum shift (theory - reco): {momentum_shift:.4f} MeV", flush=True)
            
            # Convolved theory overlay (already normalized to n_events, no binwidth multiplication needed)
            ax.plot(x_grid, theory_normalized_to_n, 'r-', linewidth=2.5,
                   label='CeLL ⊗ RLE (Floating Params)')
            
            # Also show efficiency as secondary axis
            ax2 = ax.twinx()
            ax2.plot(x_grid, conv_result['efficiency'], 'navy', linewidth=1.5, linestyle='--',
                    label='Efficiency (Chebyshev)', alpha=0.7)
            ax2.set_ylabel('Efficiency', color='navy', fontsize=13)
            ax2.tick_params(axis='y', labelcolor='navy', labelsize=12)
            
            ax.set_xlabel('Reconstructed Momentum [MeV]', fontsize=14)
            ax.set_ylabel('Events / {:.2f} MeV'.format(binwidth), fontsize=14)
            ax.set_title('Reco Spectrum vs Theory with Floating RLE Parameters', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.minorticks_on()
            ax.grid(False)
            ax.legend(loc='upper right', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(output_plot, dpi=150, bbox_inches='tight')
            logger.log(f"Plot saved to {output_plot}", "success")
            plt.close()
        
        # Optional: Fit convolved spectrum to data
        fit_result = None
        if do_fit:
            logger.log("Fitting convolved spectrum to reco data...", "info")
            # Create fit plot filename if output_plot is specified
            fit_plot = None
            if output_plot:
                base, ext = os.path.splitext(output_plot)
                fit_plot = f"{base}_fit_diagnostic{ext}"
            
            fit_result = fit_convolved_spectrum_to_data(
                reco_momenta, x_grid, theory_normalized_to_1,
                mom_range=mom_range, binwidth=binwidth,
                output_plot=fit_plot,
                rle_models=None,
                calibration_path=calibration_path
            )
            if fit_result is not None:
                logger.log(f"Fit result: success={fit_result['success']}, "
                          f"NLL={fit_result['nll']:.4f}, scale={fit_result['scale_factor']:.6f}, "
                          f"χ²/dof={fit_result['chi2_per_dof']:.4f}", "info")
                if fit_result.get('rle_params'):
                    logger.log("RLE Parameters from Fit:", "info")
                    for param_name, param_info in fit_result['rle_params'].items():
                        val = param_info['value']
                        err = param_info['error']
                        if err is not None:
                            logger.log(f"  {param_name}: {val:.6f} ± {err:.6f}", "info")
                        else:
                            logger.log(f"  {param_name}: {val:.6f}", "info")
        
        return {
            'reco_momenta': reco_filtered,
            'convolution_result': conv_result,
            'n_events': n_events,
            'constraints_used': {'loss': loss_constraints, 'resolution': res_constraints},
            'constraint_margin': constraint_margin,
            'fit_result': fit_result
        }
        
    except Exception as e:
        logger.log(f"Error in constrained overlay: {e}", "error")
        traceback.print_exc()
        return None


def overlay_convolved_theory_on_reco(reco_momenta, calibration_path="RLE/common/calibration.json",
                                     mom_range=(95, 110), binwidth=0.1, output_plot=None):
    """
    Overlay normalized convolved theory spectrum on reconstructed data
    
    Creates histogram of reco data, applies RLE convolution to theory,
    normalizes theory to number of events, and plots both overlaid.
    
    Args:
        reco_momenta (array): Reconstructed momentum values from data
        calibration_path (str): Path to calibration.json from RLE
        mom_range (tuple): (min, max) momentum range
        binwidth (float): Bin width for momentum grid and histogram
        output_plot (str): Path to save overlay plot
        
    Returns:
        dict: {
            'reco_momenta': filtered reco data,
            'convolution_result': result from apply_ce_rle_convolution,
            'n_events': number of events in reco
        }
    """
    logger = Logger(print_prefix="[overlay_convolved_theory_on_reco]", verbosity=1)
    
    try:
        # Filter reco data to mom_range
        reco_filtered = reco_momenta[(reco_momenta >= mom_range[0]) & (reco_momenta <= mom_range[1])]
        n_events = len(reco_filtered)
        logger.log(f"Filtered reco data: {n_events} events in [{mom_range[0]}, {mom_range[1]}] MeV", "info")
        
        # Get convolved theory
        logger.log("Generating convolved theory spectrum...", "info")
        conv_result = apply_ce_rle_convolution(calibration_path=calibration_path,
                                                mom_range=mom_range, binwidth=binwidth,
                                                output_plot=None)  # Don't save conv plot here
        
        if conv_result is None:
            logger.log("Failed to generate convolution result", "error")
            return None
        
        x_grid = conv_result['x_grid']
        after_resolution = conv_result['after_resolution']  # Use spectrum BEFORE efficiency
        final_spectrum = conv_result['final']  # Keep for reference
        
        # Normalize spectrum BEFORE efficiency is applied
        # The efficiency will then modulate the final shape without inflating values
        integral_before_eff = np.trapz(after_resolution, x_grid)
        if integral_before_eff > 0:
            normalization_factor = n_events / integral_before_eff
            theory_normalized = after_resolution * normalization_factor
        else:
            logger.log("Warning: after-resolution spectrum integral is 0 or negative", "warning")
            theory_normalized = after_resolution
            normalization_factor = 0
        
        logger.log(f"=== NORMALIZATION DEBUG ===", "info")
        logger.log(f"Number of filtered reco events: {n_events}", "info")
        logger.log(f"After-resolution spectrum integral (before efficiency): {integral_before_eff:.8f}", "info")
        logger.log(f"Normalization factor (n_events / after_resolution_integral): {normalization_factor:.8f}", "info")
        logger.log(f"Theory max value after normalization: {np.max(theory_normalized):.6f}", "info")
        logger.log(f"Theory min value after normalization: {np.min(theory_normalized):.6f}", "info")
        logger.log(f"Theory mean value after normalization: {np.mean(theory_normalized):.6f}", "info")
        logger.log(f"Integral of normalized theory: {np.trapz(theory_normalized, x_grid):.8f} (should ≈ {n_events})", "info")
        logger.log(f"Reco histogram integral (sum of counts): {np.sum(np.histogram(reco_filtered, bins=int((mom_range[1] - mom_range[0]) / binwidth), range=mom_range)[0])}", "info")
        logger.log(f"Efficiency range: [{np.min(conv_result['efficiency']):.3f}, {np.max(conv_result['efficiency']):.3f}]", "info")
        logger.log(f"===========================", "info")
        
        # Create overlay plot
        if output_plot:
            logger.log(f"Creating overlay plot at {output_plot}...", "info")
            fig, ax = plt.subplots(figsize=(11, 8))
            
            # Reco histogram
            n_bins = int((mom_range[1] - mom_range[0]) / binwidth)
            counts, edges, patches = ax.hist(reco_filtered, bins=n_bins, range=mom_range,
                                              label=f'Reco Data ({n_events} events)',
                                              alpha=0.7, color='orange', edgecolor='none')
            
            # Convolved theory overlay (with efficiency modulation)
            # Multiply by binwidth to convert from density to bin counts to match histogram
            theory_with_eff = (theory_normalized * conv_result['efficiency']) * binwidth
            ax.plot(x_grid, theory_with_eff, 'r-', linewidth=2.5,
                   label='CeLL Theory ⊗ RLE (Landau+GCB) × Efficiency')
            
            # Also show efficiency as secondary axis
            ax2 = ax.twinx()
            ax2.plot(x_grid, conv_result['efficiency'], 'navy', linewidth=1.5, linestyle='--',
                    label='Efficiency (Chebyshev)', alpha=0.7)
            ax2.set_ylabel('Efficiency', color='navy', fontsize=13)
            ax2.tick_params(axis='y', labelcolor='navy', labelsize=12)
            
            ax.set_xlabel('Reconstructed Momentum [MeV]', fontsize=14)
            ax.set_ylabel('Events / {:.2f} MeV'.format(binwidth), fontsize=14)
            ax.set_title('Reconstructed Spectrum vs RLE-Convolved Theory', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            ax.minorticks_on()
            ax.grid(False)
            ax.legend(loc='upper right', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(output_plot, dpi=150, bbox_inches='tight')
            logger.log(f"Overlay plot saved to {output_plot}", "success")
            plt.close()
        
        return {
            'reco_momenta': reco_filtered,
            'convolution_result': conv_result,
            'n_events': n_events,
            'theory_normalized': theory_normalized
        }
        
    except Exception as e:
        logger.log(f"Error in overlay: {e}", "error")
        traceback.print_exc()
        return None
