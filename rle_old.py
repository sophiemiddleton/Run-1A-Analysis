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
    
    def generate_efficiency(self, data_flat_uncut, data_flat_cut, output_dir="./common"):
        """
        Generate efficiency.pkl from data already loaded and processed
        
        Efficiency is computed as: (flat e- passing cuts) / (all flat e-)
        This is a separate multiplicative factor from resolution/loss.
        
        Args:
            data_flat_uncut (dict): Uncut flat electron data (all startCode==173, rank==0, nhits>0)
            data_flat_cut (dict): Same data after physics cuts applied
            output_dir (str): Output directory for pkl files
            
        Returns:
            tuple: (efficiency array, momentum bin edges)
        """
        self.logger.log("Generating efficiency (cut vs uncut flat electrons)", "info")
        
        # Calculate efficiency vs momentum
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.3))
        
        gen_mom_all = np.array(ak.flatten(
            self.vector.get_mag(data_flat_uncut['trkmc']["trkmcsim"], 'mom'), 
            axis=None
        ))
        h_all, edges = np.histogram(gen_mom_all, 63, (79.8, 105.))
        
        # Histogram of flat electrons after cuts
        gen_mom_cut = np.array(ak.flatten(
            self.vector.get_mag(data_flat_cut['trkmc']["trkmcsim"], 'mom'), 
            axis=None
        ))
        h_cut, _ = np.histogram(gen_mom_cut, edges)
        
        # Compute efficiency with divide-by-zero protection
        efficiency = np.divide(h_cut, h_all, where=(h_all>0), out=np.zeros_like(h_all, dtype=float))
        gen_mom_sel = gen_mom_all
        h_sel, _ = np.histogram(gen_mom_sel, 63, (79.8, 105.))
        h_eff = np.divide(h_sel, h_all, where=h_all!=0, out=np.zeros_like(h_sel, dtype=float))
        
        ax.stairs(h_eff, edges, label="Flat electrons")
        ax.set_xlabel("Generated Momentum (MeV)")
        ax.set_ylabel("Efficiency")
        ax.legend()
        plt.savefig(f"{output_dir}/flat_efficiency.png")
        self.logger.log(f"Efficiency plot saved to {output_dir}/flat_efficiency.png", "info")
        
        # Save efficiency
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/efficiency_MDC2025an.pkl"
        with open(output_file, 'wb') as f:
            pkl.dump([h_eff, edges], f)
        
        self.logger.log(f"Saved efficiency_MDC2025an.pkl: {h_eff.shape}", "success")
        return h_eff, edges
    
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
                        for param in result.params:
                            try:
                                # Try to get errors if available
                                errors = result.errors()
                                if param in errors:
                                    lower_err = abs(errors[param]['lower'])
                                    upper_err = abs(errors[param]['upper'])
                                    avg_err = (lower_err + upper_err) / 2
                                    self.logger.log(f"        {param.name:12s} = {float(param.value()):10.6f} ± {avg_err:.6f}", "info")
                                else:
                                    self.logger.log(f"        {param.name:12s} = {float(param.value()):10.6f}", "info")
                            except:
                                self.logger.log(f"        {param.name:12s} = {float(param.value()):10.6f}", "info")
                        
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
                        
                    except Exception as e:
                        self.logger.log(f"      Fit failed: {str(e)}", "warn")
                    
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel('Events')
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
        
        self.logger.log(f"Saved fit plots to {fit_dir}/", "success")
        return results




def generate_rle_calibration(combined_data_uncut, combined_data_cut, output_dir="./common", run_fits=True):
    """
    Generate RLE calibration from already-processed data
    
    Properly factorizes:
    - Efficiency: (flat e- passing cuts) / (all flat e-)
    - Resolution: p_reco - p_mc (on all flat e-)
    - Loss: p_mc - p_gen (on all flat e-)
    
    Call this after data has been loaded and analyzed in process.py
    
    Args:
        combined_data_uncut (dict): Combined processed data BEFORE cuts (needed for res/loss)
        combined_data_cut (dict): Combined processed data AFTER cuts (needed for efficiency)
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
    
    # Select flat electrons (uncut)
    flat_e_sim_uncut = ((combined_data_uncut['trkmc']["trkmcsim"]["startCode"] == 173) & 
                        (combined_data_uncut['trkmc']["trkmcsim"]["rank"] == 0) & 
                        (combined_data_uncut['trkmc']["trkmcsim"]["nhits"] > 0))
    flat_e_trk_uncut = ak.any(flat_e_sim_uncut, axis=-1)
    flat_e_evt_uncut = ak.any(flat_e_trk_uncut, axis=-1)
    
    data_flat_uncut = {
        'trkmc': combined_data_uncut['trkmc'][flat_e_evt_uncut],
        'trkfit': combined_data_uncut['trkfit'][flat_e_evt_uncut],
        'trk': combined_data_uncut['trk'][flat_e_evt_uncut]
    }
    
    # Select flat electrons (cut) - already have cuts applied
    flat_e_sim_cut = ((combined_data_cut['trkmc']["trkmcsim"]["startCode"] == 173) & 
                      (combined_data_cut['trkmc']["trkmcsim"]["rank"] == 0) & 
                      (combined_data_cut['trkmc']["trkmcsim"]["nhits"] > 0))
    flat_e_trk_cut = ak.any(flat_e_sim_cut, axis=-1)
    flat_e_evt_cut = ak.any(flat_e_trk_cut, axis=-1)
    
    data_flat_cut = {
        'trkmc': combined_data_cut['trkmc'][flat_e_evt_cut],
        'trkfit': combined_data_cut['trkfit'][flat_e_evt_cut],
        'trk': combined_data_cut['trk'][flat_e_evt_cut]
    }
    
    # Step 1: Generate efficiency (cut vs uncut)
    rle.logger.log("\nStep 1: Generating efficiency (cut vs uncut)...", "info")
    h_eff, edges = rle.generate_efficiency(data_flat_uncut, data_flat_cut, output_dir)
    
    # Step 2: Generate skimmed data (from UNCUT flat electrons)
    rle.logger.log("\nStep 2: Generating skimmed data (uncut flat e-)...", "info")
    skimmed_path = f"{output_dir}/skimmed_flat_mom_MDC2025an.pkl"
    data_flat = rle.generate_skimmed_data(data_flat_uncut, skimmed_path)
    
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
    rle.logger.log(f"  - skimmed_flat_mom_MDC2025an.pkl", "info")
    if fit_results:
        rle.logger.log(f"  - fits/*.png (resolution/loss plots)", "info")
    
    return {
        'efficiency': (h_eff, edges),
        'skimmed_data': data_flat,
        'fit_plots': fit_results
    }
