"""
Resolution, Loss, Efficiency (RLE) calculation module for Mu2e analysis

This module provides a RLE class that encapsulates all resolution, loss, and efficiency
calculations. It can be imported into process.py to ensure consistency between
signal processing and calibration sample processing.
"""

import sys
import gc
import numpy as np
import awkward as ak
import pickle as pkl
import matplotlib.pyplot as plt
from collections import OrderedDict
import warnings

# Ensure py-fitter is available
sys.path.append('/exp/mu2e/app/users/sdittmer/LikelihoodAnalysis/py-fitter')

try:
    import zfit
    import hist as hist
    import mplhep
except ImportError as e:
    warnings.warn(f"Optional fitting packages not available: {e}")

from analyze import Analyze
from pyutils.pyprocess import Processor
from pyutils.pyvector import Vector
from pyutils.pyselect import Select
from pyutils.pylogger import Logger


class RLE:
    """
    Resolution, Loss, Efficiency calculator for tracker calibration
    
    Generates calibration parameters from flat electron sample that can be used
    in CE/DIO signal fitting.
    """
    
    def __init__(self, verbosity=1, sign="minus", cut_switch=None, use_remote=True):
        """
        Initialize RLE calculator
        
        Args:
            verbosity (int): Logging verbosity level
            sign (str): Particle sign ("minus" or "plus")
            cut_switch (list): Active cuts configuration
            use_remote (bool): Use remote files via mdh
        """
        self.verbosity = verbosity
        self.sign = sign
        self.use_remote = use_remote
        
        # Default cuts (from process.py main signal config)
        if cut_switch is None:
            cut_switch = [
                True,  # 0 is_reco_electron
                True,  # 1 has_downstream
                True,  # 2 has trk front
                True,  # 3 good_trkqpid
                True,  # 4 good_trkqual
                False, # 5 within_t0
                True,  # 6 within_t0err
                True,  # 7 has_hits
                False, # 8 within_lhr_maxl
                False, # 9 within_d0
                False, # 10 within_pitch_angle
                True,  # 11 has_st
                True,  # 12 no_opa
                True,  # 13 no_crv_veto
                True,  # 14 no_crv_quality
                True,  # 15 no_crv_timewindow
                True,  # 16 pz/pt
                True,  # 17 triggers
                False, # 18 within_mom_time
                False, # 19 early time
                False  # 20 reflected
            ]
        
        self.cut_switch = cut_switch
        
        # Initialize logger
        self.logger = Logger(print_prefix="[RLE]", verbosity=self.verbosity)
        
        # Initialize analysis tools
        self.analyse = Analyze(verbosity=0, sign=sign, cut_switch=cut_switch)
        self.processor = Processor(use_remote=use_remote, verbosity=0)
        self.vector = Vector()
        self.selector = Select()
        
        # Branches needed for RLE analysis
        self.branches = { 
            "crv" : ["crvcoincs.time"],
            "trk" : ["trk.nactive", "trk.pdg", "trkqual.result"],
            "trkfit" : ["trksegs", "trksegsmc", "trksegpars_lh"],
            "trkmc" : ["trkmcsim"]
        }
        
        # Fitting parameters (from flat_res.py)
        self.acbtype = "gcb"
        self.landau_loss = True
        self.conv_resloss = True
        self.binwidth_eval = 0.1
        self.p_bins = [95., 97., 99., 101., 103., 105.]
        self.planes = ['entrance', 'middle', 'exit']
        
        self.logger.log("Initialized RLE calculator", "info")
    
    def process_files(self, file_list_path):
        """
        Process list of ROOT files and combine data
        
        Args:
            file_list_path (str): Path to file list (one file per line)
            
        Returns:
            dict: Combined data from all files with analysis cuts applied
        """
        self.logger.log(f"Loading files from {file_list_path}", "info")
        
        with open(file_list_path, 'r') as f:
            file_list = [line.strip() for line in f if line.strip()]
        
        self.logger.log(f"Processing {len(file_list)} files", "info")
        
        combined_data = None
        successful_files = 0
        
        for idx, file_name in enumerate(file_list):
            just_filename = file_name.split('/')[-1] if '/' in file_name else file_name
            
            try:
                # Load raw data
                data = self.processor.process_data(file_name=file_name, branches=self.branches)
                
                # Apply analysis with same cuts as main signal
                analysis_result = self.analyse.execute(data, file_name)
                
                if analysis_result is None:
                    self.logger.log(f"[{idx+1}/{len(file_list)}] {just_filename}: analysis failed", "warning")
                    continue
                
                filtered_data = analysis_result["filtered_data"]
                successful_files += 1
                
                # Combine with previous files
                if combined_data is None:
                    combined_data = filtered_data
                else:
                    for key in combined_data.keys():
                        if key == "cutflow":
                            continue
                        
                        if isinstance(combined_data[key], dict):
                            for subkey in combined_data[key].keys():
                                if subkey in filtered_data[key]:
                                    combined_data[key][subkey] = ak.concatenate([
                                        combined_data[key][subkey], 
                                        filtered_data[key][subkey]
                                    ])
                        else:
                            combined_data[key] = ak.concatenate([
                                combined_data[key], 
                                filtered_data[key]
                            ])
                
                if (idx + 1) % 10 == 0 or (idx + 1) == len(file_list):
                    self.logger.log(f"[{idx+1}/{len(file_list)}] Processed {successful_files} files", "info")
                
                gc.collect()
                
            except Exception as e:
                self.logger.log(f"[{idx+1}/{len(file_list)}] {just_filename}: {e}", "warning")
                continue
        
        if combined_data is None:
            self.logger.log("ERROR: No data loaded successfully!", "error")
            return None
        
        self.logger.log(f"Successfully loaded {successful_files}/{len(file_list)} files", "success")
        return combined_data
    
    def generate_efficiency(self, data, output_dir="./common"):
        """
        Generate efficiency.pkl from combined data
        
        Args:
            data (dict): Combined processed data
            output_dir (str): Output directory for pkl files
            
        Returns:
            tuple: (efficiency array, momentum bin edges)
        """
        self.logger.log("Generating efficiency calibration", "info")
        
        # Select flat e- gen particles
        flat_e = ((data['trkmc']["trkmcsim"]["startCode"] == 173) & 
                  (data['trkmc']["trkmcsim"]["rank"] == 0) & 
                  (data['trkmc']["trkmcsim"]["nhits"] > 0))
        
        data_flat = {
            'trkmc': data['trkmc'][flat_e],
            'trkfit': data['trkfit'][ak.any(flat_e, axis=-1)],
            'trk': data['trk'][ak.any(flat_e, axis=-1)]
        }
        
        # Calculate efficiency vs momentum
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.3))
        
        gen_mom_all = np.array(ak.flatten(
            self.vector.get_mag(data_flat['trkmc']["trkmcsim"], 'mom'), 
            axis=None
        ))
        gen_mom_all = gen_mom_all[~np.isnan(gen_mom_all)]
        
        # Normalize by per-bin expected count assuming 1,500,000 events uniformly over 70-120 MeV
        # (100 files × 15,000 events per file)
        # Only histogram range 79.8-105, so scale expected accordingly
        n_bins = 63
        hist_range = (79.8, 105.)
        full_range = (70, 120)  # Assumed uniform generation range
        total_events_assumed = 1500000  # 100 files × 15,000 per file
        fraction_in_hist_range = (hist_range[1] - hist_range[0]) / (full_range[1] - full_range[0])
        expected_in_hist_range = total_events_assumed * fraction_in_hist_range
        per_bin_expected = expected_in_hist_range / n_bins
        h_all, edges = np.histogram(gen_mom_all, n_bins, hist_range)
        h_eff = h_all / per_bin_expected
        
        ax.stairs(h_eff, edges, label="Flat electrons")
        ax.set_xlabel("Generated Momentum (MeV)")
        ax.set_ylabel("Efficiency")
        
        # Add total efficiency text box
        total_eff = np.sum(h_all) / 1500000
        ax.text(0.98, 0.97, f'Total Efficiency: {total_eff:.4f}', 
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend()
        plt.savefig(f"{output_dir}/flat_efficiency.png")
        self.logger.log(f"Efficiency plot saved to {output_dir}/flat_efficiency.png", "info")
        
        # Save efficiency
        output_file = f"{output_dir}/efficiency.pkl"
        with open(output_file, 'wb') as f:
            pkl.dump([h_eff, edges], f)
        
        self.logger.log(f"Saved efficiency.pkl: {h_eff.shape}", "success")
        return h_eff, edges
    
    def generate_skimmed_data(self, data, output_file="skimmed_flat_mom_v2.pkl"):
        """
        Extract and save skimmed momentum data at each tracker plane
        
        Args:
            data (dict): Combined processed data
            output_file (str): Output file path
            
        Returns:
            dict: Skimmed data with gen/mc/reco momenta per plane
        """
        self.logger.log("Generating skimmed momentum data", "info")
        
        # Select flat e- gen particles
        flat_e = ((data['trkmc']["trkmcsim"]["startCode"] == 173) & 
                  (data['trkmc']["trkmcsim"]["rank"] == 0) & 
                  (data['trkmc']["trkmcsim"]["nhits"] > 0))
        
        data_flat_mcs = data['trkmc'][flat_e]
        trk_flat_e = ak.any(flat_e, axis=-1)
        data_flat_fit = data['trkfit'][trk_flat_e]
        data_flat_trk = data['trk'][trk_flat_e]
        data_flat_mc = data['trkmc'][trk_flat_e]
        
        data_flat = {}
        
        for sid, plane in enumerate(self.planes):
            self.logger.log(f"  Processing {plane} plane...", "info")
            
            # Select tracks at plane
            at_plane_reco = self.selector.select_surface(data_flat_fit, sid=sid, branch_name="trksegs")
            at_plane_mc = self.selector.select_surface(data_flat_fit, sid=sid, branch_name="trksegsmc")
            
            good_track = (ak.sum(at_plane_reco, axis=2) >= 1)
            good_track = (good_track) & (ak.sum(at_plane_mc, axis=2) == 1)
            
            gc.collect()
            
            # Extract momenta
            reco_mom = self.vector.get_mag(
                data_flat_fit["trksegs"].mask[(at_plane_reco) & (good_track)], 'mom'
            )
            reco_mom = ak.nan_to_none(reco_mom)
            reco_mom = ak.drop_none(reco_mom)
            reco_mom = np.array(ak.flatten(reco_mom, axis=None))
            
            mc_mom = self.vector.get_mag(
                data_flat_fit["trksegsmc"].mask[(at_plane_mc) & (good_track)], 'mom'
            )
            mc_mom = ak.nan_to_none(mc_mom)
            mc_mom = ak.drop_none(mc_mom)
            mc_mom = np.array(ak.flatten(mc_mom, axis=None))
            
            gen_mom = self.vector.get_mag(
                data_flat_mc["trkmcsim"].mask[good_track], 'mom'
            )
            gen_mom = ak.nan_to_none(gen_mom)
            gen_mom = ak.drop_none(gen_mom)
            gen_mom = np.array(ak.flatten(gen_mom, axis=None))
            
            self.logger.log(f"    Reco: {len(reco_mom)}, MC: {len(mc_mom)}, Gen: {len(gen_mom)}", "debug")
            
            data_flat[plane] = {'reco': reco_mom, 'mc': mc_mom, 'gen': gen_mom}
            gc.collect()
        
        # Save skimmed data
        with open(output_file, 'wb') as f:
            pkl.dump(data_flat, f)
        
        self.logger.log(f"Saved skimmed data to {output_file}", "success")
        return data_flat
    
    def load_efficiency(self, pkl_file):
        """Load pre-calculated efficiency"""
        with open(pkl_file, 'rb') as f:
            h_eff, edges = pkl.load(f)
        return h_eff, edges
    
    def load_skimmed_data(self, pkl_file):
        """Load pre-calculated skimmed data"""
        with open(pkl_file, 'rb') as f:
            data_flat = pkl.load(f)
        return data_flat
    
    def load_fit_parameters(self, pkl_file):
        """Load pre-fit resolution/loss parameters"""
        with open(pkl_file, 'rb') as f:
            fitpars = pkl.load(f)
        return fitpars
    
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
        # Import zfit functions only if needed
        try:
            import zfit
            import hist as hist
            import mplhep
        except ImportError:
            self.logger.log("ERROR: zfit/hist/mplhep not available for fitting", "error")
            return None
        
        self.logger.log("Starting resolution/loss fitting", "info")
        
        # Load skimmed data
        with open(skimmed_data_path, 'rb') as f:
            dict_flat = pkl.load(f)
        
        # Create output subdirectory for fits
        import os
        fit_dir = f"{output_dir}/fits"
        os.makedirs(fit_dir, exist_ok=True)
        
        results = {}
        
        for fit_type in fit_types:
            self.logger.log(f"  Fitting {fit_type}...", "info")
            
            fig_results = []
            
            for plane in self.planes:
                self.logger.log(f"    {plane} plane", "debug")
                
                if fit_type == 'res':
                    mom_in = dict_flat[plane]['mc']
                    mom_out = dict_flat[plane]['reco']
                    res_range = (-1, 1)
                    xlabel = "p_reco - p_mc (MeV)"
                elif fit_type == 'loss':
                    mom_in = dict_flat[plane]['gen']
                    mom_out = dict_flat[plane]['mc']
                    res_range = (-5, 5)
                    xlabel = "p_mc - p_gen (MeV)"
                else:  # resloss
                    mom_in = dict_flat[plane]['gen']
                    mom_out = dict_flat[plane]['reco']
                    res_range = (-6, 6)
                    xlabel = "p_reco - p_gen (MeV)"
                
                res_slice = mom_out - mom_in
                
                # Create histogram
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                counts, bins, patches = ax.hist(res_slice, bins=50, label=f'{plane} plane')
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Count')
                ax.set_title(f'{fit_type.upper()} - {plane} plane')
                ax.legend()
                
                fig.savefig(f"{fit_dir}/{fit_type}_{plane}.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                fig_results.append(f"{fit_dir}/{fit_type}_{plane}.png")
            
            results[fit_type] = fig_results
        
        self.logger.log(f"Saved fit plots to {fit_dir}/", "success")
        return results
    
    def run_full_calibration(self, file_list_path, output_dir="./common", run_fits=True):
        """
        Run complete calibration: process files, generate efficiency, skimmed data, and fits
        
        Args:
            file_list_path (str): Path to file list
            output_dir (str): Output directory
            run_fits (bool): Run resolution/loss fitting
            
        Returns:
            dict: Results including efficiency, skimmed data, and fit plots
        """
        self.logger.log("="*60, "info")
        self.logger.log("STARTING FULL RLE CALIBRATION", "info")
        self.logger.log("="*60, "info")
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Process files
        data = self.process_files(file_list_path)
        if data is None:
            return None
        
        # Step 2: Generate efficiency
        self.logger.log("\nStep 2: Generating efficiency...", "info")
        h_eff, edges = self.generate_efficiency(data, output_dir)
        
        # Step 3: Generate skimmed data
        self.logger.log("\nStep 3: Generating skimmed data...", "info")
        skimmed_path = f"{output_dir}/skimmed_flat_mom_v2.pkl"
        data_flat = self.generate_skimmed_data(data, skimmed_path)
        
        # Step 4: Fit resolution/loss (optional)
        fit_results = None
        if run_fits:
            self.logger.log("\nStep 4: Fitting resolution and loss...", "info")
            try:
                fit_results = self.fit_resolution_loss(skimmed_path, output_dir, ['res', 'loss'])
            except Exception as e:
                self.logger.log(f"Fitting failed: {e}", "warning")
        
        self.logger.log("="*60, "success")
        self.logger.log("CALIBRATION COMPLETE", "success")
        self.logger.log("="*60, "success")
        self.logger.log(f"\nOutput files saved to: {output_dir}/", "info")
        self.logger.log(f"  - efficiency.pkl", "info")
        self.logger.log(f"  - flat_efficiency.png", "info")
        self.logger.log(f"  - skimmed_flat_mom_v2.pkl", "info")
        if fit_results:
            self.logger.log(f"  - fits/*.png (resolution/loss plots)", "info")
        
        return {
            'efficiency': (h_eff, edges),
            'skimmed_data': data_flat,
            'fit_plots': fit_results
        }


if __name__ == "__main__":
    # Example usage - generates all plots by default
    rle = RLE(verbosity=1)
    results = rle.run_full_calibration(
        "file_lists/FlateMinus.txt",
        output_dir="common",
        run_fits=True  # Generate fit plots
    )
    
    if results:
        print("\n✓ Calibration complete!")
        print(f"  Efficiency plots: common/flat_efficiency.png")
        print(f"  Fit plots: common/fits/*.png")
        print(f"  Data files:")
        print(f"    - common/efficiency.pkl")
        print(f"    - common/skimmed_flat_mom_v2.pkl")
