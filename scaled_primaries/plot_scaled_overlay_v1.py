"""
Plot scaled component distributions with data overlay.

This script loads multiple MC component files (DIO, cosmic, RPC, etc.) and a data file,
applies cuts from analyze.py, scales MC components to a target event count, and overlays
data as scatter points on top of the histograms.

Usage:
    # Standard variable access
    python plot_scaled_overlay.py --variable <var_name> \
                                   --output <output_file.pdf> \
                                   --target-events <N> \
                                   [--range <lo> <hi>] \
                                   [--bins <N>] \
                                   [--cut-lo <value>] \
                                   [--cut-hi <value>] \
                                   --dio <file> \
                                   --cosmic <file> \
                                   --rpc-ext <file> \
                                   --rpc-int <file> \
                                   --rmc-ext <file> \
                                   --rmc-int <file> \
                                   --ipa <file> \
                                   --data <file>

    # Special variables with preprocessing (e.g., momentum at tracker front)
    python plot_scaled_overlay.py --variable recomom_ttfront \
                                   --output <output_file.pdf> \
                                   --target-events <N> \
                                   --dio <file> --cosmic <file> --data <file>

Special Variables:
    - "recomom_ttfront": Reconstructed momentum at tracker front
    - "recomom_mc_ttfront": MC true momentum at tracker front
"""

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import awkward as ak
from pathlib import Path
import sys
from datetime import datetime

# Import analysis utilities
import os

from process import AnaProcessor
from pyutils.pylogger import Logger
from pyutils.pyselect import Select
from pyutils.pyvector import Vector

# Publication-style matplotlib defaults
import matplotlib.font_manager as mfm
preferred_serifs = ['DejaVu Serif', 'Times New Roman', 'Times', 'Palatino']
available_fonts = {f.name for f in mfm.fontManager.ttflist}
chosen_serif = next((f for f in preferred_serifs if f in available_fonts), 'DejaVu Serif')

try:
    # Use numpy's loadtxt to load columns cleanly skipping the header row
    fc_data = np.loadtxt('/exp/mu2e/app/users/sophie/newOffline/Run-1A-Analysis/scaled_primaries/FC.csv', delimiter=',', skiprows=1)
    fc_backgrounds = fc_data[:, 0]
    fc_average_ul = fc_data[:, 1]
except FileNotFoundError:
    raise FileNotFoundError("Please make sure 'FC.csv' is generated and in the working directory.")

# Load the Feldman-Cousins Look-Up Table from the CSV file

def get_fc_average_limit(b):
    """
    Looks up the average FC 90% CL Upper Limit using linear interpolation 
    over the values loaded from the CSV file.
    """
    if b <= fc_backgrounds[-1]:
        return np.interp(b, fc_backgrounds, fc_average_ul)
    else:
        # Asymptotic safety fallback if background optimization expands past 0.105
        return 1.25 + 1.645 * np.sqrt(b)

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': [chosen_serif],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
})


class ScaledOverlayPlotter:
    """Plot scaled MC components with data overlay"""
    
    def __init__(self, verbosity=1, jobs=1):
        """Initialize the plotter
        
        Args:
            verbosity: Verbosity level for logging
            jobs: Number of parallel jobs for file processing
        """
        self.logger = Logger(print_prefix="[ScaledOverlayPlotter]", verbosity=verbosity)
        self.components = {}
        self.data = None
        self.jobs = jobs
        
        # Default component yields after standard cuts (physics expectations)
        # These can be overridden via set_component_yields()
        self.default_yields = {
            'dio': 5.87e3,           # DIO > 95
            'cosmic': 500.5,         # Cosmics
            'rpc_ext': 1.18,         # RPC External
            'rpc_int': 1.49,         # RPC Internal
            'rmc_ext': None,         # RMC External (not specified, will auto-scale)
            'rmc_int': None,         # RMC Internal (not specified, will auto-scale)
            'ipa': None,            # IPA/CE after cuts
            'ce': 65             # CE/signal (not specified, will auto-scale)
        }
        self.component_yields = self.default_yields.copy()


        
    def set_component_yields(self, yields_dict):
        """Set component-specific expected yields
        
        Args:
            yields_dict: Dict mapping component names to expected yields
                        {
                            'dio': 5.87e3,
                            'cosmic': 500.5,
                            ...
                        }
        """
        self.component_yields.update(yields_dict)
        self.logger.log(f"Set component yields: {self.component_yields}", "info")
        
    def process_file(self, file_path, sign="minus", location="disk"):
        """Process a single file list with cuts applied
        
        Args:
            file_path: Path to a file list (text file containing ROOT file paths)
            sign: Charge sign ("minus" or "plus")
            location: Location of files ('disk' or 'local')
            
        Returns:
            Processed data after cuts
        """
        self.logger.log(f"Processing file list: {file_path}", "info")
        new= [
            True,  # 0 is_reco_electron
            True,  # 1 has_downstream
            True, # 2 has trk front
            True,  # 3 good_trkqpid
            True,  # 4 good_trkqual
            True,  # 5 within_t0err
            True,  # 6 has_hits
            False, # 7 within_lhr_maxl
            False, # 8 within_d0
            False, # 9 within_pitch_angle
            True,  #10 has_st
            True,  #11 no_opa
            True,  #12 no_crv_veto
            True,  #13 no_crv_quality
            True,  #14 no_crv_timewindow
            True,  #15 pz/pt
            True,  #16 triggers
            True,  #17 in_mom_range
            False, #18 within_t0_early
            False, #19 no_reflected
            True,  #20 within_t0
            False # 21 signal region cut
        ]
        try:
            # Create processor and pass file list directly (don't wrap in another file list)
            processor = AnaProcessor(
                file_list_path=file_path,
                jobs=self.jobs,
                sign=sign,
                cuts=new,
                location=location,
                proctype='overlay'
            )
            
            # Process the file(s) - execute() returns the postprocessed results dict
            results = processor.execute()
            
            if results and results.get('combined_data') is not None:
                data = results['combined_data']
                self.logger.log(f"Successfully processed {len(data)} events", "info")
                return data
            else:
                self.logger.log(f"No valid data from file list: {file_path}", "warning")
                return None
                
        except Exception as e:
            self.logger.log(f"Error processing file list {file_path}: {e}", "error")
            import traceback
            self.logger.log(f"Traceback: {traceback.format_exc()}", "debug")
            return None
    
    def load_components(self, component_files, sign="minus"):
        """Load and process all component files
        
        Args:
            component_files: Dict with component names and file paths
                {
                    'dio': 'path/to/dio.root',
                    'cosmic': 'path/to/cosmic.root',
                    ...
                }
            sign: Charge sign
        """
        self.components = {}
        
        for component_name, file_path in component_files.items():
            if file_path is None:
                self.logger.log(f"Skipping component {component_name} (no file)", "info")
                continue
                
            data = self.process_file(file_path, sign=sign)
            if data is not None:
                self.components[component_name] = data
                self.logger.log(f"Loaded component '{component_name}': {len(data)} events", "info")
            else:
                self.logger.log(f"Failed to load component '{component_name}'", "warning")
    
    def load_data(self, data_file, sign="minus"):
        """Load and process data file
        
        Args:
            data_file: Path to data file
            sign: Charge sign
        """
        self.data = self.process_file(data_file, sign=sign, location='local')
        if self.data is not None:
            self.logger.log(f"Loaded data: {len(self.data)} events", "info")
        else:
            self.logger.log(f"Failed to load data", "warning")
    
    
    def extract_variable(self, data, var_name):
        """Extract a variable from the processed data
        
        Supports special preprocessing for certain variables:
        - "recomom_ttfront": Reconstructed momentum at tracker front
        - "recomom_mc_ttfront": MC true momentum at tracker front
        - Direct field access for standard variables like "trkfit.trksegpars_lh.p"
        
        Args:
            data: Awkward array with processed data
            var_name: Name of variable (e.g., 'trk.pt', 'trkfit.p', 'recomom_ttfront', etc.)
            
        Returns:
            Flattened numpy array of variable values
        """
        try:
            # Handle special variables that require preprocessing
            if var_name.lower() == "recomom_ttfront":
                self.logger.log(f"Extracting reconstructed momentum at TT_Front", "debug")
                selector = Select(verbosity=0)
                vector = Vector()
                
                # Select segments at tracker front
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trkfit_ent = ak.mask(data['trkfit']["trksegs"], trk_front)
                
                # Get momentum magnitude
                mom_mag = vector.get_mag(trkfit_ent, 'mom')
                
                # Flatten and drop None values
                mom_mag = ak.drop_none(mom_mag)
                val = np.array(ak.flatten(mom_mag, axis=None))
                
                self.logger.log(f"Extracted {len(val)} recomom values at TT_Front", "debug")
                return val
            
            elif var_name.lower() == "recomom_mc_ttfront":
                self.logger.log(f"Extracting MC true momentum at TT_Front", "debug")
                selector = Select(verbosity=0)
                vector = Vector()
                
                # Select segments at tracker front
                trk_front_mc = selector.select_surface(data['trkfit'], surface_name="TT_Front", 
                                                       branch_name="trksegsmc")
                trkfit_ent_mc = ak.mask(data['trkfit']["trksegsmc"], trk_front_mc)
                
                # Get momentum magnitude
                mom_mag_mc = vector.get_mag(trkfit_ent_mc, 'mom')
                
                # Flatten and drop None values
                mom_mag_mc = ak.drop_none(mom_mag_mc)
                val = np.array(ak.flatten(mom_mag_mc, axis=None))
                
                self.logger.log(f"Extracted {len(val)} MC momentum values at TT_Front", "debug")
                return val
            
            else:
                # Standard field access for nested paths like 'trk.pt' or 'trkfit.trksegpars_lh.p'
                parts = var_name.split('.')
                val = data
                
                for part in parts:
                    val = val[part]
                
                # Flatten and drop None values
                val = ak.drop_none(val)
                val = np.array(ak.flatten(val, axis=None))
                
                self.logger.log(f"Extracted {len(val)} values for '{var_name}'", "debug")
                return val
            
        except Exception as e:
            self.logger.log(f"Error extracting '{var_name}': {e}", "error")
            import traceback
            self.logger.log(f"Traceback: {traceback.format_exc()}", "debug")
            return None
    
    def plot_scaled_overlay(self, variable_name, output_file=None, 
                           target_events=None, nbins=22,
                           cut_lo=None, cut_hi=None, use_log=False,
                           density=False, title=None, use_component_yields=True,
                           display_range=None, logo_path=None):
        """Create overlay plot with scaled components and data points
        
        Args:
            variable_name: Name of variable to plot
            output_file: Path to save plot
            target_events: Target number of events for scaling (ignored if use_component_yields=True)
            nbins: Number of bins (created across full auto-detected data range for scaling across whole sample)
            cut_lo: Lower cut line position
            cut_hi: Upper cut line position
            use_log: Use log scale on y-axis
            density: Normalize to density
            title: Plot title
            use_component_yields: If True, use process-specific yields; if False, scale all to target_events
            display_range: Tuple (lo, hi) for plot display range (only controls plot area, not binning)
            logo_path: Path to Mu2e logo image file to display in top-left corner
        """
        if not self.components:
            self.logger.log("No components loaded", "error")
            return None
        
        if self.data is None:
            self.logger.log("No data loaded", "warning")
        
        # Extract variable from all components
        component_data = {}
        max_events = 0
        
        for comp_name, comp_data in self.components.items():
            var_data = self.extract_variable(comp_data, variable_name)
            if var_data is not None and len(var_data) > 0:
                component_data[comp_name] = var_data
                max_events = max(max_events, len(var_data))
                self.logger.log(f"Component '{comp_name}': {len(var_data)} events", "info")
            elif var_data is not None and len(var_data) == 0:
                self.logger.log(f"Component '{comp_name}': extracted 0 events after filtering (skipping)", "warning")
            else:
                self.logger.log(f"Failed to extract {variable_name} from {comp_name}", "warning")
        
        if not component_data:
            self.logger.log("No valid component data to plot", "error")
            return None
        
        # Extract data variable if available
        data_var = None
        if self.data is not None:
            data_var = self.extract_variable(self.data, variable_name)
            if data_var is None:
                self.logger.log(f"Could not extract {variable_name} from data", "warning")
        
        # Auto-detect histogram range from data (always)
        all_vals = list(component_data.values())
        if data_var is not None:
            all_vals.append(data_var)
        all_combined = np.concatenate(all_vals)
        hist_range_auto = (np.min(all_combined), np.max(all_combined))
        
        # Use display_range for histogram if specified, otherwise use auto-detected range
        if display_range is not None:
            hist_range = display_range
            self.logger.log(f"Using display range for histogramming: {hist_range}", "info")
        else:
            hist_range = hist_range_auto
        
        self.logger.log(f"Auto-detected data range: {hist_range_auto[0]:.2f} - {hist_range_auto[1]:.2f}", "info")
        self.logger.log(f"Histogram range: {hist_range[0]:.2f} - {hist_range[1]:.2f}", "info")
        self.logger.log(f"Creating {nbins} bins across histogram range {hist_range[0]:.2f} - {hist_range[1]:.2f}", "info")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 9))
        
        # Component colors and styles (matching compare.py)
        component_colors = {
            'cosmic': '#1f77b4',   # blue
            'rpc_int': '#2ca02c',  # green (combined with rpc_ext)
            'rpc_ext': '#2ca02c',  # green
            'rmc_int': '#d62728',  # red
            'rmc_ext': '#9467bd',  # purple
            'ipa': '#8c564b',      # brown
            'dio': '#e377c2',      # pink
            'ce': '#ff8000',       # orange (signal)
        }
        
        # Plot stacked histograms for each component
        # First pass: collect all scaled histograms
        component_names = []
        scaled_histograms = []
        bin_edges = None
        
        for comp_name, var_data in component_data.items():
            # Safety check for empty data
            if len(var_data) == 0:
                self.logger.log(f"Skipping {comp_name}: no valid data", "warning")
                continue
            
            # Determine scaling factor
            if use_component_yields and comp_name in self.component_yields and self.component_yields[comp_name] is not None:
                # Use physics-motivated yield for this component
                target_yield = self.component_yields[comp_name]
                scale_factor = target_yield / len(var_data)
                self.logger.log(f"Scaling {comp_name}: {len(var_data)} events -> {target_yield:.1f} (factor: {scale_factor:.4f})", "debug")
            else:
                # Fall back to uniform scaling with target_events or max_events
                if target_events is None:
                    target_events = max_events
                scale_factor = target_events / len(var_data)
                self.logger.log(f"Scaling {comp_name}: {len(var_data)} events -> {target_events} (factor: {scale_factor:.4f})", "debug")
            
            # Create histogram across full range (whole sample)
            counts, bins = np.histogram(var_data, bins=nbins, range=hist_range)
            bin_edges = bins
            
            # Scale the histogram
            if density:
                bin_width = (hist_range[1] - hist_range[0]) / nbins
                scaled_counts = (counts * scale_factor) / (np.sum(counts) * bin_width)
            else:
                scaled_counts = counts * scale_factor
            
            component_names.append(comp_name)
            scaled_histograms.append(scaled_counts)
        
        # Plot stacked histogram
        bin_width = (hist_range[1] - hist_range[0]) / nbins
        bin_edges_plot = np.linspace(hist_range[0], hist_range[1], nbins + 1)
        bin_centers = 0.5 * (bin_edges_plot[:-1] + bin_edges_plot[1:])
        
        # Reorder components for stacking: cosmic first, then others
        desired_order = ['cosmic', 'dio', 'rpc_ext', 'rpc_int', 'rmc_ext', 'rmc_int', 'ipa', 'ce']
        component_order_dict = {name: hist for name, hist in zip(component_names, scaled_histograms)}
        
        # Map component names to display names for legend
        display_names = {
            'cosmic': 'Cosmic Induced',
            'dio': 'Decay in Orbit (DIO)',
            'rpc_ext': 'Radiative Pion Capture (RPC)',
            'rpc_int': None,  # Will be skipped (combined with rpc_ext)
            'rmc_ext': 'rmc_ext',
            'rmc_int': 'rmc_int',
            'ipa': 'ipa',
            'ce': 'Conversion Electron (Signal)'
        }
        
        ordered_components = []
        for comp_name in desired_order:
            if comp_name in component_order_dict:
                ordered_components.append((comp_name, component_order_dict[comp_name]))
        
        signal = component_order_dict.get('ce', np.zeros(nbins))
        background_total = np.zeros(nbins)
        for comp_name, scaled_counts in ordered_components:
            if comp_name != 'ce':
                background_total += scaled_counts
            
        # Create stacked bar plot
        bottom = np.zeros(nbins)
        for comp_name, scaled_counts in ordered_components:
            color = component_colors.get(comp_name, 'C0')
            # Use display name if available, skip if None (e.g., rpc_int combined with rpc_ext)
            display_label = display_names.get(comp_name, comp_name)
            if display_label is not None:
                ax.bar(bin_centers, scaled_counts, width=bin_width, bottom=bottom,
                       label=display_label, color=color, alpha=1.0, edgecolor='none')
            else:
                # Plot without label (for rpc_int which is combined with rpc_ext)
                ax.bar(bin_centers, scaled_counts, width=bin_width, bottom=bottom,
                       color=color, alpha=1.0, edgecolor='none')
            bottom += scaled_counts
            print (f"{comp_name} scaled counts: {scaled_counts}")
        # Overlay data as scatter points with error bars
        if data_var is not None and len(data_var) > 0:
            # Create histogram of data across full range (whole sample)
            data_counts, data_bins = np.histogram(data_var, bins=nbins, range=hist_range)
            
            # Scale data histogram if needed
            if density:
                bin_width = (hist_range[1] - hist_range[0]) / nbins
                data_scaled = data_counts / (np.sum(data_counts) * bin_width)
            else:
                data_scaled = data_counts
            
            # Calculate Poisson errors
            data_errors = np.sqrt(data_counts)
            if density:
                bin_width = (hist_range[1] - hist_range[0]) / nbins
                data_errors = data_errors / (np.sum(data_counts) * bin_width)
            
            # Plot data as points
            bin_centers = 0.5 * (data_bins[:-1] + data_bins[1:])
            mask_nonzero = data_scaled > 0
            ax.errorbar(bin_centers[mask_nonzero], data_scaled[mask_nonzero],
                       yerr=data_errors[mask_nonzero], fmt='o', capsize=3,
                       capthick=1.5, markersize=5, color='black', elinewidth=1.2,
                       label='Mock Data (pseudo-experiment)', zorder=10)
        print("scaled data", data_scaled)


        # Optimization Grid Search
        best_significance = 0
        best_low_idx = 0
        best_high_idx = 0
        total_generated_signal = sum(signal) # Total generated signal events before cuts (for efficiency calculation)
        print("total generated signal", total_generated_signal)
        best_sensitivity = float('inf')  # Initialize to infinity for minimization
        for low_idx in range(nbins):
            for high_idx in range(low_idx + 1, nbins + 1):
                B_window = np.sum(background_total[low_idx:high_idx])
                signal_passed = np.sum(signal[low_idx:high_idx])
                efficiency = signal_passed / (total_generated_signal)
                
                if efficiency == 0:
                    continue
                    
                # Get the strict average upper limit mapping from your CSV-loaded table
                expected_mu_90 = get_fc_average_limit(B_window)
                
                # Minimize the minimum detectable branching ratio factor (<mu_90> / efficiency)
                sensitivity_metric = expected_mu_90 / (efficiency)
                print(low_idx,high_idx,B_window,signal_passed, expected_mu_90 , (efficiency), sensitivity_metric)
                if sensitivity_metric < best_sensitivity:
                    best_sensitivity = sensitivity_metric
                    best_low_idx = low_idx
                    best_high_idx = high_idx

        # 4. Extract optimized selection cuts
        optimized_low_cut = bin_edges[best_low_idx]
        optimized_high_cut = bin_edges[best_high_idx]

        print("--- Optimization with CSV-Loaded Feldman-Cousins Table ---")
        print(f"Optimal Window Boundaries: {optimized_low_cut:.2f} to {optimized_high_cut:.2f} MeV/c")
        print(f"Signal Efficiency: {np.sum(signal[best_low_idx:best_high_idx])/total_generated_signal * 100:.2f}%")
        print(f"Expected Background in Window: {np.sum(background_total[best_low_idx:best_high_idx]):.4f} events")
        print(f"Optimized Sensitivity Metric (<mu_90> / eff): {best_sensitivity:.4f}")
        
        # Calculate background in nominal signal region (103.6 - 104.9 MeV/c)
        nominal_lo = 103.6
        nominal_hi = 104.9
        nominal_lo_idx = np.searchsorted(bin_edges, nominal_lo)
        nominal_hi_idx = np.searchsorted(bin_edges, nominal_hi)
        background_nominal = np.sum(background_total[nominal_lo_idx:nominal_hi_idx])
        signal_nominal = np.sum(signal[nominal_lo_idx:nominal_hi_idx])
        print(f"Background in Nominal Region ({nominal_lo:.1f} - {nominal_hi:.1f} MeV/c): {background_nominal:.4f} events")
        print(f"Signal in Nominal Region ({nominal_lo:.1f} - {nominal_hi:.1f} MeV/c): {signal_nominal:.4f} events")



        # Always use log scale for y-axis
        ax.set_yscale('log')
        ax.set_ylim(ymin=5, ymax=150)
        legend_fs = mpl.rcParams.get('legend.fontsize', 24)
        # Add logo if provided or use default if available
        logo_to_use = logo_path
        if logo_to_use is None:
            # Check for default logo in current directory
            default_logo = Path("mu2e_logo_oval.png")
            if default_logo.exists():
                logo_to_use = str(default_logo)
        
        if logo_to_use is not None and Path(logo_to_use).exists():
            try:
                from PIL import Image
                logo = Image.open(logo_to_use)
                # Create inset axis for logo on far left before Preliminary label
                # [left, bottom, width, height] in figure coordinates
                ax_logo = fig.add_axes([0.02, 0.93, 0.1, 0.09])
                ax_logo.imshow(logo)
                ax_logo.axis('off')
                self.logger.log(f"Added logo from: {logo_to_use}", "info")
            except ImportError:
                self.logger.log("PIL/Pillow not available for logo display", "warning")
            except Exception as e:
                self.logger.log(f"Could not load logo from {logo_to_use}: {e}", "warning")
        
        # Add Preliminary label on plot canvas
        ax.text(0.15, 0.98, "Mu2e Simulation (Preliminary - Summer 2026)", fontsize=legend_fs, fontweight='bold',
           ha='left', va='top', transform=ax.figure.transFigure, zorder=100)

        
        # Add annotation for signal region if variable is momentum at tracker front
        ax.text(0.32, 0.97, r"$R_{\mu e} = 1 \times 10^{-13}$" + "\n" + "t = 28 days" + "\n" + r"$N_{\mathrm{POT}} = 7.3 \times 10^{18}$", fontsize=legend_fs, 
           ha='right', va='top', transform=ax.transAxes, zorder=100,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgrey', edgecolor='black', alpha=0.8))
        # Add cut lines if specified
        if cut_lo is not None:
            ax.axvline(cut_lo, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        if cut_hi is not None:
            ax.axvline(cut_hi, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Map variable names to human-readable labels
        xlabel_map = {
            'recomom_ttfront': 'Reconstructed Momentum at Tracker Entrance [MeV/c]',
            'recomom_mc_ttfront': 'MC Momentum at Tracker Entrance [MeV/c]',
        }
        title_fs = mpl.rcParams.get('axes.titlesize', 24)
        xlabel = xlabel_map.get(variable_name.lower(), variable_name)
        ax.set_xlabel(xlabel, fontsize=title_fs)
        ax.set_ylabel('Events per bin' if not density else 'Density', fontsize=title_fs)
        """
        if title:
            ax.set_title(title, fontsize=axes.titlesize)
        else:
            ax.set_title(f'{variable_name} (scaled to {target_events} events)', fontsize=axes.titlesize)
        """
        # Set x-axis limits (use histogram range which respects display_range if provided)
        ax.set_xlim(hist_range)
        ax.legend(loc='upper right', framealpha=0.9)
        #ax.grid(True, alpha=0.3)
        
        # Adjust layout with space for preliminary label at top
        fig.subplots_adjust(top=0.97, bottom=0.1, left=0.1, right=0.95)
        fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.97])
        
        # Save figure
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
            self.logger.log(f"Saved plot to: {output_file}", "info")
        
        return fig, ax


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Plot scaled MC components with data overlay"
    )
    
    # Required arguments
    parser.add_argument('--variable', required=True, 
                       help='Variable to plot (e.g., "trk.pt", "trkfit.trksegpars_lh.p")')
    parser.add_argument('--data', required=True,
                       help='Path to data file')
    
    # Output
    parser.add_argument('--output', '-o', default='plot_scaled_overlay.pdf',
                       help='Output file path (default: plot_scaled_overlay.pdf)')
    
    # Component files
    parser.add_argument('--dio', help='Path to DIO component file')
    parser.add_argument('--cosmic', help='Path to cosmic component file')
    parser.add_argument('--rpc-ext', help='Path to external RPC component file')
    parser.add_argument('--rpc-int', help='Path to internal RPC component file')
    parser.add_argument('--rmc-ext', help='Path to external RMC component file')
    parser.add_argument('--rmc-int', help='Path to internal RMC component file')
    parser.add_argument('--ipa', help='Path to IPA component file')
    parser.add_argument('--ce', '--signal', dest='ce', help='Path to CE/signal component file')
    
    # Plot options
    parser.add_argument('--target-events', type=int,
                       help='Target number of events for uniform scaling (default: use max)')
    
    # Component-specific yields (physics expectations after cuts)
    parser.add_argument('--dio-yield', type=float, 
                       help='Expected DIO yield after cuts (default: 5.87e3)')
    parser.add_argument('--cosmic-yield', type=float,
                       help='Expected cosmic ray yield after cuts (default: 500.5)')
    parser.add_argument('--rpc-ext-yield', type=float,
                       help='Expected external RPC yield after cuts (default: 1.18)')
    parser.add_argument('--rpc-int-yield', type=float,
                       help='Expected internal RPC yield after cuts (default: 1.49)')
    parser.add_argument('--rmc-ext-yield', type=float,
                       help='Expected external RMC yield after cuts')
    parser.add_argument('--rmc-int-yield', type=float,
                       help='Expected internal RMC yield after cuts')
    parser.add_argument('--ipa-yield', type=float,
                       help='Expected IPA/CE yield after cuts (default: 64.87)')
    parser.add_argument('--ce-yield', type=float,
                       help='Expected CE/signal yield after cuts')
    parser.add_argument('--uniform-scaling', action='store_true',
                       help='Use uniform scaling (--target-events) instead of physics-motivated yields')
    
    parser.add_argument('--range', type=float, nargs=2, metavar=('LO', 'HI'),
                       help='Plot display range only (scaling done over full auto-detected data range)')
    parser.add_argument('--bins', type=int, default=22,
                       help='Number of bins across full data range (default: 22)')
    parser.add_argument('--cut-lo', type=float,
                       help='Lower cut line position')
    parser.add_argument('--cut-hi', type=float,
                       help='Upper cut line position')
    parser.add_argument('--log', action='store_true',
                       help='Use log scale on y-axis')
    parser.add_argument('--density', action='store_true',
                       help='Normalize to density')
    parser.add_argument('--title',
                       help='Plot title')
    parser.add_argument('--sign', default='minus', choices=['minus', 'plus'],
                       help='Charge sign (default: minus)')
    parser.add_argument('--verbosity', type=int, default=1,
                       help='Verbosity level (default: 1)')
    parser.add_argument('--jobs', type=int, default=1,
                       help='Number of parallel jobs for file processing (default: 1)')
    parser.add_argument('--logo', help='Path to Mu2e logo image file (PNG, JPG, or PDF)')
    
    args = parser.parse_args()
    
    # Create plotter with specified number of jobs
    plotter = ScaledOverlayPlotter(verbosity=args.verbosity, jobs=args.jobs)
    
    # Set component yields if provided
    custom_yields = {}
    if args.dio_yield is not None:
        custom_yields['dio'] = args.dio_yield
    if args.cosmic_yield is not None:
        custom_yields['cosmic'] = args.cosmic_yield
    if args.rpc_ext_yield is not None:
        custom_yields['rpc_ext'] = args.rpc_ext_yield
    if args.rpc_int_yield is not None:
        custom_yields['rpc_int'] = args.rpc_int_yield
    if args.rmc_ext_yield is not None:
        custom_yields['rmc_ext'] = args.rmc_ext_yield
    if args.rmc_int_yield is not None:
        custom_yields['rmc_int'] = args.rmc_int_yield
    if args.ipa_yield is not None:
        custom_yields['ipa'] = args.ipa_yield
    if args.ce_yield is not None:
        custom_yields['ce'] = args.ce_yield
    
    if custom_yields:
        plotter.set_component_yields(custom_yields)
    
    # Load components
    component_files = {
        'dio': args.dio,
        'cosmic': args.cosmic,
        'rpc_ext': args.rpc_ext,
        'rpc_int': args.rpc_int,
        'rmc_ext': args.rmc_ext,
        'rmc_int': args.rmc_int,
        'ipa': args.ipa,
        'ce': args.ce,
    }
    
    plotter.load_components(component_files, sign=args.sign)
    plotter.load_data(args.data, sign=args.sign)
    
    # Create plot
    plotter.plot_scaled_overlay(
        variable_name=args.variable,
        output_file=args.output,
        target_events=args.target_events,
        nbins=args.bins,
        cut_lo=args.cut_lo,
        cut_hi=args.cut_hi,
        use_log=args.log,
        density=args.density,
        title=args.title,
        use_component_yields=not args.uniform_scaling,  # Use physics yields by default
        display_range=tuple(args.range) if args.range else None,
        logo_path=args.logo
    )
    
    print(f"Plot saved to: {args.output}")


if __name__ == '__main__':
    main()
