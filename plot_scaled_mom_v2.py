"""
Plot scaled component distributions with on-the-fly CLs optimization.

This script loads multiple MC component files, applies custom analysis cuts, 
replaces the standard 'cosmic' background with a smoothed high-statistics 
Chebyshev polynomial model defined over [95, 115], scales all active MC 
components to their target yields, and plots them in a unified stacked histogram.

Example:

python plot_scaled_mom_v2.py     --variable "recomom_ttfront"     --dio file_lists/DIOtail95_MDC2025an_best_nomix.txt     --cosmic file_lists/Cosimcs_MDC2025an_nomix.txt     --ce file_lists/CeMLL_MDC2025an_best_nomix.txt     --output plots/recomom_ttfront_with_signal.png     --range 99 106     --bins 34     --title ""     --dio-yield 4760     --cosmic-yield 393     --rpc-ext-yield 0     --rpc-int-yield 0     --ce-yield 62     --jobs 16     --cut-lo 103.34     --cut-hi 104.74     --verbosity 2  --data file_lists/MDS3c_1e-13_2.txt 
"""

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import awkward as ak
from pathlib import Path
import sys
from datetime import datetime
import os

from scipy.stats import poisson
from scipy.optimize import brentq
from numpy.polynomial import chebyshev

# Import analysis utilities
from process import AnaProcessor
from pyutils.pylogger import Logger
from pyutils.pyselect import Select
from pyutils.pyvector import Vector

# Publication-style matplotlib defaults
import matplotlib.font_manager as mfm
preferred_serifs = ['DejaVu Serif', 'Times New Roman', 'Times', 'Palatino']
available_fonts = {f.name for f in mfm.fontManager.ttflist}
chosen_serif = next((f for f in preferred_serifs if f in available_fonts), 'DejaVu Serif')


def dynamic_cls_upper_limit(b, alpha=0.10):
    """
    Calculates the median expected 90% CL limit for an expected background b
    using the standard counting experiment CLs method.
    """
    if b <= 0:
        return -np.log(alpha)
    
    n_median = int(poisson.median(b))
    
    def cls_target(mu):
        cl_sb = poisson.cdf(n_median, mu + b)
        cl_b = poisson.cdf(n_median, b)
        if cl_b == 0:
            return -alpha
        return (cl_sb / cl_b) - alpha

    try:
        return brentq(cls_target, 0, 25 + 4 * np.sqrt(b))
    except ValueError:
        return 1.645 * np.sqrt(b)


mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': [chosen_serif],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
})


class ScaledOverlayPlotter:
    """Plot stacked scaled MC components with custom smooth cosmic model"""
    
    def __init__(self, verbosity=1, jobs=1):
        self.logger = Logger(print_prefix="[ScaledOverlayPlotter]", verbosity=verbosity)
        self.components = {}
        self.data = None
        self.jobs = jobs
        
        self.default_yields = {
            'dio': 5.87e3,           
            'cosmic': 500.5,  # Target normalization for the smooth cosmic model within the plot range       
            'rpc_ext': None,         
            'rpc_int': None,         
            'rmc_ext': None,         
            'rmc_int': None,         
            'ipa': None,            
            'ce': 65             
        }
        self.component_yields = self.default_yields.copy()

    def set_component_yields(self, yields_dict):
        self.component_yields.update(yields_dict)
        self.logger.log(f"Set component yields: {self.component_yields}", "info")
        
    def process_file(self, file_path, sign="minus", location="disk"):
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
            False, # 21 signal region cut
            False # or trigger select
        ]
        try:
            processor = AnaProcessor(
                file_list_path=file_path,
                jobs=self.jobs,
                sign=sign,
                cuts=new,
                location=location,
                proctype='overlay'
            )
            results = processor.execute()
            if results and results.get('combined_data') is not None:
                data = results['combined_data']
                self.logger.log(f"Successfully processed {len(data)} events", "info")
                return data
            else:
                return None
        except Exception as e:
            self.logger.log(f"Error processing file list {file_path}: {e}", "error")
            return None
    
    def load_components(self, component_files, sign="minus"):
        self.components = {}
        for component_name, file_path in component_files.items():
            if component_name == 'cosmic':
                continue  # Skip raw cosmic file loading since it will be modeled analytically
            if file_path is None:
                continue
            data = self.process_file(file_path, sign=sign)
            if data is not None:
                self.components[component_name] = data
    
    def load_data(self, data_file, sign="minus"):
        """Load and process data file"""
        self.data = self.process_file(data_file, sign=sign, location='local')
        if self.data is not None:
            self.logger.log(f"Loaded data: {len(self.data)} events", "info")
        else:
            self.logger.log(f"Failed to load data", "warning")
    
    def extract_variable(self, data, var_name):
        try:
            if var_name.lower() == "recomom_ttfront":
                selector = Select(verbosity=0)
                vector = Vector()
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trkfit_ent = ak.mask(data['trkfit']["trksegs"], trk_front)
                mom_mag = vector.get_mag(trkfit_ent, 'mom')
                return np.array(ak.flatten(ak.drop_none(mom_mag), axis=None))
            elif var_name.lower() == "recomom_mc_ttfront":
                selector = Select(verbosity=0)
                vector = Vector()
                trk_front_mc = selector.select_surface(data['trkfit'], surface_name="TT_Front", branch_name="trksegsmc")
                trkfit_ent_mc = ak.mask(data['trkfit']["trksegsmc"], trk_front_mc)
                mom_mag_mc = vector.get_mag(trkfit_ent_mc, 'mom')
                return np.array(ak.flatten(ak.drop_none(mom_mag_mc), axis=None))
            else:
                parts = var_name.split('.')
                val = data
                for part in parts:
                    val = val[part]
                return np.array(ak.flatten(ak.drop_none(val), axis=None))
        except Exception as e:
            self.logger.log(f"Error extracting '{var_name}': {e}", "error")
            return None
    
    def plot_scaled_overlay(self, variable_name, output_file=None, 
                           target_events=None, nbins=22,
                           cut_lo=None, cut_hi=None, use_log=False,
                           density=False, title=None, use_component_yields=True,
                           display_range=None, logo_path=None):
        
        # 1. Extract variables from all loaded MC components (DIO, RPC, CE, etc.)
        component_data = {}
        max_events = 0
        all_vals = []
        
        for comp_name, comp_data in self.components.items():
            var_data = self.extract_variable(comp_data, variable_name)
            if var_data is not None and len(var_data) > 0:
                component_data[comp_name] = var_data
                all_vals.append(var_data)
                max_events = max(max_events, len(var_data))
        
        # Extract data variable if data is loaded
        data_var = None
        if self.data is not None:
            data_var = self.extract_variable(self.data, variable_name)
            if data_var is not None and len(data_var) > 0:
                all_vals.append(data_var)
        
        # 2. Establish uniform histogram plotting boundaries
        if display_range is not None:
            hist_range = display_range
        elif len(all_vals) > 0:
            combined_array = np.concatenate(all_vals)
            hist_range = (np.min(combined_array), np.max(combined_array))
        else:
            hist_range = (99.0, 106.0) # Fallback window
            
        x_start, x_end = hist_range[0], hist_range[1]
        bin_width = (x_end - x_start) / nbins
        bin_edges_plot = np.linspace(x_start, x_end, nbins + 1)
        bin_centers = 0.5 * (bin_edges_plot[:-1] + bin_edges_plot[1:])
        
        # 3. Generate high-statistics Chebyshev cosmic sample defined over [95, 115]
        large_sample_size = 100000
        if use_component_yields and 'cosmic' in self.component_yields and self.component_yields['cosmic'] is not None:
            target_cosmic_yield = self.component_yields['cosmic']
        else:
            target_cosmic_yield = target_events if target_events else 500.5
        
        c1_mid = 0.219
        c2_mid = -0.108803
        cheb_coefficients = [0.0, c1_mid, c2_mid]
        
        # The wider domain where the Chebyshev polynomial parameters are defined
        def_start, def_end = 95.0, 115.0
        
        # Evaluate over dense grid to shift out negative phase configurations
        x_grid_def = np.linspace(def_start, def_end, 2000)
        x_grid_standard = 2 * (x_grid_def - def_start) / (def_end - def_start) - 1
        
        y_cheb_shape = chebyshev.chebval(x_grid_standard, cheb_coefficients)
        shift = 0.0
        if np.min(y_cheb_shape) < 0:
            shift = -np.min(y_cheb_shape)
            
        y_cheb_shape_shifted = y_cheb_shape + shift
        max_cheb_shape = np.max(y_cheb_shape_shifted)
        
        np.random.seed(42)
        cosmic_samples = []
        while len(cosmic_samples) < large_sample_size:
            # Propose random numbers across the full 95-115 definition range
            x_test_target = np.random.uniform(def_start, def_end, large_sample_size)
            # Map those proposed values to standard [-1, 1]
            x_test_standard = 2 * (x_test_target - def_start) / (def_end - def_start) - 1
            
            y_vals = chebyshev.chebval(x_test_standard, cheb_coefficients) + shift
            y_test = np.random.uniform(0, max_cheb_shape, large_sample_size)
            
            valid_samples = x_test_target[y_test < y_vals]
            cosmic_samples.extend(valid_samples)
            
        cosmic_samples = np.array(cosmic_samples[:large_sample_size])
        cosmic_weights = np.ones_like(cosmic_samples) * (target_cosmic_yield / large_sample_size)
        
        # Bin the generated smooth cosmic data into the plotting array
        cosmic_counts, _ = np.histogram(cosmic_samples, bins=nbins, range=hist_range, weights=cosmic_weights)
        
        # Print cosmic component info
        print(f"\nComponent Scaling Summary:")
        print(f"-" * 70)
        print(f"{'Component':<15} {'Detected Events':<20} {'Target Yield':<15} {'Scale Factor':<15}")
        print(f"-" * 70)
        print(f"{'cosmic':<15} {large_sample_size:<20} {target_cosmic_yield:<15.4f} {target_cosmic_yield / large_sample_size:<15.6f}")
        
        # 4. Bin and scale all available MC components (DIO, RPC, CE, etc.) using identical ranges
        scaled_histograms = {'cosmic': cosmic_counts}
        
        for comp_name, var_data in component_data.items():
            if use_component_yields and comp_name in self.component_yields and self.component_yields[comp_name] is not None:
                scale_factor = self.component_yields[comp_name] / len(var_data)
                target_yield = self.component_yields[comp_name]
            else:
                target_yield = target_events if target_events else max_events
                scale_factor = target_yield / len(var_data)
            
            counts, _ = np.histogram(var_data, bins=nbins, range=hist_range)
            scaled_histograms[comp_name] = counts * scale_factor
            
            # Print scaling info for this component
            print(f"{comp_name:<15} {len(var_data):<20} {target_yield:<15.4f} {scale_factor:<15.6f}")
        print(f"-" * 70 + "\n")
        
        # 5. Render Stacked Canvas Figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 9))
        component_colors = {
            'cosmic': '#1f77b4', 'rpc_int': '#2ca02c', 'rpc_ext': '#2ca02c',
            'rmc_int': '#d62728', 'rmc_ext': '#9467bd', 'ipa': '#8c564b',
            'dio': '#e377c2', 'ce': '#ff8000'
        }
        
        desired_order = ['cosmic', 'dio', 'rpc_ext', 'rpc_int', 'rmc_ext', 'rmc_int', 'ipa', 'ce']
        display_names = {
            'cosmic': 'Cosmic-Induced', 'dio': 'DIO',
            'rpc_ext': 'RPC', 'rpc_int': None,
            'rmc_ext': 'RMC Ext', 'rmc_int': 'RMC Int', 'ipa': 'IPA',
            'ce': 'Signal'
        }
        
        bottom = np.zeros(nbins)
        for comp_name in desired_order:
            if comp_name not in scaled_histograms:
                continue
            scaled_counts = scaled_histograms[comp_name]
            color = component_colors.get(comp_name, 'C0')
            display_label = display_names.get(comp_name, comp_name)
            
            ax.bar(bin_centers, scaled_counts, width=bin_width, bottom=bottom,
                   label=display_label, color=color, alpha=1.0, edgecolor='none')
            bottom += scaled_counts

        # Plot data if available
        if data_var is not None and len(data_var) > 0:
            data_counts, _ = np.histogram(data_var, bins=nbins, range=hist_range)
            data_errors = np.sqrt(data_counts)
            mask_nonzero = data_counts > 0
            ax.errorbar(bin_centers[mask_nonzero], data_counts[mask_nonzero],
                       yerr=data_errors[mask_nonzero], fmt='o', capsize=3,
                       capthick=1.5, markersize=5, color='black', elinewidth=1.2,
                       label='Mock Data', zorder=10)

        # 6. Sensitivity Windows Optimization Analysis
        signal = scaled_histograms.get('ce', np.zeros(nbins))
        background_total = np.sum([counts for name, counts in scaled_histograms.items() if name != 'ce'], axis=0)
        
        best_sensitivity = float('inf')  
        best_low_idx, best_high_idx = 0, nbins
        total_generated_signal = np.sum(signal) 
        
        if total_generated_signal > 0:
            bg_cumsum = np.concatenate(([0], np.cumsum(background_total)))
            sig_cumsum = np.concatenate(([0], np.cumsum(signal)))
            max_possible_bkg = bg_cumsum[-1]

            bkg_evaluation_grid = np.linspace(0.0, max_possible_bkg + 0.1, 500)
            get_cls_limit_vec = np.vectorize(dynamic_cls_upper_limit)
            cls_limits_grid = get_cls_limit_vec(bkg_evaluation_grid)

            for low_idx in range(nbins):
                for high_idx in range(low_idx + 1, nbins + 1):
                    B_window = max(0.0, bg_cumsum[high_idx] - bg_cumsum[low_idx])
                    signal_passed = sig_cumsum[high_idx] - sig_cumsum[low_idx]
                    efficiency = signal_passed / total_generated_signal
                    if efficiency <= 0: continue
                        
                    expected_mu_90 = np.interp(B_window, bkg_evaluation_grid, cls_limits_grid)
                    sensitivity_metric = expected_mu_90 / efficiency
                    
                    if sensitivity_metric < best_sensitivity:
                        best_sensitivity = sensitivity_metric
                        best_low_idx = low_idx
                        best_high_idx = high_idx

        optimized_low_cut = bin_edges_plot[best_low_idx]
        optimized_high_cut = bin_edges_plot[best_high_idx]

        print("\n" + "="*50)
        print("--- Optimization with Boundary Mapped Smooth Cosmic Component ---")
        print(f"Shape Definition Bounds: {def_start} to {def_end} MeV/c")
        print(f"Active Plot Window Range: {x_start:.2f} to {x_end:.2f} MeV/c")
        print(f"Optimal Window Boundaries: {optimized_low_cut:.3f} to {optimized_high_cut:.3f} MeV/c")
        print(f"Signal Efficiency inside Plot Window: {(np.sum(signal[best_low_idx:best_high_idx]) / total_generated_signal) * 100:.2f}%")
        print(f"Expected Background in Window: {np.sum(background_total[best_low_idx:best_high_idx]):.4f} events")
        print("="*50 + "\n")
        
        # Display settings configuration
        #ax.set_yscale('log')
        ax.set_ylim(ymin=1, ymax=45)
        legend_fs = mpl.rcParams.get('legend.fontsize', 14)
        logo_to_use = logo_path if logo_path else ("mu2e_logo_oval.png" if Path("mu2e_logo_oval.png").exists() else None)
        # --- 1. Mu2e Simulation Tag ---
        # Keep at x=0.02 (gives a clean, tight margin from the left axis)
        ax.text(0.08, 0.97, "Mu2e Simulation", 
                fontsize=20, fontweight='bold', ha='left', va='top', 
                transform=ax.transAxes, zorder=100)
        ax.text(0.08, 0.93, "Preliminary\n(Summer 2026)", 
                fontsize=20, fontweight='bold', fontstyle='italic', ha='left', va='top', 
                transform=ax.transAxes, zorder=100)
        # --- 2. Logo (Moved cleanly past "Simulation") ---
        if logo_to_use:
            try:
                from PIL import Image
                logo = Image.open(logo_to_use)
                # Shifted x right to 0.42 to completely clear the bold text
                # Adjusted y to 0.945 to perfectly line up with the first text line
                # Shrink layout slightly to [0.08, 0.045] to avoid vertical stretching
                ax_logo = ax.inset_axes([0.65, 0.60, 0.2, 0.15]) 
                ax_logo.imshow(logo)
                ax_logo.axis('off')
            except Exception: pass

        # --- 3. R_mue Bounding Box (Shifted slightly right and down) ---
        # Shifted x from 0.02 to 0.04 to add breathing room from the left axis line
        # Shifted y down slightly from 0.88 to 0.86 to avoid squeezing the text above it
        ax.text(0.08, 0.82, r"$R_{\mu e} = 1 \times 10^{-13}$" + "\n" + "t = 1 month" + "\n" + r"$N_{\mathrm{POT}} = 7.3 \times 10^{18}$", 
                fontsize=legend_fs, ha='left', va='top', 
                transform=ax.transAxes, zorder=100, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgrey', edgecolor='black', alpha=0.8))


        # Define your styling once
        line_style = dict(color='black', linestyle='--', linewidth=1.5, alpha=0.7)

        # Set your desired y-limits here
        y_min = 0  
        y_max = 25 
        enterlabel=True
        for cut in (cut_lo, cut_hi):
            if cut is not None:
                if enterlabel:
                    ax.vlines(x=cut, ymin=y_min,  label="Signal Region", ymax=y_max, **line_style)
                    enterlabel=False
                else:
                    ax.vlines(x=cut, ymin=y_min,  ymax=y_max, **line_style)
        
        xlabel_map = {'recomom_ttfront': 'Reconstructed Momentum [MeV/c]', 'recomom_mc_ttfront': 'MC Momentum at Tracker Entrance [MeV/c]'}
        ax.set_xlabel(xlabel_map.get(variable_name.lower(), variable_name))
        ax.set_ylabel(f'Events per {bin_width:.2f} MeV/c')
        ax.set_xlim(hist_range)
        handles, labels = ax.get_legend_handles_labels()

        ax.legend(
            handles[::-1], 
            labels[::-1], 
            loc='upper right', 
            framealpha=0.9, 
            fontsize=legend_fs
        )
        
        fig.subplots_adjust(top=0.98, bottom=0.12, left=0.12, right=0.95)
        if output_file:
            fig.savefig(output_file, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved successfully to: {output_file}")
        else:
            plt.show()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Plot scaled MC components with data overlay"
    )
    
    # Required arguments
    parser.add_argument('--variable', required=True, 
                       help='Variable to plot (e.g., "trk.pt", "trkfit.trksegpars_lh.p")')
    parser.add_argument('--data', default=None,
                       help='Path to data file (optional)')
    
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

    if args.data:
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
