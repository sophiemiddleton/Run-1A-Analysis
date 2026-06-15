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

    # Batch mode: run all analysis variables as overlays
    python plot_scaled_overlay.py --batch-variables \
                                   --output-dir <output_directory> \
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
    - "nst": Number of straw tube hits
    - "nopa": Number of OPA hits
    - "tandip": tan(dip) angle
    - "d0": Impact parameter
    - "px_pz" or "px/pz": Ratio of transverse to longitudinal momentum
    - "trkpid": Track PID score
    - "trkqual": Track quality score
    
Batch Mode Variables (--batch-variables):
    - NST: Number of straw tube hits
    - NOPA: Number of OPA hits
    - tandip: tan(Dip) angle
    - d0: Impact parameter
    - px_pz: Transverse to longitudinal momentum ratio
    - trkpid: Track PID score
    - trkqual: Track quality
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
            False,  # 3 good_trkqpid
            False,  # 4 good_trkqual
            True,  # 5 within_t0err
            True,  # 6 has_hits
            False, # 7 within_lhr_maxl
            False, # 8 within_d0
            False, # 9 within_pitch_angle
            False,  #10 has_st
            False,  #11 no_opa
            False,  #12 no_crv_veto
            False,  #13 no_crv_quality
            False,  #14 no_crv_timewindow
            False,  #15 pz/pt
            True,  #16 triggers
            True,  #17 in_mom_range
            False, #18 within_t0_early
            False, #19 no_reflected
            False,  #20 within_t0
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
    
"""
Plot scaled component distributions with data overlay with built-in CLs Optimization.

This script loads multiple MC component files (DIO, cosmic, RPC, etc.) and a data file,
applies cuts from analyze.py, scales MC components to a target event count, and overlays
data as scatter points on top of the histograms.
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
        # At exactly zero background, the median observation is 0.
        # The equation simplifies directly to -ln(alpha) which is 2.3026 for alpha=0.10
        return -np.log(alpha)
    
    # Determine the median background observation under a background-only hypothesis
    n_median = int(poisson.median(b))
    
    # Target function: find where CLs(mu) - alpha = 0
    def cls_target(mu):
        cl_sb = poisson.cdf(n_median, mu + b)
        cl_b = poisson.cdf(n_median, b)
        if cl_b == 0:
            return -alpha
        return (cl_sb / cl_b) - alpha

    try:
        # Bracket the root solver safely above the expected background fluctuations
        return brentq(cls_target, 0, 25 + 4 * np.sqrt(b))
    except ValueError:
        # Asymptotic safety fallback if background optimization scale is large
        return 1.645 * np.sqrt(b)


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
        self.logger = Logger(print_prefix="[ScaledOverlayPlotter]", verbosity=verbosity)
        self.components = {}
        self.data = None
        self.jobs = jobs
        
        self.default_yields = {
            'dio': 5.87e3,           
            'cosmic': 500.5,         
            'rpc_ext': 1.18,         
            'rpc_int': 1.49,         
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
            False,  # 3 good_trkqpid
            False,  # 4 good_trkqual
            True,  # 5 within_t0err
            True,  # 6 has_hits
            False, # 7 within_lhr_maxl
            False, # 8 within_d0
            False, # 9 within_pitch_angle
            False,  #10 has_st
            False,  #11 no_opa
            False,  #12 no_crv_veto
            False,  #13 no_crv_quality
            False,  #14 no_crv_timewindow
            False,  #15 pz/pt
            True,  #16 triggers
            True,  #17 in_mom_range
            False, #18 within_t0_early
            False, #19 no_reflected
            False,  #20 within_t0
            False # 21 signal region cut
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
                self.logger.log(f"No valid data from file list: {file_path}", "warning")
                return None
        except Exception as e:
            self.logger.log(f"Error processing file list {file_path}: {e}", "error")
            return None
    
    def load_components(self, component_files, sign="minus"):
        self.components = {}
        for component_name, file_path in component_files.items():
            if file_path is None:
                continue
            data = self.process_file(file_path, sign=sign)
            if data is not None:
                self.components[component_name] = data
    
    def load_data(self, data_file, sign="minus"):
        self.data = self.process_file(data_file, sign=sign, location='local')
    
    def extract_variable(self, data, var_name):
        """Extract a variable from the processed data
        
        Supports special preprocessing for certain variables:
        - "recomom_ttfront": Reconstructed momentum at tracker front
        - "recomom_mc_ttfront": MC true momentum at tracker front
        - "nst": Number of straw tube hits
        - "nopa": Number of OPA hits
        - "tandip": tan(dip) angle
        - "d0": Impact parameter
        - "px_pz": Ratio of px to pz momentum components
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
            
            elif var_name.lower() == "nst":
                self.logger.log(f"Extracting NST (Straw Tube hits)", "debug")
                selector = Select(verbosity=0)
                
                # Count straw tube hits at ST_Foils surface
                at_st_foils = selector.select_surface(data['trkfit'], surface_name="ST_Foils")
                nst = ak.sum(at_st_foils, axis=-1)
                
                # Drop None and flatten
                nst = ak.drop_none(nst)
                val = np.array(ak.flatten(nst, axis=None))
                
                self.logger.log(f"Extracted {len(val)} NST values", "debug")
                return val
            
            elif var_name.lower() == "nopa":
                self.logger.log(f"Extracting NOPA (OPA hits)", "debug")
                selector = Select(verbosity=0)
                
                # Count OPA hits
                at_opa = selector.select_surface(data['trkfit'], surface_name="OPA")
                nopa = ak.sum(at_opa, axis=-1)
                
                # Drop None and flatten
                nopa = ak.drop_none(nopa)
                val = np.array(ak.flatten(nopa, axis=None))
                
                self.logger.log(f"Extracted {len(val)} NOPA values", "debug")
                return val
            elif var_name.lower() ==  "crv":
                dt_threshold = 150
                selector = Select(verbosity=0)
                
                # Select segments at tracker front
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                # Get track and coincidence times
                trk_times = data['trkfit']["trksegs"]["time"][trk_front]  # events × tracks × segments
                coinc_times = data["crv"]["crvcoincs.time"]                  # events × coincidences
        
                # Broadcast CRV times to match track structure, so that we can compare element-wise
                # FIXME: should use ak.broadcast
                coinc_broadcast = coinc_times[:, None, None, :]  # Add dimensions for tracks and segments
                trk_broadcast = trk_times[:, :, :, None]         # Add dimension for coincidences

                # Calculate time differences
                dt = abs(trk_broadcast - coinc_broadcast)
                val = np.array(ak.flatten(dt, axis=None))
                
                self.logger.log(f"Extracted {len(val)} dt values", "debug")
                return val
            elif var_name.lower() == "tandip":
                self.logger.log(f"Extracting tanDip (tan of dip angle)", "debug")
                selector = Select(verbosity=0)
                
                # Select segments at tracker front
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trkfit_ent = ak.mask(data['trkfit']["trksegpars_lh"], trk_front)
                
                # Get tanDip
                tandip = trkfit_ent["tanDip"]
                
                # Drop None and flatten
                tandip = ak.drop_none(tandip)
                val = np.array(ak.flatten(tandip, axis=None))
                
                self.logger.log(f"Extracted {len(val)} tanDip values", "debug")
                return val
            
            elif var_name.lower() == "d0":
                self.logger.log(f"Extracting d0 (impact parameter)", "debug")
                selector = Select(verbosity=0)
                
                # Select segments at tracker front
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trkfit_ent = ak.mask(data['trkfit']["trksegpars_lh"], trk_front)
                
                # Get d0
                d0 = trkfit_ent["d0"]
                
                # Drop None and flatten
                d0 = ak.drop_none(d0)
                val = np.array(ak.flatten(d0, axis=None))
                
                self.logger.log(f"Extracted {len(val)} d0 values", "debug")
                return val
            
            elif var_name.lower() == "px_pz" or var_name.lower() == "px/pz":

                
                vec = Vector(verbosity=0)
                # restrict to tracker-front segments for vector creation
                selector = Select(verbosity=0)
                
                # Select segments at tracker front
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")

                trkfit_ent = ak.mask(data['trkfit']["trksegs"], trk_front)
                vec3 = vec.get_vector(trkfit_ent, 'mom')
  
                px = vec3.x
                py = vec3.y
                pz = vec3.z
                pt = vec3.rho

                # per-segment ratio (guard against division by zero)
                val = ak.where(pt > 0, pz / pt, ak.zeros_like(pt))
                val = ak.drop_none(val)
                val = np.array(ak.flatten(val, axis=None))

                self.logger.log(f"Extracted {len(val)} px/pz values", "debug")
                return val
            
            elif var_name.lower() == "trkpid":
                self.logger.log(f"Extracting trkpid (track PID score)", "debug")
                selector = Select(verbosity=0)
                
                # Select segments at tracker front
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trk_data = ak.mask(data['trk'], trk_front)
                
                # Get trkpid
                trkpid = trk_data["trkpid.result"]
                
                # Drop None and flatten
                trkpid = ak.drop_none(trkpid)
                val = np.array(ak.flatten(trkpid, axis=None))
                
                self.logger.log(f"Extracted {len(val)} trkpid values", "debug")
                return val
            
            elif var_name.lower() == "trkqual":
                self.logger.log(f"Extracting trkqual (track quality)", "debug")
                selector = Select(verbosity=0)
                
                # Select segments at tracker front
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trk_data = ak.mask(data['trk'], trk_front)
                
                # Get trkqual
                trkqual = trk_data["trkqual.result"]
                
                # Drop None and flatten
                trkqual = ak.drop_none(trkqual)
                val = np.array(ak.flatten(trkqual, axis=None))
                
                self.logger.log(f"Extracted {len(val)} trkqual values", "debug")
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
        if not self.components:
            return None
        
        component_data = {}
        max_events = 0
        for comp_name, comp_data in self.components.items():
            var_data = self.extract_variable(comp_data, variable_name)
            if var_data is not None and len(var_data) > 0:
                component_data[comp_name] = var_data
                max_events = max(max_events, len(var_data))
        
        data_var = None
        if self.data is not None:
            data_var = self.extract_variable(self.data, variable_name)
        
        all_vals = list(component_data.values())
        if data_var is not None:
            all_vals.append(data_var)
        all_combined = np.concatenate(all_vals)
        hist_range_auto = (np.min(all_combined), np.max(all_combined))
        hist_range = display_range if display_range is not None else hist_range_auto
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 9))
        component_colors = {
            'cosmic': '#1f77b4', 'rpc_int': '#2ca02c', 'rpc_ext': '#2ca02c',
            'rmc_int': '#d62728', 'rmc_ext': '#9467bd', 'ipa': '#8c564b',
            'dio': '#e377c2', 'ce': '#ff8000'
        }
        
        component_names = []
        scaled_histograms = []
        bin_edges = None
        
        for comp_name, var_data in component_data.items():
            if len(var_data) == 0: continue
            if use_component_yields and comp_name in self.component_yields and self.component_yields[comp_name] is not None:
                scale_factor = self.component_yields[comp_name] / len(var_data)
            else:
                scale_factor = (target_events if target_events else max_events) / len(var_data)
            
            counts, bins = np.histogram(var_data, bins=nbins, range=hist_range)
            bin_edges = bins
            scaled_histograms.append((counts * scale_factor) / (np.sum(counts) * (hist_range[1] - hist_range[0]) / nbins) if density else counts * scale_factor)
            component_names.append(comp_name)
        
        bin_width = (hist_range[1] - hist_range[0]) / nbins
        bin_edges_plot = np.linspace(hist_range[0], hist_range[1], nbins + 1)
        bin_centers = 0.5 * (bin_edges_plot[:-1] + bin_edges_plot[1:])
        
        desired_order = ['cosmic', 'dio', 'rpc_ext', 'rpc_int', 'rmc_ext', 'rmc_int', 'ipa', 'ce']
        component_order_dict = {name: hist for name, hist in zip(component_names, scaled_histograms)}
        
        display_names = {
            'cosmic': 'Cosmic Induced', 'dio': 'DIO',
            'rpc_ext': 'RPC', 'rpc_int': None,
            'rmc_ext': 'rmc_ext', 'rmc_int': 'rmc_int', 'ipa': 'ipa',
            'ce': 'Signal'
        }
        
        ordered_components = [(c, component_order_dict[c]) for c in desired_order if c in component_order_dict]
        signal = component_order_dict.get('ce', np.zeros(nbins))
        background_total = np.sum([hist for name, hist in ordered_components if name != 'ce'], axis=0)
        
        bottom = np.zeros(nbins)
        for comp_name, scaled_counts in ordered_components:
            color = component_colors.get(comp_name, 'C0')
            display_label = display_names.get(comp_name, comp_name)
            ax.bar(bin_centers, scaled_counts, width=bin_width, bottom=bottom,
                   label=display_label, color=color, alpha=1.0, edgecolor='none')
            bottom += scaled_counts

        if data_var is not None and len(data_var) > 0:
            data_counts, data_bins = np.histogram(data_var, bins=nbins, range=hist_range)
            data_scaled = data_counts / (np.sum(data_counts) * bin_width) if density else data_counts
            data_errors = np.sqrt(data_counts) / (np.sum(data_counts) * bin_width) if density else np.sqrt(data_counts)
            mask_nonzero = data_scaled > 0
            #ax.errorbar(bin_centers[mask_nonzero], data_scaled[mask_nonzero],
            #           yerr=data_errors[mask_nonzero], fmt='o', capsize=3,
            #           capthick=1.5, markersize=5, color='black', elinewidth=1.2,
            #           label='Mock Data', zorder=10)

        # =====================================================================
        # CLs ACCELERATED SIGNAL REGION OPTIMIZATION GRID SEARCH
        # =====================================================================
        best_sensitivity = float('inf')  
        best_low_idx = 0
        best_high_idx = nbins
        total_generated_signal = np.sum(signal) 
        
        if total_generated_signal <= 0:
            self.logger.log("Cannot optimize: Total signal in distribution is zero or negative.", "error")
            best_low_idx, best_high_idx = 0, nbins
        else:
            # 1. Faster running slice execution via Vectorized Cumulative Sums
            bg_cumsum = np.concatenate(([0], np.cumsum(background_total)))
            sig_cumsum = np.concatenate(([0], np.cumsum(signal)))
            max_possible_bkg = bg_cumsum[-1]

            # 2. Build local table map in RAM once to bypass root-finder inside loops
            bkg_evaluation_grid = np.linspace(0.0, max_possible_bkg + 0.1, 500)
            get_cls_limit_vec = np.vectorize(dynamic_cls_upper_limit)
            cls_limits_grid = get_cls_limit_vec(bkg_evaluation_grid)

            for low_idx in range(nbins):
                for high_idx in range(low_idx + 1, nbins + 1):
                    B_window = max(0.0, bg_cumsum[high_idx] - bg_cumsum[low_idx])
                    signal_passed = sig_cumsum[high_idx] - sig_cumsum[low_idx]
                    efficiency = signal_passed / total_generated_signal
                    
                    if efficiency <= 0:
                        continue
                        
                    # Interpolate from our dynamically generated execution map
                    expected_mu_90 = np.interp(B_window, bkg_evaluation_grid, cls_limits_grid)
                    sensitivity_metric = expected_mu_90 / efficiency
                    
                    if sensitivity_metric < best_sensitivity:
                        best_sensitivity = sensitivity_metric
                        best_low_idx = low_idx
                        best_high_idx = high_idx

        optimized_low_cut = bin_edges[best_low_idx]
        optimized_high_cut = bin_edges[best_high_idx]

        print("\n" + "="*50)
        print("--- Optimization with On-The-Fly Accelerated CLs Math ---")
        print(f"Optimal Window Boundaries: {optimized_low_cut:.3f} to {optimized_high_cut:.3f} MeV/c")
        print(f"Signal Efficiency: {(np.sum(signal[best_low_idx:best_high_idx]) / total_generated_signal) * 100:.2f}%")
        print(f"Expected Background in Window: {np.sum(background_total[best_low_idx:best_high_idx]):.4f} events")
        print(f"Optimized Sensitivity Metric (<mu_90_CLs> / eff): {best_sensitivity:.4f}")
        print("="*50 + "\n")
        
        nominal_lo, nominal_hi = 103.6, 104.9
        nominal_lo_idx, nominal_hi_idx = np.searchsorted(bin_edges, nominal_lo), np.searchsorted(bin_edges, nominal_hi)
        print(f"Background in Nominal Region ({nominal_lo} - {nominal_hi} MeV/c): {np.sum(background_total[nominal_lo_idx:nominal_hi_idx]):.4f} events")
        print(f"Signal in Nominal Region ({nominal_lo} - {nominal_hi} MeV/c): {np.sum(signal[nominal_lo_idx:nominal_hi_idx]):.4f} events")

        ax.set_yscale('log')
        ax.set_ylim(ymin=1)
        legend_fs = mpl.rcParams.get('legend.fontsize', 24)
        
        logo_to_use = logo_path if logo_path else ("mu2e_logo_oval.png" if Path("mu2e_logo_oval.png").exists() else None)
        if logo_to_use:
            try:
                from PIL import Image
                logo = Image.open(logo_to_use)
                ax_logo = fig.add_axes([0.02, 0.93, 0.1, 0.09])
                ax_logo.imshow(logo)
                ax_logo.axis('off')
            except Exception: pass
        
        ax.text(0.15, 0.98, "Mu2e Simulation (Preliminary - Summer 2026)", fontsize=legend_fs, fontweight='bold', ha='left', va='top', transform=ax.figure.transFigure, zorder=100)
        ax.text(0.32, 0.97, r"$R_{\mu e} = 1 \times 10^{-11}$" + "\n" + "t = 28 days" + "\n" + r"$N_{\mathrm{POT}} = 7.3 \times 10^{18}$", fontsize=legend_fs, ha='right', va='top', transform=ax.transAxes, zorder=100, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgrey', edgecolor='black', alpha=0.8))
        
        if cut_lo is not None: ax.axvline(cut_lo, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        if cut_hi is not None: ax.axvline(cut_hi, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        
        xlabel_map = {'recomom_ttfront': 'Reconstructed Momentum [MeV/c]', 'recomom_mc_ttfront': 'MC Momentum at Tracker Entrance [MeV/c]'}
        ax.set_xlabel(xlabel_map.get(variable_name.lower(), variable_name), fontsize=mpl.rcParams.get('axes.titlesize', 24))
        ax.set_ylabel('Events per 0.41 MeV/c' if not density else 'Density', fontsize=mpl.rcParams.get('axes.titlesize', 24))
        ax.set_xlim(hist_range)
        ax.legend(loc='upper right', framealpha=0.9)
        
        fig.subplots_adjust(top=0.97, bottom=0.1, left=0.1, right=0.95)
        fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.97])
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
        return fig, ax


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Plot scaled MC components with data overlay"
    )
    
    # Required arguments - make variable optional for batch mode
    parser.add_argument('--variable', 
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
    
    # Batch mode for running multiple plots at once
    parser.add_argument('--batch-variables', action='store_true',
                       help='Run batch mode with predefined analysis variables (NST, NOPA, tandip, d0, px_pz, trkpid, trkqual)')
    parser.add_argument('--output-dir', default='./overlay_plots',
                       help='Output directory for batch mode plots (default: ./overlay_plots)')
    
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
    
    # Batch mode for multiple predefined variables
    if args.batch_variables:
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Define batch variables with their display parameters
        batch_vars = {
            'nst': {
                'label': 'NST',
                'bins': 15,
                'cut_lo': 1,
                'cut_hi': 1,
                'range': (0, 15)
            },
            'nopa': {
                'label': 'NOPA',
                'bins': 4,
                'cut_lo': 0,
                'cut_hi': 0,
                'range': (0, 4)
            },
            'tandip': {
                'label': 'tan(Dip)',
                'bins': 22,
                'cut_lo': 0.557,
                'cut_hi': 1.0,
                'range': (-1, 2.5)
            },
            'd0': {
                'label': 'd0',
                'bins': 22,
                'cut_lo': 100,
                'cut_hi': 100,
                'range': (0, 250)
            },
            'px_pz': {
                'label': 'px/pz',
                'bins': 22,
                'cut_lo': 0.5,
                'cut_hi': 0.95,
                'range': (0.4, 2.0)
            },
            'trkpid': {
                'label': 'trkpid',
                'bins': 22,
                'cut_lo': None,
                'cut_hi': 0.67,
                'range': None  # Will auto-detect range
            },
            'trkqual': {
                'label': 'trkqual',
                'bins': 22,
                'cut_lo': None,
                'cut_hi': 0.2,
                'range': None  # Will auto-detect range
            },
            'crv': {
                'label': '|dt|',
                'bins': 22,
                'cut_lo': -150,
                'cut_hi': 150,
                'range': (-100, 300)
            },
        }
        
        plotter.logger.log(f"\n{'='*60}", "info")
        plotter.logger.log("RUNNING BATCH OVERLAY PLOTS", "info")
        plotter.logger.log(f"{'='*60}\n", "info")
        
        for var_name, var_params in batch_vars.items():
            try:
                output_file = os.path.join(args.output_dir, f"overlay_{var_name}.pdf")
                plotter.logger.log(f"Plotting {var_params['label']}...", "info")
                
                plotter.plot_scaled_overlay(
                    variable_name=var_name,
                    output_file=output_file,
                    target_events=args.target_events,
                    nbins=var_params['bins'],
                    cut_lo=var_params['cut_lo'],
                    cut_hi=var_params['cut_hi'],
                    use_log=args.log,
                    density=args.density,
                    title=var_params['label'],
                    use_component_yields=not args.uniform_scaling,
                    display_range=var_params['range'],
                    logo_path=args.logo
                )
                
                print(f"  ✓ Plot saved to: {output_file}")
                
            except Exception as e:
                plotter.logger.log(f"Error plotting {var_name}: {e}", "error")
                print(f"  ✗ Error plotting {var_name}: {e}")
        
        plotter.logger.log(f"\n{'='*60}", "info")
        plotter.logger.log(f"Batch plots complete! Output directory: {args.output_dir}", "info")
        plotter.logger.log(f"{'='*60}\n", "info")
    
    # Single plot mode
    else:
        if args.variable is None:
            print("ERROR: --variable is required when not using --batch-variables")
            sys.exit(1)
        
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
