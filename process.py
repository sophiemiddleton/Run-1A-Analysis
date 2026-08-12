print("DEBUG: Importing modules...", flush=True)

import gc
import sys
import traceback
from datetime import datetime
import numpy as np


# Set non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse
import csv
import pickle as pkl
import zfit
from sklearn.model_selection import train_test_split

import pandas as pd
import xgboost as xgb
# this ana
from compare import Compare
from cosmics import Cosmics
from rpc import RPC
from rmc import RMC
from analyze import Analyze
from rle import (
    generate_rle_calibration,
    plot_theory_with_rle,
    apply_ce_rle_convolution,
    fit_convolved_spectrum_to_data,
    overlay_convolved_theory_on_reco_with_constraints,
    overlay_convolved_theory_on_reco
)
from RLE.rle_functions import apply_rle_convolution
from spectrum import TheorySpectrum
from pyutils.pycut import CutManager
from pyutils.pylogger import Logger
import optimize_cuts
from helper import make_HistogramPDF
from rle_v2 import RLE_v2


#from fits import Fits
from pyutils.pyprocess import Processor, Skeleton
from pyutils.pyplot import Plot
from pyutils.pyprint import Print
from pyutils.pyselect import Select
from pyutils.pyvector import Vector

class AnaProcessor(Skeleton):
    """custom file processor 
    
    This class inherits from the Skeleton defined in pyutils/pyprocess base class, which provides the 
    basic structure and methods withing the Processor framework 
    """
    def __init__(self, file_list_path, jobs=1, sign="minus", cuts=[], location='disk', proctype="ensemble"):
        """Initialise your processor with specific configuration
        
        This method sets up all the parameters needed for this specific analysis.
        """
        # Call the parent class's __init__ method first
        # This ensures we have all the base functionality properly set up
        super().__init__()

        # Now override parameters from the Skeleton with the ones we need
        self.file_list_path = file_list_path
        
        # Track file processing status
        self.current_file_index = 0
        self.total_files = 0
        self.file_list = []

        self.branches = { 
            "evt" : [
                "run",
                "subrun",
                "event",
                "trig_apr_TrkDe_80m70p",
                "trig_cpr_TrkDe_80m70p",
                "trig_tpr_TrkDe_80m70p"
            ],
            "crv" : [
                "crvcoincs.time",
                "crvcoincs.nHits",
                "crvcoincs.PEs",
                "crvcoincs.timeStart",
                "crvcoincs.timeEnd"
            ],
            "calo" : [
               "caloclusters.energyDep_"
            ],
            "trk" : [
                "trk.nactive", 
                "trk.pdg", 
                "trk.status",
                "trk.goodfit",
                "trk.opainter",
                "trk.chisq",
                "trk.ndof",
                "trkqual.valid",
                "trkqual.result",
                "trkpid.valid",
                "trkpid.result",
                "trk.fitcon"
            ],
            "trkfit" : [
                "trksegs",
                "trksegsmc",
                "trksegpars_lh"
            ],
            "trkmc" : [
              "trkmcsim",
              "trkmc.valid"
            ]
        }
        self.tree_path = "EventNtuple/ntuple"
        #self.filelist = "filelist.txt"          # text file containing list of files
        self.use_remote = True     # Use remote file via mdh
        if str(location)  == "local":
          self.use_remote = False
        self.location = str(location)     # File location
        self.max_workers = jobs      # Limit the number of workers
        self.verbosity = 2         # Set verbosity 
        self.use_processes = True  # Use processes rather than threads
        #self.schema = "path"
        
        # Now add your own analysis-specific parameters
        self.sign = sign  # Store sign for use in postprocessing
        self.proctype = proctype  # Store proctype for use in output names

        # Init analysis methods
        # Would be good to load an analysis config here 
        self.analyse = Analyze(verbosity=0, sign=sign, cut_switch=cuts)
            
        # Custom prefix for log messages from this processor
        self.logger = Logger(print_prefix="[AnaProcessor]", verbosity=1)
        self.logger.log("Initialised", "info")
    
    # ==========================================
    # Define the core processing logic
    # ==========================================
    # This method overrides the parent class's process_file method
    # It will be called automatically for each file by the execute method
    def process_file(self, file_name): 
        """Process a single ROOT file with timeout protection
        
        This method will be called for each file in our list.
        It extracts data, processes it, and returns a result.
        If a file takes too long, it will be skipped.
        
        Args:
            file_name: Path to the ROOT file to process
            
        Returns:
            A dict with processing results, or None if timeout/error
        """
        import time
        import psutil
        import os
        import threading
        import queue
        
        file_start = time.time()
        timeout_seconds = 270  # 1 minute timeout per file - if file hangs, skip it
        
        # Extract just the filename for cleaner logging
        just_filename = file_name.split('/')[-1] if '/' in file_name else file_name
        
        # Queue to capture result from worker thread
        result_queue = queue.Queue()
        
        def worker():
            """Process file in a separate thread so we can timeout"""
            try:
                # Get memory info
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Write status to file so we can track which file is being processed
                try:
                    with open('_processing_status.txt', 'w') as f:
                        f.write(f"{just_filename}\n{time.time()}\n")
                except:
                    pass  # Ignore if we can't write status
                
                self.logger.log(f"[FILE {just_filename}] Starting (mem: {mem_before:.0f}MB)", "info")
                
                # Create a local pyprocess Processor to extract data from this file
                processor = Processor(
                    use_remote=self.use_remote,     # Use remote file via mdh
                    location=self.location,         # File location
                    verbosity=self.verbosity        # Reduce output in worker threads
                )
                
                # Process the files using multithreading
                self.logger.log(f"[FILE {just_filename}] Extracting data...", "debug")
                extract_start = time.time()
                data = processor.process_data(
                    file_name = file_name,
                    branches = self.branches
                )
                extract_time = time.time() - extract_start
                
                self.logger.log(f"[FILE {just_filename}] Data extracted in {extract_time:.1f}s, running analysis...", "debug")
                
                # ---- Analysis ----            
                analysis_start = time.time()
                results = self.analyse.execute(data, file_name)
                analysis_time = time.time() - analysis_start
                
                # Debug: check result type
                if not isinstance(results, dict):
                    self.logger.log(f"[FILE {just_filename}] WARNING: analyse.execute() returned {type(results).__name__}, expected dict!", "warning")
                    if isinstance(results, tuple) and len(results) == 2:
                        self.logger.log(f"[FILE {just_filename}] Got tuple with {len(results)} elements. First element type: {type(results[0]).__name__}", "warning")
                
                elapsed = time.time() - file_start
                
                # Get memory after
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                mem_delta = mem_after - mem_before
                
                self.logger.log(f"[FILE {just_filename}] ✓ Complete in {elapsed:.1f}s (extract:{extract_time:.1f}s, analysis:{analysis_time:.1f}s, mem: {mem_before:.0f}→{mem_after:.0f}MB, +{mem_delta:.0f}MB)", "info")
                
                # Clean up local data
                del data, processor
                gc.collect()
                
                result_queue.put(('success', results))
                
            except Exception as e:
                # Handle any errors that occur during processing
                elapsed = time.time() - file_start
                self.logger.log(f"[FILE {just_filename}] ✗ ERROR after {elapsed:.1f}s: {e}", "error")
                result_queue.put(('error', None))
        
        # Start worker thread
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        
        # Wait for result with timeout
        try:
            status, result = result_queue.get(timeout=timeout_seconds)
            return result
        except queue.Empty:
            # Timeout occurred - file took too long
            elapsed = time.time() - file_start
            self.logger.log(f"[FILE {just_filename}] ⏱ TIMEOUT after {elapsed:.1f}s - skipping this file", "warning")
            return None  # Return None to skip this file


    def postprocess(self, results):
        """Combine results from all processed files

        Overrides Skeleton.postprocess() to combine filtered arrays
        and cut flows from individual file results.

        Args:
            results: List of dicts from process_file, each with "filtered_data" and "cut_stats"

        Returns:
            dict: Combined data and cut flow
        """
        import time
        
        if not results:
            return None

        print(f"\n[postprocess] Starting postprocessing of {len(results)} file results...")
        
        # Combine filtered arrays
        arrays_to_combine = []
        cut_flow_list = []
        
        print(f"[postprocess] Filtering results...")
        start_filter = time.time()
        skipped_count = 0
        for i, result in enumerate(results):
            if result is None or len(result) == 0:
                skipped_count += 1
                continue
            
            # Debug: check result type
            if not isinstance(result, dict):
                print(f"[postprocess] WARNING: Result {i} is type {type(result).__name__}, expected dict. Skipping.")
                skipped_count += 1
                continue
                
            if "filtered_data" not in result or "cut_stats" not in result:
                print(f"[postprocess] WARNING: Result {i} missing expected keys. Has keys: {list(result.keys())}. Skipping.")
                skipped_count += 1
                continue
            
            arrays_to_combine.append(result["filtered_data"])
            cut_flow_list.append(result["cut_stats"])
        
        if skipped_count > 0:
            print(f"[postprocess] ⏱ Filtered {len(arrays_to_combine)} valid results in {time.time()-start_filter:.1f}s ({skipped_count} files skipped due to timeout)")
        else:
            print(f"[postprocess] Filtered {len(arrays_to_combine)} valid results in {time.time()-start_filter:.1f}s")

        print(f"[postprocess] Concatenating {len(arrays_to_combine)} arrays...")
        start_concat = time.time()
        try:
            combined_data = ak.concatenate(arrays_to_combine) if arrays_to_combine else None
            print(f"[postprocess] Concatenation completed in {time.time()-start_concat:.1f}s")
        except Exception as e:
            print(f"[postprocess] ERROR during concatenation: {e}")
            print(f"[postprocess] Array sizes: {[len(arr) if hasattr(arr, '__len__') else 'unknown' for arr in arrays_to_combine]}")
            raise

        # Combine cut flows using CutManager
        print(f"[postprocess] Combining {len(cut_flow_list)} cut flows...")
        start_cutflow = time.time()
        try:
            cut_manager = CutManager(verbosity=0)
            combined_cut_flow = cut_manager.combine_cut_flows(cut_flow_list, format_as_df=False)
            print(f"[postprocess] Cut flows combined in {time.time()-start_cutflow:.1f}s")
        except Exception as e:
            print(f"[postprocess] ERROR during cut flow combination: {e}")
            raise
            
        df = cut_manager.format_cut_flow(combined_cut_flow)
        print("================== Total Cut Flow =======================")
        print(df.to_string(index=False))
        
        # Create filename with sign prefix
        prefix = "eplus" if str(self.sign).lower() == "plus" else "eminus"
        filename = f"{prefix}_{self.proctype}_cut_stats.csv"
        df.to_csv(filename, index=False)
        
        # Force garbage collection to free memory
        gc.collect()
        print(f"[postprocess] Postprocessing complete")

        return {
            "combined_data": combined_data,
            "combined_cut_flow": combined_cut_flow
        }

def count_particle_types(data, logger=None):
    """
    Counts the occurrences of different particle types based on
    simulation data, leveraging the properties of Awkward Arrays.

    Args:
        data (ak.Array): An Awkward Array containing simulation data,
                         including 'trkmc' with 'trkmcsim' nested field.
        logger: Optional logger instance for output

    Returns:
        list: A list containing particle type identifiers for each event.
    """
    if logger is None:
        logger = Logger(print_prefix="[count_particle_types]", verbosity=1)

    # Check for empty data
    if ak.num(data['trkmc'], axis=0) == 0:
        logger.log("No events found in the data.", "warning")
        return []

    # Vectorized approach for efficiency using Awkward Array operations
    #  This is generally faster than looping through events individually for large datasets.

    # Get startCode for the first track in each event, handling empty lists
    # Use ak.firsts to safely get the first element or None if the list is empty
    proc_codes = ak.firsts(data['trkmc']['trkmcsim', 'startCode'], axis=1) 
    gen_codes = ak.firsts(data['trkmc']['trkmcsim', 'gen'], axis=1)
    vector = Vector()

    #rhos = vector.get_rho(data['trkmc','trkmcsim'],'pos')
    vec = vector.get_vector(branch=data['trkmc','trkmcsim'],vector_name='pos')
    rhos = vec.rho
    position = ak.firsts(rhos, axis=1) 

    #position = ak.firsts(sim_pos_vec.rho, axis = 1)
    # Use vectorized comparisons and selection for counting
    dio_mask = (proc_codes == 166) & (position <= 75) # Create boolean mask for DIO events
    ipa_mask = (proc_codes == 166) & (position > 75) # Create boolean mask for IPA DIO events
    cem_mask = ((proc_codes == 168)  | (proc_codes == 167)  ) # Create boolean mask for CE events
    cep_mask = ((proc_codes == 176) | (proc_codes == 169) )  # Create boolean mask for CE events
    erpc_mask = (proc_codes == 178)  # Create boolean mask for external RPC events
    irpc_mask = (proc_codes == 179)  # Create boolean mask for internal RPC events
    ermc_mask = (proc_codes == 172)  # Create boolean mask for external RMC events
    irmc_mask = (proc_codes == 171)  # Create boolean mask for internal RMC events
    flate_mask = (proc_codes == 173)  # Create boolean mask for internal flate events
    flateplus_mask = (proc_codes == 174)  # Create boolean mask for internal flate events
    cosmic_mask = ((gen_codes == 44) | (gen_codes == 38))  # Create boolean mask for cosmic events
    #combined_rpc_mask = (proc_codes == 178) |  (proc_codes == 179) # Create boolean mask for all RPC events

    # Combine masks to identify 'other' events
    other_mask = ~(dio_mask | cem_mask | erpc_mask | irpc_mask | cosmic_mask | ipa_mask | irmc_mask | ermc_mask | cep_mask)

    # Initialize particle_count with -2 for 'others'
    particle_count = ak.zeros_like(proc_codes, dtype=int) - 2
    
    # Assign particle types based on masks
    particle_count = ak.where(dio_mask, 166, particle_count)
    particle_count = ak.where(ipa_mask, 0, particle_count)
    particle_count = ak.where(cosmic_mask, -1, particle_count)
    particle_count = ak.where(other_mask, -2, particle_count)
    particle_count = ak.where(irpc_mask, 179, particle_count)
    particle_count = ak.where(erpc_mask, 178, particle_count)
    particle_count = ak.where(irmc_mask, 171, particle_count)
    particle_count = ak.where(ermc_mask, 172, particle_count)
    particle_count = ak.where(cem_mask, 168, particle_count)
    particle_count = ak.where(cep_mask, 176, particle_count)
    particle_count = ak.where(flate_mask, 173, particle_count)
    particle_count = ak.where(flateplus_mask, 174, particle_count)
    #particle_count = ak.where(combined_rpc_mask, 999, particle_count)
    particle_count_return = particle_count
    #particle_count = ak.any(dio_mask, axis=1)
    # Count the occurrences of each particle type
    counts = {
        166: (len(particle_count[ak.any(dio_mask, axis=1)==True])),
        0: (len(particle_count[ak.any(ipa_mask, axis=1)==True])),
        168:  (len(particle_count[ak.any(cem_mask, axis=1)==True])),
        176:  (len(particle_count[ak.any(cep_mask, axis=1)==True])),
        178:  (len(particle_count[ak.any(erpc_mask, axis=1)==True])),
        179:  (len(particle_count[ak.any(irpc_mask, axis=1)==True])),
        171:  (len(particle_count[ak.any(irmc_mask, axis=1)==True])),
        172:  (len(particle_count[ak.any(ermc_mask, axis=1)==True])), 
        173:  (len(particle_count[ak.any(flate_mask, axis=1)==True])),
        174:  (len(particle_count[ak.any(flateplus_mask, axis=1)==True])), 
        -1:  (len(particle_count[ak.any(cosmic_mask, axis=1)==True])),
        -2:  (len(particle_count[ak.any(other_mask, axis=1)==True])),
        #999: (len(particle_count[ak.any(combined_rpc_mask, axis=1)==True])),
    }
      
    # Print the yields to terminal for cross-check
    logger.log("===== MC truth yields for full momentum and time range=====", "info")
    logger.log(f"N_DIO: {counts[166]}", "info")
    logger.log(f"N_IPA: {counts[0]}", "info")
    logger.log(f"N_CEM: {counts[168]}", "info")
    logger.log(f"N_CEP: {counts[176]}", "info")
    logger.log(f"N_eRPC: {counts[178]}", "info")
    logger.log(f"N_iRPC: {counts[179]}", "info")
    logger.log(f"N_eRMC: {counts[171]}", "info")
    logger.log(f"N_iRMC: {counts[172]}", "info")
    logger.log(f"N_flateminus: {counts[173]}", "info")
    logger.log(f"N_flateplus: {counts[174]}", "info")
    logger.log(f"N_cosmic: {counts[-1]}", "info")
    logger.log(f"N_others: {counts[-2]}", "info")
    
    # Now return a 1D list with one element per event corresponding to the primary trk
    #particle_count_return = ak.flatten(particle_count_return, axis=None)
    #    The mask will be True for values that are not -2.
    primary_mask = particle_count_return != -2

    # Apply the mask to the flattened array to select desired elements
    particle_count_return = particle_count_return[primary_mask]
    particle_count_return = [[sublist[0]] for sublist in particle_count_return]
    particle_count_return = ak.flatten(particle_count_return, axis=None)
    logger.log(f"returned particle count length {len(particle_count_return)}", "info")
    
    return particle_count_return, counts


def compare_datasets( files, cuts, locations, columns, signs):
    """
    Allows for different types of comparisons:
    
    1) could compare different files same cuts
    2) could compare same file different cut sets
    
    Args:
      files : list of file lists (.txt files)
      cuts : list of cut switches (True/False of each cut)
      locations : list of locations e.g. tape or disk
      columns : labels for the two things you are comparing eg. [dataset 1, dataset 2]
    """
    logger = Logger(print_prefix="[compare_datasets]", verbosity=1)

    rmax = []
    d0 = []
    tanDip = []
    t0err = []
    active = []
    trkqual = []
    recomom = []
    truemom = []
    mc_count = []
    resolutions = []
    nST = []
    nOPA = []
    originmom = []
    losses = []
    times = []
    crv = []
    trkpid = []

    chisqs = []
    ndofs = []
    chiperdof = []
    
    
    comparison = Compare()
    #fit = Fits()
    for i, fil in enumerate(files):
      ana_processor = AnaProcessor(fil, args.jobs, signs[i], cuts[i], locations[i])
      results = ana_processor.execute()
      combine_result = results["combined_data"]

      # run cat
      mc_count_array, _ = count_particle_types(combine_result, logger)
      mc_count.append(mc_count_array)

      selector = Select()
      
      # select only track front to fit to
      trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")

      # did the track intersect the ST?
      has_st  = selector.has_ST(combine_result['trkfit'])

      # did the track intersect the OPA?
      no_opa  = selector.has_OPA(combine_result['trkfit'])

      # combined mask
      trkfit_ent = ak.mask(combine_result['trkfit']["trksegs"], trk_front) #combine_result['trkfit']["trksegs"].mask[(trk_front) ] #& (no_opa) & (has_st)
          
      trk_front_mc = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front",branch_name="trksegsmc")
      trkfit_ent_mc = ak.mask(combine_result['trkfit']["trksegsmc"], trk_front_mc)#combine_result['trkfit']["trksegsmc"].mask[(trk_front_mc) ]

      # make vector mag branch
      vector = Vector()
      mom_mag = vector.get_mag(trkfit_ent ,'mom')
      
      
      #mom_mag = ak.nan_to_none(mom_mag)
      #mom_mag = ak.drop_none(mom_mag)

      time = ak.nan_to_none(trkfit_ent['time'])
      time = ak.drop_none(time)
      
      # save reconstructed momentum magnitudes for this dataset to CSV
      # write fitted-range data using WriteFittedData
      try:
        WriteFittedData(mom_mag, time, 95, 110)
      except Exception as e:
        print(f"WriteFittedData failed: {e}")


      vector = Vector()
      mom_mag_mc = vector.get_mag(trkfit_ent_mc ,'mom')
      
      # get resolution:
      resolution = comparison.compare_resolution(mom_mag,mom_mag_mc)
      
      # for loss studies:
      origin = ak.mask(combine_result['trkmc']["trkmcsim"] , (combine_result['trkmc']["trkmcsim"]["rank"] == 0) & (combine_result['trkmc']["trkmcsim"]["nhits"] > 0))
      originmom.append((vector.get_mag(origin,'mom')))

      # get resolution:
      resolution = comparison.compare_resolution(mom_mag_mc, mom_mag)
      loss  = comparison.compare_resolution( (vector.get_mag(origin,'mom')), mom_mag_mc)
      
      # plot cut distributions
      test_mask = (trk_front) & (has_st) #& (no_opa)& (has_st)
      

      print_passing_events(combine_result, test_mask, output_file="passing_events_count.txt")

      # for CRV:
      # Get track and coincidence times
      trk_times = combine_result['trkfit']["trksegs"]["time"][trk_front]  # events × tracks × segments
      coinc_times = combine_result["crv"]["crvcoincs.time"]                  # events × coincidences

      coinc_broadcast = coinc_times[:, None, None, :]  # Add dimensions for tracks and segments
      trk_broadcast = trk_times[:, :, :, None]         # Add dimension for coincidences

      # Calculate time differences
      dt = abs(trk_broadcast - coinc_broadcast)
      
      


      nST.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="ST_Foils"), axis=-1))
      nOPA.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="OPA"), axis=-1))
      rmax.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['maxr']) 
      d0.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['d0']) 
      tanDip.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['tanDip']) 
      t0err.append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['t0err']) 
      trkqual.append(ak.mask(combine_result['trk'],test_mask)["trkqual.result"])
      trkpid.append(ak.mask(combine_result['trk'],test_mask)["trkpid.result"])
      active.append(ak.mask(combine_result['trk'],test_mask)["trk.nactive"])
      chisqs_masked = ak.mask(combine_result['trk'],test_mask)["trk.chisq"]
      ndofs_masked = ak.mask(combine_result['trk'],test_mask)["trk.ndof"]
      chisqs.append(chisqs_masked)
      ndofs.append(ndofs_masked)
      chiperdof.append(chisqs_masked / ndofs_masked)

      recomom.append(mom_mag)
      times.append(time)
      losses.append(loss)
      truemom.append(mom_mag_mc)
      resolutions.append(resolution)
      crv.append(dt)
    #cosmics.fit_momentum(recomom)
    prefix = "eplus" if str(signs[i]).lower() == "plus" else "eminus"
    comparison.plot_particle_counts(mc_count, columns, plot_prefix=prefix)
    
    if signs[i] == "minus":
       startmom = 95
       endmom = 115
       nbins = 34
       comparison.plot_variable(recomom, r"Reconstructed Momentum [MeV/c]",f"{prefix}_recomom", startmom, endmom, [103.34,103.34],[104.74,104.74], mc_count,columns, nbins=nbins)
    else:
        startmom = 95
        endmom = 115
        nbins = 34
        comparison.plot_variable(recomom, r"Reconstructed Momentum [MeV/c]",f"{prefix}_recomom", startmom, endmom, [0,0],[0,0], mc_count,columns, nbins=nbins)
    comparison.plot_variable(crv, "|DT| [ns]",f"{prefix}_DT",0,300, [150,150],[150,150], mc_count,columns)
    comparison.plot_variable(nST, "nST",f"{prefix}_nST",0,15, [1,1],[1,1], mc_count,columns, 15)
    comparison.plot_variable(nOPA, "nOPA",f"{prefix}_nOPA",0,4, [0,0],[0,0], mc_count,columns,4)
    comparison.plot_variable(rmax, "rmax", f"{prefix}_rmax",300,750, [450,450],[680,680], mc_count,columns)
    comparison.plot_variable(d0, "d0", f"{prefix}_d0",-200, 250, [100,100], [100,100], mc_count,columns)
    comparison.plot_variable(tanDip, "tanDip",f"{prefix}_tanDip",-1,2.5, [0.557,0.557], [1.0,1.0],mc_count,columns)
    comparison.plot_variable(trkqual, "trkqual", f"{prefix}_trkqual", 0,1,[0.2,0.2], [0.2, 0.2], mc_count,columns)
    comparison.plot_variable(trkpid, "trkpid", f"{prefix}_trkpid", 0,1,[0.6,0.6], [0.6, 0.6], mc_count,columns)
    comparison.plot_variable(t0err, "t0err",f"{prefix}_t0err", 0,1, [0.9,0.9],[0.9,0.9], mc_count,columns)
    comparison.plot_variable(active, "nactive",f"{prefix}_nactive", 0,50, [20,20],[0.9,0.9], mc_count,columns)
    comparison.plot_variable(chisqs, "chisq",f"{prefix}_chi", 0,200, [1,1],[1,1], mc_count,columns)
    comparison.plot_variable(ndofs, "ndof",f"{prefix}_ndof", 0,50, [1,1],[1,1], mc_count,columns)
    comparison.plot_variable(chiperdof, "chi2/dof",f"{prefix}_chidof", 0,10, [0,0],[0,0], mc_count,columns)

    
    comparison.plot_variable(times, "Time at TrkEnt [ns]",f"{prefix}_time", 0, 1700, [640,640],[1650,1650], mc_count,columns)
    comparison.plot_variable(truemom, "True Momentum at TrkEnt [MeV/c]",f"{prefix}_truemom", startmom, endmom, [103.9,103.9],[105.1,105.1], mc_count,columns)


def fit_dataset(files, cuts, locations, columns, signs, proctype):
    """
    Allows for different types of comparisons:
    
    1) could compare different files same cuts
    2) could compare same file different cut sets
    
    Args:
      files : list of file lists (.txt files)
      cuts : list of cut switches (True/False of each cut)
      locations : list of locations e.g. tape or disk
      columns : labels for the two things you are comparing eg. [dataset 1, dataset 2]
    """
    logger = Logger(print_prefix="[compare_datasets]", verbosity=1)

    recomom = []
    mc_count = []
    comparison = Compare()
    resolutions = []
    resolutions_origin = []
    originmom = []
    losses = []
    times = []
    truemom = []
    #fit = Fits()
    for i, fil in enumerate(files):
      ana_processor = AnaProcessor(fil, args.jobs, signs[i], cuts[i], locations[i], proctype)
      results = ana_processor.execute()
      combine_result = results["combined_data"]

      

      # run cat
      mc_count_array, _ = count_particle_types(combine_result, logger)
      mc_count.append(mc_count_array)

      selector = Select()
      
      # select only track front to fit to
      trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")

      # did the track intersect the ST?
      has_st  = selector.has_ST(combine_result['trkfit'])

      # did the track intersect the OPA?
      no_opa  = selector.has_OPA(combine_result['trkfit'])

      # combined mask
      trkfit_ent = ak.mask(combine_result['trkfit']["trksegs"], trk_front) #combine_result['trkfit']["trksegs"].mask[(trk_front) ] #& (no_opa) & (has_st)

      # make vector mag branch
      vector = Vector()
      mom_mag = vector.get_mag(trkfit_ent ,'mom')
      recomom.append(mom_mag)
      trk_front_mc = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front",branch_name="trksegsmc")
      trkfit_ent_mc = ak.mask(combine_result['trkfit']["trksegsmc"], trk_front_mc)#combine_result['trkfit']["trksegsmc"].mask[(trk_front_mc) ]

      
      mom_mag_mc = vector.get_mag(trkfit_ent_mc ,'mom')

      time = ak.nan_to_none(trkfit_ent['time'])
      time = ak.drop_none(time)
      times.append(time)


    if proctype == "ensemble":
        WriteFittedData(recomom, times, 90, 120)
    if proctype == "cosmics":
        cosmics = Cosmics()
        cosmics.fit_momentum(recomom)
        cosmics.fit_time(times, columns)
    if proctype == "rpc":
        rpc = RPC()
        rpc.fit_momentum(recomom, columns, opt="poly")
        #rpc.CR_momentum(recomom, columns)
        rpc.fit_time(times, columns)
    if proctype == "rmc":
        rmc = RMC()
        rmc.fit_momentum(recomom, columns)
    # Generate RLE calibration parameters for ensemble
    if proctype == "rle":
        rle_results = generate_rle_calibration(combine_result, "RLE/common", run_fits=True)
    if proctype == "rle_no_fit":
        rle_results = generate_rle_calibration(combine_result, "RLE/common", run_fits=False)
    if proctype == "rle_v2":
            rlev2 = RLE_v2()
            # get resolution:
            # for loss studies:
            origin = ak.mask(combine_result['trkmc']["trkmcsim"] , (combine_result['trkmc']["trkmcsim"]["rank"] == 0) & (combine_result['trkmc']["trkmcsim"]["nhits"] > 0))
            originmom.append((vector.get_mag(origin,'mom')))
            resolution = comparison.compare_resolution(mom_mag_mc, mom_mag)
            resolution_origin = comparison.compare_resolution(mom_mag,(vector.get_mag(origin,'mom')))
            loss  = comparison.compare_resolution( (vector.get_mag(origin,'mom')), mom_mag_mc)
            #times.append(time)
            losses.append(loss)
            truemom.append(mom_mag_mc)
            resolutions_origin.append(resolution_origin)
            resolutions.append(resolution)
            rlev2.fit_momentum(originmom, 90, 120, opt="poly", label = "Origin Momentum [MeV/c]")
            rlev2.fit_momentum(losses, -5,0,opt="landau",label = "Origin Momentum - True Momentum at TrkEnt [MeV/c]")
            #rlev2.fit_time(times, 450,1695,opt="piexp",label = "time at TrkFront [ns]")
            rlev2.fit_momentum(resolutions, -2,1,opt="dscb", label = "Reco - True Momentum at TrkEnt [MeV/c]")
    
    # Apply RLE convolution to CeLL theory spectrum
    if proctype == "convolution":
        logger = Logger(print_prefix="[main:convolution]", verbosity=1)
        logger.log("Applying RLE convolution to CeLL spectrum...", "info")
        result = apply_ce_rle_convolution(
            calibration_path="RLE/common/calibration.json",
            mom_range=(95, 110),
            binwidth=0.1,
            output_plot="RLE/common/convolution_result.png"
        )
        if result:
            logger.log("Convolution successful", "success")
            # Could save result to file for later use
        else:
            logger.log("Convolution failed", "error")
    
    # Overlay RLE-convolved theory on reconstructed data
    if proctype == "overlay":
        logger = Logger(print_prefix="[main:overlay]", verbosity=1)
        logger.log("Overlaying RLE-convolved theory on reco data with fitting...", "info")
        
        # Flatten reconstructed momenta from all files
        reco_flat = ak.flatten(ak.concatenate(recomom), axis=None)
        reco_array = np.array(reco_flat)
        
        logger.log(f"Total reco events: {len(reco_array)}", "info")
        
        overlay_result = overlay_convolved_theory_on_reco_with_constraints(
            reco_momenta=reco_array,
            calibration_path="RLE/common/calibration.json",
            mom_range=(95, 110),
            binwidth=0.1,
            constraint_margin=0.20,  # Moderate ±20% margin
            do_fit=True,
            output_plot="RLE/common/reco_theory_overlay.png"
        )
        
        if overlay_result:
            logger.log(f"Overlay successful: {overlay_result['n_events']} events plotted", "success")
            if overlay_result.get('fit_result'):
                logger.log(f"  Fit: χ²/dof={overlay_result['fit_result']['chi2_per_dof']:.4f}, "
                          f"scale={overlay_result['fit_result']['scale_factor']:.6f}", "info")

        else:
            logger.log("Overlay failed", "error")

    if proctype == "eff": # flat eff
        rle = RLE_v2()
        origin = ak.mask(combine_result['trkmc']["trkmcsim"] , (combine_result['trkmc']["trkmcsim"]["rank"] == 0) & (combine_result['trkmc']["trkmcsim"]["nhits"] > 0))
        originmom.append((vector.get_mag(origin,'mom')))
            
        rle.fit_momentum(originmom, 90,120,opt="poly", label = r"$p_{gen}$ [MeV/c]",nbins=50)
        rle.fit_momentum(originmom, 101,109,opt="linear", label = r"$p_{gen}$ [MeV/c]",nbins=20)

    if proctype == "mutime":
        rle = RLE_v2()
        rle.fit_time(times, columns)

    if proctype == "CE": #endpoint
        rle = RLE_v2()
        #rle.fit_momentum(resolutions_origin, -5,5,opt="dscb", label = r"$(p_{reco} - p_{gen})$ [MeV/c]")

        convolved = rle.convolve_theory_with_resolution(theory_pdf=None, momentum_range=(90, 110), 
                                        proctype='CE', theory_params=None,
                                        data_list=recomom,
                                        calibration_path='RLE/common/calibration.json',
                                        floating_params=False, plot_label='convolution')
    if proctype == "DIO":
        rle = RLE_v2()
        #rle.fit_momentum(resolutions_origin, -5,5,opt="dscb", label = r"$(p_{reco} - p_{gen})$ [MeV/c]")

        #convolved = rle.convolve_theory_with_resolution(theory_pdf=None, momentum_range=(95, 115), 
        #                                proctype='DIO', theory_params=None,
        #                                data_list=recomom,
        #                                calibration_path='RLE/common/calibration.json',
        #                                floating_params=False, plot_label='convolution')

        convolved = rle.fit_theory_with_resolution_with_fit(theory_pdf=None, momentum_range=(95, 115), 
                                   proctype='DIO', data_list=recomom,
                                   calibration_path='RLE/common/calibration.json',
                                   error_path='RLE/common/calibration_errors.json',
                                   plot_label='fit_result')
    if proctype == "CELL": 
        rle = RLE_v2()
        rle.fit_CELL_momentum_dscb(recomom, 100,106,opt="dscb", label = r"$p_{reco} $ [MeV/c]",nbins=100)
        #rle.fit_momentum(recomom, 85,115,opt="dscb", label = r"$p_{reco} $ [MeV/c]",nbins=50)



def WriteFittedData(data, time_data, min_v, max_v):
    """ Write data used in fit to csv (i,mom,time) Note: should be in format useful to BAT"""
    flat_mom = ak.flatten(data, axis = None)
    flat_np = np.array(flat_mom)

    # Create a boolean mask where elements are greater than or equal to 85
    mask = (flat_np >= min_v) & (flat_np < max_v)

    # Use the mask to filter the array and keep only the elements where the mask is True
    filtered_array = flat_np[mask]
    file_path = 'output_data.csv'

    with open(file_path , 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for item in filtered_array:
            csv_writer.writerow([item])

def print_passing_events(combine_result, cut_mask, output_file="passing_events.txt"):
    """
    Extract and print run/subrun/event for tracks that pass all cuts
    
    Args:
        combine_result: Full combined data from AnaProcessor
        cut_mask: Boolean mask indicating which events/tracks pass all cuts (can be jagged)
        output_file: Output filename for event list
    """
    # Reduce the jagged mask to event level systematically
    # Keep applying ak.any() until we reach 1D (event level)
    event_mask = ak.Array(cut_mask)
    
    # Reduce all dimensions except the first (events)
    while event_mask.ndim > 1:
        event_mask = ak.any(event_mask, axis=-1)
    
    # Now use awkward indexing directly (no numpy conversion)
    runs = combine_result['evt']['run'][event_mask]
    subruns = combine_result['evt']['subrun'][event_mask]
    events = combine_result['evt']['event'][event_mask]
    
    # Flatten to 1D for writing
    runs_flat = ak.flatten(runs, axis=None)
    subruns_flat = ak.flatten(subruns, axis=None)
    events_flat = ak.flatten(events, axis=None)
    
    # Convert to numpy for file writing
    runs_np = np.asarray(runs_flat)
    subruns_np = np.asarray(subruns_flat)
    events_np = np.asarray(events_flat)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("run,subrun,event\n")
        for run, subrun, event in zip(runs_np, subruns_np, events_np):
            f.write(f"{int(run)},{int(subrun)},{int(event)}\n")
    
    print(f"Wrote {len(runs_np)} passing events to {output_file}")
    return runs_np, subruns_np, events_np


def optimize_cut_variable(combine_result, variable_data, signal_code=168, background_codes=None, 
                          direction='greater', n_steps=200, metric='youden', 
                          variable_name='variable', output_prefix=None, save_csv=False, make_plots=False):
    """
    Optimize a cut on a variable using signal and background data.
    
    Shows background and signal efficiencies and reports optimal cuts.
    
    Args:
        combine_result: Combined analysis result containing mc_count data
        variable_data: The variable values to optimize on (awkward or numpy array)
        signal_code: MC code identifying signal (default 168 for CE)
        background_codes: List of background MC codes; if None, all non-signal codes used
        direction: 'greater' to keep values >= threshold, 'less' for <=
        n_steps: Number of threshold steps to scan
        metric: Optimization metric ('youden' or 's_over_sqrtb')
        variable_name: Name of variable for output formatting
        output_prefix: Prefix for output files (CSV and plots)
        save_csv: If True, save scan results to CSV file
        make_plots: If True, create plots of scan results
        
    Returns:
        Dictionary with:
            - 'rows': All scan results
            - 'best': Best threshold result
            - 'signal_eff': Signal efficiency at optimum
            - 'bkg_eff': Background efficiency at optimum
            - 'optimal_threshold': Optimal threshold value
    """
    logger = Logger(print_prefix=f"[optimize_cut_variable: {variable_name}]", verbosity=1)
    
    # Get mc_count array
    mc_count_array, mc_counts_dict = count_particle_types(combine_result, logger=logger)
    
    logger.log(f"Optimizing cut on {variable_name}...", "info")
    
    # Run optimization using optimize_cuts
    rows, best = optimize_cuts.optimize_from_event_arrays(
        variable_data, 
        mc_count_array, 
        signal_code=signal_code, 
        background_codes=background_codes,
        direction=direction, 
        n_steps=n_steps, 
        metric=metric
    )
    
    if best is None:
        logger.log(f"No valid threshold found for {variable_name}", "warning")
        return None
    
    # Extract efficiency information
    signal_eff = best['tpr']
    bkg_eff = 1.0 - best['bkg_rej']  # Background efficiency = 1 - background rejection
    optimal_threshold = best['threshold']
    
    # Print results
    logger.log(f"═" * 60, "info")
    logger.log(f"Optimization Results for: {variable_name}", "info")
    logger.log(f"═" * 60, "info")
    logger.log(f"Optimal Threshold:        {optimal_threshold:.6g}", "info")
    logger.log(f"Signal Efficiency (TPR):  {signal_eff:.4f} ({best['nsig_pass']} / {best['nsig_pass'] + (best['nsig_pass'] / signal_eff - best['nsig_pass']) if signal_eff > 0 else 'N/A'})", "info")
    logger.log(f"Background Efficiency:    {bkg_eff:.4f}", "info")
    logger.log(f"Background Rejection:     {best['bkg_rej']:.4f}", "info")
    logger.log(f"Optimization Metric:      {best['metric']:.6g}", "info")
    logger.log(f"Signal Pass Count:        {best['nsig_pass']}", "info")
    logger.log(f"Background Pass Count:    {best['nbkg_pass']}", "info")
    logger.log(f"═" * 60, "info")
    
    # Save CSV if requested
    if save_csv and output_prefix:
        csv_path = f"{output_prefix}_{variable_name}_scan.csv"
        optimize_cuts.save_csv(rows, csv_path)
        logger.log(f"Saved scan results to: {csv_path}", "info")
    
    # Create plots if requested
    if make_plots and output_prefix:
        # Plot efficiency vs background rejection
        plot1_path = f"{output_prefix}_{variable_name}_eff_vs_bkg.png"
        optimize_cuts.plot_scan(rows, plot1_path, show=False)
        logger.log(f"Saved efficiency plot to: {plot1_path}", "info")
        
        # Plot efficiency vs threshold value
        plot2_path = f"{output_prefix}_{variable_name}_eff_vs_value.png"
        optimize_cuts.plot_scan_vs_value(rows, plot2_path, show=False)
        logger.log(f"Saved threshold plot to: {plot2_path}", "info")
    
    result = {
        'rows': rows,
        'best': best,
        'signal_eff': signal_eff,
        'bkg_eff': bkg_eff,
        'optimal_threshold': optimal_threshold,
        'metric_value': best['metric']
    }
    
    return result


def optimize_multiple_cuts(combine_result, variables_dict, signal_code=168, background_codes=None,
                          direction='greater', n_steps=200, metric='youden', output_prefix=None,
                          save_csv=False, make_plots=False):
    """
    Optimize cuts on multiple variables and display summary.
    
    Args:
        combine_result: Combined analysis result containing mc_count data
        variables_dict: Dictionary mapping variable names to their data arrays
                       e.g., {'maxr': maxr_data, 'd0': d0_data, 'tanDip': tandip_data}
        signal_code: MC code identifying signal (default 168 for CE)
        background_codes: List of background MC codes
        direction: 'greater' or 'less'
        n_steps: Number of threshold steps
        metric: Optimization metric
        output_prefix: Prefix for output files
        save_csv: Save scan results to CSV
        make_plots: Create visualization plots
        
    Returns:
        Dictionary mapping variable names to optimization results
    """
    logger = Logger(print_prefix="[optimize_multiple_cuts]", verbosity=1)
    
    results = {}
    
    for var_name, var_data in variables_dict.items():
        logger.log(f"Processing variable: {var_name}", "info")
        result = optimize_cut_variable(
            combine_result=combine_result,
            variable_data=var_data,
            signal_code=signal_code,
            background_codes=background_codes,
            direction=direction,
            n_steps=n_steps,
            metric=metric,
            variable_name=var_name,
            output_prefix=output_prefix,
            save_csv=save_csv,
            make_plots=make_plots
        )
        if result:
            results[var_name] = result
    
    # Print summary table
    if results:
        logger.log("\n" + "═" * 80, "info")
        logger.log("SUMMARY OF ALL OPTIMIZED CUTS", "info")
        logger.log("═" * 80, "info")
        
        # Create summary table
        summary_data = []
        for var_name, res in results.items():
            summary_data.append({
                'Variable': var_name,
                'Optimal Cut': f"{res['optimal_threshold']:.6g}",
                'Signal Eff': f"{res['signal_eff']:.4f}",
                'Bkg Eff': f"{res['bkg_eff']:.4f}",
                'Metric': f"{res['metric_value']:.6g}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        logger.log("\n" + df_summary.to_string(index=False), "info")
        logger.log("═" * 80 + "\n", "info")
    
    return results


def run_cut_optimization(file_list_path, sign="minus", cuts=None, locations='disk', jobs=1, 
                        signal_code=168, background_codes=None, n_steps=200, metric='youden',
                        output_prefix='cut_optimization', save_csv=True, make_plots=True,
                        variables_to_optimize=None):
    """
    Driving function to run complete cut optimization workflow.
    
    Loads data, extracts variables, and optimizes cuts on multiple variables.
    
    Args:
        file_list_path: Path to file list for processing
        sign: Particle sign ('minus' or 'plus')
        cuts: List of boolean cuts to apply (if None, uses default for sign)
        locations: Data location ('disk' or 'tape')
        jobs: Number of parallel jobs
        signal_code: MC code for signal (default 168 for CE)
        background_codes: List of background MC codes (default None = all non-signal)
        n_steps: Number of threshold steps in scan
        metric: Optimization metric ('youden' or 's_over_sqrtb')
        output_prefix: Prefix for output files
        save_csv: Save scan results to CSV files
        make_plots: Create visualization plots
        variables_to_optimize: Dict of {var_name: var_data} to optimize
                               If None, uses default variables (maxr, d0, tanDip, etc.)
        
    Returns:
        Dictionary with all optimization results
    """
    logger = Logger(print_prefix="[run_cut_optimization]", verbosity=1)
    
    logger.log("╔" + "═" * 78 + "╗", "info")
    logger.log("║" + " CUT OPTIMIZATION WORKFLOW ".center(78) + "║", "info")
    logger.log("╚" + "═" * 78 + "╝", "info")
    
    # Set default cuts if not provided
    if cuts is None:
        if sign == "minus" or sign == "plus":
            cuts = [
                True,  # 0: has_a_track
                True,  # 1: is_good_track
                True,  # 2: has_trk_front_seg
                True,  # 3: is_reco_electron_or_positron
                True,  # 4: has_downstream
                True,  # 5: charge_selection
                True,  # 6: or_trigger
                True,  # 7: no_upstream
                True,  # 8: upstream_veto (timing-based)
                True,  # 9: no_multi_trk_veto
                True,  # 10: good_trkpid
                True,  # 11: pz_over_pt
                True,  # 12: st_boundary
                True,  # 13: has_st
                True,  # 14: no_opa
                True,  # 15: good_trkqual
                True,  # 16: has_hits
                True,  # 17: within_t0err
                True,  # 18: no_crv_veto
                True,  # 19: in_mom_range
                True,  # 20: within_t0_475
                True,  # 21: within_t0_540
                True   # 22: within_t0_640
            ]
        
    
    # Step 2: Extract variables for optimization
    logger.log(f"\nStep 2: Extracting variables for optimization", "info")
    
    selector = Select()
    vector = Vector()
    
    # Select track front intersection
    trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")
    
    # Surface masks
    has_st = selector.has_ST(combine_result['trkfit'])
    no_opa = selector.has_OPA(combine_result['trkfit'])
    
    test_mask = (trk_front) & (has_st)
    
    # Extract common track variables
    trkfit_ent = ak.mask(combine_result['trkfit']["trksegs"], test_mask)
    trksegpars = ak.mask(combine_result['trkfit']["trksegpars_lh"], test_mask)
    trk = ak.mask(combine_result['trk'], test_mask)
    
    # Get momentum
    mom_mag = vector.get_mag(trkfit_ent, 'mom')
    
    # Default variables to optimize if not provided
    if variables_to_optimize is None:
        variables_to_optimize = {
            'trkqual': trk["trkqual.result"],
            'trkpid': trk["trkpid.result"]
        }
    
    logger.log(f"Variables extracted: {list(variables_to_optimize.keys())}", "info")
    
    # Step 3: Run optimization on all variables
    logger.log(f"\nStep 3: Running optimization", "info")
    logger.log(f"Signal code: {signal_code}, Metric: {metric}, Scan steps: {n_steps}", "info")
    
    opt_results = optimize_multiple_cuts(
        combine_result=combine_result,
        variables_dict=variables_to_optimize,
        signal_code=signal_code,
        background_codes=background_codes,
        direction='greater',
        n_steps=n_steps,
        metric=metric,
        output_prefix=output_prefix,
        save_csv=save_csv,
        make_plots=make_plots
    )
    
    # Step 4: Save summary report
    logger.log(f"\nStep 4: Saving results", "info")
    
    summary_file = f"{output_prefix}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("CUT OPTIMIZATION SUMMARY REPORT\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"File List: {file_list_path}\n")
        f.write(f"Particle Sign: {sign}\n")
        f.write(f"Data Location: {locations}\n")
        f.write(f"Signal Code: {signal_code}\n")
        f.write(f"Background Codes: {background_codes if background_codes else 'All non-signal'}\n")
        f.write(f"Optimization Metric: {metric}\n")
        f.write(f"Scan Steps: {n_steps}\n")
        f.write(f"\n{'=' * 80}\n")
        f.write("OPTIMAL CUTS:\n")
        f.write(f"{'=' * 80}\n\n")
        
        for var_name in sorted(opt_results.keys()):
            res = opt_results[var_name]
            f.write(f"{var_name}:\n")
            f.write(f"  Threshold:           {res['optimal_threshold']:.6g}\n")
            f.write(f"  Signal Efficiency:   {res['signal_eff']:.6f}\n")
            f.write(f"  Background Eff:      {res['bkg_eff']:.6f}\n")
            f.write(f"  Metric Value:        {res['metric_value']:.6g}\n\n")
    
    logger.log(f"Summary saved to: {summary_file}", "info")
    
    # Step 5: Create Python cut configuration
    config_file = f"{output_prefix}_cuts.py"
    with open(config_file, 'w') as f:
        f.write("# Auto-generated cut configuration from optimization\n\n")
        f.write("optimized_cuts = {\n")
        for var_name in sorted(opt_results.keys()):
            res = opt_results[var_name]
            f.write(f"    '{var_name}': {res['optimal_threshold']:.6g},\n")
        f.write("}\n\n")
        f.write("efficiencies = {\n")
        for var_name in sorted(opt_results.keys()):
            res = opt_results[var_name]
            f.write(f"    '{var_name}': {{'signal': {res['signal_eff']:.6f}, 'background': {res['bkg_eff']:.6f}}},\n")
        f.write("}\n")
    
    logger.log(f"Cut configuration saved to: {config_file}", "info")
    
    logger.log(f"\n╔" + "═" * 78 + "╗", "info")
    logger.log("║" + " OPTIMIZATION COMPLETE ".center(78) + "║", "info")
    logger.log("╚" + "═" * 78 + "╝\n", "info")
    
    return opt_results

def run_multi_background_optimization(args):
    """Run simple TrkPID threshold scan: signal efficiency vs background rejection.
    
    Loads signal (CE) and background (Cosmics) samples, scans TrkPID cut,
    and plots signal efficiency vs background rejection curve.
    
    Args:
        args: Command line arguments with fields:
            - jobs: Number of parallel jobs
            - sign: Particle sign (minus/plus)
            - loc: Data location (disk/local)
    """
    from optimize_cuts import scan_thresholds
    
    # Pre-selection cuts to apply before optimization
    cuts = [
        True,  # 0 is_reco_electron
        True,  # 1 has_downstream
        True,  # 2 has trk front
        False,  # 3 good_trkqpid
        True,  # 4 good_trkqual
        True,  # 5 within_t0err
        True,  # 6 has_hits
        False,  # 7 within_lhr_maxl
        False,  # 8 within_d0
        False,  # 9 within_pitch_angle
        True,  # 10 has_st
        True,  # 11 st_boundary (NEW)
        True,  # 12 no_opa
        True,   # 13 no_crv_veto
        True,   # 14 no_crv_quality
        True,   # 15 no_crv_timewindow
        True,   # 16 pz/pt
        True,   # 17 triggers
        False,  # 18 in_mom_range
        False,  # 19 within_t0_early
        False,  # 20 no_reflected
        False,  # 21 within_t0
        False,   # 22 signal_region
        False   # 23 mlp_score
    ]
    
    print("\n" + "=" * 80)
    print("TrkPID THRESHOLD SCAN: Signal vs Cosmics")
    print("=" * 80)
    
    # Load signal sample
    print("\nLoading signal (CeMLL)...", end="", flush=True)
    processor_sig = AnaProcessor(
        file_list_path="file_lists_full/CeMLL_MDC2025an_best_nomix.txt",
        jobs=args.jobs,
        sign=args.sign,
        cuts=cuts,
        location=args.loc,
        proctype="ensemble"
    )
    results_sig = processor_sig.execute()
    sig_data = results_sig["combined_data"]
    print(f" {len(sig_data)} events")
    
    # Load background sample (Cosmics)
    print("Loading background (Cosmics)...", end="", flush=True)
    processor_bkg = AnaProcessor(
        file_list_path="file_lists_full/Cosimcs_MDC2025an_nomix.txt",
        jobs=args.jobs,
        sign=args.sign,
        cuts=cuts,
        location=args.loc,
        proctype="ensemble"
    )
    results_bkg = processor_bkg.execute()
    bkg_data = results_bkg["combined_data"]
    print(f" {len(bkg_data)} events")
    
    # Extract TrkPID values
    print("\nExtracting TrkPID values...")
    sig_trkpid = ak.flatten(sig_data['trk']['trkpid.result'], axis=None)
    bkg_trkpid = ak.flatten(bkg_data['trk']['trkpid.result'], axis=None)
    
    sig_trkpid = np.asarray(sig_trkpid)
    bkg_trkpid = np.asarray(bkg_trkpid)
    
    print(f"  Signal: {len(sig_trkpid)} tracks")
    print(f"  Cosmics: {len(bkg_trkpid)} tracks")
    
    # Print TrkPID range
    sig_min, sig_max = np.nanmin(sig_trkpid), np.nanmax(sig_trkpid)
    bkg_min, bkg_max = np.nanmin(bkg_trkpid), np.nanmax(bkg_trkpid)
    overall_min = min(sig_min, bkg_min)
    overall_max = max(sig_max, bkg_max)
    print(f"\nTrkPID range:")
    print(f"  Signal: [{sig_min:.6f}, {sig_max:.6f}]")
    print(f"  Cosmics: [{bkg_min:.6f}, {bkg_max:.6f}]")
    print(f"  Overall: [{overall_min:.6f}, {overall_max:.6f}]")
    
    # Scan thresholds using existing function
    print(f"\nScanning TrkPID thresholds ({500} steps)...")
    rows = scan_thresholds(sig_trkpid, bkg_trkpid, direction='greater', n_steps=500, metric='youden')
    
    # Create output directory
    import os
    os.makedirs("cut_optimization", exist_ok=True)
    
    # Sort rows by threshold for cleaner plots
    rows_sorted = sorted(rows, key=lambda r: r['threshold'])
    
    # Calculate best threshold once to ensure consistency across all plots
    best_youden_idx = np.argmax([r['metric'] for r in rows_sorted])
    best_row = rows_sorted[best_youden_idx]
    best_threshold = best_row['threshold']
    best_tpr_val = best_row['tpr']
    best_bkg_rej_val = best_row['bkg_rej']
    best_fpr_val = 1.0 - best_bkg_rej_val
    
    # Plot: TMVA-style plot with TrkPID on x-axis and two y-axes
    fig, ax1 = plt.subplots(figsize=(11, 8))
    
    thresholds = [r['threshold'] for r in rows_sorted]
    sig_effs = [r['tpr'] for r in rows_sorted]
    bkg_rejs = [r['bkg_rej'] for r in rows_sorted]
    
    # Plot signal efficiency on left y-axis
    color1 = 'steelblue'
    ax1.set_xlabel('TrkPID Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Signal Efficiency', fontsize=12, fontweight='bold', color=color1)
    line1 = ax1.plot(thresholds, sig_effs, color=color1, linewidth=2.5, marker='o', markersize=4, label='Signal Efficiency')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, 1.05])
    ax1.set_xlim([0, max(thresholds)])
    
    # Create right y-axis for background rejection
    ax2 = ax1.twinx()
    color2 = 'darkgreen'
    ax2.set_ylabel('Background Rejection', fontsize=12, fontweight='bold', color=color2)
    line2 = ax2.plot(thresholds, bkg_rejs, color=color2, linewidth=2.5, marker='s', markersize=4, label='Background Rejection')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 1.05])
    
    # Title
    ax1.set_title('TrkPID Cut Optimization (TMVA-style)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add dashed line at optimal threshold
    ax1.axvline(best_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal (Youden): {best_threshold:.4f}')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=11, loc='center left')
    
    plt.tight_layout()
    out_path = "cut_optimization/trkpid_scan.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {out_path}")
    
    # Plot: ROC Curve (True Positive Rate vs False Positive Rate)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create FPR/TPR pairs and sort by FPR for proper ROC curve
    roc_points = [(1.0 - r['bkg_rej'], r['tpr']) for r in rows_sorted]
    roc_points_sorted = sorted(roc_points, key=lambda x: x[0])  # Sort by FPR
    
    fpr_sorted = [p[0] for p in roc_points_sorted]
    tpr_sorted = [p[1] for p in roc_points_sorted]
    
    # Plot ROC curve
    ax.plot(fpr_sorted, tpr_sorted, color='darkblue', linewidth=2.5, marker='o', markersize=4, label='TrkPID ROC')
    
    # Add diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Random Classifier')
    
    # Mark optimal point using pre-calculated values
    ax.plot(best_fpr_val, best_tpr_val, marker='*', markersize=20, color='red', 
            label=f'Optimal (Youden): FPR={best_fpr_val:.4f}, TPR={best_tpr_val:.4f}')
    
    # Debug: print red star position
    print(f"\n[ROC DEBUG] Red star position:")
    print(f"  Threshold: {best_threshold:.6f}")
    print(f"  FPR (1-bkg_rej): {best_fpr_val:.6f}")
    print(f"  TPR (sig_eff): {best_tpr_val:.6f}")
    
    # Labels and formatting
    ax.set_xlabel('False Positive Rate (1 - Background Rejection)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Signal Efficiency)', fontsize=12, fontweight='bold')
    ax.set_title('TrkPID ROC Curve: Signal vs Cosmics', fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    roc_path = "cut_optimization/trkpid_roc.png"
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to: {roc_path}")
    
    # Save CSV
    csv_path = "cut_optimization/trkpid_scan.csv"
    with open(csv_path, 'w') as f:
        f.write("threshold,signal_efficiency,background_rejection\n")
        for r in rows:
            f.write(f"{r['threshold']:.6f},{r['tpr']:.6f},{r['bkg_rej']:.6f}\n")
    print(f"Saved results to: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SCAN SUMMARY")
    print("=" * 80)
    print(f"Threshold range: {min(thresholds):.4f} to {max(thresholds):.4f}")
    print(f"Best Youden Index threshold: {best_threshold:.6f}")
    print(f"  Signal Efficiency (TPR): {best_tpr_val:.6f}")
    print(f"  Background Rejection: {best_bkg_rej_val:.6f}")
    print(f"  FPR (1-bkg_rej): {best_fpr_val:.6f}")
    print("=" * 80 + "\n")
    
    return rows

def call_mlp(files, labels, cuts, locations, columns, signs):
    """
    Train MLP on d0 and rmax features for binary classification.
    
    Args:
        files: list of filesets, e.g., [background_fileset, signal_fileset]
               Each fileset is a list of file paths
        labels: list of labels for each fileset, e.g., ["background", "signal"]
        cuts: list of cut configurations for each file
        locations: list of data locations for each file
        columns: column names to use
        signs: list of signs for each file
    
    Returns:
        trainer: trained MLPTrainer instance with scores
        scores_dict: dict with scores for each label, e.g., {"background": array, "signal": array}
    """
    from mlp import MLP, MLPTrainer, MLPDataset
    import torch
    from torch.utils.data import DataLoader
    
    logger = Logger(print_prefix="[call_mlp]", verbosity=1)
    
    # Track which file index we're on across all filesets
    all_d0_by_label = {label: [] for label in labels}
    all_rmax_by_label = {label: [] for label in labels}
    all_costheta_by_label = {label: [] for label in labels}
    file_idx = 0
    
    # Process each fileset
    for fileset_idx, fileset in enumerate(files):
        label = labels[fileset_idx]
        logger.log(f"Processing fileset '{label}'", 1)
        
        for fil in fileset:
            logger.log(f"Processing file: {fil}", 1)
            ana_processor = AnaProcessor(fil, args.jobs, signs[file_idx], cuts[file_idx], locations[file_idx])
            results = ana_processor.execute()
            combine_result = results["combined_data"]
            logger.log(f"  Loaded {len(combine_result)} events", 1)

            # run cat
            mc_count_array, _ = count_particle_types(combine_result, logger)

            selector = Select()
            
            # select only track front to fit to
            trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")

            # did the track intersect the ST?
            has_st  = selector.has_ST(combine_result['trkfit'])

            # did the track intersect the OPA?
            no_opa  = selector.has_OPA(combine_result['trkfit'])

            # combined mask
            trkfit_ent = ak.mask(combine_result['trkfit']["trksegs"], trk_front)
                
            trk_front_mc = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front",branch_name="trksegsmc")
            trkfit_ent_mc = ak.mask(combine_result['trkfit']["trksegsmc"], trk_front_mc)

            # make vector mag branch
            vector = Vector()
            mom_mag = vector.get_mag(trkfit_ent ,'mom')

            # plot cut distributions
            test_mask = (trk_front)
            all_d0_by_label[label].append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['d0'])
            all_rmax_by_label[label].append(ak.mask(combine_result['trkfit']["trksegpars_lh"],test_mask)['maxr'])
            
            # Extract cosTheta from momentum vector
            mom_vec = vector.get_vector(trkfit_ent, 'mom')  # Get momentum as vector
            p_mag = vector.get_mag(trkfit_ent, 'mom')  # Get magnitude
            pz = mom_vec.z  # Access z component
            costheta = pz / p_mag
            all_costheta_by_label[label].append(ak.mask(costheta, test_mask))
            
            # TODO: Compute dtdz_ratio = dtdz_slope / dtdz_exp
            # velocity = 300 * mom / sqrt(mom^2 + 0.511^2)
            # dtdz_exp = 1 / (velocity * cz)
            # dtdz_ratio = dtdz_slope / dtdz_exp
            # p_mag_masked = ak.mask(p_mag, test_mask)
            # costheta_masked = ak.mask(costheta, test_mask)
            # velocity = 300.0 * p_mag_masked / np.sqrt(p_mag_masked**2 + 0.511**2)
            # dtdz_exp = 1.0 / (velocity * costheta_masked)
            # trksegpars_masked = ak.mask(combine_result['trkfit']["trksegpars_lh"], test_mask)
            # dtdz_slope = trksegpars_masked['dtdz']
            # dtdz_ratio = dtdz_slope / dtdz_exp
            # all_dtdz_ratio_by_label[label].append(dtdz_ratio)
            
            file_idx += 1
    
    # Combine across files within each fileset
    combined_d0_by_label = {}
    combined_rmax_by_label = {}
    combined_costheta_by_label = {}
    
    for label in labels:
        if all_d0_by_label[label]:
            combined_d0_by_label[label] = ak.concatenate(all_d0_by_label[label])
            combined_rmax_by_label[label] = ak.concatenate(all_rmax_by_label[label])
            combined_costheta_by_label[label] = ak.concatenate(all_costheta_by_label[label])
        else:
            combined_d0_by_label[label] = ak.Array([])
            combined_rmax_by_label[label] = ak.Array([])
            combined_costheta_by_label[label] = ak.Array([])
        
        logger.log(f"{label}: {len(combined_d0_by_label[label])} events", 1)
    
    # Create combined dataset with labels for training
    all_d0_combined = []
    all_rmax_combined = []
    all_costheta_combined = []
    all_numeric_labels = []
    
    for label_idx, label in enumerate(labels):
        d0_data = ak.to_numpy(ak.flatten(combined_d0_by_label[label], axis=None))
        rmax_data = ak.to_numpy(ak.flatten(combined_rmax_by_label[label], axis=None))
        costheta_data = ak.to_numpy(ak.flatten(combined_costheta_by_label[label], axis=None))
        
        all_d0_combined.append(d0_data)
        all_rmax_combined.append(rmax_data)
        all_costheta_combined.append(costheta_data)
        all_numeric_labels.append(np.full(len(d0_data), label_idx))
    
    all_d0_combined = np.concatenate(all_d0_combined)
    all_rmax_combined = np.concatenate(all_rmax_combined)
    all_costheta_combined = np.concatenate(all_costheta_combined)
    all_numeric_labels = np.concatenate(all_numeric_labels)
    
    logger.log(f"Total dataset: {len(all_numeric_labels)} events", 1)
    logger.log(f"Creating dataset and dataloaders...", 1)
    
    # Create dataset and dataloaders
    dataset = MLPDataset(all_d0_combined, all_rmax_combined, all_costheta_combined, all_numeric_labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create and train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.log(f"Using device: {device}", 1)
    
    model = MLP(input_dim=3, hidden_dim=64, dropout_rate=0.2)
    trainer = MLPTrainer(model, device=device, learning_rate=1e-3)
    
    # Set normalization parameters from dataset
    trainer.set_normalization(dataset)
    
    logger.log("Training MLP...", 1)
    trainer.train(train_loader, val_loader, epochs=5, patience=10, verbose=True)
    
    # Get scores for each label
    scores_dict = {}
    features_dict = {"d0": {}, "rmax": {}, "costheta": {}}
    
    for label in labels:
        scores = trainer.score(combined_d0_by_label[label], combined_rmax_by_label[label], combined_costheta_by_label[label])
        scores_dict[label] = scores
        logger.log(f"{label} scores: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}", 1)
        
        # Store features for plotting
        features_dict["d0"][label] = ak.to_numpy(ak.flatten(combined_d0_by_label[label], axis=None))
        features_dict["rmax"][label] = ak.to_numpy(ak.flatten(combined_rmax_by_label[label], axis=None))
        features_dict["costheta"][label] = ak.to_numpy(ak.flatten(combined_costheta_by_label[label], axis=None))
    
    # Plot score distributions
    from mlp import plot_mlp_scores, plot_mlp_features, optimize_mlp_cut, plot_mlp_optimization, plot_mlp_tmva_style
    plot_mlp_scores(scores_dict, labels, output_path="mlp_scores.png")
    plot_mlp_features(features_dict, labels, output_path="mlp_features.png")
    
    # Optimize MLP cut for S/sqrt(S+B)
    logger.log("Optimizing MLP cut...", 1)
    optimal_result = optimize_mlp_cut(scores_dict, signal_label="signal", background_label="background")
    
    logger.log("=" * 80, 1)
    logger.log("MLP CUT OPTIMIZATION RESULTS", 1)
    logger.log("=" * 80, 1)
    logger.log(f"Optimal threshold: {optimal_result['threshold']:.6f}", 1)
    logger.log(f"S/√(S+B): {optimal_result['s_over_sqrt_sb']:.6f}", 1)
    logger.log(f"Signal efficiency: {optimal_result['signal_efficiency']:.4f}", 1)
    logger.log(f"  ({optimal_result['signal_count']} / {optimal_result['total_signal']} signal events)", 1)
    logger.log(f"Background rejection: {optimal_result['background_rejection']:.4f}", 1)
    logger.log(f"  ({optimal_result['total_background'] - optimal_result['background_count']} / {optimal_result['total_background']} background events rejected)", 1)
    logger.log("=" * 80, 1)
    
    # Plot optimization results
    plot_mlp_optimization(scores_dict, optimal_result, output_path="mlp_optimization.png")
    
    # Plot TMVA-style curve
    plot_mlp_tmva_style(scores_dict, optimal_result, output_path="mlp_tmva.png")
    
    return trainer, scores_dict, optimal_result 
      
# Create an instance of our custom processor
def  main(args):
  """ main driver function to run analysis
  """
  print("Running main function")
  new = []
  old= [
      True,  # 0 is_reco_electron
      True,  # 1 has_downstream
      True, # 2 has trk front
      True,  # 3 good_trkqpid
      True,  # 4 good_trkqual
      True,  # 5 within_t0err
      True,  # 6 has_hits
      True, # 7 within_lhr_maxl
      True, # 8 within_d0
      True, # 9 within_pitch_angle
      False,  #10 has_st
      False,  #11 no_opa
      True,  #12 no_crv_veto
      True,  #13 no_crv_quality
      True,  #14 no_crv_timewindow
      True,  #15 pz/pt
      True,  #16 triggers
      True,  #17 in_mom_range
      False, #18 within_t0_early
      False, #19 no_reflected
      True,  #20 within_t0
      True, # 21 signal region cut
      True, # or trigger select
      True #MLP score
  ]
  if args.sign == "minus":
    new= [
      True,  # 0 has_a_track (event has >=1 track)
      True,  # 1 is_good_track
      True,  # 2 has_trk_front_seg
      True,  # 3 is_reco_electron_or_positron (generic e/e+)
      True,  # 4 has_downstream
      True,  # 5 charge_selection (PDG = 11, electrons)
      True,  # 6 or_trigger
      True,  # 7 no_upstream
      False, # 8 alt_hypothesis_upstream_veto
      True,  # 9 no_multi_trk_veto
      True,  #10 good_trkpid
      True,  #11 pz_over_pt
      True,  #12 has_st
      True,  #13 no_opa
      True,  #14 good_trkqual
      True,  #15 has_hits
      True,  #16 within_t0err
      True,  #17 no_crv_veto
      True,  #18 in_mom_range
      True,  #19 within_t0_475 (475-1650 ns)
      True, #20 within_t0_540 (540-1650 ns)
      True, #21 within_t0_640 (640-1650 ns)
      True   #22 signal_region
    ]
  if args.sign == "plus":
    new= [
      True,  # 0 has_a_track (event has >=1 track)
      True,  # 1 is_good_track
      True,  # 2 has_trk_front_seg
      True,  # 3 is_reco_electron_or_positron (generic e/e+)
      True,  # 4 has_downstream
      True,  # 5 charge_selection (PDG = -11, positrons)
      True,  # 6 or_trigger
      True,  # 7 no_upstream
      False, # 8 alt_hypothesis_upstream_veto
      True,  # 9 no_multi_trk_veto
      True,  #10 good_trkpid
      True,  #11 pz_over_pt
      True,  #12 has_st
      True,  #13 no_opa
      False, #14 good_trkqual (disabled for e+)
      True,  #15 has_hits
      True,  #16 within_t0err
      True,  #17 no_crv_veto
      True,  #18 in_mom_range
      False, #19 within_t0_475 (475-1650 ns) (disabled for e+)
      True,  #20 within_t0_540 (540-1650 ns)
      False, #21 within_t0_640 (640-1650 ns)
      False  #22 signal_region (disabled for e+)
    ]

  if args.proctype == "train":
    
    files = [
        ["file_lists/Cosmics_MDC2025an_nomix.txt"],  # Background fileset
        ["file_lists/signal.txt"]       # Signal fileset
    ]
    labels = ["background", "signal"]
    signs = [args.sign, args.sign]
    locations = [args.loc, args.loc]
    columns = []
    
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
      False,  #10 has_st
      False,  #11 no_opa
      False,  #12 no_crv_veto
      False,  #13 no_crv_quality
      False,  #14 no_crv_timewindow
      False,  #15 pz/pt
      False,  #16 all triggers - DEPRECATED!!!!!
      False,  #17 in_mom_range
      False, #18 within_t0_early
      True, #19 no_reflected
      False,  #20 within_t0
      False, # 21 signal region cut
      False   # 22 mlp_score
    ]
    cuts = [new,new]

    trainer, scores_dict, optimal_result = call_mlp(files, labels, cuts, locations, columns, signs)

    import torch
    torch.save(trainer.model.state_dict(), "mlp_model.pth")
    
    # Save normalization parameters
    import json
    norm_params = {
        "d0_mean": float(trainer.d0_mean),
        "d0_std": float(trainer.d0_std),
        "rmax_mean": float(trainer.rmax_mean),
        "rmax_std": float(trainer.rmax_std),
        "costheta_mean": float(trainer.costheta_mean),
        "costheta_std": float(trainer.costheta_std),
    }
    with open("mlp_normalization.json", "w") as f:
        json.dump(norm_params, f, indent=2)
    print("Model saved to mlp_model.pth")
    print("Normalization params saved to mlp_normalization.json")

  print("starting main function with cuts:", new)

  if args.proctype == "optimize_cuts":
    run_multi_background_optimization(args)
    return

  if args.proctype == "rpc":
    files = ["file_lists_full/IntRPC_MDC2025an_nomix.txt","file_lists_full/ExtRPC_MDC2025an_nomix.txt"]
    signs = ["minus","minus"]
    locations = [args.loc,args.loc]
    columns = ["Internal RPC e-","External RPC e-"]
    cuts = [new,new]
    fit_dataset(files, cuts, locations, columns, signs, args.proctype)

    print("Done plotting")
    return

  if args.proctype == "cosmics-compare":
    files = ["file_lists/Cosimcs_MDC2025an_nomix.txt","file_lists/OffSpill_MDC2025an.txt"]
    signs = [args.sign,args.sign]
    locations = [args.loc,args.loc]
    columns = ["OnSpill Cosmics","OffSpill Cosmics"]
    cuts = [new,new]
    fit_dataset(files, cuts, locations, columns, signs, args.proctype)

    print("Done plotting")
    return

  files = [args.file]
  signs = [args.sign]
  locations = [args.loc]
  columns = ["Run-1"]
  cuts = [new]
  #compare_datasets(files, cuts, locations, columns, signs)

  fit_dataset(files, cuts, locations, columns, signs, args.proctype)
  """
  if args.proctype == "CELL":
    fig, ax = plot_theory_with_rle(
      files=files,
      cuts=cuts,
      locations=locations,
      signs=signs,
      jobs=args.jobs,
      rle_calib_dir="RLE/common"
    )
  """
  
  
  
  print("Done plotting")
  return
  
def PrintArgs(args):
  """
  prints users input parameters
  """
  print("========= [process.py]✅  Analyzing with user opts: ===========")
  print("file:", args.file)
  print("number of processes (njobs - optimal is 1 per file):", args.jobs)
  print("verbose: ", args.verbose)
  print("proctype:", args.proctype)



if __name__ == "__main__":
    print("DEBUG: Starting script", flush=True)
    # list of input arguments, defaults should be overridden
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file", type=str, required=True, help="filename or file list name (text file list,fullpaths)")
    parser.add_argument("--loc", type=str, required=False, default='disk', help="location of files")
    parser.add_argument("--sign", type=str, required=False, default='minus', help="sign of the signal being sought in words (default: minus)")
    parser.add_argument("--proctype", type=str, required=False, default='ensemble', help="process type: 'ensemble', 'cosmics', 'rpc', 'rle', 'convolution', 'overlay', 'train' (default: ensemble)")
    parser.add_argument("--jobs", type=int, required=False, default=1,help="use if more than one file, should be nfiles")
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    
    print("DEBUG: Parsing arguments", flush=True)
    args = parser.parse_args()
    print(f"DEBUG: Parsed args - file={args.file}, jobs={args.jobs}, sign={args.sign}", flush=True)

    # if verbose print the user input
    if(args.verbose > 0):
      PrintArgs(args)
    
    print("DEBUG: Calling main()", flush=True)
    # run main function
    main(args)
    print("DEBUG: Script completed", flush=True)