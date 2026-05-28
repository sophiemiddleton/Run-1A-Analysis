print("DEBUG: Importing modules...", flush=True)

import gc
import sys
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
from analyze import Analyze
from rle import generate_rle_calibration
from spectrum import TheorySpectrum
from pyutils.pycut import CutManager
from pyutils.pylogger import Logger
import optimize_cuts
from helper import make_HistogramPDF


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
            "trk" : [
                "trk.nactive", 
                "trk.pdg", 
                "trk.status",
                "trkqual.valid",
                "trkqual.result",
                "trkpid.valid",
                "trkpid.result",
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
        timeout_seconds = 60  # 1 minute timeout per file - if file hangs, skip it
        
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
      
      # save reconstructed momentum magnitudes for this dataset to CSV
      # write fitted-range data using WriteFittedData
      try:
        WriteFittedData(mom_mag, 95, 110)
      except Exception as e:
        print(f"WriteFittedData failed: {e}")

      #mom_mag = ak.nan_to_none(mom_mag)
      #mom_mag = ak.drop_none(mom_mag)

      time = ak.nan_to_none(trkfit_ent['time'])
      time = ak.drop_none(time)
      
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
      recomom.append(mom_mag)
      times.append(time)
      losses.append(loss)
      truemom.append(mom_mag_mc)
      resolutions.append(resolution)
      crv.append(dt)
    cosmics.fit_momentum(recomom)
    prefix = "eplus" if str(signs[i]).lower() == "plus" else "eminus"
    comparison.plot_particle_counts(mc_count, columns, plot_prefix=prefix)
    
    if signs[i] == "minus":
       startmom = 98
       endmom = 110
       nbins = 25
       comparison.plot_variable(recomom, r"$p_e$ [MeV/c]",f"{prefix}_recomom", startmom, endmom, [103.6,103.6],[104.8,104.8], mc_count,columns, nbins=nbins)
    else:
        startmom = 85
        endmom = 97
        nbins = 15
        comparison.plot_variable(recomom, r"$p_e$ [MeV/c]",f"{prefix}_recomom", startmom, endmom, [90.85,90.85],[92.1,92.1], mc_count,columns, nbins=nbins)
    comparison.plot_variable(crv, "|DT| [ns]",f"{prefix}_DT",0,300, [150,150],[150,150], mc_count,columns)
    comparison.plot_variable(nST, "nST",f"{prefix}_nST",0,15, [1,1],[1,1], mc_count,columns, 15)
    comparison.plot_variable(nOPA, "nOPA",f"{prefix}_nOPA",0,4, [0,0],[0,0], mc_count,columns,4)
    comparison.plot_variable(rmax, "rmax", f"{prefix}_rmax",300,750, [450,450],[680,680], mc_count,columns)
    comparison.plot_variable(d0, "d0", f"{prefix}_d0",0, 250, [100,100], [100,100], mc_count,columns)
    comparison.plot_variable(tanDip, "tanDip",f"{prefix}_tanDip",-1,2.5, [0.557,0.557], [1.0,1.0],mc_count,columns)
    comparison.plot_variable(trkqual, "trkqual", f"{prefix}_trkqual", 0,1,[0.2,0.2], [0.2, 0.2], mc_count,columns)
    comparison.plot_variable(trkpid, "trkpid", f"{prefix}_trkpid", 0,1,[0.6,0.6], [0.6, 0.6], mc_count,columns)
    comparison.plot_variable(t0err, "t0err",f"{prefix}_t0err", 0,1, [0.9,0.9],[0.9,0.9], mc_count,columns)
    comparison.plot_variable(active, "nactive",f"{prefix}_nactive", 0,50, [20,20],[0.9,0.9], mc_count,columns)
    
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

    if proctype == "ensemble":
        WriteFittedData(recomom, 95, 110)
    if proctype == "cosmics":
        cosmics = Cosmics()
        cosmics.fit_momentum(recomom)
    if proctype == "rpc":
        rpc = RPC()
        rpc.fit_momentum(recomom, columns)
    # Generate RLE calibration parameters for ensemble
    if proctype == "rle":
        rle_results = generate_rle_calibration(combine_result, "RLE/common", run_fits=True)

def plot_theory_with_rle(files, cuts, locations, signs, jobs=1, rle_calib_dir="RLE/common", 
                        mom_range=(90, 120), binwidth=0.1, output_file=None):
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
        
        print(f"\n{'='*70}", flush=True)
        print(f"[plot_theory_with_rle] RLE CALIBRATION PIPELINE SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"[plot_theory_with_rle] Theory: CeLL Leading Log (E_MAX={104.969:.3f} MeV)", flush=True)
        print(f"[plot_theory_with_rle] Momentum range for plotting: {mom_range}", flush=True)
        print(f"[plot_theory_with_rle] Loading calibration from: {rle_calib_dir}", flush=True)
        print(f"{'='*70}\n", flush=True)
        
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
            
            # ===== PRINT RESOLUTION PARAMETERS =====
            print(f"\n[plot_theory_with_rle] ===== RESOLUTION PARAMETERS =====", flush=True)
            print(f"[plot_theory_with_rle] Bins: {res_nbins}", flush=True)
            print(f"[plot_theory_with_rle] Range: [{res_min_physical:.4f}, {res_max_physical:.4f}] MeV", flush=True)
            print(f"[plot_theory_with_rle] Mean (trimmed): {res_mean:.4f} MeV", flush=True)
            print(f"[plot_theory_with_rle] Std (trimmed): {res_std:.4f} MeV", flush=True)
            print(f"[plot_theory_with_rle] Events used: {len(res_trimmed)} (from {len(res_data)} total)", flush=True)
            print(f"[plot_theory_with_rle] Percentile range: 10-90th", flush=True)
            print(f"[plot_theory_with_rle] ======================================\n", flush=True)
            
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
            
            # ===== PRINT LOSS PARAMETERS =====
            print(f"\n[plot_theory_with_rle] ===== LOSS PARAMETERS =====", flush=True)
            print(f"[plot_theory_with_rle] Bins: {loss_nbins}", flush=True)
            print(f"[plot_theory_with_rle] Range: [{loss_min_physical:.4f}, {loss_max_physical:.4f}] MeV", flush=True)
            print(f"[plot_theory_with_rle] Mean (trimmed): {loss_mean:.4f} MeV", flush=True)
            print(f"[plot_theory_with_rle] Std (trimmed): {loss_std:.4f} MeV", flush=True)
            print(f"[plot_theory_with_rle] Events used: {len(loss_trimmed)} (from {len(loss_data)} total)", flush=True)
            print(f"[plot_theory_with_rle] Percentile range: 10-90th", flush=True)
            print(f"[plot_theory_with_rle] =====================================\n", flush=True)
            
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
        import traceback
        traceback.print_exc()
        return None, None

def WriteFittedData(data, min_v, max_v):
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
        if sign == "minus":
            cuts= [
                True,  # 0 is_reco_electron
                True,  # 1 has_downstream
                True, # 2 has trk front
                False,  # 3 good_trkqpid
                False,  # 4 good_trkqual
                False, # 5 within_t0
                True,  # 6 within_t0err
                True,  # 7 has_hits
                False, # 8 within_lhr_maxl
                False, # 9 within_d0
                False, # 10 within_pitch_angle
                False,  #11 has_st
                False,  #12 no_opa
                False,  #13 no_crv_veto
                False,  #14 no_crv_quality
                False,  #15 no_crv_timewindow
                False,  #16 pz/pt
                True,  #17 triggers
                False,  #18 within_mom_time
                False, #19 early time
                False #20 reflected
                ]
        else:
            cuts = [True, True, True, True, True, False, True, True, False, False, False, True, True, True, True, True, True, False, False, False, False]
    
    # Step 1: Load and process data
    logger.log(f"\nStep 1: Loading data from {file_list_path}", "info")
    ana_processor = AnaProcessor(file_list_path, jobs=jobs, sign=sign, cuts=cuts, location=locations)
    results = ana_processor.execute()
    combine_result = results["combined_data"]
    
    logger.log(f"Data loaded successfully", "info")
    
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


# Create an instance of our custom processor
def  main(args):
  """ main driver function to run analysis
  """
  print("Running main function")
  new = []
  
  if args.sign == "minus":
    new= [
      True,  # 0 is_reco_electron
      True,  # 1 has_downstream
      True, # 2 has trk front
      True,  # 3 good_trkqpid
      True,  # 4 good_trkqual
      False, # 5 within_t0
      True,  # 6 within_t0err
      True,  # 7 has_hits
      False, # 8 within_lhr_maxl
      False, # 9 within_d0
      False, # 10 within_pitch_angle
      True,  #11 has_st
      True,  #12 no_opa
      True,  #13 no_crv_veto
      True,  #14 no_crv_quality
      True,  #15 no_crv_timewindow
      True,  #16 pz/pt
      True,  #17 triggers
      False,  #18 within_mom_time
      False, #19 early time
      False #20 reflected
    ]
  if args.sign == "plus":
    new= [
      True,  # 0 is_reco_electron
      True,  # 1 has_downstream
      True, # 2 has trk front
      True,  # 3 good_trkqpid
      False,  # 4 good_trkqual
      False, # 5 within_t0
      True,  # 6 within_t0err
      True,  # 7 has_hits
      False, # 8 within_lhr_maxl
      False, # 9 within_d0
      False, # 10 within_pitch_angle
      True,  #11 has_st
      True,  #12 no_opa
      True,  #13 no_crv_veto
      True,  #14 no_crv_quality
      True,  #15 no_crv_timewindow
      True,  #16 pz/pt
      True,  #17 triggers
      False,  #18 within_mom_time
      False, #19 early time
      False #20 reflected
    ]

  print("starting main function with cuts:", new)

  files = [args.file]
  signs = [args.sign]
  locations = [args.loc]
  columns = ["Run1A"]
  cuts = [new]
  #compare_datasets(files, cuts, locations, columns, signs)
  fit_dataset(files, cuts, locations, columns, signs, args.proctype)
  if args.proctype == "CE":
    fig, ax = plot_theory_with_rle(
      files=files,
      cuts=cuts,
      locations=locations,
      signs=signs,
      jobs=args.jobs,
      rle_calib_dir="RLE/common"
    )
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
    parser.add_argument("--proctype", type=str, required=False, default='ensemble', help="process type (default: ensemble)")
    parser.add_argument("--jobs", type=int, required=False, default=1,help="use if more than one file, should be nfiles")
    parser.add_argument("--verbose", type=int, default=1, help="verbose")
    
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




