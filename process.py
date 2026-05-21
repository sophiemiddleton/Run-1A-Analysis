from platform import processor

import gc
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb

# this ana
from compare import Compare
from cosmics import Cosmics
from analyze import Analyze
from pyutils.pycut import CutManager
from pyutils.pylogger import Logger

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


# Create an instance of our custom processor
def  main(args):
  """ main driver function to run analysis
  """
  
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

  old = [
    True,  # 0 is_reco_electron
    True,  # 1 has_downstream
    True, # 2 has trk front
    True,  # 3 good_trkqual
    True,  # 4 good_trkpid
    True, # 5 within_t0
    True,  # 6 within_t0err
    True,  # 7 has_hits
    True, # 8 within_lhr_maxl
    True, # 9 within_d0
    True, # 10 within_pitch_angle
    False,  #11 has_st
    False,  #12 no_opa
    True,  #13 no_crv_veto
    False,  #14 no_crv_quality
    False,  #15 no_crv_timewindow
    False,  #16 pz/pt
    False,  #17 triggers
    False,  #18 within_mom_time
    False #19 early time
  ]
  

  files = [args.file]
  signs = [args.sign]
  locations = [args.loc]
  columns = ["Run1A"]
  cuts = [new]
  #compare_datasets(files, [new], locations, columns, signs)
  fit_dataset(files, cuts, locations, columns, signs, args.proctype)
  return
  
def PrintArgs(args):
  """
  prints users input parameters
  """
  print("========= [RefAna/pyCount/process.py]✅  Analyzing with user opts: ===========")
  print("file:", args.file)
  print("number of processes (njobs - optimal is 1 per file):", args.jobs)
  print("verbose: ", args.verbose)
  print("proctype:", args.proctype)
if __name__ == "__main__":
    # list of input arguments, defaults should be overridden
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file", type=str, required=True, help="filename or file list name (text file list,fullpaths)")
    parser.add_argument("--loc", type=str, required=False, default='disk', help="location of files")
    parser.add_argument("--sign", type=str, required="minus", help="sign of the signal being sought in words")
    parser.add_argument("--proctype", type=str, required=False, default='ensemble', help="process type (default: ensemble)")
    parser.add_argument("--jobs", type=int, required=False, default=1,help="use if more than one file, should be nfiles")
    parser.add_argument("--verbose", type=int, default=1, help="verbose")
    args = parser.parse_args()
    (args) = parser.parse_args()

    # if verbose print the user input
    if(args.verbose > 0):
      PrintArgs(args)
    
    # run main function
    main(args)




