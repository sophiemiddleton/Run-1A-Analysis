import sys 
import gc
import numpy as np
import awkward as ak
import pickle as pkl
import matplotlib.pyplot as plt

sys.path.append('/exp/mu2e/app/users/sdittmer/LikelihoodAnalysis/py-fitter')
from analyze import Analyze
from pyutils.pyprocess import Processor, Skeleton
from pyutils.pyvector import Vector
from pyutils.pyselect import Select

analyse = Analyze(verbosity=0)

processor = Processor(use_remote=False, verbosity=0)
            
# Process the files using multithreading
my_branches = { 
    "crv" : [
        "crvcoincs.time"
    ],
    "trk" : [
        "trk.nactive", 
        "trk.pdg", 
        "trkqual.result"
    ],
    "trkfit" : [
        "trksegs",
        "trksegsmc",
        "trksegpars_lh"
    ],
    "trkmc" : [
        "trkmcsim"
    ]
}

data = processor.process_data(file_name = "/exp/mu2e/data/users/sdittmer/SignalShape/ntuple_flat.root", branches = my_branches)
gc.collect()

# Selecting flat e- gen particles
flat_e = (data['trkmc']["trkmcsim"]["startCode"] == 173) & (data['trkmc']["trkmcsim"]["rank"] == 0) & (data['trkmc']["trkmcsim"]["nhits"] > 0)
data['trkmc'] = data['trkmc'][flat_e]
trk_flat_e = ak.any(flat_e, axis=-1)
data['trkfit'] = data['trkfit'][trk_flat_e]
data['trkmc']  = data['trkmc'][trk_flat_e]
data['trk']    = data['trk'][trk_flat_e]

# Apply reco selection
flat_sel = analyse.execute(data, "/exp/mu2e/data/users/sdittmer/SignalShape/ntuple_flat.root")

# Calculate efficiency
fig, ax = plt.subplots(1,1,figsize=(5.2,4.3))
vector = Vector()
gen_mom_all = np.array(ak.flatten(vector.get_mag(data['trkmc']["trkmcsim"].mask[(data['cutflow'] >= 1)],'mom'),axis=None))
h_all,edges = np.histogram(gen_mom_all,63,(79.8,105.))
for icut, cut_name in enumerate(analyse.cut_names):
  if icut == 0: continue
  gen_mom_sel = np.array(ak.flatten(vector.get_mag(data['trkmc']["trkmcsim"].mask[(data['cutflow'] >= icut+1)],'mom'),axis=None))
  h_sel,_ = np.histogram(gen_mom_sel,63,(79.8,105.))
  h_eff = h_sel/h_all
  ax.stairs(h_eff,edges,label=cut_name)
ax.legend()
plt.savefig("flat_efficiency.png")
print(h_eff)
print(edges)
with open(f'../common/efficiency.pkl','wb') as f:
    pkl.dump([h_eff,edges],f)

# Clean up
gc.collect()

data_flat = {}

for sid, plane in enumerate(['entrance','middle','exit']):
  print(plane)
  # select only track front to fit to
  selector = Select()
  at_plane_reco = selector.select_surface(flat_sel['trkfit'], sid=sid, branch_name="trksegs")
  at_plane_mc   = selector.select_surface(flat_sel['trkfit'], sid=sid, branch_name="trksegsmc")

  good_track = (ak.sum(at_plane_reco,axis=2) >= 1)
  good_track = (good_track) & (ak.sum(at_plane_mc,axis=2) == 1)

  gc.collect()
  
  # make vector mag branch

  reco_mom = vector.get_mag(flat_sel['trkfit']["trksegs"].mask[(at_plane_reco) & (good_track)],'mom')
  reco_mom = ak.nan_to_none(reco_mom)
  reco_mom = ak.drop_none(reco_mom)
  reco_mom = np.array(ak.flatten(reco_mom,axis=None))
  print(f'Number of tracks at {plane}: {len(reco_mom)}')
    
  mc_mom = vector.get_mag(flat_sel['trkfit']["trksegsmc"].mask[(at_plane_mc) & (good_track)],'mom')
  mc_mom = ak.nan_to_none(mc_mom)
  mc_mom = ak.drop_none(mc_mom)
  mc_mom = np.array(ak.flatten(mc_mom,axis=None))
  print(f'Number of MC tracks at {plane}: {len(mc_mom)}')

  gen_mom = vector.get_mag(flat_sel['trkmc']["trkmcsim"].mask[(good_track)],'mom')
  gen_mom = ak.nan_to_none(gen_mom)
  gen_mom = ak.drop_none(gen_mom)
  gen_mom = np.array(ak.flatten(gen_mom,axis=None))
  print(f'Number of gen tracks at {plane}: {len(gen_mom)}')

  data_flat[plane] = {'reco' : reco_mom, 'mc' : mc_mom, 'gen' : gen_mom}

  gc.collect()
  
print('Loaded everything')

# Save skimmed data
with open(f'skimmed_flat_mom_v2.pkl','wb') as f:
    pkl.dump(data_flat,f)

