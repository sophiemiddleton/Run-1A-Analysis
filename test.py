import tensorflow as tf
from pyutils.pyprocess import Processor
file_list_path="nts.mu2e.CosmicSignalOnSpill-reco-ntuple.MDC2025-002.001430_00000002.root"
branches=["event"]
processor = Processor(use_remote=True, location="tape")
data = processor.process_data(file_name=file_list_path, branches=branches)
print(data)
