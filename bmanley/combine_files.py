import numpy as np
import os
import argparse


directory = 'mc_data/data'
npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
	
if not npy_files:	print("No .npy files found in the directory.")

data_list = [np.load(os.path.join(directory, f)) for f in npy_files]
combined_data = np.concatenate(data_list, axis=0)

output_file = 'mc_data/mc_data_trial3_root_s40.npy'
np.save(output_file, combined_data)
print(f"Combined {len(npy_files)} files into {output_file}")