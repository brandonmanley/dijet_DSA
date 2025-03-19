import numpy as np
import os

input_directory = 'mc_data/roots_70/data'
output_directory = 'mc_data/'

npy_files = [f for f in os.listdir(input_directory) if f.endswith('.npy')]
if not npy_files:	print("No .npy files found in the directory.")

data_list = [np.load(os.path.join(input_directory, f)) for f in npy_files]
combined_data = np.concatenate(data_list, axis=0)

output_file = output_directory + 'mc_data_roots70.npy'
np.save(output_file, combined_data)
print(f"Combined {len(npy_files)} files into {output_file}")