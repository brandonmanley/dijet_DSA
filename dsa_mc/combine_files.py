import numpy as np
import os, sys

if len(sys.argv) != 3: 
		print("Usage: python combine_files.py <data directory> <output file>")
		sys.exit(1)


data_dir = sys.argv[1]
output_file = sys.argv[2]

npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
if not npy_files:	print("No .npy files found in the directory.")

data_list = [np.load(os.path.join(data_dir, f)) for f in npy_files]
combined_data = np.concatenate(data_list, axis=0)

np.save(output_file, combined_data)
print(f"Combined {len(npy_files)} files from {data_dir} into {output_file}")