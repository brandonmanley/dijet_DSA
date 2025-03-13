import subprocess
import time
import os, sys


if len(sys.argv) != 5: 
		print("Usage: python submit_mc.py <output_directory> <n submissions> <sample_size> <root s (GeV)>")
		sys.exit(1)

output_dir = sys.argv[1]
num_submissions = int(sys.argv[2])
sample_size = int(sys.argv[3])
root_s = float(sys.argv[4])

if not os.path.exists(output_dir): 
	os.mkdir(output_dir)
	os.mkdir(output_dir + '/data')
	os.mkdir(output_dir + '/logs')

for i in range(num_submissions):

	arguments = f"{sample_size} {output_dir}/data/mc_data_{i}.npy {root_s}"
	command = f"nohup python dsa_mc.py {arguments} > {output_dir}/logs/output_{i}.log 2>&1 &"
	print(f"Submitting: {command}")
	
	process = subprocess.Popen(command, shell=True)
	process.wait()
	time.sleep(5)