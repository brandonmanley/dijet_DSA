import subprocess
import time

script_to_submit = "dsa_mc.py"
num_submissions = 5
sample_size = 1000
run_num = 0
root_s = 40

for i in range(num_submissions):

	arguments = f"{sample_size} mc_data/data/mc_data_{run_num}_{i}.npy {root_s}"
	command = f"nohup python {script_to_submit} {arguments} > mc_data/logs/output{i}.log 2>&1 &"
	print(f"Submitting: {command}")
	
	process = subprocess.Popen(command, shell=True)
	process.wait()
	time.sleep(5)