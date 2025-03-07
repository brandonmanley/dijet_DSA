import pandas as pd
import numpy as np
import time

# Create sample data
n = 10**6  # Number of rows
data = {
    's': np.random.random(n),
    'n': np.random.random(n),
    'eta': np.random.random(n),
}

# Pandas DataFrame
df = pd.DataFrame(data)

# Dictionary of NumPy arrays
data_dict = {key: np.array(value) for key, value in data.items()}

# Example target value for filtering
target_eta = 0.5

# Pandas filtering
start = time.time()
for i in range(1000):
    filtered_df = df[df['eta'].abs() - target_eta < 0.05]
print("Pandas filtering time:", time.time() - start)

# Dictionary (NumPy) filtering
start = time.time()
for i in range(1000):
    mask = np.abs(data_dict['eta'] - target_eta) < 0.05
    filtered_dict = {key: value[mask] for key, value in data_dict.items()}
print("NumPy dictionary filtering time:", time.time() - start)