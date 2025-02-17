import pandas as pd
from tqdm import tqdm
import psutil
import time
import os

directory = os.getcwd()

input = os.path.join(directory, 'ukb677942.csv')
temp_output = os.path.join(directory, 'temp_balanced_data.csv')

diabetes_columns = ['2443-0.0', '2443-1.0', '2443-2.0', '2443-3.0']
gestational_diabetes_columns = ['4041-0.0', '4041-1.0', '4041-2.0', '4041-3.0', '10844-0.0']
print("File paths and columns added")

# Function to check memory usage and pause if necessary
def monitor_memory(threshold=96):
    while psutil.virtual_memory().percent > threshold:
        print("Memory usage high, pausing for 1 minute...")
        time.sleep(60)

# Reduce chunk size to manage memory usage
chunk_size = 1000
total_chunks = 500000 // chunk_size + 1  # Estimate the number of chunks

# Process and save chunks incrementally
print("Processing and saving chunks...")
for i, chunk in enumerate(tqdm(pd.read_csv(input, low_memory=False, chunksize=chunk_size), total=total_chunks)):
    monitor_memory()  # Check and pause if memory usage is high
    # Process each chunk
    chunk['is_diabetic'] = chunk[diabetes_columns].eq(1).any(axis=1).map({True: 'yes', False: 'no'})
    
    # Append chunk to CSV
    if i == 0:
        chunk.to_csv(temp_output, index=False, mode='w', header=True)
    else:
        chunk.to_csv(temp_output, index=False, mode='a', header=False)

print("Dataset processed and saved in chunks.")
