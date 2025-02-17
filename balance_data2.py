import pandas as pd
from tqdm import tqdm
import os

directory = os.getcwd()

temp_output = os.path.join(directory, 'temp_balanced_data.csv')
incremental_output = os.path.join(directory, 'incrementally_balanced_data.csv')

chunk_size = 1000  # Adjust as needed

# Initialize counters and placeholders
total_diabetics = 0
total_non_diabetics = 0

print("Loading, shuffling, and balancing the dataset in chunks...")

for chunk in tqdm(pd.read_csv(temp_output, chunksize=chunk_size), desc="Loading chunks"):
    # Shuffle each chunk
    chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split chunk into diabetic and non-diabetic
    diabetic_chunk = chunk[chunk['is_diabetic'] == 'yes']
    non_diabetic_chunk = chunk[chunk['is_diabetic'] == 'no']
    
    # Count diabetics and non-diabetics
    total_diabetics += len(diabetic_chunk)
    total_non_diabetics += len(non_diabetic_chunk)
    
    # Determine number to append to final balanced set
    n_diabetics_to_append = min(len(diabetic_chunk), len(non_diabetic_chunk))
    
    # Append balanced data to final output file
    if n_diabetics_to_append > 0:
        balanced_chunk = pd.concat([diabetic_chunk.head(n_diabetics_to_append), non_diabetic_chunk.head(n_diabetics_to_append)])
        balanced_chunk = balanced_chunk.sample(frac=1, random_state=42).reset_index(drop=True)
        
        if total_diabetics == len(diabetic_chunk) and total_non_diabetics == len(non_diabetic_chunk):
            balanced_chunk.to_csv(incremental_output, index=False, mode='w', header=True)
        else:
            balanced_chunk.to_csv(incremental_output, index=False, mode='a', header=False)

print("Balanced dataset defined and saved incrementally.")

print(f"Total diabetics: {total_diabetics}")
print(f"Total non-diabetics: {total_non_diabetics}")
