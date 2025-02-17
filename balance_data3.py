import pandas as pd
from tqdm import tqdm
import os

directory = os.getcwd()

incremental_output = os.path.join(directory, 'incrementally_balanced_data.csv')
final_output = os.path.join(directory, 'big_balanced_data.csv')

chunk_size = 1000  # Adjust as needed

# Initialize counts for diabetics and non-diabetics
diabetic_count = 0
non_diabetic_count = 0
diabetic_chunks = []
non_diabetic_chunks = []

print("Loading, sorting, and collecting chunks...")

# Read the incrementally balanced data in chunks
for chunk in tqdm(pd.read_csv(incremental_output, chunksize=chunk_size), desc="Loading chunks"):
    # Sort each chunk by 'is_diabetic'
    chunk_sorted = chunk.sort_values(by='is_diabetic', ascending=False)
    
    # Split chunk into diabetic and non-diabetic
    diabetic_chunk = chunk_sorted[chunk_sorted['is_diabetic'] == 'yes']
    non_diabetic_chunk = chunk_sorted[chunk_sorted['is_diabetic'] == 'no']
    
    # Collect counts
    diabetic_count += len(diabetic_chunk)
    non_diabetic_count += len(non_diabetic_chunk)
    
    # Store sorted chunks in lists
    diabetic_chunks.append(diabetic_chunk)
    non_diabetic_chunks.append(non_diabetic_chunk)

print(f"Total diabetics: {diabetic_count}")
print(f"Total non-diabetics: {non_diabetic_count}")

# Calculate the number of rows for the balanced dataset
n_total = 2 * diabetic_count

# Initialize writer for the final balanced output
print("Writing the final balanced dataset...")
with open(final_output, 'w', newline='') as f:
    header_written = False
    
    # Write balanced chunks incrementally
    for diabetic_chunk, non_diabetic_chunk in zip(diabetic_chunks, non_diabetic_chunks):
        # Determine number to append to final balanced set
        n_diabetics_to_append = len(diabetic_chunk)
        n_non_diabetics_to_append = len(non_diabetic_chunk)
        
        # Create a balanced chunk
        balanced_chunk = pd.concat([
            diabetic_chunk.head(n_diabetics_to_append),
            non_diabetic_chunk.head(n_non_diabetics_to_append)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Write the balanced chunk to the final output file
        if not header_written:
            balanced_chunk.to_csv(f, index=False, header=True)
            header_written = True
        else:
            balanced_chunk.to_csv(f, index=False, header=False, mode='a')

print(f"Balanced dataset with approximately {n_total} rows has been written to: {final_output}")
