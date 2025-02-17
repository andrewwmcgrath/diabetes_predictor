import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# Define the file path
file_path = r"C:\Users\andre\Dissertation\Saved to Local Drive\big_balanced_data.csv"

try:
    # Load the data
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with shape: {df.shape}")
    
    # Extract column names
    columns = df.columns[1:]  # Skip the first column if it's an ID column
    
    descriptions = []
    
    for column in columns:
        field_id = column.split('-')[0]
        url = f"https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id={field_id}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            description_tag = soup.find('td', text='Description:').find_next_sibling('td')
            
            if description_tag:
                description = description_tag.text.strip()
            else:
                description = "Description not found"
            
            descriptions.append(description)
            print(f"Field ID {field_id}: {description}")
            
            # Add a delay to avoid overwhelming the server
            time.sleep(1)
        
        except Exception as e:
            print(f"Failed to fetch description for Field ID {field_id}: {e}")
            descriptions.append("Description not found")
    
    # Add the descriptions as a new row in the DataFrame
    new_row = pd.DataFrame([[''] + descriptions], columns=df.columns)
    df = pd.concat([new_row, df], ignore_index=True)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(file_path, index=False)
    print(f"Updated data with descriptions has been written to: {output_file_path}")

except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
