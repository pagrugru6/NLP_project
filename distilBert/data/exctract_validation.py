import json
from datasets import load_dataset

# Define the output file path
output_file_path = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/validation.json'

print(f"Validation data has been saved to {output_file_path}")

# Load the dataset
dataset = load_dataset("coastalcph/tydi_xor_rc")

# Extract the validation data
validation_data = dataset['validation']

# Convert the validation data to a list of dictionaries
validation_data_list = [dict(record) for record in validation_data]

# Save the validation data to a JSON file
with open(output_file_path, 'w') as f:
    json.dump(validation_data_list, f, indent=4)
