import json
from datasets import load_dataset

# Define the output file path
output_file_path = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/train.json'

print(f"Training data has been saved to {output_file_path}")

# Load the dataset
dataset = load_dataset("coastalcph/tydi_xor_rc")

# Extract the training data
train_data = dataset['train']

# Convert the training data to a list of dictionaries
train_data_list = [dict(record) for record in train_data]

# Define the output file path
output_file_path = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/train.json'

# Save the training data to a JSON file
with open(output_file_path, 'w') as f:
    json.dump(train_data_list, f, indent=4)
