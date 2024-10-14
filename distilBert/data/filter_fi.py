import json

def filter_fi_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fi_data = [entry for entry in data if entry['lang'] == 'fi']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fi_data, f, ensure_ascii=False, indent=4)

# Define paths to your train and validation files
train_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/train.json'
validation_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/validation.json'

# Define paths to the output files
train_fi_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/train_fi.json'
validation_fi_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/validation_fi.json'

# Filter and save the fi data
filter_fi_data(train_file, train_fi_file)
filter_fi_data(validation_file, validation_fi_file)