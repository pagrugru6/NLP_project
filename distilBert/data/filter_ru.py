import json

def filter_ru_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ru_data = [entry for entry in data if entry['lang'] == 'ru']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ru_data, f, ensure_ascii=False, indent=4)

# Define paths to your train and validation files
train_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/train.json'
validation_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/validation.json'

# Define paths to the output files
train_ru_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/train_ru.json'
validation_ru_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/validation_ru.json'

# Filter and save the ru data
filter_ru_data(train_file, train_ru_file)
filter_ru_data(validation_file, validation_ru_file)