import os

def count_lines(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for line in file)

def main():
    data_folder = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data'
    files_to_check = [
        'train_fi.json', 'train_ja.json', 'train_ru.json',
        'validation_fi.json', 'validation_ja.json', 'validation_ru.json'
    ]

    for file_name in files_to_check:
        file_path = os.path.join(data_folder, file_name)
        lines_count = count_lines(file_path)
        print(f"Number of lines in {file_name}: {lines_count}")

if __name__ == "__main__":
    main()