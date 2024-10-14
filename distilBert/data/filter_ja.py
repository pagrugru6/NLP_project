import json
import os

# the data looks like this: 
    # {
    #     "question": "\u0989\u0987\u0995\u09bf\u09b2\u09bf\u0995\u09b8 \u0995\u09a4 \u09b8\u09be\u09b2\u09c7 \u09b8\u09b0\u09cd\u09ac\u09aa\u09cd\u09b0\u09a5\u09ae \u0987\u09a8\u09cd\u099f\u09be\u09b0\u09a8\u09c7\u099f\u09c7 \u09aa\u09cd\u09b0\u09a5\u09ae \u09a4\u09a5\u09cd\u09af \u09aa\u09cd\u09b0\u09a6\u09b0\u09cd\u09b6\u09a8 \u0995\u09b0\u09c7 ?",
    #     "context": "WikiLeaks () is an international non-profit organisation that publishes secret information, news leaks, and classified media provided by anonymous sources. Its website, initiated in 2006 in Iceland by the organisation Sunshine Press, claims a database of 10 million documents in 10 years since its launch. Julian Assange, an Australian Internet activist, is generally described as its founder and director. Kristinn Hrafnsson is its editor-in-chief.",
    #     "lang": "bn",
    #     "answerable": true,
    #     "answer_start": 182,
    #     "answer": "2006",
    #     "answer_inlang": null
    # },

# extract the train and validation data from the json files that have lang == ja only, in two new train and validation ja files: 
def filter_ja_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ja_data = [entry for entry in data if entry['lang'] == 'ja']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ja_data, f, ensure_ascii=False, indent=4)

# Define paths to your train and validation files
train_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/train.json'
validation_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/validation.json'

# Define paths to the output files
train_ja_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/train_ja.json'
validation_ja_file = '/Users/PAG/Desktop/Copenhagen 2024/NLP/NLP_Project/week3/data/validation_ja.json'

# Filter and save the ja data
filter_ja_data(train_file, train_ja_file)
filter_ja_data(validation_file, validation_ja_file)

