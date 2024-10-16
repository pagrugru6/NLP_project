import json

with open("../data/train/edited_ds_ja.json", "r", encoding='utf-8') as f:
    translated_questions = json.load(f)
print((translated_questions['word_overlap']))

