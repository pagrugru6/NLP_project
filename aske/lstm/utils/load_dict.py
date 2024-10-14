import json

with open("translated_questions.json", "r", encoding='utf-8') as f:
    translated_questions = json.load(f)

print(translated_questions['translated'])
