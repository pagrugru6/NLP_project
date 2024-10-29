import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

df = pd.read_parquet('translated_ru_rows.parquet')

model_name = 'facebook/m2m100_418M'
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

tokenizer.src_lang = "en"
target_lang = "ru"

def translate_text(text):
    if text:
        encoded_text = tokenizer(text, return_tensors='pt').to(device)
        translated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
        return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return text

df['answer_inlang'] = df['answer'].apply(translate_text)

df.to_parquet('translated_ru_rows_v2.parquet')
