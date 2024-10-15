# The purpose of this file is to grab and translate the question from the
# dataset and save it to a file. In all the languages


from datasets import load_dataset, Dataset
import sys
sys.path.append('../utils')
from nlp_utils import *
from transformers import MarianMTModel, MarianTokenizer
import json

ds = load_dataset("coastalcph/tydi_xor_rc")
ds_val = ds['validation']
ds_train = ds['train']
ds_ru = ds_train.filter(filter_ru), "ru"
ds_fi = ds_train.filter(filter_fi), "fi"
ds_ja = ds_train.filter(filter_ja), "ja"
ds_ru_val = ds_val.filter(filter_ru), "ru"
ds_fi_val = ds_val.filter(filter_fi), "fi"
ds_ja_val = ds_val.filter(filter_ja), "ja"
# Load a translation model (e.g., English to French)
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = "val" # val
train_list = [ds_ru, ds_fi, ds_ja]
val_list = [ds_ru_val, ds_fi_val, ds_ja_val]
if dataset == 'val':
    ds_lst = val_list
else:
    ds_lst = train_list
def edit_questions(ds):
    ds['translated'] = translate(ds['question'])
    return ds

def save_to_json(ds, filename):
    with open(f"../data/{dataset}/{filename}", "w", encoding='utf-8') as f:
        json.dump(ds.to_dict(), f)

for curr_ds, lang in ds_lst:
    model_name = f"Helsinki-NLP/opus-mt-{lang}-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    # Define a translation function
    def translate(crazy_text):
        inputs = tokenizer(crazy_text, return_tensors="pt", padding=True,
                           truncation=True).to(device)
        translated = model.generate(**inputs)
        translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return translations[0]


    # curr_ds = curr_ds.select(range(2))
    curr_ds = curr_ds.map(edit_questions)
    curr_ds = curr_ds.map(add_word_overlap)
    # save the translated questions to a file
    save_to_json(curr_ds, f"edited_ds_{lang}.json")
