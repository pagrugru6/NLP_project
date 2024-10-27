import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForTokenClassification

from util import CustomDataset, tokenize_and_align_labels


model_save_path = "./model_weights"
tokenizer = BertTokenizer.from_pretrained(model_save_path)
model = BertForTokenClassification.from_pretrained(model_save_path)
model.eval()  


MAX_LENGTH = 512

data = pd.read_parquet('../../../Translated_Questions/translated_fi_rows.parquet', 
                       columns=['context', 'question', 'answerable', 'answer', 'lang'], 
                       filters=[('lang', 'in', ['fi'])])


train_data, valid_data = train_test_split(data, test_size=0.1, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def extract_first_contiguous_answer(tokens, pred_labels):
    answer_tokens = []
    found_answer = False
    
    for token, label in zip(tokens, pred_labels):
        if label == 1 and not found_answer:
            answer_tokens.append(token)
        elif label == 0 and answer_tokens:
            found_answer = True
            break

    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer

for context, question, answerable, real_answer, lang in list(valid_data.itertuples(index=False, name=None))[:200]:
  
  inputs = tokenizer(context, question, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors="pt")

  
  inputs = {key: val.to(device) for key, val in inputs.items()}

  
  with torch.no_grad():  
      outputs = model(**inputs)
      
  
  logits = outputs.logits
  predictions = torch.argmax(logits, dim=-1).cpu().numpy()

  
  input_ids = inputs['input_ids'].cpu().numpy()[0]  
  pred_labels = predictions[0]  

  
  tokens = tokenizer.convert_ids_to_tokens(input_ids)

  
  answer_tokens = [token for token, label in zip(tokens, pred_labels) if label == 1 and token != '[PAD]']

  if(len(answer_tokens) != 0):
    print("Real answers: ",real_answer,"\nPredicted answer: ",answer_tokens,"\n")
    