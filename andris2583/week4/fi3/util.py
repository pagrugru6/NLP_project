from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report


MAX_LENGTH = 512

def tokenize_and_align_labels(data, tokenizer, max_length=512):
    tokenized_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for i, context in enumerate(data['context']):
        context = data['context'].iloc[i]
        question = data['question'].iloc[i]
        answer = data['answer'].iloc[i]

      
        encoded_input = tokenizer(
            question,
            context,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True, 
        )

      
        offsets = encoded_input.pop("offset_mapping")
        
      
        labels = [0] * len(encoded_input['input_ids'])

      
        answer_start_idx = context.find(answer)
        if answer_start_idx != -1:
            answer_end_idx = answer_start_idx + len(answer)
            
          
            for idx, (start, end) in enumerate(offsets):
                if start >= answer_start_idx and end <= answer_end_idx:
                    labels[idx] = 1 

      
        if len(labels) < max_length:
            labels += [-100] * (max_length - len(labels))

      
        tokenized_inputs["input_ids"].append(encoded_input["input_ids"])
        tokenized_inputs["attention_mask"].append(encoded_input["attention_mask"])
        tokenized_inputs["labels"].append(labels)

  
    tokenized_inputs["input_ids"] = torch.tensor(tokenized_inputs["input_ids"])
    tokenized_inputs["attention_mask"] = torch.tensor(tokenized_inputs["attention_mask"])
    tokenized_inputs["labels"] = torch.tensor(tokenized_inputs["labels"])

    return tokenized_inputs

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item
