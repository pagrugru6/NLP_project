from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# Initialize the tokenizer

MAX_LENGTH = 512

# Tokenize and align labels
def tokenize_and_align_labels(data,tokenizer):
    tokenized_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for i, context in enumerate(list(data['context'])):
        question = list(data['question'])[i]
        answer = list(data['answer'])[i]

        # Tokenize the context and question
        tokenized_context = tokenizer.tokenize(context)
        tokenized_question = tokenizer.tokenize(question)

        # Skip example if length exceeds MAX_LENGTH
        if len(tokenized_context) + len(tokenized_question) > MAX_LENGTH:
            continue

        # Tokenize context and answer
        context_tokens = tokenizer.tokenize(context)
        answer_tokens = tokenizer.tokenize(answer)
        
        label_ids = [0] * len(context_tokens)

        # Find the start index of the answer in the context
        start_index = context.find(answer)
        if start_index != -1:
            pre_answer_context = context[:start_index]
            pre_answer_tokens = tokenizer.tokenize(pre_answer_context)
            answer_start_token_idx = len(pre_answer_tokens)

            for j in range(0, len(answer_tokens)):
                label_ids[answer_start_token_idx + j] = 1  # Label answer tokens as '1'

        # Tokenize and encode for BERT
        input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
        attention_mask = [1] * len(input_ids)
        
        # Add padding if necessary
        padding_length = MAX_LENGTH - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        label_ids += [-100] * padding_length  # Ignore padded tokens during loss computation

        # Append to the tokenized inputs
        tokenized_inputs['input_ids'].append(input_ids)
        tokenized_inputs['attention_mask'].append(attention_mask)
        tokenized_inputs['labels'].append(label_ids)
    
    # Convert lists to tensors
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
