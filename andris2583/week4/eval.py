from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from util import CustomDataset, tokenize_and_align_labels


model_save_path = "./model_weights"

# Load the trained model and tokenizer
print("Loading model and tokenizer from the saved checkpoint...")
model = BertForTokenClassification.from_pretrained(model_save_path)
tokenizer = BertTokenizer.from_pretrained(model_save_path)

valid_data = pd.read_parquet('./validation.parquet', columns=['context', 'question', 'answerable', 'answer', 'lang'], filters=[('lang', 'in', ['ja'])])
tokenized_valid_data = tokenize_and_align_labels(valid_data, tokenizer)
valid_dataset = CustomDataset(tokenized_valid_data['input_ids'], tokenized_valid_data['attention_mask'], tokenized_valid_data['labels'])


# TrainingArguments (reuse for evaluation)
training_args = TrainingArguments(
    output_dir=model_save_path,         # Output directory
    per_device_eval_batch_size=64,      # Batch size for evaluation
    eval_strategy="epoch",              # Ensure the evaluation strategy is set
    logging_dir='./logs',               # Directory for storing logs
)

# Trainer setup (reuse for evaluation)
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=valid_dataset  # Validation data
)

# Perform evaluation
print("Evaluating the model on the validation dataset...")
eval_results = trainer.evaluate()

# Print evaluation results
print(f"Evaluation results: {eval_results}")