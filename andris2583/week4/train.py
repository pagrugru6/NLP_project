from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

from util import CustomDataset, tokenize_and_align_labels


model_save_path = "./model_weights"  # Path to save model checkpoints

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Load data
# train_data = pd.read_parquet('./train.parquet', columns=['context', 'question', 'answerable', 'answer', 'lang'], filters=[('lang', 'in', ['ja'])])
# valid_data = pd.read_parquet('./validation.parquet', columns=['context', 'question', 'answerable', 'answer', 'lang'], filters=[('lang', 'in', ['ja'])])
data = pd.read_parquet('./translated_ja_rows.parquet', columns=['context', 'question', 'answerable', 'answer', 'lang'], filters=[('lang', 'in', ['ja'])])

# tokenized_train_data = tokenize_and_align_labels(train_data, tokenizer)
# tokenized_valid_data = tokenize_and_align_labels(valid_data, tokenizer)
tokenized_data = tokenize_and_align_labels(data, tokenizer)

# Create TensorDataset and DataLoader
batch_size = 16
# train_dataset = CustomDataset(tokenized_train_data['input_ids'], tokenized_train_data['attention_mask'], tokenized_train_data['labels'])
# valid_dataset = CustomDataset(tokenized_valid_data['input_ids'], tokenized_valid_data['attention_mask'], tokenized_valid_data['labels'])
dataset = CustomDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], tokenized_data['labels'])

# Initialize the model
model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

# Load previously saved model if available
if os.path.exists(os.path.join(model_save_path, 'pytorch_model.bin')):
    print("Loading saved model weights...")
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'pytorch_model.bin')))

# Training arguments
training_args = TrainingArguments(
    output_dir=model_save_path,         # Output directory
    num_train_epochs=3,                 # Number of training epochs
    per_device_train_batch_size=batch_size,  # Batch size for training
    per_device_eval_batch_size=64,      # Batch size for evaluation
    warmup_steps=500,                   # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                  # Strength of weight decay
    logging_dir='./logs',               # Directory for storing logs
    logging_steps=10,                   # Log every 10 steps
    save_strategy="epoch",              # Save model at the end of each epoch
    eval_strategy="epoch",        # Evaluate at the end of each epoch
    load_best_model_at_end=True,        # Load the best model at the end of training
    save_total_limit=2                  # Limit the number of saved checkpoints
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Training data
    eval_dataset=valid_dataset    # Validation data
)

# Train the model
trainer.train()

# Save the trained model
print("Saving the model...")
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)