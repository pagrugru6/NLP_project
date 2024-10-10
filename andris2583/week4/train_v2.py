import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import numpy as np

# Load data
train_data = pd.read_parquet('../train.parquet', columns=['context', 'question', 'answerable', 'answer', 'lang'], filters=[('lang', 'in', ['ja','fi','ru'])])
valid_data = pd.read_parquet('../validation.parquet', columns=['context', 'question', 'answerable', 'answer', 'lang'], filters=[('lang', 'in', ['ja','fi','ru'])])

# Define Dataset class
class ContextAnswerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):  # Reduced max_length
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context = self.data.iloc[idx]['context']
        answer = self.data.iloc[idx]['answer']
        
        # Tokenize context and answer
        encoding = self.tokenizer(context, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = self.create_labels(context, answer, encoding)
        
        encoding = {key: val.squeeze() for key, val in encoding.items()}  # remove batch dimension
        encoding['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return encoding
    
    def create_labels(self, context, answer, encoding):
        """ Create labels where tokens are classified as 1 if part of the answer, else 0 """
        labels = np.zeros(len(encoding['input_ids'][0]), dtype=int)  # 0 = not part of answer, 1 = part of answer
        
        # Tokenize answer and search for it within the context
        answer_tokens = self.tokenizer.tokenize(answer)
        answer_ids = self.tokenizer.convert_tokens_to_ids(answer_tokens)
        context_ids = encoding['input_ids'][0].tolist()
        
        for i in range(len(context_ids) - len(answer_ids) + 1):
            if context_ids[i:i+len(answer_ids)] == answer_ids:
                labels[i:i+len(answer_ids)] = 1
        
        return labels

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

# Create datasets
train_dataset = ContextAnswerDataset(train_data, tokenizer)
valid_dataset = ContextAnswerDataset(valid_data, tokenizer)

# Model: BertForTokenClassification
model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,  # Reduced number of epochs
    per_device_train_batch_size=4,  # Reduced batch size
    per_device_eval_batch_size=4,  # Reduced batch size
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    fp16=True,  # Enable mixed precision training if using a compatible GPU
    gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch size
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
