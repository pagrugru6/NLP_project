from datasets import Dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch import nn
import torch
import pandas as pd
import random
from tqdm import tqdm
from functools import partial

MODEL_NAME = 'google/mt5-small'
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(samples, tokenizer=None, max_input_length=512, max_target_length=64):
    """
    Tokenize input data in the format: "[context] [question]" and target as the answer.
    """
    # Treat samples as a dictionary with lists
    contexts = samples['context']
    questions = samples['question']
    answers = samples['answer_inlang']

    # Prepare the input text as "[context] [question]"
    inputs = [f"{context} {question}" for context, question in zip(contexts, questions)]
    
    # Tokenize inputs (context + question)
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, padding="max_length"
    )

    # Tokenize targets (answers)
    labels = tokenizer(
        answers, max_length=max_target_length, truncation=True, padding="max_length"
    ).input_ids

    model_inputs["labels"] = labels
    return model_inputs


def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def train(model, train_dl, optimizer, scheduler, n_epochs, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    for ep in range(n_epochs):
        running_loss = 0.0
        optimizer.zero_grad()

        # Iterate through each batch
        for batch in tqdm(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps

            # Backward pass with mixed precision
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        print(f"Epoch {ep+1}/{n_epochs} Loss: {running_loss/len(train_dl)}")


# Data loading
train_data = pd.read_parquet('../..train.parquet', columns=['context', 'question', 'answerable', 'answer', 'lang','answer_inlang'], filters=[('lang', 'in', ['ja']),('answer_inlang', '!=', 'null')])
valid_data = pd.read_parquet('../..validation.parquet', columns=['context', 'question', 'answerable', 'answer', 'lang','answer_inlang'], filters=[('lang', 'in', ['ja']),('answer_inlang', '!=', 'null')])

train_data = Dataset.from_pandas(train_data)
valid_data = Dataset.from_pandas(valid_data)

# Tokenization
tokenized_train = train_data.map(partial(prepare_data, tokenizer=tokenizer), batched=True)
tokenized_valid = valid_data.map(partial(prepare_data, tokenizer=tokenizer), batched=True)

train_dl = DataLoader(tokenized_train, collate_fn=collate_fn, shuffle=True, batch_size=4)
valid_dl = DataLoader(tokenized_valid, collate_fn=collate_fn, shuffle=False, batch_size=4)

# Model setup
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
n_epochs = 3
accumulation_steps = 4

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=n_epochs * len(train_dl))

# Training
train(model, train_dl, optimizer, scheduler, n_epochs, device)

# Save the model
model.save_pretrained('./mt5_model')
tokenizer.save_pretrained('./mt5_model')
