from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch import nn
import torch
import pandas as pd
import random
from tqdm import tqdm
from functools import partial
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, default_data_collator

MODEL_NAME = 'google/mt5-small'
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(samples, tokenizer=None, max_input_length=64, max_target_length=32):
    english_answers = samples['answer']
    inlang_answers = samples['answer_inlang']
    
    model_inputs = tokenizer(
        english_answers, max_length=max_input_length, truncation=True, padding="max_length"
    )
    
    labels = tokenizer(
        inlang_answers, max_length=max_target_length, truncation=True, padding="max_length"
    ).input_ids

    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_example]
        for label_example in labels
    ]
    
    model_inputs["labels"] = labels
    return model_inputs

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

data = pd.read_parquet('../../../../Translated_Questions/Only_Answers/translated_ja_answers.parquet', 
                       columns=['context', 'question', 'answerable','answer', 'lang','answer_inlang'], 
                       filters=[('lang', 'in', ['ja'])])

train_data, valid_data = train_test_split(data, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_data)
valid_dataset = Dataset.from_pandas(valid_data)

tokenized_train = train_dataset.map(partial(prepare_data, tokenizer=tokenizer), batched=True)
tokenized_valid = valid_dataset.map(partial(prepare_data, tokenizer=tokenizer), batched=True)

model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

training_args = TrainingArguments(
    output_dir="./mt5_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=7,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    save_total_limit=1,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)
trainer.train()

model.save_pretrained('./mt5_model')
tokenizer.save_pretrained('./mt5_model')
