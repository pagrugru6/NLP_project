from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch import nn
import torch
import pandas as pd
import random
from tqdm import tqdm
from functools import partial
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, default_data_collator, EarlyStoppingCallback  

MODEL_NAME = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(samples, tokenizer=None, max_input_length=128, max_target_length=16):
    inputs = samples['question']
    answers = samples['answer_inlang']
    

    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, padding="max_length"
    )
    

    labels = tokenizer(
        answers, max_length=max_target_length, truncation=True, padding="max_length"
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

no_answers = data[data['answer_inlang'] == 'ノー']
other_answers = data[data['answer_inlang'] != 'ノー']
undersampled_no_answers = no_answers.sample(frac=0.1, random_state=42)
data = pd.concat([undersampled_no_answers, other_answers])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

train_data, valid_data = train_test_split(data, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_data)
valid_dataset = Dataset.from_pandas(valid_data)

tokenized_train = train_dataset.map(partial(prepare_data, tokenizer=tokenizer), batched=True)
tokenized_valid = valid_dataset.map(partial(prepare_data, tokenizer=tokenizer), batched=True)

model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

training_args = TrainingArguments(
    output_dir="./bart_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=0,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    weight_decay=0.01,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)
trainer.train()

model.save_pretrained('./bart_model')
tokenizer.save_pretrained('./bart_model')
