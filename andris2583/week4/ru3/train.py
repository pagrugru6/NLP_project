from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, Trainer, TrainingArguments
from util import CustomDataset, tokenize_and_align_labels
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import BertTokenizerFast


model_save_path = "./model_weights"  

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_parquet('../../../Translated_Questions/translated_ru_rows.parquet', 
                       columns=['context', 'question', 'answerable', 'answer', 'lang'], 
                       filters=[('lang', 'in', ['ru'])])


train_data, valid_data = train_test_split(data, test_size=0.1, random_state=42)


tokenized_train_data = tokenize_and_align_labels(train_data, tokenizer)
tokenized_valid_data = tokenize_and_align_labels(valid_data, tokenizer)

batch_size = 16
train_dataset = CustomDataset(tokenized_train_data['input_ids'], 
                              tokenized_train_data['attention_mask'], 
                              tokenized_train_data['labels'])

valid_dataset = CustomDataset(tokenized_valid_data['input_ids'], 
                              tokenized_valid_data['attention_mask'], 
                              tokenized_valid_data['labels'])


model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        
        loss_fct = CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=model_save_path,            
    num_train_epochs=15,  
    per_device_train_batch_size=batch_size,  
    per_device_eval_batch_size=64,         
    warmup_steps=500,                      
    weight_decay=0.01,                     
    logging_dir='./logs',                  
    logging_steps=10,                      
    save_strategy="epoch",                 
    eval_strategy="epoch",                 
    load_best_model_at_end=True,           
    save_total_limit=0,                    
    learning_rate=1e-5,  
    gradient_accumulation_steps=8,  
    fp16=True                              
)


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  
    eval_dataset=valid_dataset    
)


trainer.train()

print("Saving the model...")
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("Evaluating the model on the validation dataset...")
eval_results = trainer.evaluate()

def get_predictions(model, dataloader):
    model.eval()
    predictions, true_labels = [], []

    for batch in dataloader:
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        
        if isinstance(input_ids, list) or isinstance(input_ids, str):
            input_ids = torch.tensor(input_ids)
        if isinstance(attention_mask, list) or isinstance(attention_mask, str):
            attention_mask = torch.tensor(attention_mask)
        if isinstance(labels, list) or isinstance(labels, str):
            labels = torch.tensor(labels)
        
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        
        predicted_labels = torch.argmax(logits, dim=2)

        
        predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels


valid_dataloader = DataLoader(valid_dataset, batch_size=64)


predictions, true_labels = get_predictions(model, valid_dataloader)


flattened_predictions = [pred for batch in predictions for pred in batch]
flattened_true_labels = [true for batch in true_labels for true in batch]


filtered_preds = [pred for pred, label in zip(flattened_predictions, flattened_true_labels) if label != -100]
filtered_labels = [label for label in flattened_true_labels if label != -100]


precision, recall, f1, _ = precision_recall_fscore_support(filtered_labels, filtered_preds, average='weighted')


report = classification_report(filtered_labels, filtered_preds)


print(f"Evaluation Results:\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}")
print(f"\nClassification Report:\n{report}")
