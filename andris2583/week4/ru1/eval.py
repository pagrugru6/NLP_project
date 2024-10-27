from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, classification_report
import pandas as pd
from torch.utils.data import DataLoader
from util import CustomDataset, tokenize_and_align_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_save_path = "./model_weights"


print("Loading model and tokenizer from the saved checkpoint...")
model = BertForTokenClassification.from_pretrained(model_save_path).to(device)  
tokenizer = BertTokenizerFast.from_pretrained(model_save_path)


data = pd.read_parquet('../../../Translated_Questions/translated_ru_rows.parquet', 
                       columns=['context', 'question', 'answerable', 'answer', 'lang'], 
                       filters=[('lang', 'in', ['ru'])])


train_data, valid_data = train_test_split(data, test_size=0.1, random_state=42)

tokenized_valid_data = tokenize_and_align_labels(valid_data, tokenizer)

valid_dataset = CustomDataset(tokenized_valid_data['input_ids'], 
                              tokenized_valid_data['attention_mask'], 
                              tokenized_valid_data['labels'])


training_args = TrainingArguments(
    output_dir=model_save_path,         
    per_device_eval_batch_size=64,      
    eval_strategy="epoch",              
    logging_dir='./logs',               
)


trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=valid_dataset  
)


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

