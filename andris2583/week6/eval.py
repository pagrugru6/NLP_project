from sklearn.model_selection import train_test_split
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from functools import partial
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = './bart_model/'
model = BartForConditionalGeneration.from_pretrained(model_dir).to(device)
tokenizer = BartTokenizer.from_pretrained(model_dir)

data = pd.concat([pd.read_parquet('../../Translated_Questions//translated_fi_rows.parquet', 
                       columns=['context', 'question', 'answerable','answer', 'lang','answer_inlang'], 
                       filters=[('lang', 'in', ['fi'])]),pd.read_parquet('../../Translated_Questions//translated_ru_rows.parquet', 
                       columns=['context', 'question', 'answerable','answer', 'lang','answer_inlang'], 
                       filters=[('lang', 'in', ['ru'])]),pd.read_parquet('../../Translated_Questions//translated_ja_rows.parquet', 
                       columns=['context', 'question', 'answerable','answer', 'lang','answer_inlang'], 
                       filters=[('lang', 'in', ['ja'])])])

train_data, valid_data = train_test_split(data, test_size=0.1, random_state=42)

valid_data = valid_data[valid_data['answer'].notnull()]

valid_data['id'] = valid_data.index.astype(str)  
valid_dataset = Dataset.from_pandas(valid_data)

def prepare_data(samples, tokenizer=None, max_input_length=256, max_target_length=64):
    contexts = samples['context']
    questions = samples['question']
    answers = samples['answer']

    inputs = [f"{context} {question}" for context, question in zip(contexts, questions)]
    
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
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def generate_answers(model, valid_dl):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(valid_dl, desc="Generating answers"):
            batch = {k: v.to(device) for k, v in batch.items()}

            generated_ids = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=64,
                num_beams=4
            )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_predictions.extend(preds)
    return all_predictions

tokenized_valid = valid_dataset.map(partial(prepare_data, tokenizer=tokenizer), batched=True)
valid_dl = DataLoader(tokenized_valid, collate_fn=collate_fn, shuffle=False, batch_size=4)

generated_answers = generate_answers(model, valid_dl)
predictions = [{'id': str(i), 'prediction_text': pred} for i, pred in enumerate(generated_answers)]
gold = [{'id': example['id'], 'answers': example['answer']} for _, example in valid_data.iterrows()]

def compute_metrics(predictions, references, tokenizer):
    exact_match = total = bleu_score_total = 0
    chencherry = SmoothingFunction()

    for pred, ref in zip(predictions, references):
        total += 1
        pred_text = pred["prediction_text"]
        true_text = ref["answers"] if ref["answers"] is not None else ""

        exact_match += exact_match_score(pred_text, true_text)

        if true_text:  # Proceed only if true_text is not empty or None
            pred_tokens = tokenizer.tokenize(pred_text)
            ref_tokens = tokenizer.tokenize(true_text)
            bleu_score_total += sentence_bleu([ref_tokens], pred_tokens, smoothing_function=chencherry.method2)
        else:
            total -= 1  # Adjust total count if skipping this instance

    exact_match = 100.0 * exact_match / total if total > 0 else 0
    avg_bleu = bleu_score_total / total if total > 0 else 0

    return {'exact_match': exact_match, 'bleu': avg_bleu}

def exact_match_score(prediction, ground_truth):
    if ground_truth == None or prediction == None:
        return False
    return prediction.strip().lower() == ground_truth.strip().lower()

results = compute_metrics(predictions, gold, tokenizer)

for i, prediction in enumerate(predictions):
    answer = valid_data.iloc[i]['answer']
    predicted_answer = prediction['prediction_text']
    
    print(f"Example {i + 1}")
    print(f"Answer: {answer}")
    print(f"Predicted Answer: {predicted_answer}\n")
    
print(f"Evaluation Results:\nExact Match: {results['exact_match']:.2f}%\nBLEU Score: {results['bleu']:.4f}")
