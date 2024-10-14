from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast

# Load the checkpoint
model = DistilBertForQuestionAnswering.from_pretrained('./results/checkpoint-432')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')

# Save the model and tokenizer to the models folder
model.save_pretrained('./models/distilbert_ja')
tokenizer.save_pretrained('./models/distilbert_ja')
