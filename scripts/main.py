from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = AutoModelForSequenceClassification.from_pretrained("../training/results/checkpoint-297", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
model = AutoModelForSequenceClassification.from_pretrained("trainers/results/roberta-base/checkpoint-297", local_files_only=True)
model.eval()

id2label = {0: 'Eminem', 1: 'Future', 2: 'Hozier', 3: 'The Weeknd'}

def main(lyrics_text):
    inputs = tokenizer(lyrics_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = torch.argmax(outputs.logits).item()
        print(f"Predicted Artist: {id2label[predicted_class_id]}")
        return id2label[predicted_class_id]