from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
model = AutoModelForSequenceClassification.from_pretrained(
    "models/results/albert-base-v2.vanilla/checkpoint-1734", local_files_only=True
)
model.eval()


id2label = {
    0: "Ariana Grande",
    1: "Billie Eilish",
    2: "Drake",
    3: "Ed Sheeran",
    4: "Eminem",
    5: "Future",
    6: "Hozier",
    7: "Lil Baby",
    8: "Post Malone",
    9: "Sza",
    10: "Taylor Swift",
    11: "The Weeknd",
    12: "Travis Scott",
}
label2id = {
    "Ariana Grande": 0,
    "Billie Eilish": 1,
    "Drake": 2,
    "Ed Sheeran": 3,
    "Eminem": 4,
    "Future": 5,
    "Hozier": 6,
    "Lil Baby": 7,
    "Post Malone": 8,
    "Sza": 9,
    "Taylor Swift": 10,
    "The Weeknd": 11,
    "Travis Scott": 12,
}


def main(lyrics_text):
    inputs = tokenizer(lyrics_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = int(torch.argmax(outputs.logits).item())
        print(f"Predicted Artist: {id2label[predicted_class_id]}")
        return id2label[predicted_class_id]
