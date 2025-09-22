# Lyrics Artist Classification with ALBERT

This project predicts the **artist of a song from its lyrics** using transformer-based models.  
We began with a *vanilla ALBERT* pipeline and improved it with a **flattening strategy** to handle long songs fairly.

---

## 📂 Workflow

### 1. Data
- **Scraped lyrics** with the [Genius API](https://docs.genius.com/) per artist.  
- **Cleaned** text by removing section headers (`[Chorus]`, `[Verse]`), uploader tags, and extra whitespace.  
- Stored as JSON:  
  ```json
  {"text": "lyrics ...", "label": "ArtistName"}
  ```

### 2. Baseline: Vanilla ALBERT
- Model: `albert-base-v2` (HuggingFace).  
- Stratified splits: train/val/test.  
- Used sliding windows (max length 256, stride 192) so long lyrics were broken into overlapping chunks.  
- Worked, but confusion matrices showed uneven performance across artists.

### 3. Flattening: Making It Fair
Sliding windows gave long songs many more training/eval samples. To fix this:
- **Per-sample weights** – each chunk weighted `1 / (#chunks of its song)` so every song contributes equally.  
- **Per-sample evaluation** – averaged logits across chunks before predicting → fair confusion matrices.  
- **Optional cap** – limited max chunks per song (e.g. 4) to prevent flooding.

---

## 📊 Results
- **Vanilla**: good accuracy but biased toward artists with longer songs.  
- **Flattened**: more balanced per-class performance; confusion matrix is much fairer.  

---

## 🚀 Run
```bash
pip install -r requirements.txt
python train_vanilla.py
```

Outputs:  
- `metrics_*_chunk_level.json` – raw chunk-level scores  
- `confusion_*_per_sample.npy`, `classification_*_per_sample.txt` – per-song results

---

## 🙌 Credits
- Lyrics: Genius API  
- Models: HuggingFace Transformers  
- Author: Jaivanth Melanaturu