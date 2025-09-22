# Lyrics Artist Classification with ALBERT

This project predicts the **artist of a song from its lyrics** using transformer-based models. This is a hobby project that made me realize that this system is as good as the LLM that produces lyrics in the style of the select artist. Prompt tuning will make the generated lyrics much more cohesive and better represent the artist the LLM wants to mimic.
I began with a *vanilla ALBERT* pipeline and improved it with a **flattening strategy** to handle long songs fairly.

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

### 4. Using GEMMA 4b for lyrics generation
I used Gemma locally using Ollama to generate lyrics based on the prompts given for the artist style. Once the lyrics are generated, the lryics are classified by Albert. (highly flawed, but interesting stuff to investigate.)

---

## 📊 Results
- **Vanilla**: Good accuracy but biased toward artists with longer songs.  
- **Flattened**: More balanced per-class performance; confusion matrix is much fairer.
- *System (LLM + ALBERT Classification)*: Couldn't see how well the system performs because of the time it might take to generate lyrics for each artist multiple time and then classify. The score will be a mystery for now, but maybe I will try to investigate this later on.

---

## 🚀 Run
```bash
pip install -r requirements.txt
```
## To train:
```bash
python scripts/trainers/albert_vanilla.py
python scripts/trainers/albert_weighted.py
```
## To generate graphs:
```bash
python scripts/graph_gen.py
```

## To run the system:
```bash
# Use the Eminem continuation (default)
python predict_artist.py

# Use the Hozier-style continuation
python predict_artist.py --prompt hozier

# Use the Weeknd-style prompt
python predict_artist.py --prompt weeknd

# Use the “future” continuation prompt (your neon-lights seed)
python predict_artist.py --prompt future

# Bypass presets and pass a custom prompt
python predict_artist.py --text "Write minimalistic lyrics about memory and rain in the style of a 90s alt band..."
```

Outputs:  
- `metrics_*_chunk_level.json` – raw chunk-level scores  
- `confusion_*_per_sample.npy`, `classification_*_per_sample.txt` – per-song results

---

## 🙌 Credits
- Lyrics: Genius API  
- Models: HuggingFace Transformers  
- Author: Jaivanth Melanaturu