# eval_confusion_minimal.py
# pip install torch transformers datasets scikit-learn matplotlib sentencepiece
from transformers import DataCollatorWithPadding
import os, json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import torch

# ----------------- EDIT THESE -----------------
MODEL_DIR = "results_flatten/albert-base-v2.vanilla/checkpoint-1734"            # your checkpoint/run folder
DATA_PATH = "data/lyrics_dataset.json"          # your dataset with fields: text, label
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 32
SHORTEN_LABELS_TO = 18      # 0 = no shorten
SORT_BY_SUPPORT = True
# ----------------------------------------------

def load_label_names(model_dir, model):
    # Prefer model.config.id2label
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        if isinstance(id2label, dict):
            keys = sorted(id2label.keys(), key=lambda k: int(k))
            return [id2label[k] for k in keys]
        return list(id2label)
    # Fallback to labels.json
    with open(os.path.join(model_dir, "labels.json")) as f:
        meta = json.load(f)
    id2 = meta["id2label"]
    if isinstance(id2, dict):
        keys = sorted(id2.keys(), key=lambda k: int(k))
        return [id2[k] for k in keys]
    return id2

def make_splits(path, seed, val_ratio, test_ratio):
    raw = load_dataset("json", data_files=path, split="train")
    raw = raw.class_encode_column("label")
    ds = raw.train_test_split(test_size=test_ratio, seed=seed, stratify_by_column="label")
    tmp = ds["train"].train_test_split(test_size=val_ratio, seed=seed, stratify_by_column="label")
    return DatasetDict(train=tmp["train"], validation=tmp["test"], test=ds["test"])

def tokenize_split(ds, tok):
    def prep(batch):
        enc = tok(batch["text"], truncation=True, max_length=MAX_LENGTH)
        enc["labels"] = batch["label"]
        return enc
    keep_cols = ["text", "label"]
    return ds.map(prep, batched=True, remove_columns=[c for c in ds.column_names if c not in keep_cols])

@torch.no_grad()
def predict_logits(model, ds):
    logits_all = []
    n = len(ds)
    for i in range(0, n, BATCH_SIZE):
        j0, j1 = i, min(i + BATCH_SIZE, n)

        # build list[dict] of features for the collator
        examples = []
        for j in range(j0, j1):
            ex = {}
            if "input_ids" in ds.column_names:       ex["input_ids"] = ds["input_ids"][j]
            if "attention_mask" in ds.column_names:  ex["attention_mask"] = ds["attention_mask"][j]
            if "token_type_ids" in ds.column_names:  ex["token_type_ids"] = ds["token_type_ids"][j]
            examples.append(ex)

        batch = collator(examples)  # pads to the longest in the batch
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(**batch).logits
        logits_all.append(logits.cpu().numpy())

    return np.concatenate(logits_all, axis=0)

def plot_cm(cm, labels, title, path, cmap="Blues", vmin=None, vmax=None, annotate=True):
    lbls = labels
    if SHORTEN_LABELS_TO and SHORTEN_LABELS_TO > 3:
        lbls = [n if len(n) <= SHORTEN_LABELS_TO else n[:SHORTEN_LABELS_TO-1] + "…" for n in labels]
    size = max(8, min(18, 0.35 * len(lbls)))
    plt.figure(figsize=(size, size))
    im = plt.imshow(cm, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(lbls)), lbls, rotation=90)
    plt.yticks(range(len(lbls)), lbls)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if annotate and len(lbls) <= 50:
        thresh = (np.nanmax(cm) + np.nanmin(cm)) / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                txt = f"{val:.2f}" if cm.dtype.kind == 'f' else str(int(val))
                plt.text(j, i, txt, ha="center", va="center",
                         color="black" if val < thresh else "white", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[saved]", path)

def eval_split(name, ds_tok, out_dir, label_names):
    y_true = np.array(ds_tok["labels"])
    logits = predict_logits(model, ds_tok)
    y_pred = np.argmax(logits, axis=-1)

    # Metrics
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    per_class = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"metrics_{name}.json"), "w") as f:
        json.dump({
            "split": name,
            "macro": {"precision": p_m, "recall": r_m, "f1": f_m},
            "weighted": {"precision": p_w, "recall": r_w, "f1": f_w},
            "per_class": {k: v for k, v in per_class.items() if k in label_names}
        }, f, indent=2)

    # Confusion matrices
    labels_idx = list(range(len(label_names)))
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm_norm   = confusion_matrix(y_true, y_pred, labels=labels_idx, normalize="true")

    # Sort by support (optional)
    if SORT_BY_SUPPORT:
        support = np.bincount(y_true, minlength=len(label_names))
        order = np.argsort(-support)
        cm_counts = cm_counts[order][:, order]
        cm_norm   = cm_norm[order][:, order]
        label_ordered = [label_names[i] for i in order]
    else:
        label_ordered = label_names

    # Plots
    plot_cm(cm_counts, label_ordered, f"Confusion Matrix ({name}) — counts",
            os.path.join(out_dir, f"confusion_counts_{name}.png"), cmap="Blues")
    plot_cm(cm_norm,   label_ordered, f"Confusion Matrix ({name}) — normalized (rows sum to 1)",
            os.path.join(out_dir, f"confusion_norm_{name}.png"), cmap="YlOrRd", vmin=0.0, vmax=1.0)
    cm_off = cm_norm.copy(); np.fill_diagonal(cm_off, 0.0)
    plot_cm(cm_off,    label_ordered, f"Confusions Only ({name}) — normalized, diagonal masked",
            os.path.join(out_dir, f"confusion_offdiag_{name}.png"), cmap="Reds", vmin=0.0, vmax=max(cm_off.max(), 1e-6), annotate=False)

# ---- Run ----
tok = AutoTokenizer.from_pretrained("albert/albert-base-v2", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)  # pads per batch

label_names = load_label_names(MODEL_DIR, model)

dds = make_splits(DATA_PATH, seed=SEED, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)
val_tok  = tokenize_split(dds["validation"], tok)
test_tok = tokenize_split(dds["test"], tok)

OUT_DIR = os.path.join(MODEL_DIR, "metrics_eval")
eval_split("validation", val_tok, OUT_DIR, label_names)
eval_split("test",       test_tok, OUT_DIR, label_names)

print("Done. Files in:", OUT_DIR)
