import argparse, os, json, random, numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def build_label_maps(dataset, label_key="label"):
    labels = sorted(list(set(dataset[label_key])))
    id2label = {i: lbl for i, lbl in enumerate(labels)}
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    return labels, id2label, label2id

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    # Also report macro-F1 for class balance visibility
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_weighted": f1, "f1_macro": f1_macro, "precision_w": p, "recall_w": r}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSON/JSONL dataset with fields: text, label")
    ap.add_argument("--model", default="roberta-base", help="HF model id (e.g., roberta-base, distilbert-base-uncased)")
    ap.add_argument("--out", default="results/vanilla", help="Output dir for checkpoints & logs")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation split from train portion")
    ap.add_argument("--test_ratio", type=float, default=0.1, help="Test split from full dataset")
    ap.add_argument("--max_length", type=int, default=256)
    return ap.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    # 1) Load full dataset (single JSON or JSONL file)
    #    load_dataset infers JSON vs JSONL automatically.
    raw = load_dataset("json", data_files=args.data, split="train")

    # 2) Split: first carve out test, then make a val split from the remaining
    ds = raw.train_test_split(test_size=args.test_ratio, seed=args.seed, stratify_by_column="label")
    tmp = ds["train"].train_test_split(test_size=args.val_ratio, seed=args.seed, stratify_by_column="label")
    dds = DatasetDict(train=tmp["train"], validation=tmp["test"], test=ds["test"])

    # 3) Labels
    all_labels, id2label, label2id = build_label_maps(dds["train"])

    # 4) Tokenizer & encode
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def preprocess(batch):
        enc = tok(batch["text"], truncation=True, max_length=args.max_length)
        enc["labels"] = [label2id[lbl] for lbl in batch["label"]]
        return enc

    dds = dds.map(preprocess, batched=True, remove_columns=dds["train"].column_names)
    collator = DataCollatorWithPadding(tokenizer=tok)

    # 5) Fresh model init (key point: starts from base weights, not your old checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # 6) Training args — explicitly do NOT resume; overwrite_output_dir ensures fresh run
    training_args = TrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,       # blow away any old state under this folder
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        report_to=["none"],              # set to ["tensorboard"] if you log TB
        seed=args.seed,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dds["train"],
        eval_dataset=dds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # 7) Train from scratch (fresh base model) — no resume, no checkpoint path
    trainer.train()  # DON'T pass resume_from_checkpoint

    # 8) Evaluate & save label mapping
    metrics = trainer.evaluate(dds["validation"])
    test_metrics = trainer.evaluate(dds["test"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics_val.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "metrics_test.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(args.out, "labels.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print("Val:", metrics)
    print("Test:", test_metrics)

if __name__ == "__main__":
    main()