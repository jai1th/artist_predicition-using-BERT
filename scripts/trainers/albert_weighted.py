# train_vanilla.py (per-sample fair training + per-sample evaluation)
import inspect
import os, json, random
from dataclasses import dataclass, asdict
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from collections import Counter, defaultdict
import torch, torch.nn.functional as F

# =============== CONFIG ===============
@dataclass
class Config:
    data: str = "D:/projects/lyr_pred/artist_predicition-using-BERT/ubernew/data/lyrics_dataset.json"
    model: str = "albert/albert-base-v2"          # requires: pip install sentencepiece
    out: str = "results_flatten/albert-base-v2.vanilla"
    batch: int = 16
    epochs: int = 3
    lr: float = 1.5e-5
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    seed: int = 42
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    max_length: int = 256
    # token windows (True = slide windows; False = single window w/ truncation)
    windowing: bool = True
    stride: int = 192                     # used only if windowing=True; <= max_length
    dropout: float = 0.2
    logging_steps: int = 50
    num_workers: int = 2
    report_to_tb: bool = False
    # NEW: cap chunks per original sample (None to disable; try 4)
    max_chunks_per_sample: int | None = None
# ======================================

CONFIG = Config()

def seeder(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def compute_metrics(eval_pred):
    # NOTE: this is chunk-level; we also dump per-sample metrics below
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    _, _, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m, "precision_w": p, "recall_w": r}

def train_args(cfg: Config):
    base_kwargs = dict(
        output_dir=cfg.out,
        overwrite_output_dir=True,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch,
        per_device_eval_batch_size=cfg.batch,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        seed=cfg.seed,
        dataloader_num_workers=cfg.num_workers,
        warmup_ratio=cfg.warmup_ratio,
        save_total_limit=2,
        report_to=(["none"] if not cfg.report_to_tb else ["tensorboard"]),
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    safe_kwargs = {k: v for k, v in base_kwargs.items() if k in allowed}
    return TrainingArguments(**safe_kwargs)

def build_preprocess(cfg: Config, tok: AutoTokenizer):
    """Returns a preprocess(batch) that also tracks sample_idx and (optionally) caps chunks per sample."""
    if cfg.windowing:
        def preprocess(batch):
            enc = tok(
                batch["text"],
                truncation=True,
                max_length=cfg.max_length,
                stride=cfg.stride,
                return_overflowing_tokens=True,
                padding=False
            )
            overflow = enc.pop("overflow_to_sample_mapping")  # list[int], points to original row in this batch
            if cfg.max_chunks_per_sample is not None:
                keep, seen = [], Counter()
                for i, src in enumerate(overflow):
                    if seen[src] < cfg.max_chunks_per_sample:
                        keep.append(i); seen[src] += 1
                for k in list(enc.keys()):
                    enc[k] = [enc[k][i] for i in keep]
                overflow = [overflow[i] for i in keep]

            enc["labels"] = [batch["label"][i] for i in overflow]
            # retain mapping to aggregate per original sample later
            enc["sample_idx"] = overflow
            return enc
    else:
        def preprocess(batch):
            enc = tok(batch["text"], truncation=True, max_length=cfg.max_length)
            enc["labels"] = batch["label"]
            # 1:1 mapping per row in this batch; we represent sample_idx as the row index in this batch
            enc["sample_idx"] = list(range(len(batch["text"])))
            return enc
    return preprocess

def compute_per_example_weights(ds_split):
    """
    For windowed datasets: each chunk gets weight = 1 / (#chunks of its original sample).
    This ensures each original sample contributes equally in expectation.
    Normalized so mean weight ~ 1 to keep loss scale stable.
    """
    sids = np.array(ds_split["sample_idx"])
    uniq, counts = np.unique(sids, return_counts=True)
    sid_to_count = dict(zip(uniq, counts))
    w = np.array([1.0 / sid_to_count[sid] for sid in sids], dtype=np.float32)
    w *= (len(w) / w.sum())
    return w.tolist()

def confusion_per_sample(trainer: Trainer, ds, id2label: dict, outdir: str, tag: str):
    """Aggregate chunk logits per original sample before building confusion matrix."""
    pred = trainer.predict(ds)
    logits = pred.predictions
    y_chunk = pred.label_ids
    sample_idx = np.array(ds["sample_idx"])

    by_idx_logits = defaultdict(list)
    by_idx_label  = {}
    for logit, lab, sid in zip(logits, y_chunk, sample_idx):
        by_idx_logits[int(sid)].append(logit)
        by_idx_label[int(sid)] = int(lab)  # same for all chunks

    y_true, y_pred = [], []
    for sid, ll in by_idx_logits.items():
        m = np.mean(ll, axis=0)
        y_pred.append(int(np.argmax(m)))
        y_true.append(by_idx_label[sid])

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=3
    )

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"confusion_{tag}_per_sample.npy"), cm)
    with open(os.path.join(outdir, f"classification_{tag}_per_sample.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    # also return a compact dict for quick viewing
    # (not strictly necessary, but can be handy if you log elsewhere)
    return {
        "support": dict(Counter(y_true)),
        "macro_f1": precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2],
        "weighted_f1": precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2],
        "acc": accuracy_score(y_true, y_pred)
    }

class PerExampleWeightedTrainer(Trainer):
    """
    Trainer that uses per-example weights (from 'per_ex_weight') so each original sample
    contributes equally when windowing created multiple chunks.
    Do NOT combine with class weights here since the dataset is balanced.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        ex_w = inputs.pop("per_ex_weight", None)  # [B] float tensor if present
        outputs = model(**inputs)
        logits = outputs.logits
        ce = F.cross_entropy(logits, labels, reduction="none")  # per-example CE
        if ex_w is not None:
            ex_w = ex_w.to(logits.device)
            loss = (ce * ex_w).mean()
        else:
            loss = ce.mean()
        return (loss, outputs) if return_outputs else loss

def main(cfg: Config):
    seeder(cfg.seed)

    # 1) Load + encode labels to ClassLabel (needed for stratify)
    raw = load_dataset("json", data_files=cfg.data, split="train")
    raw = raw.class_encode_column("label")

    # 2) Stratified splits
    ds = raw.train_test_split(test_size=cfg.test_ratio, seed=cfg.seed, stratify_by_column="label")
    tmp = ds["train"].train_test_split(test_size=cfg.val_ratio, seed=cfg.seed, stratify_by_column="label")
    dds = DatasetDict(train=tmp["train"], validation=tmp["test"], test=ds["test"])

    # 3) Labels
    label_names = dds["train"].features["label"].names
    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in enumerate(label_names)}

    # 4) Tokenizer & preprocess
    tok = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
    preprocess = build_preprocess(cfg, tok)
    dds = dds.map(preprocess, batched=True, remove_columns=dds["train"].column_names)

    # 5) Per-example weights (so each original sample contributes equally in training)
    train_weights = compute_per_example_weights(dds["train"])
    dds["train"] = dds["train"].add_column("per_ex_weight", train_weights)

    # (optional) You can add to val too; not used in loss but keeps signature uniform
    val_weights = compute_per_example_weights(dds["validation"])
    dds["validation"] = dds["validation"].add_column("per_ex_weight", val_weights)

    collator = DataCollatorWithPadding(tokenizer=tok)

    # 6) Model (fresh) + apply dropout from config
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model, num_labels=len(label_names), id2label=id2label, label2id=label2id
    )
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = cfg.dropout
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = cfg.dropout

    # 7) Training args
    training_args = train_args(cfg)

    # 8) Trainer
    trainer = PerExampleWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dds["train"],
        eval_dataset=dds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # 9) Train
    trainer.train()

    # 10) Evaluate (chunk-level, for quick sanity)
    os.makedirs(cfg.out, exist_ok=True)
    metrics_val = trainer.evaluate(dds["validation"])
    metrics_test = trainer.evaluate(dds["test"])
    with open(os.path.join(cfg.out, "metrics_val_chunk_level.json"), "w") as f:
        json.dump(metrics_val, f, indent=2)
    with open(os.path.join(cfg.out, "metrics_test_chunk_level.json"), "w") as f:
        json.dump(metrics_test, f, indent=2)

    # 11) Evaluate per-sample (aggregated confusion matrix & report)
    summary_val = confusion_per_sample(trainer, dds["validation"], id2label, cfg.out, "val")
    summary_test = confusion_per_sample(trainer, dds["test"], id2label, cfg.out, "test")
    with open(os.path.join(cfg.out, "metrics_val_per_sample.json"), "w") as f:
        json.dump(summary_val, f, indent=2)
    with open(os.path.join(cfg.out, "metrics_test_per_sample.json"), "w") as f:
        json.dump(summary_test, f, indent=2)

    # 12) Save label maps & run config
    with open(os.path.join(cfg.out, "labels.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)
    with open(os.path.join(cfg.out, "run_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("Saved:", cfg.out)
    print("Val (chunk-level):", metrics_val)
    print("Test (chunk-level):", metrics_test)
    print("Val (per-sample):", summary_val)
    print("Test (per-sample):", summary_test)

if __name__ == "__main__":
    main(CONFIG)
