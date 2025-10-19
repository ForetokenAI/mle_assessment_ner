# backend_evaluation_example.py
# ------------------------------------------------------
# This script illustrates how Foretoken evaluates submissions.
# The actual backend uses the same logic but runs on a hidden
# out-of-domain (OOD) test set not shared with candidates.
# ------------------------------------------------------

import json, time, torch, os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import f1_score, classification_report
from dotenv import load_dotenv

# ------------------------------------------------------
# 1. Environment setup
# ------------------------------------------------------
load_dotenv()

SUBMIT_DIR = Path(os.getenv("SUBMIT_DIR"))
TEST_PATH = Path(os.getenv("TEST_PATH"))

print(f"🔍 Loading model from: {SUBMIT_DIR}")
print(f"🧪 Using test file: {TEST_PATH}")

# ------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f]

test_data = load_jsonl(TEST_PATH)
tokens_list = [ex["tokens"] for ex in test_data]
true_tags = [ex["ner_tags"] for ex in test_data]

# ------------------------------------------------------
# 3. Load model + tokenizer
# ------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForTokenClassification.from_pretrained(SUBMIT_DIR).to(device)
tokenizer = AutoTokenizer.from_pretrained(SUBMIT_DIR)
id2label = model.config.id2label

# ------------------------------------------------------
# 4. Inference
# ------------------------------------------------------
pred_tags = []
for tokens_ in tokens_list:
    enc = tokenizer(tokens_, is_split_into_words=True,
                    return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = logits.argmax(-1).cpu().tolist()[0]

    # Align back to words
    word_ids = enc.word_ids(batch_index=0)
    labels, prev = [], None
    for wid, pid in zip(word_ids, preds):
        if wid is None or wid == prev:
            continue
        labels.append(id2label[pid])
        prev = wid
    pred_tags.append(labels)

# ------------------------------------------------------
# 5. Compute metrics
# ------------------------------------------------------
ood_f1 = f1_score(true_tags, pred_tags)
print("\n=== Backend Evaluation ===")
print(classification_report(true_tags, pred_tags))
print(f"OOD Macro F1: {ood_f1:.4f}")

# Latency measurement (optional)
sample = " ".join(tokens_list[0])
inputs = tokenizer(sample, return_tensors="pt", truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
_ = model(**inputs)
if device == "cuda": torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad(): _ = model(**inputs)
if device == "cuda": torch.cuda.synchronize()
latency = round(time.time() - t0, 4)
print(f"Latency (s): {latency:.4f}")

# ------------------------------------------------------
# 6. Save results
# ------------------------------------------------------
results = {"ood_macro_f1": ood_f1, "latency_sec": latency}
Path("evaluation_results.json").write_text(json.dumps(results, indent=2))
print("\nSaved evaluation_results.json")

