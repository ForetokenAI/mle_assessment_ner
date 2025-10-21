# backend_evaluation_example.py
# ------------------------------------------------------
# This script illustrates how Foretoken evaluates submissions.
# The actual backend uses the same logic but runs on a hidden
# out-of-domain (OOD) test set not shared here.
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
TEST_PATH_OOD = Path(os.getenv("TEST_PATH_OOD"))
TEST_PATH_ID = Path(os.getenv("TEST_PATH_ID"))

print(f"🔍 Loading model from: {SUBMIT_DIR}")
print(f"🧪 Using test files:\n - OOD: {TEST_PATH_OOD}\n - ID:  {TEST_PATH_ID}")

# ------------------------------------------------------
# Helper: load dataset
# ------------------------------------------------------
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f]

def evaluate_split(test_path, split_name, model, tokenizer, device):
    test_data = load_jsonl(test_path)
    tokens_list = [ex["tokens"] for ex in test_data]
    true_tags = [ex["ner_tags"] for ex in test_data]

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
            labels.append(model.config.id2label[pid])
            prev = wid
        pred_tags.append(labels)

    f1 = f1_score(true_tags, pred_tags)
    print(f"\n=== {split_name.upper()} Evaluation ===")
    print(classification_report(true_tags, pred_tags))
    print(f"{split_name.upper()} Macro F1: {f1:.4f}")
    return f1

# ------------------------------------------------------
# 2. Load model + tokenizer
# ------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = AutoModelForTokenClassification.from_pretrained(
        SUBMIT_DIR, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
except Exception:
    model = AutoModelForTokenClassification.from_pretrained(SUBMIT_DIR).to(device)

tokenizer = AutoTokenizer.from_pretrained(SUBMIT_DIR)

# ------------------------------------------------------
# 3. Evaluate both splits
# ------------------------------------------------------
id_f1 = evaluate_split(TEST_PATH_ID, "id", model, tokenizer, device)
ood_f1 = evaluate_split(TEST_PATH_OOD, "ood", model, tokenizer, device)

# ------------------------------------------------------
# 4. Latency (optional)
# ------------------------------------------------------
sample = "Nike shoes"
inputs = tokenizer(sample, return_tensors="pt", truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
_ = model(**inputs)
if device == "cuda": torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    _ = model(**inputs)
if device == "cuda": torch.cuda.synchronize()
latency = round(time.time() - t0, 4)
print(f"\nLatency (s): {latency:.4f}")

# ------------------------------------------------------
# 5. Save results
# ------------------------------------------------------
results = {
    "id_macro_f1": id_f1,
    "ood_macro_f1": ood_f1,
    "latency_sec": latency
}
Path("evaluation_results.json").write_text(json.dumps(results, indent=2))
print("\n✅ Saved evaluation_results.json")