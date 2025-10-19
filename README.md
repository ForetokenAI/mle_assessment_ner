# Foretoken NER Assessment

> Build a small **Named Entity Recognition (NER)** model that tags product attributes (e.g., BRAND, COLOR, TYPE, SIZE, MATERIAL, GENDER).  
> Train on the provided data; verify locally on the **ID test**. Final scoring uses a **hidden OOD test** (unseen brands/types/colors) on our backend.

---

## 🚀 Quick Start

### Open in Colab (recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1krxadAmN6R6Chpq0su5nXRE3qkFkVeK4?usp=sharing)

1. Click the badge → **File → Save a copy in Drive**  
2. Run the cells in the notebook  
3. When prompted, **upload** the two dataset files you received:  
   - `product_ner_train.jsonl`  
   - `product_ner_test_id.jsonl`  
4. At the end, the notebook creates and downloads **`submission.zip`** automatically  
5. Return to the Foretoken assessment webpage for next steps.

### Optional: Run locally
If you prefer local execution, first download the notebook from Colab (**File → Download .ipynb**) and ensure you have the two dataset files in the same folder. Then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install transformers datasets seqeval accelerate torch --extra-index-url https://download.pytorch.org/whl/cu121
````

Open the notebook in Jupyter/VSCode and run. You’ll need to adapt the data-loading cell to read the two JSONL files from disk.

---

## 🎯 Your Task

1. Implement tokenization + label alignment (`is_split_into_words=True`; use `-100` for subword continuations and special tokens).
2. Choose any open-source Hugging Face encoder (`bert-base-cased`, `roberta-base`, `deberta-v3-base`, etc.).
3. Train within the suggested **training budget** below.
4. Evaluate on the visible **ID test** for sanity check.
5. Return to the Foretoken assessment webpage for next steps.

---

## 🧠 Training Budget (Guideline)

* **Epochs:** ≤ 4
* **Batch size:** ≤ 16
* **Sequence length:** typical ≤ 128
* **Compute:** Any GPU or CPU is fine (Colab or local)

> We measure *generalization and efficiency*, not raw compute time.
> Latency scores are normalized by hardware during scoring.

---

## 🧾 Evaluation Metrics (Backend)

| Metric                   | Purpose                             | Source          |
| ------------------------ | ----------------------------------- | --------------- |
| **OOD Macro F1**         | Primary score (unseen products)     | Hidden OOD test |
| **ID Macro F1**          | Sanity check (pipeline correctness) | Visible ID test |
| **Latency (normalized)** | Efficiency measure                  | Backend         |

Final score (example weighting):

```
final_score = 100 * (0.7 * OOD_F1 + 0.2 * ID_F1 + 0.1 * latency_score)
```

---

## 📦 What’s inside `submission.zip`

```
submission/
├─ model/
│  ├─ config.json
│  ├─ pytorch_model.bin
│  ├─ tokenizer.json
│  ├─ tokenizer_config.json
│  └─ special_tokens_map.json   (if applicable)
└─ results.json                 # {"macro_f1_id": <float>, "notes": "...optional..."}
```

> The OOD test set is **not** provided. It is used only for final scoring on our backend.

---

## ⚙️ Notes & Rules

* Use only the provided dataset and standard open-source libraries.
* You’re free to choose your model architecture or light augmentations.
* No paid APIs or external training data.
* You can train on Colab, local GPU, or CPU — all valid.
* Hardware differences are normalized during scoring.

---

## ⚠️ Common Pitfalls

| Issue                       | Explanation                                                          | Fix                                                                                                    |
| --------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Token alignment errors**  | Subwords not mapped to correct labels                                | Use `is_split_into_words=True` and assign `-100` to ignored positions                                  |
| **RoBERTa tokenizer crash** | Assertion: `add_prefix_space=True` required for pre-tokenized inputs | For RoBERTa tokenizers, use:<br>`AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)` |
| **No GPU detected**         | Colab CPU-only runtime                                               | *Runtime → Change runtime type → GPU*                                                                  |
| **Overtraining**            | Too many epochs inflate ID F1 but hurt OOD F1                        | Keep ≤ 4 epochs or use early stopping                                                                  |
