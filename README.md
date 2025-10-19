# Foretoken NER Assessment

> Build a small **Named Entity Recognition (NER)** model that tags product attributes (e.g., BRAND, COLOR, TYPE, SIZE, MATERIAL, GENDER).  
> Train on the provided data; verify locally on the **ID test**. Final scoring uses a **hidden OOD test** (unseen brands/types/colors) on our backend.

---

## 🚀 Quick Start

### Option A — Open in Colab (recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_PUBLIC_NOTEBOOK_LINK)

1. Click the badge → **File → Save a copy in Drive**  
2. Run the cells in `starter_notebook.ipynb`  
3. At the end, download the `submission/` folder and upload it along with your notebook

### Option B — Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install transformers datasets seqeval accelerate torch --extra-index-url https://download.pytorch.org/whl/cu121
```

Then open `starter_notebook.ipynb` in Jupyter or VSCode and run.

---

## 📁 Repo Contents

```
foretoken_ner_assessment/
│
├─ README.md
├─ starter_notebook.ipynb          # minimal scaffold with TODOs
│
├─ data/
│  ├─ product_ner_train_clean.jsonl    # training data
│  └─ product_ner_test_id_clean.jsonl  # visible test set (ID)
│
├─ sample_submission/
│  ├─ model/                           # example structure only (empty)
│  └─ results.json                     # example schema only
│
└─ utils/
   └─ evaluation.py                    # optional helper (not required)
```

> The OOD test set is **not** provided. It is used only for final scoring.

---

## 🎯 Your Task

1. Implement tokenization + label alignment (`is_split_into_words=True`; use `-100` for subword continuations and special tokens).
2. Choose any open-source Hugging Face encoder (`bert-base-cased`, `roberta-base`, `deberta-v3-base`, etc.).
3. Train within the suggested **training budget** below.
4. Evaluate on the visible **ID test** for sanity check.
5. Save artifacts under `./submission/` and upload:

   * `submission/model/` (from `save_pretrained`)
   * `submission/results.json`
   * your notebook (`.ipynb`)

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

## 📦 Submission Structure

### 1. Model directory

```
submission/model/
├─ config.json
├─ pytorch_model.bin
├─ tokenizer.json
├─ tokenizer_config.json
└─ special_tokens_map.json
```

### 2. Results JSON

```json
{
  "macro_f1_id": 0.9312,
  "notes": "optional: anything notable about your approach"
}
```

### 3. Notebook

Your executed `.ipynb` (used to produce the model).

---

## ⚙️ Notes & Rules

* Use only the provided dataset and standard open-source libraries.
* You’re free to choose your model architecture or light augmentations.
* No paid APIs or external training data.
* You can train on Colab, local GPU, or CPU — all valid.
* Hardware differences are normalized during scoring.

---

## ⚠️ Common Pitfalls

| Issue                       | Explanation                                                          | Fix                                                                                                                     |
| --------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Token alignment errors**  | Subwords not mapped to correct labels                                | Use `is_split_into_words=True` and assign `-100` to ignored positions                                                   |
| **RoBERTa tokenizer crash** | Assertion: `add_prefix_space=True` required for pre-tokenized inputs | When using RoBERTa, instantiate tokenizer as:<br>`AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)` |
| **No GPU detected**         | Colab CPU-only runtime                                               | Go to *Runtime → Change runtime type → GPU*                                                                             |
| **Overtraining**            | Too many epochs inflate ID F1 but hurt OOD F1                        | Stick to ≤4 epochs or apply early stopping                                                                              |

---

## 📬 Contact

If something is broken (e.g., dataset path or runtime error), reply to your invite email with a short description and a screenshot/log snippet.
