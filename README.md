# ML Text Classification Project

> **Based on:** "Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors" (Jiang et al., ACL Findings 2023)

This project reproduces the original gzip + kNN compression-based text classification method and extends it with a broader evaluation framework and traditional baseline comparisons.

---

## What This Project Does

### Original Method (Reproduced)
- Computes **Normalized Compression Distance (NCD)** between texts using `gzip`
- Classifies test examples via **k-Nearest Neighbors** (k=2 by default)
- Requires **no training, no parameters, no tokenizer**

### Our Extensions
1. **Richer evaluation metrics** beyond accuracy:
   - Macro-averaged Precision, Recall, F1-score
   - Confusion Matrix
   - Full Classification Report
   - Results saved automatically to `results/`

2. **Traditional baseline comparison** under identical train/test splits:
   - TF-IDF + Logistic Regression
   - TF-IDF + Naive Bayes
   - TF-IDF + SVM

---

## Project Structure

```
ml-text-classification-project/
├── code/
│   ├── main_text.py          # Entry point for gzip+kNN method
│   ├── baseline_models.py    # TF-IDF + LR / NB / SVM baselines
│   ├── compressors.py        # NCD computation logic
│   ├── data.py               # Dataset loading utilities
│   ├── experiments.py        # Core experiment runner
│   ├── utils.py              # Shared helpers
│   └── requirements.txt
├── results/                  # Saved evaluation outputs (auto-generated)
├── README.md
└── .gitignore
```

---

## Datasets

The following datasets were evaluated in this project:

| Dataset | Type | Status |
|---|---|---|
| AG News | In-distribution | ✅ Tested |
| 20News | In-distribution | ✅ Tested |
| KirundiNews | Out-of-distribution (low-resource) | ✅ Tested |
| SwahiliNews | Out-of-distribution (low-resource) | ✅ Tested |
| DBpedia, YahooAnswers, Ohsumed, R8, R52, SogouNews, DengueFilipino | Various | ⚠️ Loader compatibility issues |

> Some datasets from the original paper could not be loaded due to outdated download links or Hugging Face caching issues.

---

## Results Summary

All experiments use a balanced per-class sampling strategy. Metrics are macro-averaged.

### AG News (100 train / 100 test per class)

| Method | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **gzip + kNN** | **0.6500** | **0.6572** | **0.6500** | **0.6502** |
| TF-IDF + Logistic Regression | 0.6375 | 0.6351 | 0.6375 | 0.6327 |
| TF-IDF + Naive Bayes | 0.6000 | 0.5936 | 0.6000 | 0.5918 |
| TF-IDF + SVM | 0.6375 | 0.6358 | 0.6375 | 0.6347 |

### 20News (20 train / 20 test per class)

| Method | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| gzip + kNN | — | — | — | — |
| TF-IDF + Logistic Regression | 0.5975 | 0.6284 | 0.5975 | 0.5952 |
| TF-IDF + Naive Bayes | 0.4950 | 0.6715 | 0.4950 | 0.5004 |
| **TF-IDF + SVM** | **0.6250** | **0.6323** | **0.6250** | **0.6175** |

> KirundiNews and SwahiliNews results are saved in `results/`. Sample sizes were reduced to match the minimum available class size.

---

## Setup & Installation

**Requirements:** Python 3.11

```bash
# 1. Clone the repository
git clone https://github.com/WadhaAlotaibi8/ml-text-classification-project.git
cd ml-text-classification-project

# 2. Create and activate a virtual environment
py -3.11 -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r code/requirements.txt

# If needed, install manually:
pip install unidecode "portalocker>=2.0.0"
```

---

## How to Run

### Compression-based method (gzip + kNN)

```bash
cd code

# Default: AG_NEWS, 100 train/test per class, gzip, k=2
python main_text.py

# Custom dataset and sample size
python main_text.py --dataset AG_NEWS --num_train 100 --num_test 100 --compressor gzip --k 2

# Other datasets
python main_text.py --dataset 20News
python main_text.py --dataset kirnews --num_train 10 --num_test 10
python main_text.py --dataset swahili
```

**Arguments:**
| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset name | `AG_NEWS` |
| `--num_train` | Training samples per class | `100` |
| `--num_test` | Test samples per class | `100` |
| `--compressor` | Compressor to use (`gzip`, `zlib`, `bz2`) | `gzip` |
| `--k` | Number of nearest neighbors | `2` |
| `--para` | Enable parallel execution | off |

### Traditional baselines (TF-IDF + LR / NB / SVM)

```bash
cd code

python baseline_models.py AG_NEWS
python baseline_models.py 20News
python baseline_models.py kirnews
python baseline_models.py swahili
```

Results are saved automatically to `results/`.

---

## Key Findings

- On **AG News**, gzip + kNN outperforms all three traditional baselines, showing compression-based similarity captures topical signal effectively in small-sample settings without any feature engineering.
- On **20News** (20 categories), SVM is the strongest baseline, benefiting from TF-IDF's high-dimensional sparse representation and margin-based boundaries over many classes.
- On **low-resource languages** (KirundiNews, SwahiliNews), gzip + kNN is language-agnostic — it needs no tokenizer or vocabulary — which is an advantage over TF-IDF-based methods.
- Extended evaluation metrics (precision, recall, F1) reveal per-class performance differences that accuracy alone conceals.

---

## Reference

```bibtex
@inproceedings{jiang-etal-2023-low,
    title = {"Low-Resource" Text Classification: A Parameter-Free Classification Method with Compressors},
    author = {Jiang, Zhiying and Yang, Matthew and Tsirlin, Mikhail and Tang, Raphael and Dai, Yiqin and Lin, Jimmy},
    booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
    year = {2023},
    pages = {6810--6828}
}
```

Original paper: https://aclanthology.org/2023.findings-acl.426/  
Original code: https://github.com/bazingagin/npc_gzip

---

## Authors

- Wadhhaa Alotaibi
- Soukaina Alwosaibae
