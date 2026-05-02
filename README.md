# ML Text Classification Project
 
> **Reproducing and extending:** "Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors" — Jiang et al., ACL Findings 2023  
> **Original paper:** https://aclanthology.org/2023.findings-acl.426/  
> **Original code:** https://github.com/bazingagin/npc_gzip
 
---
 
## What This Project Does
 
### Core method (reproduced)
Classifies text using **Normalized Compression Distance (NCD)** + **k-Nearest Neighbors** — no training, no vocabulary, no tokenizer.
 
```
NCD(x, y) = ( C(xy) − min(C(x), C(y)) ) / max(C(x), C(y))
```
 
A lower NCD means the two texts are more similar. After computing NCD between a test sample and all training samples, the method predicts via majority vote among the k nearest neighbours (default k=2).
 
### Our three extensions
 
This project extends the original work in the following ways:

- Added precision, recall, and F1-score in addition to accuracy.
- Added confusion matrices and classification reports.
- Added TF-IDF baseline models:
   1. Logistic Regression
   2. Naive Bayes
   3. SVM
- Compared three compressors:
   1. gzip
   2. bz2
   3. lzma
- Fixed a kNN tie-breaking issue so each test sample receives one final predicted label.
- Saved results automatically in the results/ folder.
---
 
## Summary Results
 
## Summary

| Dataset | gzip+kNN F1 | Best Baseline F1 | Best Method |
|---------|-------------|------------------|-------------|
| AG News | 0.6502 | 0.6604 | SVM |
| 20News | 0.5302 | 0.5852 | SVM |
| KirundiNews | 0.8318 | 0.5278 | gzip+kNN |
| SwahiliNews | 0.8487 | 0.7976 | gzip+kNN |


 
> **Key finding:** gzip+kNN wins decisively on low-resource non-English datasets. On English datasets, TF-IDF+SVM is slightly stronger. Choose based on language and available labelled data.
 
---
 
### Compressor Comparison — AG News (20 train / 20 test per class, same split)
 
| Compressor | Accuracy | Precision | Recall | F1-score | Time (s) |
|------------|----------|-----------|--------|----------|----------|
| **gzip** | **0.7000** | **0.7009** | **0.7000** | **0.6935** | **0.82** |
| bz2 | 0.5625 | 0.5722 | 0.5625 | 0.5603 | 1.56 |
| lzma | 0.5750 | 0.6105 | 0.5750 | 0.5757 | 38.44 |
 
**gzip wins on every metric and is 47× faster than lzma.**
 
The 0.133 F1 gap between gzip and bz2 is larger than the entire gap between gzip+kNN and SVM on AG News (0.010) — the compressor is a more consequential design choice than the classification method.
 
---
 
## Project Structure
 
```
ml-text-classification-project/
├── code/
│   ├── main_text.py          # gzip+kNN entry point (bug-fixed, seed, CSV save)
│   ├── baseline_models.py    # TF-IDF baselines + comparison plot
│   ├── experiments.py        # KnnExpText class (3 bug fixes applied)
│   ├── compressors.py        # NCD compression (get_compressed_len_fast added)
│   ├── data.py               # Dataset loading utilities
│   ├── utils.py              # NCD, CLM, CDM distance functions + helpers
│   ├── requirements.txt      # Pinned dependencies (Python 3.11)
│   └── run_all.sh            # One command to reproduce all results
├── results/                  # Auto-generated outputs (CSV, PNG)
└── README.md
```
 
---
 
## Setup
 
**Requires Python 3.11**
 
```bash
git clone https://github.com/WadhaAlotaibi8/ml-text-classification-project.git
cd ml-text-classification-project
 
python3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
 
pip install -r code/requirements.txt
```
 
---
 
## How to Run
  
### Individual runs — gzip + kNN
```bash
cd code
 
python main_text.py --dataset AG_NEWS --num_train 20 --num_test 20 --seed 42
python main_text.py --dataset 20News  --num_train 20 --num_test 100 --seed 42
python main_text.py --dataset kirnews --seed 42
python main_text.py --dataset swahili --num_train 20 --num_test 20 --seed 42
 
# Compressor comparison
python main_text.py --dataset AG_NEWS --compressor bz2  --num_train 20 --num_test 20
python main_text.py --dataset AG_NEWS --compressor lzma --num_train 20 --num_test 20
```
 
### Baselines
```bash
cd code
python baseline_models.py AG_NEWS 20
python baseline_models.py 20News  20
python baseline_models.py kirnews 3
python baseline_models.py swahili 20
```
 
### CLI Arguments
 
| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | AG_NEWS, 20News, kirnews, swahili | `AG_NEWS` |
| `--num_train` | Training samples per class | `100` |
| `--num_test` | Test samples per class | `100` |
| `--compressor` | `gzip`, `bz2`, `lzma`, `zlib` | `gzip` |
| `--k` | Number of nearest neighbours | `2` |
| `--seed` | Random seed | `42` |
| `--para` | Parallel mode (pathos) | off |
| `--results_dir` | Output directory | `../results` |
 
 
## Reference
 
```bibtex
@inproceedings{jiang-etal-2023-low,
  title     = {"Low-Resource" Text Classification: A Parameter-Free Classification Method with Compressors},
  author    = {Jiang, Zhiying and Yang, Matthew and Tsirlin, Mikhail and Tang, Raphael and Dai, Yiqin and Lin, Jimmy},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
  year      = {2023},
  pages     = {6810--6828}
}
```
 
## Authors
- Wadhhaa Alotaibi  
- Soukaina Alwosaibae