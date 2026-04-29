# ML Text Classification Project

> **Based on:** "Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors" (Jiang et al., ACL Findings 2023)

This project reproduces the original gzip + kNN compression-based text classification method and extends it with a broader evaluation framework and traditional baseline comparisons.

---

## What This Project Does

### Original Method
- Computes **Normalized Compression Distance (NCD)** between texts using `gzip`
- Classifies test examples via **k-Nearest Neighbors** (k=2 by default)
- Requires **no training, no parameters, no tokenizer**

### Our Extensions
1. **Richer evaluation metrics** beyond accuracy:
   - Macro-averaged Precision, Recall, F1-score
   - Confusion Matrix
   - Full Classification Report

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
│   ├── requirements.txt
│   └── data/
│       └── datasets/
├── results/                  # Saved evaluation outputs (auto-generated)
├── README.md
└── .gitignore
```

## Project Description:
This project performs text classification using a compression-based KNN approach instead of a neural network.

The workflow is:
- Load the dataset
- Sample train and test examples
- Compare each test text with training texts using a compression-based distance
- Predict the class label using K-nearest neighbors
- Print evaluation results
- Save the results into the results folder
In addition to the original compression-based method, this project also includes traditional machine learning baselines using TF-IDF features with:

-Logistic Regression
-Naive Bayes
-SVM

## Environment:
This project was tested using:
- Python 3.11

It is recommended to use a virtual environment.

## Installation:
1. Clone the repository
cd ml-text-classification-project


2. Create and activate a virtual environment
Windows PowerShell:
py -3.11 -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1


3. Install dependencies
pip install -r code/requirements.txt
``` 

If needed, install these packages manually:
pip install unidecode
pip install "portalocker>=2.0.0"


## How to Run the Code:
Step 1: Open the project folder

Step 2: Activate the virtual environment

Windows PowerShell:
.\.venv\Scripts\Activate.ps1

Step 3: Move to the code folder
cd code

Step 4: Run the code
python main_text.py
```

Default run:
This uses the default settings:
- Dataset: AG_NEWS
- Train samples per class: 100
- Test samples per class: 100
- Compressor: gzip
- K: 2

Example run:
python main_text.py --dataset AG_NEWS --num_train 100 --num_test 100 --compressor gzip --k 2
``` 

Example with another dataset:
python main_text.py --dataset kirnews --num_train 10 --num_test 10

Main Arguments:
- --dataset : dataset name
- --num_train : number of training samples per class
- --num_test : number of testing samples per class
- --compressor : compressor used for similarity calculation
- --k : number of nearest neighbors
- --data_dir : path to the dataset folder
- --para : enables parallel execution



## Extension Added:
Compared to the original repository, this version adds:
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report
- Saving evaluation results to a file in the results folder
- Traditional baseline comparison using:
    -TF-IDF + Logistic Regression
    -TF-IDF + Naive Bayes
    -TF-IDF + SVM
    

## Dataset Notes:
The project structure supports multiple datasets, but some older dataset download links may fail because of outdated sources or compatibility issues.


## Authors:
- Wadhhaa Alotaibi
- Soukaina Alwosaibae
