# ML Text Classification Project

This project is based on the paper **“Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors”**.

The goal of this project is to reproduce the original compression-based text classification method and extend it by adding more evaluation outputs, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report
- Saving experiment results into the `results` folder
- Comparison with traditional baselines:
  - TF-IDF + Logistic Regression
  - TF-IDF + Naive Bayes
  - TF-IDF + SVM

---

## Project Structure

```text
ml-text-classification-project/
├── code/
│   ├── compressors.py
│   ├── data.py
│   ├── experiments.py
│   ├── main_text.py
│   ├── baseline_models.py
│   ├── requirements.txt
│   ├── utils.py
│   └── data/
│       └── datasets/
├── results/
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
.\.venv\Scripts\Activate.ps1


3. Install dependencies
pip install -r code/requirements.txt

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

Default run:
This uses the default settings:
- Dataset: AG_NEWS
- Train samples per class: 100
- Test samples per class: 100
- Compressor: gzip
- K: 2

Example run:
python main_text.py --dataset AG_NEWS --num_train 100 --num_test 100 --compressor gzip --k 2

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
- Soukiana Alwosaibae

## Reference:
This project is based on:
Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors

