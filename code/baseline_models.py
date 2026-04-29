import os
import csv
import random
import sys

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from data import (
    load_20news,
    load_swahili,
    load_kinnews_kirnews,
    pick_n_sample_from_each_class_given_dataset,
    pick_n_sample_from_each_class,
)
import random
import numpy as np

random.seed(42)
np.random.seed(42)

def normalize_dataset_name(dataset_name):
    name_map = {
        "AG_NEWS": "AG_NEWS",
        "ag_news": "AG_NEWS",
        "agnews": "AG_NEWS",

        "20News": "20News",
        "20news": "20News",

        "kirnews": "kirnews",
        "KirundiNews": "kirnews",
        "kirundi": "kirnews",

        "swahili": "swahili",
        "SwahiliNews": "swahili",
        "swahilinews": "swahili",
    }

    if dataset_name not in name_map:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    return name_map[dataset_name]


def load_selected_dataset(dataset_name, num_train, num_test):
    dataset_name = normalize_dataset_name(dataset_name)

    if dataset_name == "AG_NEWS":
        train_file = os.path.join("data", "datasets", "AG_NEWS", "train.csv")
        test_file = os.path.join("data", "datasets", "AG_NEWS", "test.csv")

        train_texts, train_labels = pick_n_sample_from_each_class(train_file, num_train)
        test_texts, test_labels = pick_n_sample_from_each_class(test_file, num_test)

    elif dataset_name == "20News":
        train_ds, test_ds = load_20news()

        train_texts, train_labels = pick_n_sample_from_each_class_given_dataset(train_ds, num_train)
        test_texts, test_labels = pick_n_sample_from_each_class_given_dataset(test_ds, num_test)

    elif dataset_name == "kirnews":
        train_ds, test_ds = load_kinnews_kirnews(
            dataset_name="kinnews_kirnews",
            data_split="kirnews_cleaned"
        )

        train_texts, train_labels = pick_n_sample_from_each_class_given_dataset(train_ds, num_train)
        test_texts, test_labels = pick_n_sample_from_each_class_given_dataset(test_ds, num_test)

    elif dataset_name == "swahili":
        train_ds, test_ds = load_swahili()

        train_texts, train_labels = pick_n_sample_from_each_class_given_dataset(train_ds, num_train)
        test_texts, test_labels = pick_n_sample_from_each_class_given_dataset(test_ds, num_test)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_texts, train_labels, test_texts, test_labels


def get_compressor_result(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)

    compressor_results = {
        "AG_NEWS": {
            "method": "gzip + kNN",
            "accuracy": 0.6500,
            "precision": 0.6572,
            "recall": 0.6500,
            "f1": 0.6502,
        },
        "kirnews": {
            "method": "gzip + kNN",
            "accuracy": 0.8080,
            "precision": 0.8256,
            "recall": 0.8613,
            "f1": 0.8318,
        },
        "swahili": {
            "method": "gzip + kNN",
            "accuracy": 0.8500,
            "precision": 0.8524,
            "recall": 0.8500,
            "f1": 0.8487,
        },
        "20News": {
            "method": "gzip + kNN",
            "accuracy": 0.5300,
            "precision": 0.5357,
            "recall": 0.5300,
            "f1": 0.5302,
        },
    }

    return compressor_results.get(dataset_name, None)


def evaluate_model(model, model_name, X_train, train_labels, X_test, test_labels):
    model.fit(X_train, train_labels)
    predictions = model.predict(X_test)

    acc = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="macro", zero_division=0)
    recall = recall_score(test_labels, predictions, average="macro", zero_division=0)
    f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)

    cm = confusion_matrix(test_labels, predictions)
    report = classification_report(test_labels, predictions, zero_division=0)

    print(f"\nBaseline: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return {
        "method": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_full_comparison_plot(results, dataset_name):
    methods = [result["method"] for result in results]
    accuracy = [result["accuracy"] for result in results]
    precision = [result["precision"] for result in results]
    recall = [result["recall"] for result in results]
    f1 = [result["f1"] for result in results]

    x = range(len(methods))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar([i - 1.5 * width for i in x], accuracy, width=width, label="Accuracy")
    plt.bar([i - 0.5 * width for i in x], precision, width=width, label="Precision")
    plt.bar([i + 0.5 * width for i in x], recall, width=width, label="Recall")
    plt.bar([i + 1.5 * width for i in x], f1, width=width, label="F1-score")

    plt.xticks(list(x), methods, rotation=15)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(f"Method Comparison on {dataset_name}")
    plt.legend()
    plt.tight_layout()

    os.makedirs("../results", exist_ok=True)
    plt.savefig(f"../results/{dataset_name}_full_method_comparison.png")
    plt.close()


def run_all_baselines(dataset_name="AG_NEWS", num_train=20, num_test=20, random_seed=42):
    dataset_name = normalize_dataset_name(dataset_name)
    random.seed(random_seed)

    train_texts, train_labels, test_texts, test_labels = load_selected_dataset(
        dataset_name, num_train, num_test
    )

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    models = [
        ("TF-IDF + Logistic Regression", LogisticRegression(max_iter=1000, random_state=random_seed)),
        ("TF-IDF + Naive Bayes", MultinomialNB()),
        ("TF-IDF + SVM", LinearSVC(random_state=random_seed, dual="auto")),
    ]

    results = []

    compressor_result = get_compressor_result(dataset_name)
    if compressor_result is not None:
        results.append(compressor_result)
        print("\nCompression-Based Method:")
        print(f"Accuracy: {compressor_result['accuracy']:.4f}")
        print(f"Precision: {compressor_result['precision']:.4f}")
        print(f"Recall: {compressor_result['recall']:.4f}")
        print(f"F1-score: {compressor_result['f1']:.4f}")

    for model_name, model in models:
        result = evaluate_model(model, model_name, X_train, train_labels, X_test, test_labels)
        results.append(result)

    output_file = f"../results/{dataset_name}_all_methods_results.csv"
    os.makedirs("../results", exist_ok=True)

    header = ["Method", "Dataset", "Train/Class", "Test/Class", "Accuracy", "Precision", "Recall", "F1"]

    print("\n" + ",".join(header))

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for result in results:
            row = [
                result["method"],
                dataset_name,
                str(num_train),
                str(num_test),
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1']:.4f}",
            ]
            writer.writerow(row)
            print(",".join(row))

    save_full_comparison_plot(results, dataset_name)

    print(f"\nAll results saved in {output_file}")
    print(f"Comparison plot saved in ../results/{dataset_name}_full_method_comparison.png")


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "AG_NEWS"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    run_all_baselines(dataset_name=dataset_name, num_train=sample_size, num_test=sample_size)