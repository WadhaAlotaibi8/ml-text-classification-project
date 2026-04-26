import os
import csv
import random
import sys

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


def load_selected_dataset(dataset_name, num_train, num_test):
    if dataset_name == "AG_NEWS":
        train_file = os.path.join("data", "datasets", "AG_NEWS", "train.csv")
        test_file = os.path.join("data", "datasets", "AG_NEWS", "test.csv")
        train_texts, train_labels = pick_n_sample_from_each_class(train_file, num_train)
        test_texts, test_labels = pick_n_sample_from_each_class(test_file, num_test)

    elif dataset_name == "20News":
        train_ds, test_ds = load_20news()
        train_texts, train_labels = pick_n_sample_from_each_class_given_dataset(train_ds, num_train)
        test_texts, test_labels = pick_n_sample_from_each_class_given_dataset(test_ds, num_test)

    elif dataset_name == "SwahiliNews":
        train_ds, test_ds = load_swahili()
        train_texts, train_labels = pick_n_sample_from_each_class_given_dataset(train_ds, num_train)
        test_texts, test_labels = pick_n_sample_from_each_class_given_dataset(test_ds, num_test)

    elif dataset_name == "KirundiNews":
        train_ds, test_ds = load_kinnews_kirnews(
            dataset_name="kinnews_kirnews",
            data_split="kirnews_cleaned"
        )
        train_texts, train_labels = pick_n_sample_from_each_class_given_dataset(train_ds, num_train)
        test_texts, test_labels = pick_n_sample_from_each_class_given_dataset(test_ds, num_test)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_texts, train_labels, test_texts, test_labels


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


def run_all_baselines(dataset_name="AG_NEWS", num_train=20, num_test=20, random_seed=42):
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

    for model_name, model in models:
        result = evaluate_model(model, model_name, X_train, train_labels, X_test, test_labels)
        results.append(result)

    os.makedirs("../results", exist_ok=True)

    output_file = f"../results/{dataset_name}_baseline_models_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Dataset", "Train/Class", "Test/Class", "Accuracy", "Precision", "Recall", "F1"])
        for result in results:
            writer.writerow([
                result["method"],
                dataset_name,
                num_train,
                num_test,
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1']:.4f}",
            ])

    print(f"\nAll baseline results saved in {output_file}")


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "AG_NEWS"
    run_all_baselines(dataset_name=dataset_name, num_train=20, num_test=20)