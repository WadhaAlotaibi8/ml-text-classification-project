# main_text.py — Entry point for compression-based kNN text classification experiments.
#
# CHANGES FROM ORIGINAL:
#   1. Replaced wildcard imports (from x import *) with explicit imports.
#   2. Results are now saved to timestamped CSV files in ../results/.
#   3. Added dataset-loading error handling with clear messages.
#   4. Added timestamps to plot filenames so plots are not overwritten.
#   5. Replaced eval(args.dataset) with explicit dataset loader mapping.

import argparse
import os
import time
from datetime import datetime
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from compressors import DefaultCompressor
from data import (
    load_20news,
    load_custom_dataset,
    load_filipino,
    load_kinnews_kirnews,
    load_ohsumed,
    load_ohsumed_single,
    load_r8,
    load_swahili,
    pick_n_sample_from_each_class_given_dataset,
    read_torch_text_labels,
)
from experiments import KnnExpText
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torchtext.datasets import (
    AG_NEWS,
    IMDB,
    AmazonReviewPolarity,
    DBpedia,
    SogouNews,
    YahooAnswers,
    YelpReviewPolarity,
)
from utils import NCD, agg_by_concat_space


def plot_results(
    test_label,
    predictions,
    dataset_name,
    compressor_name,
    output_dir="../results",
    timestamp=None,
):
    """Saves confusion matrix and per-class metric bar chart to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cm = confusion_matrix(test_label, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix — {dataset_name} ({compressor_name})")
    plt.savefig(
        os.path.join(
            output_dir,
            f"{dataset_name}_{compressor_name}_{timestamp}_confusion_matrix.png",
        )
    )
    plt.close()

    report = classification_report(test_label, predictions, output_dict=True, zero_division=0)
    class_labels, precision_vals, recall_vals, f1_vals = [], [], [], []

    for key in report:
        if str(key).isdigit() or (isinstance(key, str) and key.replace("-", "").isdigit()):
            class_labels.append(str(key))
            precision_vals.append(report[key]["precision"])
            recall_vals.append(report[key]["recall"])
            f1_vals.append(report[key]["f1-score"])

    if class_labels:
        x = range(len(class_labels))
        width = 0.25
        plt.figure(figsize=(10, 6))
        plt.bar([i - width for i in x], precision_vals, width=width, label="Precision")
        plt.bar(list(x), recall_vals, width=width, label="Recall")
        plt.bar([i + width for i in x], f1_vals, width=width, label="F1-score")
        plt.xticks(list(x), class_labels)
        plt.ylim(0, 1)
        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.title(f"Per-Class Metrics — {dataset_name} ({compressor_name})")
        plt.legend()
        plt.savefig(
            os.path.join(
                output_dir,
                f"{dataset_name}_{compressor_name}_{timestamp}_class_metrics.png",
            )
        )
        plt.close()


def run_knn_experiment(
    compressor_name: str,
    test_data: list,
    test_label: list,
    train_data: list,
    train_label: list,
    agg_func: Callable,
    dis_func: Callable,
    k: int,
    para: bool = False,
) -> dict:
    """
    Runs one compression-based kNN experiment and returns a metrics dict.
    """
    cp = DefaultCompressor(compressor_name)
    knn_exp_ins = KnnExpText(agg_func, cp, dis_func)

    if para:
        with Pool(5) as p:
            pred_correct_pair = p.map(
                partial(knn_exp_ins.combine_dis_acc_single, k, train_data, train_label),
                test_data,
                test_label,
            )
    else:
        pred_correct_pair = [
            knn_exp_ins.combine_dis_acc_single(k, train_data, train_label, x, y)
            for x, y in zip(test_data, test_label)
        ]

    predictions = [pair[0] for pair in pred_correct_pair]

    acc = accuracy_score(test_label, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_label, predictions, average="macro", zero_division=0
    )

    return {
        "predictions": predictions,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def non_neural_knn_exp(
    compressor_name: str,
    test_data: list,
    test_label: list,
    train_data: list,
    train_label: list,
    agg_func: Callable,
    dis_func: Callable,
    k: int,
    dataset_name: str,
    para: bool = False,
    output_dir: str = "../results",
):
    """
    Main experiment: runs compression-based kNN, prints metrics,
    saves results to CSV and plots.
    """
    print(f"\nKNN with compressor={compressor_name}, k={k}")
    start = time.time()

    metrics = run_knn_experiment(
        compressor_name, test_data, test_label,
        train_data, train_label, agg_func, dis_func, k, para
    )

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_label, metrics["predictions"]))
    print("\nClassification Report:")
    print(classification_report(test_label, metrics["predictions"], zero_division=0))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_results(
        test_label,
        metrics["predictions"],
        dataset_name,
        compressor_name,
        output_dir,
        timestamp=ts,
    )

    _save_results_csv(
        metrics,
        dataset_name,
        compressor_name,
        k,
        len(train_label),
        len(test_label),
        output_dir,
        timestamp=ts,
    )

    print(f"Results saved to {output_dir}/")
    print(f"Time: {time.time() - start:.1f}s")


def _save_results_csv(metrics, dataset_name, compressor_name, k,
                      n_train, n_test, output_dir, timestamp=None):
    """Saves a single-row results CSV with a timestamp so runs are never overwritten."""
    os.makedirs(output_dir, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_path = os.path.join(
        output_dir,
        f"{dataset_name}_{compressor_name}_k{k}_{timestamp}.csv",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("method,dataset,compressor,k,n_train,n_test,accuracy,precision,recall,f1\n")
        f.write(
            f"gzip+kNN,{dataset_name},{compressor_name},{k},{n_train},{n_test},"
            f"{metrics['accuracy']:.4f},{metrics['precision']:.4f},"
            f"{metrics['recall']:.4f},{metrics['f1']:.4f}\n"
        )
    print(f"Numeric results saved to {out_path}")


def record_distance(
    compressor_name, test_data, test_portion_name, train_data,
    agg_func, dis_func, out_dir, para=True,
):
    """Records full distance matrix to a .npy file for later reuse."""
    print(f"compressor={compressor_name}")
    numpy_dir = os.path.join(out_dir, compressor_name)
    os.makedirs(numpy_dir, exist_ok=True)
    out_fn = os.path.join(numpy_dir, test_portion_name)
    cp = DefaultCompressor(compressor_name)
    knn_exp = KnnExpText(agg_func, cp, dis_func)
    start = time.time()

    if para:
        with Pool(6) as p:
            distance_for_selected_test = p.map(
                partial(knn_exp.calc_dis_single_multi, train_data), test_data
            )
        np.save(out_fn, np.array(distance_for_selected_test))
    else:
        knn_exp.calc_dis(test_data, train_data=train_data)
        np.save(out_fn, np.array(knn_exp.distance_matrix))

    print(f"spent: {time.time() - start:.1f}s")


def non_neural_knn_exp_given_dis(dis_matrix, k, test_label, train_label, rand=False):
    knn_exp = KnnExpText(None, None, None)
    _, correct = knn_exp.calc_acc(
        k, test_label, train_label=train_label,
        provided_distance_matrix=dis_matrix, rand=rand,
    )
    return correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/datasets")
    parser.add_argument("--dataset", default="AG_NEWS")
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--num_train", type=int, default=100)
    parser.add_argument("--compressor", default="gzip")
    parser.add_argument("--all_test", action="store_true", default=False)
    parser.add_argument("--all_train", action="store_true", default=False)
    parser.add_argument("--para", action="store_true", default=False)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--output_dir", default="text_exp_output")
    parser.add_argument("--results_dir", default="../results")
    parser.add_argument("--test_idx_fn", default=None)
    parser.add_argument("--test_idx_start", type=int, default=None)
    parser.add_argument("--test_idx_end", type=int, default=None)
    parser.add_argument("--distance_fn", default=None)
    parser.add_argument("--score", action="store_true", default=False)
    parser.add_argument("--k", default=2, type=int)
    parser.add_argument("--class_num", default=5, type=int)
    parser.add_argument("--random", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    train_idx_fn = os.path.join(
        args.output_dir,
        f"{args.dataset}_train_indicies_{args.num_train}per_class",
    )
    test_idx_fn = os.path.join(
        args.output_dir,
        f"{args.dataset}_test_indicies_{args.num_test}per_class",
    )

    torchtext_loaders = {
        "AG_NEWS": AG_NEWS,
        "IMDB": IMDB,
        "AmazonReviewPolarity": AmazonReviewPolarity,
        "DBpedia": DBpedia,
        "SogouNews": SogouNews,
        "YahooAnswers": YahooAnswers,
        "YelpReviewPolarity": YelpReviewPolarity,
    }

    custom_loaders = {
        "20News": lambda: load_20news(),
        "Ohsumed": lambda: load_ohsumed(args.data_dir),
        "Ohsumed_single": lambda: load_ohsumed_single(args.data_dir),
        "R8": lambda: load_r8(args.data_dir),
        "R52": lambda: load_r8(args.data_dir),
        "kinnews": lambda: load_kinnews_kirnews("kinnews_kirnews", "kinnews_cleaned"),
        "kirnews": lambda: load_kinnews_kirnews("kinnews_kirnews", "kirnews_cleaned"),
        "swahili": lambda: load_swahili(),
        "filipino": lambda: load_filipino(args.data_dir),
        "custom": lambda: load_custom_dataset(args.data_dir),
    }

    try:
        if args.dataset in torchtext_loaders:
            dataset_pair = torchtext_loaders[args.dataset](root=args.data_dir)
        elif args.dataset in custom_loaders:
            dataset_pair = custom_loaders[args.dataset]()
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    except Exception as e:
        print(f"Error: failed to load dataset '{args.dataset}'.")
        print(f"Reason: {e}")
        raise SystemExit(1)

    if not args.all_test:
        if args.test_idx_fn is not None:
            try:
                test_idx = np.load(args.test_idx_fn)
                test_data, test_labels = read_torch_text_labels(dataset_pair[1], test_idx)
            except FileNotFoundError:
                print("No generated indices file for test set provided.")
                raise SystemExit(1)
        elif args.test_idx_start is not None:
            test_idx = list(range(args.test_idx_start, args.test_idx_end))
            test_data, test_labels = read_torch_text_labels(dataset_pair[1], test_idx)
        else:
            test_data, test_labels = pick_n_sample_from_each_class_given_dataset(
                dataset_pair[1], args.num_test, test_idx_fn
            )
    else:
        test_data, test_labels = read_torch_text_labels(
            dataset_pair[1], range(len(dataset_pair[1]))
        )

    if not args.all_train:
        if args.test_idx_fn is not None or args.test_idx_start is not None:
            train_idx = np.load(train_idx_fn + ".npy")
            train_data, train_labels = read_torch_text_labels(dataset_pair[0], train_idx)
        else:
            train_data, train_labels = pick_n_sample_from_each_class_given_dataset(
                dataset_pair[0], args.num_train, train_idx_fn
            )
    else:
        train_data, train_labels = read_torch_text_labels(
            dataset_pair[0], range(len(dataset_pair[0]))
        )

    print(f"Dataset   : {args.dataset}")
    print(f"Train     : {len(train_data)} samples")
    print(f"Test      : {len(test_data)} samples")
    print(f"k         : {args.k}")
    print(f"Compressor: {args.compressor}")

    if not args.record:
        non_neural_knn_exp(
            args.compressor, test_data, test_labels,
            train_data, train_labels,
            agg_by_concat_space, NCD,
            args.k, args.dataset,
            para=args.para, output_dir=args.results_dir,
        )
    else:
        if not args.score:
            start_idx = args.test_idx_start if args.test_idx_start is not None else 0
            for i in range(0, len(test_data), 100):
                print(f"from {start_idx + i} to {start_idx + i + 100}")
                output_rel_fn = f"test_dis_idx_from_{start_idx + i}_to_{start_idx + i + 100}"
                output_dir = os.path.join(args.output_dir, "distance", args.dataset)
                record_distance(
                    args.compressor,
                    np.array(test_data)[i:i + 100],
                    output_rel_fn,
                    train_data,
                    agg_by_concat_space,
                    NCD,
                    output_dir,
                    para=args.para,
                )
        else:
            if os.path.isdir(args.distance_fn):
                from tqdm import tqdm
                all_correct = 0
                total_num = 0
                for fn in tqdm(os.listdir(args.distance_fn)):
                    if fn.endswith(".npy"):
                        dis_matrix = np.load(os.path.join(args.distance_fn, fn))
                        start_idx = int(fn.split(".")[0].split("_")[-3])
                        end_idx = int(fn.split(".")[0].split("_")[-1])
                        sub_test_labels = test_labels[start_idx:end_idx]
                        correct = non_neural_knn_exp_given_dis(
                            dis_matrix, args.k, sub_test_labels, train_labels,
                            rand=args.random,
                        )
                        all_correct += sum(correct)
                        total_num += len(correct)
                        del dis_matrix
                print(f"Altogether Accuracy: {all_correct / total_num:.4f}")
            else:
                dis_matrix = np.load(args.distance_fn)
                non_neural_knn_exp_given_dis(
                    dis_matrix, args.k, test_labels, train_labels, rand=args.random
                )