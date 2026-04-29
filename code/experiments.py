# Experiment framework
#
# CHANGES FROM ORIGINAL:
#   1. CRITICAL BUG FIX — calc_acc(): tie-breaking now uses the nearest
#      neighbour's label instead of marking the sample correct whenever ANY
#      tied label matches the true label. The original behaviour was
#      equivalent to top-k accuracy rather than true top-1 kNN prediction.
#
#   2. CRITICAL BUG FIX — combine_dis_acc(): same tie-breaking fix as above.
#
#   3. CRITICAL BUG FIX — combine_dis_acc_single():
#        a) Switched np.argpartition → np.argsort.
#           argpartition returns the k smallest indices but NOT in sorted order,
#           so sorted_idx[0] was not guaranteed to be the nearest neighbour.
#        b) Same tie-breaking fix as above.
#
# WHAT THE BUG WAS:
#   When k=2 and the two nearest neighbours had different labels, the old code
#   checked all tied labels and marked the prediction as correct if ANY tied
#   label matched the true label. That behaves like top-2 correctness, not
#   true kNN top-1 classification.
#
# WHAT IT AFFECTED:
#   It could inflate:
#   - accuracy
#   - precision
#   - recall
#   - F1-score
#
# HOW IT WAS SOLVED:
#   On ties, choose ONE final label only:
#   - if rand=True, choose randomly among tied labels
#   - otherwise, use the label of the closest neighbour among the tied labels
#   Then compare that one final predicted label with the true label.

import operator
import random
from collections import defaultdict
from typing import Any, Callable, Optional

import numpy as np
from compressors import DefaultCompressor
from tqdm import tqdm


def quiet_tqdm(iterable, *args, **kwargs):
    return tqdm(iterable, *args, disable=True, **kwargs)


class KnnExpText:
    def __init__(
        self,
        aggregation_function: Callable,
        compressor: DefaultCompressor,
        distance_function: Callable,
    ) -> None:
        self.aggregation_func = aggregation_function
        self.compressor = compressor
        self.distance_func = distance_function
        self.distance_matrix: list = []

    def calc_dis(
        self, data: list, train_data: Optional[list] = None, fast: bool = False
    ) -> None:
        """
        Calculates distances between `data` and itself (or `train_data` if given)
        and stores them in self.distance_matrix.
        """
        data_to_compare = data if train_data is None else train_data

        for i, t1 in quiet_tqdm(enumerate(data)):
            distance4i = []
            if fast:
                t1_compressed = self.compressor.get_compressed_len_fast(t1)
            else:
                t1_compressed = self.compressor.get_compressed_len(t1)

            for j, t2 in enumerate(data_to_compare):
                if fast:
                    t2_compressed = self.compressor.get_compressed_len_fast(t2)
                    t1t2_compressed = self.compressor.get_compressed_len_fast(
                        self.aggregation_func(t1, t2)
                    )
                else:
                    t2_compressed = self.compressor.get_compressed_len(t2)
                    t1t2_compressed = self.compressor.get_compressed_len(
                        self.aggregation_func(t1, t2)
                    )

                distance = self.distance_func(
                    t1_compressed, t2_compressed, t1t2_compressed
                )
                distance4i.append(distance)

            self.distance_matrix.append(distance4i)

    def calc_dis_with_single_compressed_given(
        self, data: list, data_len: list = None, train_data: Optional[list] = None
    ) -> None:
        """
        Calculates distances using pre-computed individual lengths.
        """
        data_to_compare = data if train_data is None else train_data

        for i, t1 in quiet_tqdm(enumerate(data)):
            distance4i = []
            t1_compressed = self.compressor.get_compressed_len_given_prob(t1, data_len[i])

            for j, t2 in quiet_tqdm(enumerate(data_to_compare)):
                t2_compressed = self.compressor.get_compressed_len_given_prob(
                    t2, data_len[j]
                )
                t1t2_compressed = self.compressor.get_compressed_len(
                    self.aggregation_func(t1, t2)
                )

                distance = self.distance_func(
                    t1_compressed, t2_compressed, t1t2_compressed
                )
                distance4i.append(distance)

            self.distance_matrix.append(distance4i)

    def calc_dis_single(self, t1: str, t2: str) -> float:
        """Returns NCD between a single pair of texts."""
        t1_compressed = self.compressor.get_compressed_len(t1)
        t2_compressed = self.compressor.get_compressed_len(t2)
        t1t2_compressed = self.compressor.get_compressed_len(
            self.aggregation_func(t1, t2)
        )
        return self.distance_func(t1_compressed, t2_compressed, t1t2_compressed)

    def calc_dis_single_multi(self, train_data: list, datum: str) -> list:
        """Returns a list of NCD values between `datum` and every item in `train_data`."""
        distance4i = []
        t1_compressed = self.compressor.get_compressed_len(datum)

        for j, t2 in quiet_tqdm(enumerate(train_data)):
            t2_compressed = self.compressor.get_compressed_len(t2)
            t1t2_compressed = self.compressor.get_compressed_len(
                self.aggregation_func(datum, t2)
            )
            distance = self.distance_func(t1_compressed, t2_compressed, t1t2_compressed)
            distance4i.append(distance)

        return distance4i

    def calc_dis_with_vector(self, data: list, train_data: Optional[list] = None):
        """Calculates distances when data points are already vectors."""
        data_to_compare = train_data if train_data is not None else data

        for i, t1 in quiet_tqdm(enumerate(data)):
            distance4i = [self.distance_func(t1, t2) for t2 in data_to_compare]
            self.distance_matrix.append(distance4i)

    def calc_acc(
        self,
        k: int,
        label: list,
        train_label: Optional[list] = None,
        provided_distance_matrix: Optional[list] = None,
        rand: bool = False,
    ) -> tuple:
        """
        Predicts labels using kNN and returns (predictions, correctness flags).

        Arguments:
            k: number of nearest neighbours.
            label: true labels for the test set.
            train_label: if given, treats self.distance_matrix as test-vs-train.
            provided_distance_matrix: optional pre-computed matrix.
            rand: if True, break ties randomly; if False (default), use the
                  nearest neighbour's label as the tie-breaker.

        Returns:
            (pred, correct): predictions and per-sample 0/1 correctness.
        """
        if provided_distance_matrix is not None:
            self.distance_matrix = provided_distance_matrix

        correct = []
        pred = []

        if train_label is not None:
            compare_label = train_label
            start = 0
            end = k
        else:
            compare_label = label
            start = 1
            end = k + 1

        for i in range(len(self.distance_matrix)):
            sorted_idx = np.argsort(np.array(self.distance_matrix[i]))

            pred_labels = defaultdict(int)
            for j in range(start, end):
                pred_l = compare_label[sorted_idx[j]]
                pred_labels[pred_l] += 1

            sorted_pred_lab = sorted(
                pred_labels.items(), key=operator.itemgetter(1), reverse=True
            )
            most_count = sorted_pred_lab[0][1]

            # OLD BUGGY LOGIC:
            # This treated ties like top-k correctness.
            #
            # if_right = 0
            # for pair in sorted_pred_lab:
            #     if pair[1] < most_count:
            #         break
            #     if pair[0] == label[i]:
            #         if_right = 1

            # FIX:
            # collect all tied labels with the same highest vote count
            tied_labels = [p[0] for p in sorted_pred_lab if p[1] == most_count]

            if rand:
                most_label = random.choice(tied_labels)
            else:
                if len(tied_labels) > 1:
                    # Tie -> choose the label of the closest neighbour
                    most_label = compare_label[sorted_idx[start]]
                else:
                    most_label = sorted_pred_lab[0][0]

            if_right = 1 if most_label == label[i] else 0

            pred.append(most_label)
            correct.append(if_right)

        print("Accuracy is {}".format(sum(correct) / len(correct)))
        return pred, correct

    def combine_dis_acc(
        self,
        k: int,
        data: list,
        label: list,
        train_data: Optional[list] = None,
        train_label: Optional[list] = None,
        rand: bool = False,
    ) -> tuple:
        """
        Calculates distance and accuracy in a single pass (no distance matrix stored).
        """
        correct = []
        pred = []

        if train_label is not None:
            compare_label = train_label
            start = 0
            end = k
        else:
            compare_label = label
            start = 1
            end = k + 1

        data_to_compare = train_data if train_data is not None else data

        for i, t1 in quiet_tqdm(enumerate(data)):
            distance4i = self.calc_dis_single_multi(data_to_compare, t1)
            sorted_idx = np.argsort(np.array(distance4i))

            pred_labels = defaultdict(int)
            for j in range(start, end):
                pred_l = compare_label[sorted_idx[j]]
                pred_labels[pred_l] += 1

            sorted_pred_lab = sorted(
                pred_labels.items(), key=operator.itemgetter(1), reverse=True
            )
            most_count = sorted_pred_lab[0][1]

            # OLD BUGGY LOGIC:
            # if_right = 0
            # for pair in sorted_pred_lab:
            #     if pair[1] < most_count:
            #         break
            #     if pair[0] == label[i]:
            #         if_right = 1

            # FIX:
            tied_labels = [p[0] for p in sorted_pred_lab if p[1] == most_count]

            if rand:
                most_label = random.choice(tied_labels)
            else:
                if len(tied_labels) > 1:
                    most_label = compare_label[sorted_idx[start]]
                else:
                    most_label = sorted_pred_lab[0][0]

            if_right = 1 if most_label == label[i] else 0

            pred.append(most_label)
            correct.append(if_right)

        print("Accuracy is {}".format(sum(correct) / len(correct)))
        return pred, correct

    def combine_dis_acc_single(
        self,
        k: int,
        train_data: list,
        train_label: list,
        datum: str,
        label: Any,
        rand: bool = False,
    ) -> tuple:
        """
        Calculates distance and accuracy for a single test datum.
        Used by the parallelised path in main_text.py via Pool.map.
        """
        distance4i = self.calc_dis_single_multi(train_data, datum)

        # OLD BUGGY LOGIC:
        # sorted_idx = np.argpartition(np.array(distance4i), range(k))
        #
        # Problem:
        # argpartition does not guarantee full sorted order,
        # so sorted_idx[0] was not guaranteed to be the nearest neighbour.

        # FIX:
        sorted_idx = np.argsort(np.array(distance4i))

        pred_labels = defaultdict(int)
        for j in range(k):
            pred_l = train_label[sorted_idx[j]]
            pred_labels[pred_l] += 1

        sorted_pred_lab = sorted(
            pred_labels.items(), key=operator.itemgetter(1), reverse=True
        )
        most_count = sorted_pred_lab[0][1]

        # OLD BUGGY LOGIC:
        # if_right = 0
        # for pair in sorted_pred_lab:
        #     if pair[1] < most_count:
        #         break
        #     if pair[0] == label:
        #         if_right = 1

        # FIX:
        tied_labels = [p[0] for p in sorted_pred_lab if p[1] == most_count]

        if rand:
            most_label = random.choice(tied_labels)
        else:
            if len(tied_labels) > 1:
                most_label = train_label[sorted_idx[0]]
            else:
                most_label = sorted_pred_lab[0][0]

        if_right = 1 if most_label == label else 0

        return most_label, if_right