# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""evaluation metrics"""
import os
from sklearn import metrics
import numpy as np

epsilon = 1e-8


def sklearn_auc_macro(gt, score):
    return metrics.roc_auc_score(gt, score, average='macro')


def sklearn_auc_micro(gt, score):
    return metrics.roc_auc_score(gt, score, average='micro')


def sklearn_f1_macro(gt, predict):
    return metrics.f1_score(gt, predict, average='macro')


def sklearn_f1_micro(gt, predict):
    return metrics.f1_score(gt, predict, average='micro')


def np_metrics(gt, predict, score=None, path=None):
    """numpy metrics"""
    try:
        sk_auc_macro = sklearn_auc_macro(gt, score)
    except ValueError:
        sk_auc_macro = -1
    sk_auc_micro = sklearn_auc_micro(gt, score)

    sk_f1_macro = sklearn_f1_macro(gt, predict)
    sk_f1_micro = sklearn_f1_micro(gt, predict)

    lab_sensitivity = label_sensitivity(gt, predict)
    lab_specificity = label_specificity(gt, predict)

    ex_subset_acc = example_subset_accuracy(gt, predict)
    ex_acc = example_accuracy(gt, predict)
    ex_precision = example_precision(gt, predict)
    ex_recall = example_recall(gt, predict)
    ex_f1 = compute_f1(ex_precision, ex_recall)

    lab_acc_macro, lab_acc_macro_list = label_accuracy_macro(
        gt, predict, average=False)
    lab_precision_macro, lab_precision_macro_list = label_precision_macro(
        gt, predict, average=False)
    lab_recall_macro, lab_recall_macro_list = label_recall_macro(
        gt, predict, average=False)
    lab_f1_macro, f1_list, f1_list_mean = label_f1_macro(
        gt, predict, average=False)

    lab_acc_micro = label_accuracy_micro(gt, predict)
    lab_precision_micro = label_precision_micro(gt, predict)
    lab_recall_micro = label_recall_micro(gt, predict)
    lab_f1_micro = compute_f1(lab_precision_micro, lab_recall_micro)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "eval.txt"), 'a+') as f:
        f.write("--------------------------------------------\n")
        f.write("example_subset_accuracy:   %.4f\n" % ex_subset_acc)
        f.write("example_accuracy:          %.4f\n" % ex_acc)
        f.write("example_precision:         %.4f\n" % ex_precision)
        f.write("example_recall:            %.4f\n" % ex_recall)
        f.write("example_f1:                %.4f\n" % ex_f1)

        f.write("label_accuracy_macro:      %.4f\n" % lab_acc_macro)
        f.write("label_precision_macro:     %.4f\n" % lab_precision_macro)
        f.write("label_recall_macro:        %.4f\n" % lab_recall_macro)
        f.write("label_f1_macro:            %.4f\n" % lab_f1_macro)

        f.write("label_accuracy_micro:      %.4f\n" % lab_acc_micro)
        f.write("label_precision_micro:     %.4f\n" % lab_precision_micro)
        f.write("label_recall_micro:        %.4f\n" % lab_recall_micro)
        f.write("label_f1_micro:            %.4f\n" % lab_f1_micro)

        f.write("sk_auc_macro:              %.4f\n" % sk_auc_macro)
        f.write("sk_auc_micro:              %.4f\n" % sk_auc_micro)
        f.write("sk_f1_macro:               %.4f\n" % sk_f1_macro)
        f.write("sk_f1_micro:               %.4f\n" % sk_f1_micro)
        f.write("lab_sensitivity:           %.4f\n" % lab_sensitivity)
        f.write("lab_specificity:           %.4f\n" % lab_specificity)

        f.write("\nlabel_f1_average: %.4f\n" % f1_list_mean)

        f.write("label_accuracy_macro: \n")
        for i, v in enumerate(lab_acc_macro_list):
            f.write("(label:%d,label_accuracy: %.4f)\n" % (i, v))
        f.write("label_precious_macro: \n")
        for i, v in enumerate(lab_precision_macro_list):
            f.write("(label:%d,lab_precision:  %.4f)\n" % (i, v))
        f.write("label_recall_macro: \n")
        for i, v in enumerate(lab_recall_macro_list):
            f.write("(label:%d,lab_recall:     %.4f)\n" % (i, v))
        f.write("label_f1_macro: \n")
        for i, v in enumerate(f1_list):
            f.write("(label:%d,lab_f1:         %.4f)\n" % (i, v))
    return sk_f1_macro, sk_f1_micro, sk_auc_macro



def threshold_tensor_batch(predict, base=0.5):
    '''make sure at least one label for batch'''
    p_max = np.max(predict, axis=1)
    pivot = np.ones(p_max.shape) * base
    pivot = pivot.astype(np.float32)
    threshold = np.minimum(p_max, pivot)
    pd_threshold = np.greater_equal(predict, threshold[:, np.newaxis])
    return pd_threshold


def compute_f1(precision, recall):
    return 2 * precision * recall / (precision + recall + epsilon)


def example_subset_accuracy(gt, predict):
    ex_equal = np.all(np.equal(gt, predict), axis=1).astype("float32")
    return np.mean(ex_equal)


def example_accuracy(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_or = np.sum(np.logical_or(gt, predict), axis=1).astype("float32")
    return np.mean((ex_and + epsilon) / (ex_or + epsilon))


def example_precision(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_predict = np.sum(predict, axis=1).astype("float32")
    return np.mean((ex_and + epsilon) / (ex_predict + epsilon))


def example_recall(gt, predict):
    ex_and = np.sum(np.logical_and(gt, predict), axis=1).astype("float32")
    ex_gt = np.sum(gt, axis=1).astype("float32")
    return np.mean((ex_and + epsilon) / (ex_gt + epsilon))


def example_f1(gt, predict):
    p = example_precision(gt, predict)
    r = example_recall(gt, predict)
    return (2 * p * r) / (p + r + epsilon)


def _label_quantity(gt, predict):
    tp = np.sum(np.logical_and(gt, predict), axis=0)
    fp = np.sum(np.logical_and(1 - gt, predict), axis=0)
    tn = np.sum(np.logical_and(1 - gt, 1 - predict), axis=0)
    fn = np.sum(np.logical_and(gt, 1 - predict), axis=0)
    return np.stack([tp, fp, tn, fn], axis=0).astype("float")


def label_accuracy_macro(gt, predict, average=True):
    quantity = _label_quantity(gt, predict)
    tp_tn = np.add(quantity[0], quantity[2])
    tp_fp_tn_fn = np.sum(quantity, axis=0)
    if average:
        return np.mean((tp_tn + epsilon) / (tp_fp_tn_fn + epsilon))
    return np.mean((tp_tn + epsilon) / (tp_fp_tn_fn + epsilon)), (tp_tn + epsilon) / (tp_fp_tn_fn + epsilon)


def label_precision_macro(gt, predict, average=True):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fp = np.add(quantity[0], quantity[1])
    if average:
        return np.mean((tp + epsilon) / (tp_fp + epsilon))
    return np.mean((tp + epsilon) / (tp_fp + epsilon)), (tp + epsilon) / (tp_fp + epsilon)


def label_recall_macro(gt, predict, average=True):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fn = np.add(quantity[0], quantity[3])
    if average:
        return np.mean((tp + epsilon) / (tp_fn + epsilon))
    return np.mean((tp + epsilon) / (tp_fn + epsilon)), (tp + epsilon) / (tp_fn + epsilon)


def label_f1_macro(gt, predict, average=True):
    p, plist = label_precision_macro(gt, predict, average=False)
    r, rlist = label_recall_macro(gt, predict, average=False)
    f1_list = (2 * plist * rlist) / (plist + rlist + epsilon)
    if average:
        return (2 * p * r) / (p + r + epsilon)
    return (2 * p * r) / (p + r + epsilon), f1_list, np.mean(f1_list)


def label_accuracy_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, sum_tn, sum_fn = np.sum(quantity, axis=1)
    return (sum_tp + sum_tn + epsilon) / (
        sum_tp + sum_fp + sum_tn + sum_fn + epsilon)


def label_precision_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, sum_fp, _, _ = np.sum(quantity, axis=1)
    return (sum_tp + epsilon) / (sum_tp + sum_fp + epsilon)


def label_recall_micro(gt, predict):
    quantity = _label_quantity(gt, predict)
    sum_tp, _, _, sum_fn = np.sum(quantity, axis=1)
    return (sum_tp + epsilon) / (sum_tp + sum_fn + epsilon)


def label_f1_micro(gt, predict):
    p = label_precision_micro(gt, predict)
    r = label_recall_micro(gt, predict)
    return (2 * p * r) / (p + r + epsilon)


def label_sensitivity(gt, predict):
    return label_recall_micro(gt, predict)


def label_specificity(gt, predict):
    quantity = _label_quantity(gt, predict)
    _, sum_fp, sum_tn, _ = np.sum(quantity, axis=1)
    return (sum_tn + epsilon) / (sum_tn + sum_fp + epsilon)


def single_label_accuracy(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp_tn = np.add(quantity[0], quantity[2])
    tp_fp_tn_fn = np.sum(quantity, axis=0)
    return (tp_tn + epsilon) / (tp_fp_tn_fn + epsilon)


def single_label_precision(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fp = np.add(quantity[0], quantity[1])
    return (tp + epsilon) / (tp_fp + epsilon)


def single_label_recall(gt, predict):
    quantity = _label_quantity(gt, predict)
    tp = quantity[0]
    tp_fn = np.add(quantity[0], quantity[3])
    return (tp + epsilon) / (tp_fn + epsilon)


def print_metrics(gt, predict):
    """print metrics results"""
    ex_subset_acc = example_subset_accuracy(gt, predict)
    ex_acc = example_accuracy(gt, predict)
    ex_precision = example_precision(gt, predict)
    ex_recall = example_recall(gt, predict)
    ex_f1 = compute_f1(ex_precision, ex_recall)

    lab_acc_macro = label_accuracy_macro(gt, predict)
    lab_precision_macro = label_precision_macro(gt, predict)
    lab_recall_macro = label_recall_macro(gt, predict)
    lab_f1_macro = compute_f1(lab_precision_macro, lab_recall_macro)

    lab_acc_micro = label_accuracy_micro(gt, predict)
    lab_precision_micro = label_precision_micro(gt, predict)
    lab_recall_micro = label_recall_micro(gt, predict)
    lab_f1_micro = compute_f1(lab_precision_micro, lab_recall_micro)

    print("example_subset_accuracy:", ex_subset_acc)
    print("example_accuracy:", ex_acc)
    print("example_precision:", ex_precision)
    print("example_recall:", ex_recall)
    print("example_f1:", ex_f1)

    print("label_accuracy_macro:", lab_acc_macro)
    print("label_precision_macro:", lab_precision_macro)
    print("label_recall_macro:", lab_recall_macro)
    print("label_f1_macro:", lab_f1_macro)

    print("label_accuracy_micro:", lab_acc_micro)
    print("label_precision_micro:", lab_precision_micro)
    print("label_recall_micro:", lab_recall_micro)
    print("label_f1_micro:", lab_f1_micro)


def write_metrics(gt, predict, path):
    """write metrics results"""
    ex_subset_acc = example_subset_accuracy(gt, predict)
    ex_acc = example_accuracy(gt, predict)
    ex_precision = example_precision(gt, predict)
    ex_recall = example_recall(gt, predict)
    ex_f1 = compute_f1(ex_precision, ex_recall)

    lab_acc_macro = label_accuracy_macro(gt, predict)
    lab_precision_macro = label_precision_macro(gt, predict)
    lab_recall_macro = label_recall_macro(gt, predict)
    lab_f1_macro = compute_f1(lab_precision_macro, lab_recall_macro)

    lab_acc_micro = label_accuracy_micro(gt, predict)
    lab_precision_micro = label_precision_micro(gt, predict)
    lab_recall_micro = label_recall_micro(gt, predict)
    lab_f1_micro = compute_f1(lab_precision_micro, lab_recall_micro)

    with open(path, 'w') as f:
        f.write("example_subset_accuracy:   %.4f\n" % ex_subset_acc)
        f.write("example_accuracy:          %.4f\n" % ex_acc)
        f.write("example_precision:         %.4f\n" % ex_precision)
        f.write("example_recall:            %.4f\n" % ex_recall)
        f.write("example_f1:                %.4f\n" % ex_f1)

        f.write("label_accuracy_macro:      %.4f\n" % lab_acc_macro)
        f.write("label_precision_macro:     %.4f\n" % lab_precision_macro)
        f.write("label_recall_macro:        %.4f\n" % lab_recall_macro)
        f.write("label_f1_macro:            %.4f\n" % lab_f1_macro)

        f.write("label_accuracy_micro:      %.4f\n" % lab_acc_micro)
        f.write("label_precision_micro:     %.4f\n" % lab_precision_micro)
        f.write("label_recall_micro:        %.4f\n" % lab_recall_micro)
        f.write("label_f1_micro:            %.4f\n" % lab_f1_micro)
