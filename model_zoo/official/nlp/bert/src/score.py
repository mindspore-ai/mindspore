# Copyright 2020 Huawei Technologies Co., Ltd
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

"""
Calculate average F1 score among labels.
"""

import json

def get_f1_score_for_each_label(pre_lines, gold_lines, label):
    """
    Get F1 score for each label.
    Args:
        pre_lines: listed label info from pre_file.
        gold_lines: listed label info from gold_file.
        label:

    Returns:
        F1 score for this label.
    """
    TP = 0
    FP = 0
    FN = 0
    index = 0
    while index < len(pre_lines):
        pre_line = pre_lines[index].get(label, {})
        gold_line = gold_lines[index].get(label, {})
        for sample in pre_line:
            if sample in gold_line:
                TP += 1
            else:
                FP += 1
        for sample in gold_line:
            if sample not in pre_line:
                FN += 1
        index += 1
    f1 = 2 * TP / (2 * TP + FP + FN)
    return f1


def get_f1_score(labels, pre_file, gold_file):
    """
    Get F1 scores for each label.
    Args:
        labels: list of labels.
        pre_file: prediction file.
        gold_file: ground truth file.

    Returns:
        average F1 score on all labels.
    """
    pre_lines = [json.loads(line.strip())['label'] for line in open(pre_file) if line.strip()]
    gold_lines = [json.loads(line.strip())['label'] for line in open(gold_file) if line.strip()]
    if len(pre_lines) != len(gold_lines):
        raise ValueError("pre file and gold file have different line count.")
    f1_sum = 0
    for label in labels:
        f1 = get_f1_score_for_each_label(pre_lines, gold_lines, label)
        print('label: %s, F1: %.6f' % (label, f1))
        f1_sum += f1

    return f1_sum/len(labels)


def get_result(labels, pre_file, gold_file):
    avg = get_f1_score(labels, pre_file=pre_file, gold_file=gold_file)
    print("avg F1: %.6f" % avg)
