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
"""metric method for downstream task"""

import string
import re
from collections import Counter
import numpy as np

from .rouge_score import get_rouge_score
from .bleu import compute_bleu


class LastWordAccuracy():
    """
    LastWordAccuracy class is for lambada task (predict the final word of sentence)
    """

    def __init__(self):
        self.acc_num = 0
        self.total_num = 0

    def normalize(self, word):
        """normalization"""
        word = word.lstrip()
        word = word.rstrip()

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return remove_punc(lower(word))

    def update(self, predict_label, gold_label):
        if isinstance(predict_label, str) and isinstance(gold_label, str):
            predict_label = [predict_label]
            gold_label = [gold_label]
        for predict_word, gold_word in zip(predict_label, gold_label):
            self.total_num += 1
            if self.normalize(predict_word) == self.normalize(gold_word):
                self.acc_num += 1


class Accuracy():
    """
    calculate accuracy
    """

    def __init__(self):
        self.acc_num = 0
        self.total_num = 0

    def update(self, logits, labels):
        """accuracy update"""
        labels = np.reshape(labels, -1)
        logits_id = np.argmax(logits, axis=-1)
        print(" | Preict Label: {}   Gold Label: {}".format(logits_id, labels))
        self.acc_num += np.sum(labels == logits_id)
        self.total_num += len(labels)
        print("\n| Accuracy = {} \n".format(self.acc_num / self.total_num))


class F1():
    """calculate F1 score"""

    def __init__(self):
        self.f1_score = 0.0

    def get_normalize_answer_token(self, string_):
        """Lower text and remove punctuation, article and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(char for char in text if char not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(string_)))).split()

    def update(self, pred_answer, gold_answer):
        """F1 update"""
        common = Counter(pred_answer) & Counter(gold_answer)
        num_same = sum(common.values())
        # the number of same tokens between pred_answer and gold_answer
        precision = 1.0 * num_same / len(pred_answer) if pred_answer else 0
        recall = 1.0 * num_same / len(gold_answer) if gold_answer else 0
        if ' '.join(pred_answer).strip() == "" and ' '.join(gold_answer).strip() == "":
            self.f1_score += 1
        else:
            self.f1_score += 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0.0

        print('| precision: {}, recall: {}\n'.format(precision, recall))


class BLEU():
    """calculate BLEU score"""

    def __init__(self, tokenizer=None, max_order=4, smooth=True):
        self.bleu = 0.0
        self.total_num = 0
        self.tokenizer = tokenizer
        self.max_order = max_order
        self.smooth = smooth

    def sum_bleu(self, references, translations, max_order, smooth):
        """calculate the sum of bleu score"""
        all_result = []
        bleu_avg = 0.0
        for refer, trans in zip(references, translations):
            result = compute_bleu([[refer]], [trans], max_order, smooth)
            all_result.append(result)
            bleu_avg += result[0]
        bleu_avg /= len(references)
        return bleu_avg, all_result

    def update(self, hypotheses, references):
        """BLEU update"""
        hypo_l = []
        ref_l = []
        if self.tokenizer is not None:
            for hypo, ref in zip(hypotheses, references):
                if ref.strip() == '':
                    print("Reference is None, skip it !")
                    continue
                if hypo.strip() == '':
                    print("translation is None, skip it !")
                    continue
                hypo_l.append(self.tokenizer.encode(hypo))
                ref_l.append(self.tokenizer.encode(ref))

        if hypo_l and ref_l:
            hypotheses = hypo_l
            references = ref_l

            bleu_avg, _ = self.sum_bleu(references, hypotheses, self.max_order, self.smooth)
            self.bleu += bleu_avg * 100

        self.total_num += 1

        print("============== BLEU: {} ==============".format(float(self.bleu / self.total_num)))


class Rouge():
    '''
    Get Rouge Score
    '''

    def __init__(self):
        self.Rouge1 = 0.0
        self.Rouge2 = 0.0
        self.RougeL = 0.0
        self.total_num = 0

    def update(self, hypothesis, targets):
        scores = get_rouge_score(hypothesis, targets)
        self.Rouge1 += scores['rouge-1']['f'] * 100
        self.Rouge2 += scores['rouge-2']['f'] * 100
        self.RougeL += scores['rouge-l']['f'] * 100
        self.total_num += 1

        print("=============== ROUGE: {} ===============".format(
            (self.Rouge1 + self.Rouge2 + self.RougeL) / float(3.0 * self.total_num)))
