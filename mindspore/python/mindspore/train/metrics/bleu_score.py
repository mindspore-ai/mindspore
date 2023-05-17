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
"""BleuScore."""
from __future__ import absolute_import

from collections import Counter
import numpy as np

from mindspore import _checkparam as validator
from mindspore.train.metrics.metric import Metric, rearrange_inputs


class BleuScore(Metric):
    """
    Calculates the BLEU score. BLEU (bilingual evaluation understudy) is a metric for evaluating
    the quality of text translated by machine.

    Args:
        n_gram (int): The n_gram value ranges from 1 to 4. Default: ``4`` .
        smooth (bool): Whether or not to apply smoothing. Default: ``False`` .

    Raises:
        ValueError: If the value range of n_gram is not from 1 to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.train import BleuScore
        >>>
        >>> candidate_corpus = [['i', 'have', 'a', 'pen', 'on', 'my', 'desk']]
        >>> reference_corpus = [[['i', 'have', 'a', 'pen', 'in', 'my', 'desk'],
        ...                      ['there', 'is', 'a', 'pen', 'on', 'the', 'desk']]]
        >>> metric = BleuScore()
        >>> metric.clear()
        >>> metric.update(candidate_corpus, reference_corpus)
        >>> bleu_score = metric.eval()
        >>> print(bleu_score)
        0.5946035575013605
    """
    def __init__(self, n_gram=4, smooth=False):
        super().__init__()
        self.n_gram = validator.check_value_type("n_gram", n_gram, [int])
        if self.n_gram > 4 or self.n_gram < 1:
            raise ValueError("For 'BleuScore', the argument 'n_gram' should range from 1 to 4, "
                             "but got {}.".format(n_gram))

        self.smooth = validator.check_value_type("smooth", smooth, [bool])
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self._numerator = np.zeros(self.n_gram)
        self._denominator = np.zeros(self.n_gram)
        self._precision_scores = np.zeros(self.n_gram)
        self._c = 0.0
        self._r = 0.0
        self._trans_len = 0
        self._ref_len = 0
        self._is_update = False

    def _count_ngram(self, ngram_input_list, n_gram):
        """
        Counting how many times each word appears in a given text with ngram.

        Args:
            ngram_input_list (list): A list of translated text or reference texts.
            n_gram (int): gram value ranges from 1 to 4.

        Return:
            ngram_counter: a collections.Counter object of ngram.
        """

        ngram_counter = Counter()

        for i in range(1, n_gram + 1):
            for j in range(len(ngram_input_list) - i + 1):
                ngram_key = tuple(ngram_input_list[j:(i + j)])
                ngram_counter[ngram_key] += 1

        return ngram_counter

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the internal evaluation result with `candidate_corpus` and `reference_corpus`.

        Args:
            inputs(iterator): Input `candidate_corpus` and `reference_corpus`.
                `candidate_corpus` and `reference_corpus` are
                both a list. The `candidate_corpus` is an iterable of machine translated corpus. The
                `reference_corpus` is an iterable object of iterables of reference corpus.

        Raises:
            ValueError: If the number of inputs is not 2.
            ValueError: If the lengths of `candidate_corpus` and `reference_corpus` are not equal.
        """
        if len(inputs) != 2:
            raise ValueError("For 'BleuScore.update', it needs 2 inputs (candidate_corpus, reference_corpus), "
                             "but got {}.".format(len(inputs)))
        candidate_corpus = inputs[0]
        reference_corpus = inputs[1]
        if len(candidate_corpus) != len(reference_corpus):
            raise ValueError("For 'BleuScore.update', 'translate_corpus' (inputs[0]) and 'reference_corpus' "
                             "(inputs[1]) should be equal in length, but got {}, {}"
                             .format(len(candidate_corpus), len(reference_corpus)))

        for (candidate, references) in zip(candidate_corpus, reference_corpus):
            self._c += len(candidate)
            ref_len_list = [len(ref) for ref in references]
            ref_len_diff = [abs(len(candidate) - x) for x in ref_len_list]
            self._r += ref_len_list[ref_len_diff.index(min(ref_len_diff))]
            translation_counter = self._count_ngram(candidate, self.n_gram)
            reference_counter = Counter()

            for ref in references:
                reference_counter |= self._count_ngram(ref, self.n_gram)

            ngram_counter_clip = translation_counter & reference_counter

            for counter_clip in ngram_counter_clip:
                self._numerator[len(counter_clip) - 1] += ngram_counter_clip[counter_clip]

            for counter in translation_counter:
                self._denominator[len(counter) - 1] += translation_counter[counter]

        self._trans_len = np.array(self._c)
        self._ref_len = np.array(self._r)
        self._is_update = True

    def eval(self):
        """
         Computes the bleu score.

         Returns:
             numpy.float64, the bleu score.

         Raises:
            RuntimeError: If the update method is not called first, an error will be reported.

        """
        if self._is_update is False:
            raise RuntimeError("Please call the 'update' method before calling 'eval' method.")
        if min(self._numerator) == 0.0:
            return np.array(0.0)

        if self.smooth:
            precision_scores = np.add(self._numerator, np.ones(self.n_gram)) / np.add(self._denominator,
                                                                                      np.ones(self.n_gram))
        else:
            precision_scores = self._numerator / self._denominator

        log_precision_scores = np.array([1.0 / self.n_gram] * self.n_gram) * np.log(precision_scores)
        geometric_mean = np.exp(np.sum(log_precision_scores))
        brevity_penalty = np.array(1.0) if self._c > self._r else np.exp(1 - (self._ref_len / self._trans_len))
        bleu = brevity_penalty * geometric_mean

        return bleu
