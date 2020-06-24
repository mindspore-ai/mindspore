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
"""Calculate Perplexity score under N-gram language model."""
from typing import Union

import numpy as np

NINF = -1.0 * 1e9


def ngram_ppl(prob: Union[np.ndarray, list], log_softmax=False, index: float = np.e):
    """
    Calculate Perplexity(PPL) score under N-gram language model.

    Please make sure the sum of `prob` is 1.
    Otherwise, assign `normalize=True`.

    The number of N is depended by model.

    Args:
        prob (Union[list, np.ndarray]): Prediction probability
            of the sentence.
        log_softmax (bool): If sum of `prob` is not 1, please
            set normalize=True.
        index (float): Base number of log softmax.

    Returns:
        float, ppl score.
    """
    eps = 1e-8
    if not isinstance(prob, (np.ndarray, list)):
        raise TypeError("`prob` must be type of list or np.ndarray.")
    if not isinstance(prob, np.ndarray):
        prob = np.array(prob)
    if prob.shape[0] == 0:
        raise ValueError("`prob` length must greater than 0.")

    p = 1.0
    sen_len = 0
    for t in range(prob.shape[0]):
        s = prob[t]
        if s <= NINF:
            break
        if log_softmax:
            s = np.power(index, s)
        p *= (1 / (s + eps))
        sen_len += 1

    if sen_len == 0:
        return np.inf

    return pow(p, 1 / sen_len)
