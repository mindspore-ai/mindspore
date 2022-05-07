# Copyright 2022 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
from scipy.special import log_softmax

from mindspore import Tensor, context
from mindspore.ops import operations as P


def logsumexp(a, b):
    if a < b:
        a, b = b, a

    if b == -np.inf:
        return a
    return a + np.log(1 + np.exp(b - a))


def np_ctc_loss_1_batch(y, target):
    """
        y: shape time_series, classes
        target: 1d array
    """
    labels = [0]
    for i in target:
        labels.append(i)
        labels.append(0)

    labels = np.array(labels)
    time_series = y.shape[0]
    labels_len = len(labels)
    alpha = np.full((time_series, labels_len), -np.inf)

    alpha[0, 0] = y[0, labels[0]]
    alpha[0, 1] = y[0, labels[1]]

    for t in range(1, time_series):
        for s in range(labels_len):
            label = labels[s]

            a = alpha[t - 1, s]
            if s - 1 >= 0:
                a = logsumexp(a, alpha[t - 1, s - 1])
            if s - 2 >= 0 and label != 0 and label != labels[s - 2]:
                a = logsumexp(a, alpha[t - 1, s - 2])
            alpha[t, s] = a + y[t, label]

    neg_log_likelihood = -logsumexp(alpha[-1, -1], alpha[-1, -2])
    return neg_log_likelihood


def np_ctc_loss(logits, targets, logits_lens, target_lens, reduction: str = 'mean'):
    assert len(logits.shape) == 3
    _, batch, _ = logits.shape

    if len(targets.shape) == 1:
        processed_target = []
        i = 0
        for target_len in target_lens:
            processed_target.append(targets[i:i + target_len])
            i += target_len
        targets = processed_target

    result = np.array([np_ctc_loss_1_batch(logits[:logits_lens[b], b, :], targets[b][:target_lens[b]])
                       for b in range(batch)])

    if reduction == 'mean':
        return (result / target_lens).mean()
    if reduction == 'sum':
        return np.sum(result)
    return result


def compare_to_numpy(method, input_matrix, target, input_lengths, target_lengths):
    expected = np_ctc_loss(input_matrix, target, input_lengths, target_lengths, method)

    input_matrix = Tensor(input_matrix)
    target = Tensor(target)
    input_lengths = Tensor(input_lengths)
    target_lengths = Tensor(target_lengths)

    loss, _ = P.CTCLossV2(blank=0, reduction=method)(input_matrix, target, input_lengths, target_lengths)
    assert np.allclose(loss.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("batch", [1, 10])
@pytest.mark.parametrize("data_type", [np.float64])
def test_ctc_loss_v2_un_padded(batch, data_type):
    """
    Feature: Test CTCLossV2.
    Description: The input is padded and the target target_sequences maybe equal to input_sequences
    Expectation: Result matches the numpy implemented version.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    np.random.seed(0)
    method = 'none'
    input_sequences = 10
    classes = 5
    target_sequences = input_sequences
    target_sequences_min = 1

    input_matrix = log_softmax(np.random.randn(input_sequences, batch, classes), 2).astype(data_type)
    input_lengths = np.full(shape=(batch,), fill_value=input_sequences, dtype=np.int64)
    target_lengths = np.random.randint(low=target_sequences_min, high=target_sequences, size=(batch,), dtype=np.int64)
    target = np.random.randint(low=1, high=classes, size=(batch, np.max(target_lengths)), dtype=np.int64)

    compare_to_numpy(method, input_matrix, target, input_lengths, target_lengths)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("batch", [1, 10])
@pytest.mark.parametrize("data_type", [np.float32])
def test_ctc_loss_v2_padded(batch, data_type):
    """
    Feature: Test CTCLossV2.
    Description: The input is un-padded and the target target_sequences is shorter than input_sequences
    Expectation: Result matches the numpy implemented version.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    np.random.seed(0)
    method = 'none'
    input_sequences = 50
    input_sequences_min = 35
    classes = 20
    target_sequences = 30  # Target sequence length of longest target in batch (padding length)
    target_sequences_min = 10  # Minimum target length, for demonstration purposes

    # Initialize random batch of input vectors, for *size = (input_sequences,N,classes)
    input_matrix = log_softmax(np.random.randn(input_sequences, batch, classes), 2).astype(data_type)

    # Initialize random batch of targets (0 = blank, 1:classes = classes)
    target = np.random.randint(low=1, high=classes, size=(batch, target_sequences), dtype=np.int64)

    input_lengths = np.random.randint(low=input_sequences_min, high=input_sequences, size=(batch,), dtype=np.int64)
    target_lengths = np.random.randint(low=target_sequences_min, high=target_sequences, size=(batch,), dtype=np.int64)

    compare_to_numpy(method, input_matrix, target, input_lengths, target_lengths)
