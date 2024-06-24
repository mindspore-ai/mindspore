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
from tests.mark_utils import arg_mark

import pytest
import numpy as np
from scipy.special import log_softmax

from mindspore import Tensor, context, nn
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation


class Net(nn.Cell):
    def __init__(self, blank, reduction):
        super(Net, self).__init__()
        self.loss = P.CTCLossV2(blank=blank, reduction=reduction)

    def construct(self, input_matrix, target, input_lengths, target_lengths):
        x, _ = self.loss(input_matrix, target, input_lengths, target_lengths)
        return x


class GradData(nn.Cell):
    def __init__(self, network):
        super(GradData, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=False)
        self.network = network

    def construct(self, probs, indices, labels, input_lengths):
        return self.grad(self.network)(probs, indices, labels, input_lengths)[0]


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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("batch", [1, 10])
@pytest.mark.parametrize("data_type", [np.float64])
def test_ctc_loss_v2_un_padded(batch, data_type):
    """
    Feature: Test CTCLossV2.
    Description: The input is padded and the target target_sequences maybe equal to input_sequences
    Expectation: Result matches the numpy implemented version.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ctc_loss_v2_un_padded_grad():
    """
    Feature: Test CTCLossV2.
    Description: The input is padded and the target target_sequences maybe equal to input_sequences
    Expectation: Result matches the numpy implemented version.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np.random.seed(0)
    batch = 10
    data_type = np.float64

    method = 'none'
    input_sequences = 5
    classes = 3
    target_sequences = input_sequences
    target_sequences_min = 1

    input_matrix = log_softmax(np.random.randn(input_sequences, batch, classes), 2).astype(data_type)
    input_lengths = np.full(shape=(batch,), fill_value=input_sequences, dtype=np.int64)
    target_lengths = np.random.randint(low=target_sequences_min, high=target_sequences, size=(batch,), dtype=np.int64)
    target = np.random.randint(low=1, high=classes, size=(batch, np.max(target_lengths)), dtype=np.int64)

    input_matrix = Tensor(input_matrix)
    target = Tensor(target)
    input_lengths = Tensor(input_lengths)
    target_lengths = Tensor(target_lengths)

    net = Net(blank=0, reduction=method)
    loss = net(input_matrix, target, input_lengths, target_lengths)
    print(np.mean(loss.asnumpy()))

    expected_grad = np.array([[[2.21999385e-01, 1.49367328e-01, -3.71366713e-01],
                               [1.21524177e-01, -1.44682444e-01, 2.31582675e-02],
                               [-2.72130267e-01, 6.46665395e-02, 2.07463727e-01],
                               [-2.08145533e-01, 1.66320573e-01, 4.18249597e-02],
                               [-6.40181898e-02, 2.33895304e-01, -1.69877115e-01],
                               [1.66409322e-01, -2.88602261e-01, 1.22192939e-01],
                               [7.30902539e-01, 2.27492367e-01, -9.58394906e-01],
                               [4.02847968e-01, -5.02608798e-01, 9.97608303e-02],
                               [np.nan, np.nan, np.nan],
                               [3.54625312e-02, -4.78671463e-01, 4.43208932e-01]],

                              [[-2.47471377e-01, 4.80327110e-01, -2.32855733e-01],
                               [-7.19044568e-04, -3.50625805e-01, 3.51344849e-01],
                               [-7.06902188e-03, -8.43104544e-02, 9.13794763e-02],
                               [-9.66319775e-02, 2.63241041e-01, -1.66609063e-01],
                               [-1.33521992e-01, 8.99922273e-01, -7.66400281e-01],
                               [3.06852929e-02, -2.73818291e-02, -3.30346375e-03],
                               [-8.59374974e-01, 5.70923102e-01, 2.88451871e-01],
                               [-3.81211648e-01, 2.52157946e-01, 1.29053702e-01],
                               [np.nan, np.nan, np.nan],
                               [5.36556015e-03, -5.05351326e-02, 4.51695724e-02]],

                              [[-2.80587684e-01, 4.22536598e-01, -1.41948913e-01],
                               [7.27670649e-03, 1.89209901e-01, -1.96486607e-01],
                               [3.08220162e-02, -2.15289462e-01, 1.84467446e-01],
                               [-3.79525891e-01, 4.86187906e-01, -1.06662015e-01],
                               [-1.89419282e-01, 5.92301671e-02, 1.30189115e-01],
                               [-5.41303138e-02, 2.43931812e-01, -1.89801498e-01],
                               [3.48393864e-01, 5.03232105e-01, -8.51625969e-01],
                               [5.76510068e-01, -6.26906613e-01, 5.03965454e-02],
                               [np.nan, np.nan, np.nan],
                               [4.65760597e-02, 5.56476308e-02, -1.02223690e-01]],

                              [[-2.14977953e-01, 6.41234229e-01, -4.26256276e-01],
                               [5.60402917e-02, 1.95392934e-01, -2.51433226e-01],
                               [1.01160497e-01, -2.41139199e-01, 1.39978702e-01],
                               [-6.77672365e-01, 7.89331393e-01, -1.11659028e-01],
                               [-3.17830985e-01, 8.17107077e-01, -4.99276092e-01],
                               [-6.53148112e-02, 7.75629981e-02, -1.22481869e-02],
                               [-6.13690461e-01, 2.48194256e-01, 3.65496205e-01],
                               [2.56408238e-01, 4.37941616e-02, -3.00202400e-01],
                               [np.nan, np.nan, np.nan],
                               [-3.73922094e-02, 3.48893393e-01, -3.11501184e-01]],

                              [[6.82148555e-02, 1.06153890e-01, -1.74368746e-01],
                               [-3.82024509e-02, 9.73708746e-02, -5.91684237e-02],
                               [-3.38166563e-02, -1.84766114e-01, 2.18582770e-01],
                               [-3.88080543e-01, 1.25803041e-01, 2.62277502e-01],
                               [-5.94619350e-02, 4.98396907e-01, -4.38934972e-01],
                               [-4.53783646e-01, 3.90447024e-01, 6.33366220e-02],
                               [7.26180335e-01, 1.63813757e-01, -8.89994092e-01],
                               [3.35863956e-01, -7.44304322e-01, 4.08440366e-01],
                               [np.nan, np.nan, np.nan],
                               [-7.70374805e-02, 6.78337545e-02, 9.20372600e-03]]])

    grad = GradData(net)(input_matrix, target, input_lengths, target_lengths)

    print(grad.shape)
    print(grad)

    np.allclose(grad.asnumpy(), expected_grad)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("batch", [1, 10])
@pytest.mark.parametrize("data_type", [np.float32])
def test_ctc_loss_v2_padded(batch, data_type):
    """
    Feature: Test CTCLossV2.
    Description: The input is un-padded and the target target_sequences is shorter than input_sequences
    Expectation: Result matches the numpy implemented version.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
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
