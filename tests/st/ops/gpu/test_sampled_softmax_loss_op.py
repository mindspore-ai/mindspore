# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor

def generate_test_data(num_classes, batch_size, sampled):
    dim = 10
    weights_s = np.linspace(start=1, stop=num_classes * dim, num=num_classes * dim)
    weights_s = np.reshape(weights_s, (num_classes, dim)).astype(np.float32) / 100.0
    biases_s = np.linspace(start=1, stop=num_classes, num=num_classes)
    biases_s = np.reshape(biases_s, (num_classes,)).astype(np.float32) / 100.0
    hidden_acts_s = np.linspace(start=1, stop=batch_size * dim, num=batch_size * dim)
    hidden_acts_s = np.reshape(
        hidden_acts_s, (batch_size, dim)).astype(np.float32) / 100.0

    true_exp = np.full([batch_size, 1], fill_value=0.5, dtype=np.float32)
    sampled_exp = np.full([len(sampled)], fill_value=0.5, dtype=np.float32)
    sampled_values = (Tensor(sampled), Tensor(true_exp), Tensor(sampled_exp))
    return weights_s, biases_s, hidden_acts_s, sampled_values


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sampled_softmax_loss_assigned_sampler():
    np.random.seed(0)
    num_classes = 7
    batch_size = 3
    labels = [0, 1, 2]
    (weights, biases, hidden_acts, sampled_vals) = generate_test_data(
        num_classes=num_classes,
        batch_size=batch_size,
        sampled=[4, 0, 2, 3])

    def case_not_remove_accidental_hits():
        loss = nn.SampledSoftmaxLoss(
            num_sampled=4,
            num_classes=num_classes,
            num_true=1,
            sampled_values=sampled_vals,
            remove_accidental_hits=False)

        got_sampled_softmax_loss = loss(Tensor(weights), Tensor(biases),
                                        Tensor(labels), Tensor(hidden_acts))
        exp_sampled_softmax_loss = np.array(
            [1.7318448, 1.8015041, 1.7211525]).astype(np.float32)
        assert np.allclose(got_sampled_softmax_loss.asnumpy(),
                           exp_sampled_softmax_loss)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    case_not_remove_accidental_hits()

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    case_not_remove_accidental_hits()

    (weights, biases, hidden_acts, sampled_vals) = generate_test_data(
        num_classes=num_classes,
        batch_size=batch_size,
        sampled=[4, 5, 6, 3])

    def case_remove_accidental_hits():
        loss = nn.SampledSoftmaxLoss(
            num_sampled=4,
            num_classes=num_classes,
            num_true=1,
            sampled_values=sampled_vals,
            remove_accidental_hits=True)

        got_sampled_softmax_loss = loss(Tensor(weights), Tensor(biases),
                                        Tensor(labels), Tensor(hidden_acts))
        exp_sampled_softmax_loss = np.array(
            [[1.85211, 2.10999, 2.20862]]).astype(np.float32)
        assert np.allclose(got_sampled_softmax_loss.asnumpy(),
                           exp_sampled_softmax_loss)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    case_remove_accidental_hits()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    case_remove_accidental_hits()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sampled_softmax_loss_none_sampler():
    np.random.seed(0)
    num_classes = 7
    batch_size = 3
    labels = [0, 1, 2]
    (weights, biases, hidden_acts, _) = generate_test_data(
        num_classes=num_classes,
        batch_size=batch_size,
        sampled=[4, 0, 2, 3])

    def case_no_sampler():
        loss = nn.SampledSoftmaxLoss(
            num_sampled=4,
            num_classes=num_classes,
            num_true=1,
            sampled_values=None,
            seed=1,
            remove_accidental_hits=False)

        got_sampled_softmax_loss = loss(Tensor(weights), Tensor(biases),
                                        Tensor(labels), Tensor(hidden_acts))
        exp_sampled_softmax_loss = np.array(
            [1.7345718, 1.820291, 1.7704818]).astype(np.float32)
        assert np.allclose(got_sampled_softmax_loss.asnumpy(),
                           exp_sampled_softmax_loss)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    case_no_sampler()

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    case_no_sampler()

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sampledsoftmaxloss_reduction_invalid():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    # Check 'reduction'
    with pytest.raises(ValueError):
        nn.SampledSoftmaxLoss(num_sampled=4, num_classes=7, reduction="")

    with pytest.raises(ValueError):
        nn.SampledSoftmaxLoss(num_sampled=4, num_classes=7, reduction="invalid")

    # reduction can be None, as defined in Loss
    # with pytest.raises(ValueError):
    #     nn.SampledSoftmaxLoss(num_sampled=4, num_classes=7, reduction=None)  #

    # Check 'num_true'
    with pytest.raises(ValueError):
        nn.SampledSoftmaxLoss(num_sampled=4, num_classes=7, num_true=0)

    # Check 'sampled_values'
    with pytest.raises(ValueError):
        sampled_values_more_para = (Tensor(np.array([1])), Tensor(np.array([1])),
                                    Tensor(np.array([1])), Tensor(np.array([1])))
        nn.SampledSoftmaxLoss(num_sampled=4, num_classes=7,
                              sampled_values=sampled_values_more_para)

    with pytest.raises(TypeError):
        sampled_values_wrong_type = Tensor(np.array([1]))
        nn.SampledSoftmaxLoss(num_sampled=4, num_classes=7,
                              sampled_values=sampled_values_wrong_type)

if __name__ == "__main__":
    test_sampled_softmax_loss_assigned_sampler()
    test_sampled_softmax_loss_none_sampler()
