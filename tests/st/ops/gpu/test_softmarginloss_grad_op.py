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

import numpy as np
import pytest

import mindspore
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


class Net(nn.Cell):
    def __init__(self, reduction="mean"):
        super(Net, self).__init__()
        self.softmarginlossgrad = G.SoftMarginLossGrad(reduction)

    def construct(self, logits, labels, dout):
        return self.softmarginlossgrad(logits, labels, dout)


def assert_func(logits, labels, dout_none, dout_other, expect_none, expect_mean, expect_sum, epsilon):
    loss = Net(reduction="none")
    output = loss(logits, labels, dout_none).asnumpy()
    assert np.allclose(output, expect_none, epsilon)

    loss = Net(reduction="mean")
    output = loss(logits, labels, dout_other).asnumpy()
    assert np.allclose(output, expect_mean, epsilon)

    loss = Net(reduction="sum")
    output = loss(logits, labels, dout_other).asnumpy()
    assert np.allclose(output, expect_sum, epsilon)


def softmarginlossgrad_testset():
    np_types = [np.float16, np.float32, np.float64]
    mindspore_types = [mindspore.float16, mindspore.float32, mindspore.float64]
    epsilons = [1e-3, 1e-4, 1e-5]

    for i in range(2):
        logits = Tensor(np.array([[0.3000, 0.7000, 0.4000, 0.9000],
                                  [0.5000, 0.5000, 0.5000, 0.5000]]), mindspore_types[i])
        labels = Tensor(np.array([[-1., 1., 1., -1.],
                                  [1., -1., -1., 1.]]), mindspore_types[i])
        dout_none = Tensor(np.array([[-1.6353, -0.4148, 2.0491, -0.8399],
                                     [0.5377, -1.7429, -0.3595, -0.2867]]), mindspore_types[i])
        dout_other = Tensor(np.array(2), mindspore_types[i])

        expect_none = np.array([[-0.9393858, 0.1376357, -0.8223291, -0.5971265],
                                [-0.2030036, -1.0848844, -0.2237741, 0.1082409]]).astype(np_types[i])
        expect_mean = np.array([[0.1436106, -0.0829531, -0.1003281, 0.1777374],
                                [-0.0943852, 0.1556148, 0.1556148, -0.0943852]]).astype(np_types[i])
        expect_sum = np.array([[1.1488850, -0.6636245, -0.8026247, 1.4218990],
                               [-0.7550813, 1.2449187, 1.2449187, -0.7550813]]).astype(np_types[i])

        assert_func(logits, labels, dout_none, dout_other, expect_none,
                    expect_mean, expect_sum, epsilons[i])


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_softmarginlossgrad_graph():
    """
    Feature: SoftMarginLossGrad
    Description: Test float16,float32,float64 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    softmarginlossgrad_testset()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_soft_margin_loss_pynative():
    """
    Feature: SoftMarginLossGrad
    Description: Another test float16,float32,float64 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    softmarginlossgrad_testset()
