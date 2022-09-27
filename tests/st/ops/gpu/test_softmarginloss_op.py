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
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, reduction="mean"):
        super(Net, self).__init__()
        self.softmarginloss = P.SoftMarginLoss(reduction)

    def construct(self, logits, labels):
        return self.softmarginloss(logits, labels)


def assert_func(logits, labels, expect_none, expect_mean, expect_sum, epsilon):
    loss = Net(reduction="none")
    output = loss(logits, labels)
    output = output.asnumpy()
    assert np.allclose(output, expect_none, epsilon)

    loss = Net(reduction="mean")
    output = loss(logits, labels).asnumpy()
    assert np.allclose(output, expect_mean, epsilon)

    loss = Net(reduction="sum")
    output = loss(logits, labels).asnumpy()
    assert np.allclose(output, expect_sum, epsilon)



def softmarginloss_testset():
    np_types = [np.float16, np.float32, np.float64]
    mindspore_types = [mindspore.float16, mindspore.float32, mindspore.float64]
    epsilons = [1e-3, 1e-4, 1e-5]

    for i in range(3):
        logits = Tensor(np.array([[0.3000, 0.7000, 0.4000, 0.9000],
                                  [0.5000, 0.5000, 0.5000, 0.5000]]), mindspore_types[i])
        labels = Tensor(np.array([[-1., 1., 1., -1.],
                                  [1., -1., -1., 1.]]), mindspore_types[i])

        expect_none = np.array([[0.8543552, 0.4031860, 0.5130153, 1.2411539],
                                [0.4740770, 0.9740770, 0.9740770, 0.4740770]]).astype(np_types[i])
        expect_mean = np.array(0.7385022946508065).astype(np_types[i])
        expect_sum = np.array(5.908018357206452).astype(np_types[i])

        assert_func(logits, labels, expect_none,
                    expect_mean, expect_sum, epsilons[i])


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_softmarginloss_graph():
    """
    Feature: SoftMarginLoss
    Description: Test float16,float32,float64 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    softmarginloss_testset()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_soft_margin_loss_pynative():
    """
    Feature: SoftMarginLoss
    Description: Another test float16,float32,float64 of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    softmarginloss_testset()
