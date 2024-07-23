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
import numpy as np
import pytest

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class BiasAddGradNet(nn.Cell):
    def __init__(self):
        super(BiasAddGradNet, self).__init__()
        self.op = G.BiasAddGrad()

    def construct(self, dout):
        return self.op(dout)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
def test_bias_add_grad2d(data_type):
    """
    Feature: GPU BiasAddGrad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    dout = np.ones([2, 3]).astype(data_type)
    bias_add_grad = BiasAddGradNet()
    output = bias_add_grad(Tensor(dout))
    expect_output = np.array([2., 2., 2.]).astype(data_type)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
def test_bias_add_grad4d(data_type):
    """
    Feature: GPU BiasAddGrad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    dout = np.ones([2, 3, 4, 4]).astype(data_type)
    bias_add_grad = BiasAddGradNet()
    output = bias_add_grad(Tensor(dout))
    expect_output = np.array([32, 32, 32]).astype(data_type)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
def test_bias_add_grad5d(data_type):
    """
    Feature: GPU BiasAddGrad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    dout = np.ones([2, 3, 4, 4, 2]).astype(data_type)
    bias_add_grad = BiasAddGradNet()
    output = bias_add_grad(Tensor(dout))
    expect_output = np.array([64., 64., 64.]).astype(data_type)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bias_add_grad4d_dyn_inputs():
    """
    Feature: GPU BiasAddGrad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    bias_add_grad = BiasAddGradNet()
    dyn_input = Tensor(shape=[2, None, 4, 4], dtype=ms.float32)
    bias_add_grad.set_inputs(dyn_input)
    dout = Tensor(np.ones([2, 3, 4, 4]).astype(np.float32))
    output = bias_add_grad(dout)
    expect_output = np.array([32, 32, 32]).astype(np.float32)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"
