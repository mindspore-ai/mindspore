# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bias_add_grad = G.BiasAddGrad()

    def construct(self, dout):
        return self.bias_add_grad(dout)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type", [np.float16, np.float32, np.float64])
def test_bias_add_grad2d(data_type):
    """
    Feature: CPU BiasAddGrad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    dout = np.ones([2, 3]).astype(data_type)
    bias_add_grad = Net()
    output = bias_add_grad(Tensor(dout))
    expect_output = np.array([2., 2., 2.]).astype(data_type)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type",
                         [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
def test_bias_add_grad4d(data_type):
    """
    Feature: CPU BiasAddGrad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    dout = np.ones([2, 3, 4, 4]).astype(data_type)
    bias_add_grad = Net()
    output = bias_add_grad(Tensor(dout))
    expect_output = np.array([32, 32, 32]).astype(data_type)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type", [np.complex64, np.complex128])
def test_bias_add_grad5d(data_type):
    """
    Feature: CPU BiasAddGrad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    dout = np.ones([2, 3, 4, 4, 2]).astype(data_type)
    bias_add_grad = Net()
    output = bias_add_grad(Tensor(dout))
    expect_output = np.array([64., 64., 64.]).astype(data_type)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_bias_add_grad4d_dyn_inputs():
    """
    Feature: CPU BiasAddGrad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    bias_add_grad = Net()
    dyn_input = Tensor(shape=[2, None, 4, 4], dtype=ms.float32)
    bias_add_grad.set_inputs(dyn_input)
    dout = Tensor(np.ones([2, 3, 4, 4]).astype(np.float32))
    output = bias_add_grad(dout)
    expect_output = np.array([32, 32, 32]).astype(np.float32)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_biasaddgrad_vmap():
    """
    Feature: biasaddgrad vmap test on cpu.
    Description: test the rightness of basic biasaddgrad vmap
    Expectation: use vmap rule's result equal to manually batched.
    """

    def cal_biasaddgrad(x):
        return G.BiasAddGrad(data_format="NCHW")(x)

    vmap_biasaddgrad = vmap(cal_biasaddgrad, in_axes=(0))
    x = Tensor(np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                         [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]).astype(np.float16))
    output = vmap_biasaddgrad(x)
    expect_out = np.array([[14, 22],
                           [46, 54]]).astype(np.float16)
    assert np.allclose(output.asnumpy(), expect_out)


if __name__ == '__main__':
    test_biasaddgrad_vmap()
