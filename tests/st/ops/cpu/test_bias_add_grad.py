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
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bias_add_grad = G.BiasAddGrad()

    def construct(self, dout):
        return self.bias_add_grad(dout)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add_grad2d():
    dout = np.ones([2, 3]).astype(np.float32)
    bias_add_grad = Net()
    output = bias_add_grad(Tensor(dout))
    expect_output = np.array([2., 2., 2.]).astype(np.float32)
    print(output.asnumpy())
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add_grad4d():
    dout = np.ones([2, 3, 4, 4]).astype(np.float32)
    bias_add_grad = Net()
    output = bias_add_grad(Tensor(dout))
    expect_output = np.array([32., 32., 32.]).astype(np.float32)
    print(output.asnumpy())
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add_grad5d():
    dout = np.ones([2, 3, 4, 4, 2]).astype(np.float32)
    bias_add_grad = Net()
    output = bias_add_grad(Tensor(dout))
    expect_output = np.array([64., 64., 64.]).astype(np.float32)
    print(output.asnumpy())
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
