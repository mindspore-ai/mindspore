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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class NetEluGrad(nn.Cell):
    def __init__(self):
        super(NetEluGrad, self).__init__()
        self.elu_grad = G.EluGrad()

    def construct(self, dy, y):
        return self.elu_grad(dy, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_elu_grad_fp32():
    y = Tensor(np.array([[[[-0.3, 1, 2],
                           [1, -0.6, 1],
                           [2, 1, -2]]]]).astype(np.float32))
    dy = Tensor(np.array([[[[-11, 2, 4],
                            [-1, 1, -1],
                            [-4, 4, -4]]]]).astype(np.float32))

    expect = np.array([[[[-7.7, 2, 4],
                         [-1, 0.4, -1],
                         [-4, 4, 4]]]]).astype(np.float32)

    error = np.ones(shape=[1, 1, 3, 3]) * 1.0e-6

    elu_grad = NetEluGrad()
    output = elu_grad(dy, y)
    print(output)
    diff = np.abs(output.asnumpy() - expect)
    double_check = diff / expect
    assert np.all(double_check < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_elu_grad_fp16():
    y = Tensor(np.array([[0.5, 2, 5.5], [4.5, -2, 0]]).astype(np.float16))
    dy = Tensor(np.array([[2, 1, 1.5], [-0.5, -1, -3]]).astype(np.float16))
    expect = np.array([[2, 1, 1.5], [-0.5, 1, -3]]).astype(np.float16)
    error = np.ones(shape=[2, 3]) * 1.0e-3

    elu_grad = NetEluGrad()
    output = elu_grad(dy, y)
    print(output)
    diff = np.abs(output.asnumpy() - expect)
    double_check = diff / expect
    assert np.all(double_check < error)
