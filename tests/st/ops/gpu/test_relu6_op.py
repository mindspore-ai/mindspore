# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner


class NetReLU6(nn.Cell):
    def __init__(self):
        super(NetReLU6, self).__init__()
        self.relu6 = P.ReLU6()

    def construct(self, x):
        return self.relu6(x)


class NetRelu6Dynamic(nn.Cell):
    def __init__(self):
        super(NetRelu6Dynamic, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.relu6 = P.ReLU6()

    def construct(self, x):
        x = self.test_dynamic(x)
        return self.relu6(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu6():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [5.9, 6.1, 6],
                           [10, 1, -1]]]]).astype(np.float32))
    expect = np.array([[[[0, 1, 6,],
                         [5.9, 6, 6,],
                         [6, 1, 0.]]]]).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    relu6 = NetReLU6()
    output = relu6(x)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    relu6 = NetReLU6()
    output = relu6(x)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_relu6_dynamic():

    x1 = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype(np.float32))
    expect1 = np.array([[0, 4, 0,],
                        [2, 0, 6,]]).astype(np.float32)
    x2 = Tensor(np.array([[[[-1, 1, 10],
                            [5.9, 6.1, 6],
                            [10, 1, -1]]]]).astype(np.float32))
    expect2 = np.array([[[[0, 1, 6,],
                          [5.9, 6, 6,],
                          [6, 1, 0.]]]]).astype(np.float32)


    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    relu6 = NetRelu6Dynamic()
    output1 = relu6(x1)
    assert (output1.asnumpy() == expect1).all()
    output2 = relu6(x2)
    assert (output2.asnumpy() == expect2).all()
