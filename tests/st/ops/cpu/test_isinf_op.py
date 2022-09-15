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

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class NetIsInf(nn.Cell):
    def __init__(self):
        super(NetIsInf, self).__init__()
        self.isinf = P.IsInf()

    def construct(self, x):
        return self.isinf(x)


x1 = Tensor(np.array([3, np.log(0), 1, np.log(0)]), ms.float32)
x2 = Tensor(np.array([np.log(0), 1, np.log(0), 3]), ms.float32)
x3 = Tensor(np.array([[np.log(0), 2], [np.log(0), np.log(0)]]), ms.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_nan():
    ms_isinf = NetIsInf()
    output1 = ms_isinf(Tensor(x1))
    expect1 = [[False, True, False, True]]
    assert (output1.asnumpy() == expect1).all()

    output2 = ms_isinf(Tensor(x2))
    expect2 = [[True, False, True, False]]
    assert (output2.asnumpy() == expect2).all()

    output3 = ms_isinf(Tensor(x3))
    expect3 = [[True, False], [True, True]]
    assert (output3.asnumpy() == expect3).all()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_is_nan_cpu_dynamic_shape():
    """
    Feature: test FloatStatus op on CPU.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = NetIsInf()
    x_dyn = Tensor(shape=[1, 32, 9, None], dtype=ms.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, ms.float32))
    except_shape = (1, 32, 9, 9)
    assert output.asnumpy().shape == except_shape
