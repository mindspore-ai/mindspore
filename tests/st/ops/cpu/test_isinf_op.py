# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class NetIsInf(nn.Cell):
    def __init__(self):
        super(NetIsInf, self).__init__()
        self.isinf = P.IsInf()

    def construct(self, x):
        return self.isinf(x)


x1 = Tensor(np.array([3, np.log(0), 1, np.log(0)]), mstype.float32)
x2 = Tensor(np.array([np.log(0), 1, np.log(0), 3]), mstype.float32)
x3 = Tensor(np.array([[np.log(0), 2], [np.log(0), np.log(0)]]), mstype.float32)


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
    x_dyn = Tensor(shape=[1, 32, 9, None], dtype=mstype.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, mstype.float32))
    except_shape = (1, 32, 9, 9)
    assert output.asnumpy().shape == except_shape


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_isinf_functional_api_modes(mode):
    """
    Feature: Test isinf functional api.
    Description: Test isinf functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mstype.float32)
    output = F.isinf(x)
    expected = np.array([False, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_isinf_tensor_api_modes(mode):
    """
    Feature: Test isinf tensor api.
    Description: Test isinf tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mstype.float32)
    output = x.isinf()
    expected = np.array([False, False, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)
