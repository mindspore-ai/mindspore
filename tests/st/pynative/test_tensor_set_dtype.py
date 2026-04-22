# Copyright 2025 Huawei Technologies Co., Ltd
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
import pytest
from tests.mark_utils import arg_mark
import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore import tensor, Parameter
import mindspore.nn as nn


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.x = Parameter(ops.ones((2, 2)))

    def construct(self, y):
        result = self.x * y
        return result


class Net2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.x = Parameter(ops.ones((2, 2)), name="x")

    def construct(self, y):
        result = self.x * y
        return result


class Net3(nn.Cell):
    def __init__(self):
        super().__init__()
        self.x = Parameter(ms.Tensor([[1, 1], [1, 1]], dtype=ms.float32))

    def construct(self, y):
        result = self.x * y
        return result


class Net4(nn.Cell):
    def __init__(self):
        super().__init__()
        self.x = Parameter(ms.Tensor([[1, 1], [1, 1]], dtype=ms.float32), name="x")

    def construct(self, y):
        result = self.x * y
        return result


@arg_mark(plat_marks=['platform_gpu'],
    level_mark='level0',
    card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_set_dtype(mode):
    """
    Feature: tensor.set_dtype
    Description: Verify the result of tensor.set_dtype
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    input_y = tensor([[1, 2], [3, 4]], dtype=ms.float16)
    expected_output = tensor([[1, 2], [3, 4]], dtype=ms.float16)

    net1 = Net()
    net1.x.set_dtype(ms.float16)
    output1 = net1(input_y)
    assert output1.asnumpy().dtype == np.float16
    assert np.allclose(output1.asnumpy(), expected_output.asnumpy())

    net2 = Net2()
    net2.x.set_dtype(ms.float16)
    output2 = net2(input_y)
    assert output2.asnumpy().dtype == np.float16
    assert np.allclose(output2.asnumpy(), expected_output.asnumpy())

    net3 = Net3()
    net3.x.set_dtype(ms.float16)
    output3 = net3(input_y)
    assert output3.asnumpy().dtype == np.float16
    assert np.allclose(output3.asnumpy(), expected_output.asnumpy())

    net4 = Net4()
    net4.x.set_dtype(ms.float16)
    output4 = net4(input_y)
    assert output4.asnumpy().dtype == np.float16
    assert np.allclose(output4.asnumpy(), expected_output.asnumpy())
