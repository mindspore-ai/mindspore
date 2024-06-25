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
import pytest
import mindspore as ms
from mindspore import Tensor, context, nn, jit
from mindspore.common import dtype as mstype
import numpy as np
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_tensor_slice_1():
    """
    Feature: Test tensor slice.
    Description: Tensor getitem by a single bool value.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, input_x):
            return input_x[True] + 1.0

    x = Tensor(2.0, mstype.float32)
    net = Net()
    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_out = net(x)
    assert graph_out == pynative_out


@pytest.mark.skip(reason="this case depending on other property")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_slice_2():
    """
    Feature: Test tensor slice.
    Description: Tensor getitem by slice.
    Expectation: No exception.
    """
    data = Tensor(np.array([[2, 3.2], [3, 4.1]]).astype(np.float32))

    @jit(input_signature=(Tensor(shape=None, dtype=ms.int64), data))
    def my_index(x):
        out = x[3:2:1]
        return out

    out = my_index(data)
    assert out.shape == (0, 2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_slice_in_tuple():
    """
    Feature: Test tensor slice graph mode.
    Description: Tensor getitem by slice in tuple.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x):
            output = x[1:2:1, ...]
            return output

    data = Tensor(np.arange(27).reshape((3, 3, 3)), ms.float32)
    net = Net()
    result = net(data)
    expected = np.array([[[9, 10, 11], [12, 13, 14], [15, 16, 17]]]).astype(np.float32)
    assert np.array_equal(result.asnumpy(), expected)
