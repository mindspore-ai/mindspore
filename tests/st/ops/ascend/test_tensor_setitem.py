# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.common import dtype as mstype


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_tensor_slice_by_bool_broadcast():
    """
    Feature: Tensor-setitem-by-bool support broadcast.
    Description: Tensor-setitem-by-bool support broadcast.
    Expectation: success.
    """
    data_np = np.ones([2, 3, 4], np.float32)
    index_np = np.array([True, False])
    value = 2

    data_tensor = Tensor(data_np)
    index_tensor = Tensor(index_np)

    data_np[index_np] = value
    data_tensor[index_tensor] = value
    assert np.allclose(data_tensor.asnumpy(), data_np)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.0)
def test_tensor_slice_by_bool_nan():
    """
    Feature: Tensor-setitem-by-bool support nan.
    Description: Tensor-setitem-by-bool support nan.
    Expectation: success.
    """
    data = Tensor(np.ones([2, 3, 4], np.float32))
    index = Tensor(np.array([False, False]))
    data[index] = Tensor([np.nan])
    assert np.allclose(data.asnumpy(), np.ones([2, 3, 4], np.float32))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_tensor_setitem_empty_tuple():
    """
    Feature: Tensor setitem which index is empty tuple.
    Description: Tensor setitem.
    Expectation: success.
    """

    class Net16(Cell):
        def __init__(self):
            super().__init__()
            self.idx = ()

        def construct(self, x):
            x[self.idx] = 2
            out = x
            return out

    ms.set_context(mode=ms.GRAPH_MODE)
    net = Net16()
    x = Tensor(np.random.rand(3, 3, 2), dtype=mstype.float32)
    graph_out = net(x)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    y = Tensor(np.random.rand(3, 3, 2), dtype=mstype.float32)
    pynative_out = net(y)
    assert (graph_out == pynative_out).all()
