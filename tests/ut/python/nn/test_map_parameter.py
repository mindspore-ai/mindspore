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
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter, context
from mindspore.experimental import MapParameter
from mindspore.common.initializer import initializer


def test_basic_operations():
    """
    Feature: MapParameter
    Description: Test MapParameter basic operations.
    Expectation: MapParameter works as expected.
    """
    m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(2), default_value='zeros', name='my_map')
    assert m.name == 'my_map'
    assert m.requires_grad

    t = m.get(Tensor([1, 2, 3], dtype=ms.int32))
    assert t.dtype == ms.float32
    assert t.shape == (3, 2)
    assert np.allclose(t.asnumpy(), 0)

    t = m.get(Tensor([1, 2, 3], dtype=ms.int32), 'ones')
    assert t.dtype == ms.float32
    assert t.shape == (3, 2)
    assert np.allclose(t.asnumpy(), 1)

    m.put(Tensor([1, 2, 3], dtype=ms.int32), Tensor([[1, 1], [2, 2], [3, 3]], dtype=ms.float32))
    m.erase(Tensor([1, 2, 3], dtype=ms.int32))


def test_simple_graph_compile():
    """
    Feature: MapParameter
    Description: Test IR graph compiled with MapParameter.
    Expectation: IR graph with MapParameter created without exceptions.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            nn.Cell.__init__(self)
            self.p = Parameter(initializer('ones', (2, 3), ms.float32))
            self.m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(3,))
            self.key = Tensor([1, 2], dtype=ms.int32)
            self.default_value = Tensor([3.0, 3.0, 3.0], dtype=ms.float32)

        def construct(self, x):
            self.m.put(self.key, x)
            value = self.m.get(self.key, self.default_value)
            self.m.erase(self.key)
            return self.p + value

    context.set_context(mode=context.GRAPH_MODE)
    net = MyNet()
    t = initializer('ones', (2, 3), ms.float32)
    t = t.init_data()
    out = net(t)
    print(out)
    assert out.shape == (2, 3)
