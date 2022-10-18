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
from mindspore import context, Tensor, Parameter, ParameterTuple
from mindspore.experimental import MapParameter
from mindspore.common.initializer import initializer
from mindspore.ops import composite as C


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

    t = m.get(Tensor([1, 2, 3], dtype=ms.int32), 0)
    assert t.dtype == ms.float32
    assert t.shape == (3, 2)
    assert np.allclose(t.asnumpy(), 0)

    t = m[Tensor([1, 2, 3], dtype=ms.int32)]
    assert t.dtype == ms.float32
    assert t.shape == (3, 2)
    assert np.allclose(t.asnumpy(), 0)

    m.put(Tensor([1, 2, 3], dtype=ms.int32), Tensor([[1, 1], [2, 2], [3, 3]], dtype=ms.float32))
    m[Tensor([1, 2, 3], dtype=ms.int32)] = Tensor([[11, 11], [22, 22], [33, 33]], dtype=ms.float32)
    m.erase(Tensor([1, 2, 3], dtype=ms.int32))

    print(m)


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

        def construct(self, x):
            self.m.put(self.key, x)
            value1 = self.m.get(self.key, 0.1)
            value2 = self.m.get(self.key, 'zeros')
            value3 = self.m.get(self.key)
            value4 = self.m[self.key]
            self.m[self.key] = value4
            self.m.erase(self.key)
            keys = self.m.get_keys()
            values = self.m.get_values()
            self.m.put(keys, values)
            return self.p + value1 + value2 + value3 + value4

    context.set_context(mode=context.GRAPH_MODE)
    net = MyNet()
    t = initializer('ones', (2, 3), ms.float32)
    t = t.init_data()
    out = net(t)
    print(out)
    assert out.shape == (2, 3)


def test_export_update_api():
    """
    Feature: MapParameter
    Description: Test export update api for MapParameter.
    Expectation: Export update api works as expected.
    """
    m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(3,))
    data = m.export(full=True)
    m.update(data)


def test_map_parameter_clone():
    """
    Feature: MapParameter
    Description: Test MapParameter clone() method.
    Expectation: MapParameter cloned as expected.
    """
    m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(3,), name="map")
    p = Parameter(Tensor(1), name="param")
    params = ParameterTuple([m, p])
    cloned_params = params.clone(prefix="cloned", init='zeros')

    cloned_map = cloned_params[0]
    assert isinstance(cloned_map, MapParameter)
    assert cloned_map.name == 'cloned.map'
    assert cloned_map.key_dtype == m.key_dtype
    assert cloned_map.value_dtype == m.value_dtype
    assert cloned_map.value_shape == m.value_shape
    assert cloned_map.default_value == 'zeros'

    old_map_tensor = m._map_tensor  # pylint: disable=W0212
    new_map_tensor = cloned_map._map_tensor  # pylint: disable=W0212
    assert new_map_tensor != old_map_tensor
    assert new_map_tensor.key_dtype == old_map_tensor.key_dtype
    assert new_map_tensor.value_dtype == old_map_tensor.value_dtype
    assert new_map_tensor.value_shape == old_map_tensor.value_shape

    clone_same = cloned_map.clone(init='same')
    assert clone_same.key_dtype == m.key_dtype
    assert clone_same.value_dtype == m.value_dtype
    assert clone_same.value_shape == m.value_shape
    assert clone_same.default_value == 'zeros'


def test_grad_net():
    """
    Feature: MapParameter
    Description: Test grad graph compiled with MapParameter.
    Expectation: Grad graph for MapParameter created without exceptions.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            nn.Cell.__init__(self)
            self.p = Parameter(initializer('ones', (2, 3), ms.float32))
            self.m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(3,))
            self.key = Tensor([1, 2], dtype=ms.int32)

        def construct(self, x):
            a = self.m.get(self.key, 0.1)
            self.m.erase(self.key)
            return x * a

    class GradNet(nn.Cell):
        def __init__(self, network):
            super(GradNet, self).__init__()
            self.grad_by_list = C.GradOperation(get_by_list=True)
            self.network = network
            self.weights = ParameterTuple(network.trainable_params())

        def construct(self, *inputs):
            gout = self.grad_by_list(self.network, self.weights)(*inputs)
            return gout

    context.set_context(mode=context.GRAPH_MODE)
    net = MyNet()
    grad = GradNet(net)
    t = initializer('ones', (2, 3), ms.float32)
    t = t.init_data()
    grad(t)
