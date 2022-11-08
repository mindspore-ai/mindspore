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
import os.path

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor, Parameter, ParameterTuple
from mindspore.experimental import MapParameter
from mindspore.common.initializer import initializer
from mindspore.ops import composite as C
from mindspore import export, load


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

    t = m.get(Tensor([1, 2, 3], dtype=ms.int32))
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

    data = m.get_data()
    assert data == (None, None)

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
            value1 = self.m.get(self.key)
            value2 = self.m[self.key]
            self.m[self.key] = value2
            self.m.erase(self.key)
            keys = self.m.get_keys()
            values = self.m.get_values()
            keys_values = self.m.get_data()
            print(keys_values)
            self.m.put(keys, values)
            return self.p + value1 + value2

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
    data = m.export_data(full=True)
    m.import_data(data)


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
            self.m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(3,))
            self.key = Tensor([1, 2], dtype=ms.int32)

        def construct(self, x):
            a = self.m.get(self.key)
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


def test_map_parameter_in_init_and_construct():
    """
    Feature: MapParameter
    Description: Test new MapParameter in construct.
    Expectation: New MapTensor in construct without exceptions.
    """
    class MapTensorNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.default_value = 'zeros'
            self.map_param_1 = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(3,))
            self.key_tensor = Tensor([1, 2], dtype=ms.int32)
            self.value_tensor = Tensor([[1, 2], [1, 2]], dtype=ms.float32)
            self.map_param_2 = MapParameter(key_tensor=self.key_tensor, value_tensor=self.value_tensor,
                                            default_value=self.default_value)

        def construct(self):
            keys = self.map_param_2.get_keys()
            values = self.map_param_2.get_values()
            new_map_tensor = MapParameter(keys, values, self.default_value)
            new_data = new_map_tensor.get_data()
            return self.map_param_1, new_map_tensor, new_data

    context.set_context(mode=context.GRAPH_MODE)
    net = MapTensorNet()
    out = net()
    print("out:", out)


def test_map_parameter_get_data_api():
    """
    Feature: MapParameter
    Description: Test get_data api for MapParameter.
    Expectation: get_data api works as expected.
    """
    keys = Tensor([1, 2], dtype=ms.int32)
    values = Tensor([[1, 2], [1, 2]], dtype=ms.float32)
    map_tensor = MapParameter(key_tensor=keys, value_tensor=values, default_value='zeros')
    get_keys = map_tensor.get_keys()
    print("get_keys:", get_keys)
    get_values = map_tensor.get_values()
    print("get_values:", get_values)
    [the_keys, the_values] = map_tensor.get_data()
    print("the_keys:", the_keys)
    print("the_values:", the_values)


def test_map_parameter_filter():
    """
    Feature: MapParameter
    Description: Test IR graph compiled with MapParameter, test with filter.
    Expectation: IR graph with MapParameter created without exceptions.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            nn.Cell.__init__(self)
            self.m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(3,), permit_filter_value=2,
                                  evict_filter_value=3)

        def construct(self):
            return self.m

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = MyNet()
    out = net()
    print("out:", out)


def test_simple_graph_export_load():
    """
    Feature: MapParameter
    Description: Test IR graph export and load with MapParameter.
    Expectation: IR graph with MapParameter exported and loaded without exceptions.
    """

    class MyNet(nn.Cell):
        def __init__(self):
            nn.Cell.__init__(self)
            self.p = Parameter(initializer('ones', (2, 3), ms.float32))
            self.m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(3,))
            self.key = Tensor([1, 2], dtype=ms.int32)

        def construct(self, x):
            self.m.put(self.key, x)
            value1 = self.m.get(self.key)
            value2 = self.m[self.key]
            self.m[self.key] = value2
            self.m.erase(self.key)
            keys = self.m.get_keys()
            values = self.m.get_values()
            self.m.put(keys, values)
            return self.p + value1 + value2

    context.set_context(mode=context.GRAPH_MODE)
    net = MyNet()
    t = initializer('ones', (2, 3), ms.float32)
    t = t.init_data()
    file_path = "./map-parameter.mindir"
    export(net, t, file_name=file_path, file_format="MINDIR")
    assert os.path.isfile(file_path)
    load(file_path)
