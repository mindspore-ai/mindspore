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
import os
import os.path
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor, Parameter, save_checkpoint, load_checkpoint, ParameterTuple
from mindspore.experimental import MapParameter
from mindspore.common.initializer import initializer
from mindspore.ops import composite as C
from mindspore import export, load
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_simple_graph_compile_export():
    """
    Feature: MapParameter
    Description: Test IR graph compiled with MapParameter, and export api.
    Expectation: IR graph with MapParameter created without exceptions.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            nn.Cell.__init__(self)
            self.p = Parameter(initializer('ones', (2, 3), ms.float32))
            self.m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(3,))
            self.keys = Tensor([1, 2], dtype=ms.int32)
            self.values = Tensor([[11, 11, 11], [22, 22, 22]], dtype=ms.float32)

        def construct(self):
            self.m.put(self.keys, self.values)
            return self.p, self.m

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        net = MyNet()
        out = net()
        print("out:", out)
        data = net.m.export_data()
        print("data:", data)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('ms_type', [ms.int32, ms.int64])
def test_maptensor_put_get_export(ms_type):
    """
    Feature: MapParameter
    Description: Test IR graph compiled with MapParameter, test put, get and export api.
    Expectation: IR graph with MapParameter created without exceptions.
    """

    class MyNet(nn.Cell):
        def __init__(self, ms_type):
            nn.Cell.__init__(self)
            self.m = MapParameter(key_dtype=ms_type, value_dtype=ms.float32, value_shape=(3,))
            self.keys = Tensor([1, 2], dtype=ms_type)
            self.values = Tensor([[11, 11, 11], [22, 22, 22]], dtype=ms.float32)

        def construct(self, ms_type):
            self.m[self.keys] = self.values
            key1 = Tensor([3], dtype=ms_type)
            value1 = self.m.get(key1, True)
            key2 = Tensor([4], dtype=ms_type)
            value2 = self.m.get(key2, True)
            return value1, value2, self.m

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        net = MyNet(ms_type)
        out1, out2, out3 = net(ms_type)
        print("out1:", out1)
        print("out2:", out2)
        print("out3:", out3)
        data = net.m.export_data()
        print("data:", data)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('ms_type', [ms.int32, ms.int64])
def test_mapparameter_ckpt_save_load(ms_type):
    """
    Feature: MapParameter
    Description: Test MapParameter, test save and load
    Expectation: IR graph with MapParameter created without exceptions.
    """
    class MyNet(nn.Cell):
        def __init__(self, ms_type):
            nn.Cell.__init__(self)
            self.m = MapParameter(key_dtype=ms_type, value_dtype=ms.float32, value_shape=(3,))
            self.keys = Tensor([1, 2], dtype=ms_type)
            self.values = Tensor([[11, 11, 11], [22, 22, 22]], dtype=ms.float32)

        def construct(self, ms_type):
            self.m[self.keys] = self.values
            key1 = Tensor([3], dtype=ms_type)
            value1 = self.m.get(key1, True)
            key2 = Tensor([4], dtype=ms_type)
            value2 = self.m.get(key2, True)
            return value1, value2, self.m

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        net = MyNet(ms_type)
        net(ms_type)
        file_name = "map_parameter.ckpt"
        save_checkpoint(net, file_name)
        assert os.path.exists(file_name)
        load_checkpoint(file_name)
        os.remove(file_name)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_map_parameter_get():
    """
    Feature: MapParameter
    Description: Test get api for MapParameter.
    Expectation: get api works as expected.
    """
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        keys = Tensor([1, 2], dtype=ms.int32)
        values = Tensor([[1, 2], [1, 2]], dtype=ms.float32)
        map_tensor = MapParameter(key_tensor=keys, value_tensor=values, default_value='zeros')
        key = Tensor([3], dtype=ms.int32)
        get_value = map_tensor.get(key)
        print("get_value:", get_value)
        data1 = map_tensor.export_data(incremental=False)
        print("data1:", data1)

        map_tensor.put(Tensor([3], dtype=ms.int32), Tensor([[3, 3]], dtype=ms.float32))
        data2 = map_tensor.export_data(incremental=False)
        print("data2:", data2)
        map_tensor[Tensor([1, 2, 3], dtype=ms.int32)] = Tensor([[11, 11], [22, 22], [33, 33]], dtype=ms.float32)
        data3 = map_tensor.export_data(incremental=False)
        print("data3:", data3)
        map_tensor.erase(Tensor([1, 2, 3], dtype=ms.int32))
        data4 = map_tensor.export_data(incremental=False)
        print("data4:", data4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_map_parameter_put():
    """
    Feature: MapParameter
    Description: Test put api for MapParameter.
    Expectation: put api works as expected.
    """
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        keys = Tensor([1, 2], dtype=ms.int32)
        values = Tensor([[1, 2], [1, 2]], dtype=ms.float32)
        map_tensor = MapParameter(key_tensor=keys, value_tensor=values, default_value='zeros')
        key = Tensor([3], dtype=ms.int32)
        value = Tensor([[4, 5]], dtype=ms.float32)
        map_tensor.put(key, value)
        data1 = map_tensor.export_data(incremental=False)
        print("data1:", data1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_map_parameter_erase():
    """
    Feature: MapParameter
    Description: Test erase api for MapParameter.
    Expectation: erase api works as expected.
    """
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        keys = Tensor([1, 2], dtype=ms.int32)
        values = Tensor([[1, 2], [1, 2]], dtype=ms.float32)
        map_tensor = MapParameter(key_tensor=keys, value_tensor=values, default_value='zeros')
        key = Tensor([2], dtype=ms.int32)
        map_tensor.put(keys, values)
        map_tensor.erase(key)
        data1 = map_tensor.export_data(incremental=False)
        print("data1:", data1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_basic_operations():
    """
    Feature: MapParameter
    Description: Test MapParameter basic operations.
    Expectation: MapParameter works as expected.
    """
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        m = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(2), default_value='zeros',
                         name='my_map')
        assert m.name == 'my_map'
        assert m.requires_grad

        t = m.get(Tensor([1, 2, 3], dtype=ms.int32))
        assert t.dtype == ms.float32
        assert t.shape == (3, 2)
        assert np.allclose(t.asnumpy(), 0)

        t = m[Tensor([1, 2, 3], dtype=ms.int32)]
        assert t.dtype == ms.float32
        assert t.shape == (3, 2)
        assert np.allclose(t.asnumpy(), 0)

        m.put(Tensor([1, 2, 3], dtype=ms.int32), Tensor([[1, 1], [2, 2], [3, 3]], dtype=ms.float32))
        data = m.export_data()
        print(m)
        print("data:", data)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
            return self.p + value1 + value2, self.m

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        net = MyNet()
        t = initializer('ones', (2, 3), ms.float32)
        t = t.init_data()
        out = net(t)
        print(out)
        assert out[0].shape == (2, 3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_export_update_api():
    """
    Feature: MapParameter
    Description: Test export update api for MapParameter.
    Expectation: Export update api works as expected.
    """
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        m1 = MapParameter(key_dtype=ms.int32, value_dtype=ms.float32, value_shape=(3,))
        data1 = m1.export_data(incremental=False)
        print("data1:", data1)
        m1.import_data(data1)

        keys = Tensor([1, 2], dtype=ms.int32)
        values = Tensor([[1, 2], [1, 2]], dtype=ms.float32)
        m2 = MapParameter(key_tensor=keys, value_tensor=values, default_value='zeros')
        data2 = m2.export_data(incremental=False)
        print("data2:", data2)
        m1.import_data(data2)
        new_data1 = m1.export_data(incremental=False)
        print("new_data1:", new_data1)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        net = MyNet()
        grad = GradNet(net)
        t = initializer('ones', (2, 3), ms.float32)
        t = t.init_data()
        grad(t)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_map_parameter_in_construct():
    """
    Feature: MapParameter
    Description: Test new MapParameter in construct.
    Expectation: New MapTensor in construct without exceptions.
    """
    class MapTensorNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.default_value = 'zeros'
            self.key_tensor = Tensor([1, 2], dtype=ms.int32)
            self.value_tensor = Tensor([[1, 2], [1, 2]], dtype=ms.float32)
            self.new_key_tensor = Tensor([3, 4], dtype=ms.int32)
            self.new_value_tensor = Tensor([[3, 3], [4, 4]], dtype=ms.float32)
            self.map_tensor = MapParameter(key_tensor=self.key_tensor, value_tensor=self.value_tensor)

        def construct(self):
            new_map_tensor = MapParameter(self.key_tensor, self.value_tensor, self.default_value)
            new_map_tensor.put(self.new_key_tensor, self.new_value_tensor)
            return new_map_tensor, self.map_tensor

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        net = MapTensorNet()
        out = net()
        print("out:", out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_map_parameter_get_data_api():
    """
    Feature: MapParameter
    Description: Test get_data api for MapParameter.
    Expectation: get_data api works as expected.
    """
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        keys = Tensor([1, 2], dtype=ms.int32)
        values = Tensor([[1, 2], [1, 2]], dtype=ms.float32)
        map_tensor = MapParameter(key_tensor=keys, value_tensor=values, default_value='zeros')
        [the_keys, the_values] = map_tensor.get_data()
        print("the_keys:", the_keys)
        print("the_values:", the_values)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
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
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        net = MyNet()
        out = net()
        print("out:", out)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
            self.m.put(self.key, value2)
            return self.p + value1 + value2

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        net = MyNet()
        t = initializer('ones', (2, 3), ms.float32)
        t = t.init_data()
        file_path = "./map-parameter.mindir"
        export(net, t, file_name=file_path, file_format="MINDIR")
        assert os.path.isfile(file_path)
        load(file_path)

        file_path = "./map-parameter-incremental.mindir"
        export(net, t, file_name=file_path, file_format="MINDIR", incremental=True)
        assert os.path.isfile(file_path)
        load(file_path)
