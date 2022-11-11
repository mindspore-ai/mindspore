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
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor, Parameter
from mindspore.experimental import MapParameter
from mindspore.common.initializer import initializer


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
            return self.p, self.m

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    if not 'SAULT_ENV_TYPE' in os.environ or not "CUDA10" in os.environ['SAULT_ENV_TYPE']:
        net = MyNet()
        out = net()
        print("out:", out)
        data = net.m.export_data()
        print("data:", data)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
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
            self.m.put(self.keys, self.values)
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
