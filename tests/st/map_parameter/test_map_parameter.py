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
    net = MyNet()
    out = net()
    print("out:", out)
    data = net.m.export()
    print("data:", data)
