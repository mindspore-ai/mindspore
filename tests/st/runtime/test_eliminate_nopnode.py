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

import numpy as np
from tests.mark_utils import arg_mark
import mindspore
from mindspore import context, ops, nn, Tensor


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reshape = ops.Reshape()

    def construct(self, x, y, z):
        a = x + y
        b = self.reshape(a, (3, 2))
        c = self.reshape(z, (3, 2))
        return b + c


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_eliminate_nopnode():
    """
    Feature: eliminate nopnode.
    Description: base scene.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.ones([6, 1]), mindspore.float32)
    y = Tensor(np.ones([6, 1]), mindspore.float32)
    z = Tensor(np.ones([6, 1]), mindspore.float32)
    net = Net()
    out = net(x, y, z)
    assert out.shape == (3, 2)


class NetWithNopNodeOutput(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reshape = ops.Reshape()

    def construct(self, x, y):
        a = x + y
        return self.reshape(a, (3, 2))


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_nopnode_output():
    """
    Feature: eliminate nopnode.
    Description: base scene.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.ones([6, 1]), mindspore.float32)
    y = Tensor(np.ones([6, 1]), mindspore.float32)
    net = NetWithNopNodeOutput()
    out = net(x, y)
    assert out.shape == (3, 2)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_nopnode_dynamic_shape():
    """
    Feature: eliminate nopnode.
    Description: base scene.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x_dyn = Tensor(shape=[6, None], dtype=mindspore.float32)
    y_dyn = Tensor(shape=[6, None], dtype=mindspore.float32)
    z_dyn = Tensor(shape=[6, None], dtype=mindspore.float32)
    net = Net()
    net.set_inputs(x_dyn, y_dyn, z_dyn)
    x = Tensor(np.ones([6, 1]), mindspore.float32)
    y = Tensor(np.ones([6, 1]), mindspore.float32)
    z = Tensor(np.ones([6, 1]), mindspore.float32)
    out = net(x, y, z)
    assert out.shape == (3, 2)


class AscendNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reshape = ops.Reshape()

    def construct(self, x, y, z):
        a = self.reshape(x, (3, 2))
        while z < 3:
            z = z + 1
            b = self.reshape(y, (3, 2))
            a = a + b
        return a


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ascend_nopnode_eliminate():
    """
    Feature: eliminate nopnode.
    Description: dynamic shape scene.
    Expectation: No exception.
    """
    x = Tensor(np.ones([6, 1]), mindspore.float32)
    y = Tensor(np.ones([6, 1]), mindspore.float32)
    z = Tensor([0], mindspore.float32)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = AscendNet()
    output = net(x, y, z)
    expect = np.array([[4., 4.], [4., 4.], [4., 4.]], dtype=np.float32)
    assert output.shape == (3, 2)
    assert np.allclose(output.asnumpy(), expect)
