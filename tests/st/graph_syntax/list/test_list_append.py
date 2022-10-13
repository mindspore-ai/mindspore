# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
""" test list append operation """
import pytest
import numpy as np
from mindspore import ms_function, context, Tensor, dtype
from mindspore.nn import Cell
import mindspore.ops.operations as P


context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_append_1():
    """
    Feature: list append.
    Description: support list append.
    Expectation: No exception.
    """
    @ms_function
    def list_append():
        x = [1, 3, 4]
        x.append(2)
        return Tensor(x)

    assert np.all(list_append().asnumpy() == np.array([1, 3, 4, 2]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_append_2():
    """
    Feature: list append.
    Description: support list append.
    Expectation: No exception.
    """
    @ms_function
    def list_append():
        x = [1, 2, 3]
        x.append(4)
        x.append(6)
        return Tensor(x)

    assert np.all(list_append().asnumpy() == np.array([1, 2, 3, 4, 6]))


@pytest.mark.skip(reason='Not support list as parameter in while function yet')
@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_list():
    """
    Feature: list in while.
    Description: Infer list in while.
    Expectation: Null.
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.addn = P.AddN()

        def construct(self, x):
            y = []
            for _ in range(3):
                while x < 10:
                    y.append(x)
                    x = self.addn(y)
            return x

    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path="./listir")
    net = Net()
    x = Tensor([1], dtype.float32)
    print(net(x))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_list():
    """
    Feature: list for.
    Description: Infer list in for.
    Expectation: Null.
    """

    def convert_points_to_homogeneous(points):
        padding = [[0, 0] for _ in range(len(points.shape))]
        padding[-1][-1] = 1
        return padding

    class Net(Cell):
        def construct(self, x1, x2):
            y1 = convert_points_to_homogeneous(x1)
            y2 = convert_points_to_homogeneous(x2)
            return y1, y2

    context.set_context(mode=context.GRAPH_MODE)
    x1 = Tensor([[[-1, -1],  # left top
                  [1, -1],  # right top
                  [-1, 5],  # left bottom
                  [1, 5]]], dtype.float32)  # right bottom
    x2 = Tensor([[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]], dtype.float32)
    net = Net()
    print(net(x1, x2))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dictionary_list():
    """
    Feature: dictionary list.
    Description: Infer list in dictionary.
    Expectation: Null.
    """

    class D3rNet(Cell):
        def __init__(self):
            super().__init__()
            self.a = Tensor(np.random.randn(300, 9).astype(np.float32))
            self.b = Tensor(np.random.randn(300).astype(np.float32))
            self.c = Tensor(np.ones([300]).astype(np.int32))

        def construct(self, a, b, c):
            a_o = a * self.a
            b_o = b * self.b
            c_o = c * self.c
            pts = [[a_o, b_o, c_o]]
            bbox = []
            for i in pts:
                bbox.append({"ptx:": i})
            return bbox

    a = Tensor(np.random.randn(300, 9).astype(np.float32))
    b = Tensor(np.random.randn(300).astype(np.float32))
    c = Tensor(np.ones([300]).astype(np.int32))
    net = D3rNet()
    bbox = net(a, b, c)
    print(bbox)
