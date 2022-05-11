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
""" test list control flow """
import pytest
import mindspore.context as context
from mindspore import Tensor, dtype
from mindspore.nn import Cell
import mindspore.ops.operations as P


@pytest.mark.skip(reason='Not support list  as parameter in while function yet')
@pytest.mark.level0
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


@pytest.mark.level0
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
