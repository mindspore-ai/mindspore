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

import mindspore
from mindspore import context, nn, Tensor, ops


class TestNet(nn.Cell):
    def __init__(self):
        super(TestNet, self).__init__()
        self.op = ops.ResizeNearestNeighbor((1, 1))

    def construct(self, x):
        return self.op(x)


class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.op = ops.GradOperation(get_all=True)

    def construct(self, x):
        gradient_function = self.op(self.net)
        return gradient_function(x)


def test_convert_resize_nearest_neighbor_x_dtype_with_static_shape_input():
    """
    Feature: GE Optimization
    Description: test convert data type to int32 for ResizeNearestNeighborX with static shape input
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = Tensor(np.array([[[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]]), mindspore.float32)
    expect = np.array([[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=np.float32)
    net = TestNet()
    grad_net = GradNet(net)
    assert np.allclose(grad_net(x)[0].asnumpy(), expect, 1e-03, 1e-03)


def test_convert_resize_nearest_neighbor_x_dtype_with_dynamic_shape_input():
    """
    Feature: GE Optimization
    Description: test convert data type to int32 for ResizeNearestNeighborX with dynamic shape input
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})

    x = Tensor(np.array([[[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]]), mindspore.float32)
    expect = np.array([[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=np.float32)
    net = TestNet()
    grad_net = GradNet(net)
    dyn_x = Tensor(shape=(1, 1, None, None), dtype=mindspore.float32)
    grad_net.set_inputs(dyn_x)
    assert np.allclose(grad_net(x)[0].asnumpy(), expect, 1e-03, 1e-03)


if __name__ == "__main__":
    test_convert_resize_nearest_neighbor_x_dtype_with_static_shape_input()
    # GE don't support dynamic shape
    test_convert_resize_nearest_neighbor_x_dtype_with_dynamic_shape_input()
