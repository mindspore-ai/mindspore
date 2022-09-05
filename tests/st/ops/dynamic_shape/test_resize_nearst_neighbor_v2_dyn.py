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

import mindspore as ms
from mindspore import ops as P
from mindspore import context, Tensor, nn
from mindspore.ops.operations import array_ops as A

import numpy as np


class ResizeNearestNeighborV2Net(nn.Cell):
    def __init__(self):
        super(ResizeNearestNeighborV2Net, self).__init__()
        self.op = A.ResizeNearestNeighborV2()

    def construct(self, image, size):
        res = self.op(image, size)
        return res


class ResizeNearestNeighborV2DynamicRankNet(nn.Cell):
    def __init__(self):
        super(ResizeNearestNeighborV2DynamicRankNet, self).__init__()
        self.op = A.ResizeNearestNeighborV2()
        self.reduce_sum = P.ReduceSum(keep_dims=False)

    def construct(self, image, size, dyn_reduce_axis):
        image = self.reduce_sum(image, dyn_reduce_axis)
        res = self.op(image, size)
        return res


def case_resize_nearest_neighbor_v2_dyn(mode, device_target):
    context.set_context(mode=mode, device_target=device_target)
    net = ResizeNearestNeighborV2Net()
    x = np.random.randn(16, 32, 3, 3).astype(np.float32)
    y = np.array([9, 9]).astype(np.int32)
    static_out = net(Tensor(x), Tensor(y)).asnumpy()

    dyn_x = Tensor(shape=[None, 32, None, None], dtype=ms.float32)
    dyn_y = Tensor(shape=[None], dtype=ms.int32)
    dyn_net = ResizeNearestNeighborV2Net()
    dyn_net.set_inputs(dyn_x, dyn_y)
    dyn_out = dyn_net(Tensor(x), Tensor(y)).asnumpy()
    assert np.allclose(dyn_out, static_out, 1e-3, 1e-3)

    dyn_rank_net = ResizeNearestNeighborV2DynamicRankNet()
    reduce_axis = np.array([4], dtype=np.int64)
    dyn_x = Tensor(shape=[None, 32, None, None, 1], dtype=ms.float32)
    dyn_y = Tensor(shape=[None], dtype=ms.int32)
    dyn_reduce_axis = Tensor(shape=[None], dtype=ms.int64)
    dyn_rank_net.set_inputs(dyn_x, dyn_y, dyn_reduce_axis)
    dyn_rank_out = dyn_rank_net(Tensor(np.expand_dims(x, -1)),
                                Tensor(y), Tensor(reduce_axis)).asnumpy()
    assert np.allclose(dyn_rank_out, static_out, 1e-3, 1e-3)


def test_resize_nearest_neighbor_v2_dyn_gpu():
    """
    Feature: Test ResizeNearestNeighborV2 op on GPU.
    Description: The input shape is dynamic
    Expectation: Assert the result is equal the static result.
    """
    case_resize_nearest_neighbor_v2_dyn(context.GRAPH_MODE, "GPU")


def test_resize_nearest_neighbor_v2_dyn_cpu():
    """
    Feature: Test ResizeNearestNeighborV2 op on CPU.
    Description: The input shape is dynamic
    Expectation: Assert the result is equal the static result.
    """
    case_resize_nearest_neighbor_v2_dyn(context.GRAPH_MODE, "CPU")
