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
import pytest

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops.functional import vmap


def vmap_case():
    class Net(nn.Cell):
        def __init__(self, block_size, paddings):
            super(Net, self).__init__()
            self.space_to_batch_nd = ops.SpaceToBatchND(block_size, paddings)

        def construct(self, a):
            return self.space_to_batch_nd(a)

    class WrapNet(nn.Cell):
        def __init__(self, net, in_axes, out_axes):
            super(WrapNet, self).__init__()
            self.net = net
            self.in_axes = in_axes
            self.out_axes = out_axes

        def construct(self, input_x):
            return vmap(self.net, self.in_axes, self.out_axes)(input_x)

    block_size = [2, 2]
    paddings = [[0, 0], [0, 0]]
    input_shape = (2, 3, 1, 4, 4)
    data_np = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    net = Net(block_size, paddings)

    # test input axis and output axis are the same
    v_net_1 = WrapNet(Net(block_size, paddings), (0,), 0)
    output_v = v_net_1(Tensor(data_np)).asnumpy()

    for i in range(input_shape[0]):
        assert np.allclose(output_v[i, :, :, :, :], net(Tensor(data_np[i, :, :, :, :])).asnumpy())

    # test input axis and output axis are different
    v_net_2 = WrapNet(Net(block_size, paddings), (0,), 1)
    output_v = v_net_2(Tensor(data_np)).asnumpy()

    for i in range(input_shape[0]):
        assert np.allclose(output_v[:, i, :, :, :], net(Tensor(data_np[i, :, :, :, :])).asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_space_to_batch_nd_vmap_cpu():
    """
    Feature: test SpactToBatchND vmap on CPU.
    Description: inputs with batch.
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    vmap_case()
