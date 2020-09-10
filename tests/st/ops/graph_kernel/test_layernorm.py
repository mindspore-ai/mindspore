# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P

context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.layernorm = P.LayerNorm(1, 1)

    def construct(self, x, y, z):
        return self.layernorm(x, y, z)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_basic():
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    gamma = np.random.normal(0, 1, [3, 4, 3]).astype(np.float32)
    beta = np.random.normal(0, 1, [3, 4, 3]).astype(np.float32)
    shape_x = [2, 3, 4, 3]
    begin_norm_axis = 1

    in_rank = len(shape_x)
    if begin_norm_axis < 0:
        norm_axis = begin_norm_axis + in_rank
    else:
        norm_axis = begin_norm_axis
    norm_axes = tuple(range(norm_axis, in_rank))
    mean = np.mean(input_x, axis=norm_axes, keepdims=True)
    mean_b = np.broadcast_to(mean, shape_x)
    diff = input_x - mean_b
    square = np.square(diff)
    smean = np.mean(square, axis=norm_axes, keepdims=True)
    smean_b = np.broadcast_to(smean, shape_x)
    meps = smean_b + 1e-5
    logs = np.log(meps)
    mul = logs * (-0.5)
    rsqrt = np.exp(mul)
    out = diff * rsqrt
    bn = out * gamma + beta
    expect = (bn, mean, smean)

    net = Net()

    net_result = net(Tensor(input_x), Tensor(gamma), Tensor(beta))
    if isinstance(net_result, tuple) and len(net_result) == 3:
        result = (net_result[0].asnumpy(), net_result[1].asnumpy(), net_result[2].asnumpy())
        res0 = np.allclose(expect[0], result[0], rtol=1.e-4, atol=1.e-4, equal_nan=True)
        assert res0
        res1 = np.allclose(expect[1], result[1], rtol=1.e-4, atol=1.e-7, equal_nan=True)
        assert res1
        res2 = np.allclose(expect[2], result[2], rtol=1.e-4, atol=1.e-7, equal_nan=True)
        assert res2
    else:
        assert False
