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
import mindspore.nn as nn
import mindspore.ops.operations.array_ops as P
from mindspore.common.api import ms_function
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class PackNet(nn.Cell):
    def __init__(self):
        super(PackNet, self).__init__()
        self.stack = P.Stack(axis=2)

    @ms_function
    def construct(self, x1, x2):
        return self.stack((x1, x2))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.int32, np.int16, np.uint8, np.bool])
def test_pack_graph(dtype):
    """
    Feature: pack operation test
    Description: test pack graph float32 operation
    Expectation: pack output == expect
    """
    data_np = np.array([0] * 16).astype(dtype)
    data_np = np.reshape(data_np, (2, 2, 2, 2))
    x1 = Tensor(data_np)
    x2 = Tensor(np.arange(16).reshape(2, 2, 2, 2).astype(dtype))
    net = PackNet()
    output = net(x1, x2)
    expect = np.array([[[[[0, 0],
                          [0, 0]],
                         [[0, 1],
                          [2, 3]]],
                        [[[0, 0],
                          [0, 0]],
                         [[4, 5],
                          [6, 7]]]],
                       [[[[0, 0],
                          [0, 0]],
                         [[8, 9],
                          [10, 11]]],
                        [[[0, 0],
                          [0, 0]],
                         [[12, 13],
                          [14, 15]]]]]).astype(dtype)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pack_graph_float32_dynamic_shape():
    """
    Feature: pack operation dynamic shape test
    Description: test pack graph float32 dynamic shape operation
    Expectation: pack output == expect
    """
    data_np = np.array([0] * 16).astype(np.float32)
    data_np = np.reshape(data_np, (2, 2, 2, 2))
    x1 = Tensor(data_np)
    x2 = Tensor(np.arange(16).reshape(2, 2, 2, 2).astype(np.float32))
    net = PackNet()
    x1_dyn = Tensor(shape=[None for _ in x1.shape], dtype=x1.dtype)
    x2_dyn = Tensor(shape=[None for _ in x2.shape], dtype=x2.dtype)
    net.set_inputs(x1_dyn, x2_dyn)
    output = net(x1, x2)
    expect = np.array([[[[[0, 0],
                          [0, 0]],
                         [[0, 1],
                          [2, 3]]],
                        [[[0, 0],
                          [0, 0]],
                         [[4, 5],
                          [6, 7]]]],
                       [[[[0, 0],
                          [0, 0]],
                         [[8, 9],
                          [10, 11]]],
                        [[[0, 0],
                          [0, 0]],
                         [[12, 13],
                          [14, 15]]]]]).astype(np.float32)
    assert (output.asnumpy() == expect).all()
