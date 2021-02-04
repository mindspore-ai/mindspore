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
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter


class PackNet(nn.Cell):
    def __init__(self, nptype):
        super(PackNet, self).__init__()
        self.stack = P.Stack(axis=2)
        self.data_np = np.array([0] * 16).astype(nptype)
        self.data_np = np.reshape(self.data_np, (2, 2, 2, 2))
        self.x1 = Parameter(initializer(
            Tensor(self.data_np), [2, 2, 2, 2]), name='x1')
        self.x2 = Parameter(initializer(
            Tensor(np.arange(16).reshape(2, 2, 2, 2).astype(nptype)), [2, 2, 2, 2]), name='x2')

    @ms_function
    def construct(self):
        return self.stack((self.x1, self.x2))


def pack(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    pack_ = PackNet(nptype)
    output = pack_()
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
                          [14, 15]]]]]).astype(nptype)
    assert (output.asnumpy() == expect).all()

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pack_graph_float32():
    pack(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pack_graph_float16():
    pack(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pack_graph_int32():
    pack(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pack_graph_int16():
    pack(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pack_graph_uint8():
    pack(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pack_graph_bool():
    pack(np.bool)
