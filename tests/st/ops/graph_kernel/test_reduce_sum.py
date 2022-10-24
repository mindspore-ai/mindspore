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
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.nn import Cell


class ReduceNet(Cell):
    def __init__(self):
        super(ReduceNet, self).__init__()
        self.add = ops.Add()
        self.sum = ops.ReduceSum()

    def construct(self, x, y, axis):
        z = self.add(x, y)
        return self.sum(z, axis)


def run_case():
    np.random.seed(1)
    x = np.random.normal(0, 1, [1024, 1024]).astype(np.float32)
    y = np.random.normal(0, 1, [1024, 1024]).astype(np.float32)
    expect = np.sum(x + y, axis=(0,))

    net = ReduceNet()
    output = net(Tensor(x), Tensor(y), 0)
    assert np.allclose(expect, output.asnumpy(), 1.e-4, 1.e-4, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_reduce():
    """
    Feature: test reduce sum with enable_graph_kernel=True
    Description: reduce sum with InplaceAssign
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        enable_graph_kernel=True,
                        graph_kernel_flags="--enable_cluster_ops=ReduceSum")
    run_case()
