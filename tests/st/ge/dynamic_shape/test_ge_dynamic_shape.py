# Copyright 2024 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, context
import mindspore.ops.operations as P


class DynNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add().add_prim_attr("primitive_target", "CPU")
        self.sub = P.Sub().add_prim_attr("graph_split_group", "KernelGroup")
        self.mul = P.Mul()

    def construct(self, x, y, z):
        x_y = self.mul(x, y)
        xy = self.add(x_y, x_y)
        xyz = self.add(xy, z)
        xy_sub = self.sub(xy, z)
        xyz_z = self.mul(xyz, xy_sub)
        return self.add(xyz_z, xy)


def dyn_basic():
    context.set_context(jit_level='O2')
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = Tensor(np.ones([2, 3, 4, 3]).astype(np.float32))
    y = Tensor(np.ones([2, 3, 4, 3]).astype(np.float32))
    z = Tensor(np.ones([2, 3, 4, 3]).astype(np.float32))
    x_dyn = Tensor(shape=[2, 3, 4, None], dtype=mstype.float32)
    y_dyn = Tensor(shape=[2, 3, 4, None], dtype=mstype.float32)
    z_dyn = Tensor(shape=[2, 3, 4, None], dtype=mstype.float32)
    net = DynNet()
    net.set_inputs(x_dyn, y_dyn, z_dyn)
    output = net(x, y, z)
    expect = np.ones([2, 3, 4, 3]).astype(np.float32) * 5
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ge_dyn_shape():
    """
    Feature: convert ge graph
    Description: test ge dynamic by jit_level='O2'
    Expectation: success
    """
    dyn_basic()
