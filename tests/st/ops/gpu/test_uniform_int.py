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

import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Net(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(Net, self).__init__()
        self.uniformint = P.UniformInt(seed=seed)
        self.shape = shape

    def construct(self, a, b):
        return self.uniformint(self.shape, a, b)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_1D():
    seed = 10
    shape = (3, 2, 4)
    a = 1
    b = 5
    net = Net(shape, seed=seed)
    ta, tb = Tensor(a, mstype.int32), Tensor(b, mstype.int32)
    output = net(ta, tb)
    assert output.shape == (3, 2, 4)


class DynamicShapeNet(nn.Cell):
    def __init__(self, seed=0, seed2=0):
        super(DynamicShapeNet, self).__init__()
        self.seed = seed
        self.seed2 = seed2
        self.uniformint = P.UniformInt(seed)

    def construct(self, input_shape, a, b):
        return self.uniformint(input_shape, a, b)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_dynamic_shape():
    """
    Feature: op dynamic shape
    Description: set input_shape None and input real tensor
    Expectation: success
    """

    seed = 10
    shape = Tensor((3, 2, 4), mstype.int64)
    shape_dyn = Tensor(shape=[None], dtype=shape.dtype)
    net = DynamicShapeNet(seed)
    net.set_inputs(shape_dyn, Tensor(1, mstype.int32), Tensor(5, mstype.int32))
    output = net(shape, Tensor(1, mstype.int32), Tensor(5, mstype.int32))
    assert output.shape == (3, 2, 4)
