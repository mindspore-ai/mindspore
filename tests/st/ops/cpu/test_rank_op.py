# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class RankNet(nn.Cell):
    def __init__(self):
        super(RankNet, self).__init__()
        self.rank = ops.Rank()

    def construct(self, x):
        return self.rank(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_rank_scalar(mode):
    """
    Feature: ops.rank
    Description: Verify the result of rank for scalar tensor
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array(1).astype(np.float32))
    expect = 0
    net = RankNet()
    output = net(x)
    assert output == expect


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_rank_3d_tensor(mode):
    """
    Feature: ops.rank
    Description: Verify the result of rank for 3D tensor
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float32))
    expect = 3
    net = RankNet()
    output = net(x)
    assert output == expect


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_rank_4d_tensor(mode):
    """
    Feature: ops.rank
    Description: Verify the result of rank for 4D tensor
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32))
    expect = 4
    net = RankNet()
    output = net(x)
    assert output == expect


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_rank_dynamic_shape(mode):
    """
    Feature: ops.rank
    Description: test rank with dynamic shape.
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32))
    expect = 4

    net = RankNet()
    x_dyn = Tensor(shape=[None] * len(x.shape), dtype=x.dtype)
    net.set_inputs(x_dyn)

    output = net(x)
    assert output == expect


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_rank_invalid_input(mode):
    """
    Feature: ops.rank
    Description: Test invalid input cases of rank
    Expectation: raise TypeError
    """
    context.set_context(mode=mode)
    net = RankNet()
    with pytest.raises(TypeError):
        net(1)

    with pytest.raises(TypeError):
        net("str")
