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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, perm_in):
        super().__init__()
        super(Net, self).__init__()
        self.perm = perm_in
        self.permute = ops.permute

    def construct(self, input_):
        x = self.permute(input_, self.perm)
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_permute_dynamic_perm1(mode):
    """
    Feature: test transpose dynamic
    Description: test transpose dynamic with graph and pynative mode
    Expectation: none.
    """
    context.set_context(mode=mode)
    perm = (0, 2, 3, 1)
    in_shape = (8, 24, 1, 1)
    np_value = np.random.uniform(0, 100, size=in_shape).astype(np.float16)
    permute = Net(perm)
    real_input = Tensor(np_value)
    dyn_input = Tensor(shape=[None for _ in real_input.shape], dtype=real_input.dtype)
    permute.set_inputs(dyn_input)
    out = permute(real_input)
    return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_permute_dynamic_perm2(mode):
    """
    Feature: test transpose dynamic
    Description: test transpose dynamic with graph and pynative mode
    Expectation: none.
    """
    context.set_context(mode=mode)
    perm = (3, 2, 1, 0)
    in_shape = (8, 24, 1, 1)
    np_value = np.random.uniform(0, 100, size=in_shape).astype(np.float16)
    permute = Net(perm)
    real_input = Tensor(np_value)
    dyn_input = Tensor(shape=[None for _ in real_input.shape], dtype=real_input.dtype)
    permute.set_inputs(dyn_input)
    out = permute(real_input)
    return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_permute_perm(mode):
    """
    Feature: test transpose dynamic
    Description: test transpose dynamic with graph and pynative mode
    Expectation: none.
    """
    context.set_context(mode=mode)
    perm = (3, 2, 1, 0)
    in_shape = (8, 24, 1, 1)
    np_value = np.random.uniform(0, 100, size=in_shape).astype(np.float16)
    permute = Net(perm)
    real_input = Tensor(np_value)
    out = permute(real_input)
    return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_permute_perm2(mode):
    """
    Feature: test transpose dynamic
    Description: test transpose dynamic with graph and pynative mode
    Expectation: none.
    """
    context.set_context(mode=mode)
    perm = (2, 1, 3, 0)
    in_shape = (8, 24, 1, 1)
    np_value = np.random.uniform(0, 100, size=in_shape).astype(np.float16)
    permute = Net(perm)
    real_input = Tensor(np_value)
    out = permute(real_input)
    return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_permute_perm3(mode):
    """
    Feature: test transpose dynamic
    Description: test transpose dynamic with graph and pynative mode
    Expectation: none.
    """
    context.set_context(mode=mode)
    perm = (-1, 2, 1, 0)
    in_shape = (8, 24, 1, 1)
    np_value = np.random.uniform(0, 100, size=in_shape).astype(np.float16)
    permute = Net(perm)
    real_input = Tensor(np_value)
    out = permute(real_input)
    return out


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_permute_perm4(mode):
    """
    Feature: test transpose dynamic
    Description: test transpose dynamic with graph and pynative mode
    Expectation: none.
    """
    context.set_context(mode=mode)
    perm = (-2, 3, 0, 1)
    in_shape = (8, 24, 1, 1)
    np_value = np.random.uniform(0, 100, size=in_shape).astype(np.float16)
    permute = Net(perm)
    real_input = Tensor(np_value)
    out = permute(real_input)
    return out
