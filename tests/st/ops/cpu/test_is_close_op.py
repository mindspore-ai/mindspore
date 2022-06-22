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
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap


class Net(nn.Cell):
    def __init__(self, rtol, atol, equal_nan):
        super(Net, self).__init__()
        self.ops = P.IsClose(rtol=rtol, atol=atol, equal_nan=equal_nan)

    def construct(self, a, b):
        return self.ops(a, b)


def rand_int(*shape):
    """return an random integer array with parameter shape"""
    res = np.random.randint(low=1, high=5, size=shape)
    if isinstance(res, np.ndarray):
        return res.astype(np.float32)
    return float(res)


def compare_with_numpy(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    ms_result_graph = Net(rtol, atol, equal_nan)(Tensor(a), Tensor(b))
    # PyNative Mode
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    ms_result_pynative = Net(rtol, atol, equal_nan)(Tensor(a), Tensor(b))

    np_result = np.isclose(a, b, rtol, atol, equal_nan)
    return np.array_equal(ms_result_graph, np_result) and np.array_equal(ms_result_pynative, np_result)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('equal_nan', [True, False])
def test_net(equal_nan):
    """
    Feature: ALL TO ALL
    Description:  test cases for IsClose operator.
    Expectation: the result match numpy isclose.
    """
    a = [0, 1, 2, float('inf'), float('inf'), float('nan')]
    b = [0, 1, -2, float('-inf'), float('inf'), float('nan')]
    assert compare_with_numpy(a, b, equal_nan=equal_nan)

    a = rand_int(2, 3, 4, 5)
    diff = (np.random.random((2, 3, 4, 5)).astype("float32") - 0.5) / 1000
    b = a + diff
    assert compare_with_numpy(a, b, atol=1e-3)
    assert compare_with_numpy(a, b, atol=1e-3, rtol=1e-4)
    assert compare_with_numpy(a, b, atol=1e-2, rtol=1e-6)

    a = rand_int(2, 3, 4, 5)
    b = rand_int(4, 5)
    assert compare_with_numpy(a, b, equal_nan=equal_nan)

    a = np.array(1.0).astype("float32")
    b = np.array(1.0 + 1e-8).astype("float32")
    assert compare_with_numpy(a, b, equal_nan=equal_nan)


class VmapNet(nn.Cell):
    def __init__(self, rtol, atol, equal_nan):
        super(VmapNet, self).__init__()
        self.net = Net(rtol=rtol, atol=atol, equal_nan=equal_nan)
        self.ops = vmap(self.net, in_axes=(0, 0), out_axes=0)

    def construct(self, a, b):
        return self.ops(a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_net():
    """
    Feature: Support vmap for IsClose operator.
    Description:  test cases of vmap for IsClose operator.
    Expectation: the result match numpy isclose.
    """

    a = rand_int(2, 3, 4, 5)
    b = rand_int(2, 4, 5)
    rtol = 1e-05
    atol = 1e-08
    equal_nan = False
    np_result = np.isclose(a, b.reshape((2, 1, 4, 5)), rtol, atol, equal_nan)
    ms_result = VmapNet(rtol=rtol, atol=atol, equal_nan=equal_nan)(Tensor(a), Tensor(b))
    return np.array_equal(ms_result, np_result)
