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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import ops
from mindspore.ops.functional import vmap
from mindspore import dtype as mstype


class NetCosh(nn.Cell):
    def construct(self, x):
        return ops.cosh(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype, tol', [(np.float16, 1.e-3), (np.float32, 1.e-5), (np.float64, 1.e-8),
                                        (np.complex64, 1.e-5), (np.complex128, 1.e-8)])
def test_cosh_graph(dtype, tol):
    """
    Feature: ALL To ALL
    Description: test cases for Cosh
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype(dtype)
    input_x = Tensor(np_array)
    net = NetCosh()
    output = net(input_x)
    expect = np.cosh(np_array)
    assert np.allclose(output.asnumpy(), expect, atol=tol, rtol=tol)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype, tol', [(np.float16, 1.e-3), (np.float32, 1.e-5), (np.float64, 1.e-8),
                                        (np.complex64, 1.e-5), (np.complex128, 1.e-8)])
def test_cosh_py(dtype, tol):
    """
    Feature: ALL To ALL
    Description: test cases for Cosh
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype(dtype)
    input_x = Tensor(np_array)
    output = input_x.cosh()
    expect = np.cosh(np_array)
    assert np.allclose(output.asnumpy(), expect, atol=tol, rtol=tol)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_cosh_dynamic_shape():
    """
    Feature: test test_cosh_dynamic_shape dynamic_shape feature.
    Description: test padding test_cosh_dynamic_shape feature.
    Expectation: Success.
    """
    dtype, tol = np.float32, 1.e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype(dtype)
    input_x = Tensor(np_array)

    dynamic_net = NetCosh()
    place_holder = Tensor(shape=[None], dtype=mstype.float32)
    dynamic_net.set_inputs(place_holder)

    output = dynamic_net(input_x)
    expect = np.cosh(np_array)
    assert np.allclose(output.asnumpy(), expect, atol=tol, rtol=tol)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype, tol', [(np.float16, 1.e-3), (np.float32, 1.e-5), (np.float64, 1.e-8)])
def test_vmap_cosh(dtype, tol):
    """
    Feature: test vmap inplace operators
    Description: test vmap inplace operators
    Expectation: result is the same as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    np_array = np.random.random((3, 4, 2, 1)).astype(dtype)
    input_x = Tensor(np_array)
    net = NetCosh()
    output = vmap(net, 0)(input_x)
    expect = np.cosh(np_array)
    assert np.allclose(output.asnumpy(), expect, atol=tol, rtol=tol)
