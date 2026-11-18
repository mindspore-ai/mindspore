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
"""Test bessel."""
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
from mindspore import Tensor
import mindspore.ops as F


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dtype, eps', [(np.float16, 1.0e-3), (np.float32, 1.0e-6), (np.float64, 1.0e-6)])
def test_bessel_j0(dtype, eps):
    """
    Feature: bessel j0 function
    Description: test cases for BesselJ0
    Expectation: the result matches scipy
    """
    x = Tensor(np.array([0.5, 1., 2., 4.]).astype(dtype))
    expect = np.array(
        [0.9384698, 0.7651977, 0.22389078, -0.3971498]).astype(dtype)
    error = np.ones(shape=[4]) * eps
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    output = F.bessel_j0(x)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dtype, eps', [(np.float16, 1.0e-3), (np.float32, 1.0e-6), (np.float64, 1.0e-6)])
def test_bessel_j1(dtype, eps):
    """
    Feature: bessel j1 function
    Description: test cases for BesselJ1
    Expectation: the result matches scipy
    """
    x = Tensor(np.array([0.5, 1., 2., 4.]).astype(dtype))
    expect = np.array(
        [0.24226846, 0.44005057, 0.5767248, -0.06604332]).astype(dtype)
    error = np.ones(shape=[4]) * eps
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    output = F.bessel_j1(x)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', mem_peak=1024.01)
@pytest.mark.parametrize('dtype, eps', [(np.float16, 1.0e-3), (np.float32, 1.0e-6), (np.float64, 1.0e-6)])
def test_bessel_i0(dtype, eps):
    """
    Feature: bessel i0 function
    Description: test cases for BesselI0
    Expectation: the result matches scipy
    """
    x = Tensor(np.array([-1, -0.5, 0.5, 1]).astype(dtype))
    expect = np.array(
        [1.2660658, 1.0634834, 1.0634834, 1.2660658]).astype(dtype)
    error = np.ones(shape=[4]) * eps
    context.set_context(mode=context.GRAPH_MODE)

    output = F.bessel_i0(x)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dtype, eps', [(np.float16, 1.0e-3), (np.float32, 1.0e-6), (np.float64, 1.0e-6)])
def test_bessel_i0e(dtype, eps):
    """
    Feature: bessel i0e function
    Description: test cases for BesselI0e
    Expectation: the result matches scipy
    """
    x = Tensor(np.array([-1, -0.5, 0.5, 1]).astype(dtype))
    expect = np.array(
        [0.4657596, 0.64503527, 0.64503527, 0.4657596]).astype(dtype)
    error = np.ones(shape=[4]) * eps
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    output = F.bessel_i0e(x)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)






@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dtype, eps', [(np.float16, 1.0e-3), (np.float32, 1.0e-6), (np.float64, 1.0e-6)])
def test_bessel_y0(dtype, eps):
    """
    Feature: bessel y0 function
    Description: test cases for BesselY0
    Expectation: the result matches scipy
    """
    x = Tensor(np.array([0.5, 1., 2., 4.]).astype(dtype))
    expect = np.array(
        [-0.44451874, 0.08825696, 0.51037567, -0.01694074]).astype(dtype)
    error = np.ones(shape=[4]) * eps
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    output = F.bessel_y0(x)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dtype, eps', [(np.float16, 1.0e-3), (np.float32, 1.0e-6), (np.float64, 1.0e-6)])
def test_bessel_y1(dtype, eps):
    """
    Feature: bessel y1 function
    Description: test cases for BesselY1
    Expectation: the result matches scipy
    """
    x = Tensor(np.array([0.5, 1., 2., 4.]).astype(dtype))
    expect = np.array([-1.47147239, -0.78121282,
                       -0.10703243, 0.39792571]).astype(dtype)
    error = np.ones(shape=[4]) * eps
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    output = F.bessel_y1(x)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dtype, eps', [(np.float16, 1.0e-3), (np.float32, 1.0e-6), (np.float64, 1.0e-6)])
def test_bessel_i1(dtype, eps):
    """
    Feature: bessel i1 function
    Description: test cases for BesselI1
    Expectation: the result matches scipy
    """
    x = Tensor(np.array([-1, -0.5, 0.5, 1]).astype(dtype))
    expect = np.array(
        [-0.5651591, -0.25789431, 0.25789431, 0.5651591]).astype(dtype)
    error = np.ones(shape=[4]) * eps
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    output = F.bessel_i1(x)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dtype, eps', [(np.float16, 1.0e-3), (np.float32, 1.0e-6), (np.float64, 1.0e-6)])
def test_bessel_i1e(dtype, eps):
    """
    Feature: bessel i1e function
    Description: test cases for BesselI1e
    Expectation: the result matches scipy
    """
    x = Tensor(np.array([-1, -0.5, 0.5, 1]).astype(dtype))
    expect = np.array(
        [-0.20791042, -0.15642083, 0.15642083, 0.20791042]).astype(dtype)
    error = np.ones(shape=[4]) * eps
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    output = F.bessel_i1e(x)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)
