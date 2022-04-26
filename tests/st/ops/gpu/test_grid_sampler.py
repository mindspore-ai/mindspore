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
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops.operations.nn_ops import GridSampler2D, GridSampler3D


class Net2D(Cell):
    def __init__(self, mode, padding_mode, align_corners):
        super(Net2D, self).__init__()
        self.grid_sampler_2d = GridSampler2D(mode, padding_mode, align_corners)

    def construct(self, x0, x1):
        return self.grid_sampler_2d(x0, x1)


class Net3D(Cell):
    def __init__(self, mode, padding_mode, align_corners):
        super(Net3D, self).__init__()
        self.grid_sampler_3d = GridSampler3D(mode, padding_mode, align_corners)

    def construct(self, x0, x1):
        return self.grid_sampler_3d(x0, x1)


def run_net2d(dtype):
    in_tensor = Tensor(np.arange(36).reshape((2, 3, 3, 2)).astype(dtype))
    grid = Tensor(np.arange(0.2, 1, 0.1).reshape((2, 2, 1, 2)).astype(dtype))
    if dtype == np.float32:
        expect_out = np.array([[[[3.2], [3.7]], [[9.2], [9.7]], [[15.200001], [15.7]]],
                               [[[22.2], [22.699999]], [[28.2], [28.699999]], [[34.2], [34.7]]]])
    elif dtype == np.float64:
        expect_out = np.array([[[[3.2], [3.7]], [[9.2], [9.7]], [[15.2], [15.7]]],
                               [[[22.2], [22.7]], [[28.2], [28.7]], [[34.2], [34.7]]]])
    error_out = np.ones(shape=expect_out.shape) * 1.0e-6
    net = Net2D('bilinear', 'border', True)
    output = net(in_tensor, grid)
    diff_out = output.asnumpy() - expect_out
    assert np.all(np.abs(diff_out) < error_out)


def run_net3d(dtype):
    in_tensor = Tensor(np.arange(32).reshape((2, 2, 2, 2, 2)).astype(dtype))
    grid = Tensor(np.arange(-0.2, 1, 0.1).reshape((2, 2, 1, 1, 3)).astype(dtype))
    if dtype == np.float32:
        expect_out = np.array([[[[[3.3]], [[4.35]]], [[[11.300001]], [[12.349999]]]],
                               [[[[21.4]], [[22.449999]]], [[[29.4]], [[30.449999]]]]])
    elif dtype == np.float64:
        expect_out = np.array([[[[[3.3]], [[4.35]]], [[[11.3]], [[12.35]]]],
                               [[[[21.4]], [[22.45]]], [[[29.4]], [[30.45]]]]])
    error_out = np.ones(shape=expect_out.shape) * 1.0e-6
    net = Net3D('bilinear', 'zeros', True)
    output = net(in_tensor, grid)
    diff_out = output.asnumpy() - expect_out
    assert np.all(np.abs(diff_out) < error_out)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gridsampler2d():
    """
    Feature: GridSampler2D op.
    Description: test data type is float32 and float64 in GPU.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_net2d(np.float32)
    run_net2d(np.float64)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gridsampler3d():
    """
    Feature: GridSampler3D op.
    Description: test data type is float32 and float64 in GPU.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_net3d(np.float32)
    run_net3d(np.float64)
