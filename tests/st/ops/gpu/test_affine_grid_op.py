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

import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn

from mindspore import Tensor
from mindspore.ops.operations import array_ops
from mindspore.ops.operations import _inner_ops as inner

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="GPU")


class AffineGridNet(nn.Cell):
    def __init__(self, align_corners=False):
        super(AffineGridNet, self).__init__()
        self.affine_grid = array_ops.AffineGrid(align_corners=align_corners)

    def construct(self, theta, size):
        return self.affine_grid(theta, size)


class AffineGridDynamicShapeNet(nn.Cell):
    def __init__(self, align_corners=False):
        super(AffineGridDynamicShapeNet, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()
        self.affine_grid = array_ops.AffineGrid(align_corners=align_corners)

    def construct(self, theta, size):
        theta = self.test_dynamic(theta)
        grid = self.affine_grid(theta, size)
        return grid


def generate_nchw():
    n = np.random.randint(1, 128)
    c = np.random.randint(1, 128)
    h = np.random.randint(100, 1000)
    w = np.random.randint(100, 1000)
    return np.array([n, c, h, w])


def generate_ncdhw():
    n = np.random.randint(1, 128)
    c = np.random.randint(1, 128)
    d = np.random.randint(50, 300)
    h = np.random.randint(100, 1000)
    w = np.random.randint(100, 1000)
    return np.array([n, c, d, h, w])


def np_linspace_from_neg_one(theta, n_steps, align_corners):
    if n_steps <= 1:
        return np.array(0, dtype=theta.dtype)
    x = np.linspace(-1, 1, n_steps, dtype=theta.dtype)
    if not align_corners:
        x = x * (n_steps - 1) / n_steps
    return x


def np_affine_grid_4d(theta, size, align_corners=False):
    n, h, w = size[0], size[2], size[3]
    x = np_linspace_from_neg_one(theta, w, align_corners)
    y = np_linspace_from_neg_one(theta, h, align_corners)
    yv, xv = np.meshgrid(y, x, indexing='ij')  # (h, w)
    base_grid = np.stack((xv, yv, np.ones_like(xv)), axis=-1)  # (h, w, 3)
    base_grid = base_grid.reshape((1, h*w, 3))  # (1, hw, 3)
    theta = np.expand_dims(theta.transpose(0, 2, 1), 1)  # (n, 1, 3, 2)
    grid = np.matmul(base_grid, theta)  # (n, hw, 2)
    return grid.reshape((n, h, w, 2))


def np_affine_grid_5d(theta, size, align_corners=False):
    n, d, h, w = size[0], size[2], size[3], size[4]
    x = np_linspace_from_neg_one(theta, w, align_corners)
    y = np_linspace_from_neg_one(theta, h, align_corners)
    z = np_linspace_from_neg_one(theta, d, align_corners)
    zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')  # (d, h, w)
    base_grid = np.stack((xv, yv, zv, np.ones_like(xv)), axis=-1)  # (d, h, w, 4)
    base_grid = base_grid.reshape((1, d*h*w, 4))  # (1, dhw, 4)
    theta = np.expand_dims(theta.transpose(0, 2, 1), 1)  # (n, 1, 4, 3)
    grid = np.matmul(base_grid, theta)  # (n, dhw, 3)
    return grid.reshape((n, d, h, w, 3))


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("net", [AffineGridNet, AffineGridDynamicShapeNet])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_affine_grid_4d(net, align, dtype):
    """
    Feature: gpu backend of operator AffineGrid
    Description: generate 128 X 1080 X 1920 X 2 affine grid(case 4D).
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    # Big case, require enormous memory.
    np_nchw = (32, 1, 540, 960)
    np_theta = np.array([[[1, 0, 0], [0, 1, 0]]]).astype(dtype)
    np_theta = np.repeat(np_theta, np_nchw[0], axis=0)

    np_grid = np_affine_grid_4d(np_theta, np_nchw, align_corners=align)

    affine_grid = net(align_corners=align)
    ms_theta, ms_nchw = Tensor(np_theta), np_nchw
    ms_grid = affine_grid(ms_theta, ms_nchw)

    print(f"max error: {np.max(np_grid - ms_grid.asnumpy())}")
    if dtype == np.float16:
        atol = 1e-2
    elif dtype == np.float32:
        atol = 1e-6
    else:
        raise TypeError("Only support float16 or float32!")
    assert np.allclose(np_grid, ms_grid.asnumpy(), atol=atol)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("net", [AffineGridNet, AffineGridDynamicShapeNet])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_affine_grid_5d(net, align, dtype):
    """
    Feature: gpu backend of operator AffineGrid
    Description: generate 128 X 16 X 270 X 480 X 3 affine grid(case 5D).
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    # Big case, require enormous memory.
    np_ncdhw = (32, 1, 16, 135, 240)
    np_theta = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]).astype(dtype)
    np_theta = np.repeat(np_theta, np_ncdhw[0], axis=0)

    np_grid = np_affine_grid_5d(np_theta, np_ncdhw, align_corners=align)

    affine_grid = net(align_corners=align)
    ms_theta, ms_ncdhw = Tensor(np_theta), np_ncdhw
    ms_grid = affine_grid(ms_theta, ms_ncdhw)

    print(f"max error: {np.max(np_grid - ms_grid.asnumpy())}")
    if dtype == np.float16:
        atol = 1e-2
    elif dtype == np.float32:
        atol = 1e-6
    else:
        raise TypeError("Only support float16 or float32!")
    assert np.allclose(np_grid, ms_grid.asnumpy(), atol=atol)


if __name__ == '__main__':
    for p_n in [AffineGridNet, AffineGridDynamicShapeNet]:
        for p_a in [False, True]:
            for p_d in [np.float32, np.float16]:
                print(f"\n[CASE] net: {p_n}, align: {p_a}, dtype: {p_d}")
                print("  [4D]", end=" ")
                test_affine_grid_4d(p_n, p_a, p_d)
                print("  [5D]", end=" ")
                test_affine_grid_5d(p_n, p_a, p_d)
