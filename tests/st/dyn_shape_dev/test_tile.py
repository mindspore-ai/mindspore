# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Test Tile."""

import numpy as np
import pytest
import test_utils

from mindspore import ops
from mindspore import Tensor
import mindspore as ms


def tile_func(x, multiplies):
    return ops.operations.manually_defined.tile(x, multiplies)


@test_utils.run_with_cell
def tile_forward_func(x, multiplies):
    return tile_func(x, multiplies)


@test_utils.run_with_cell
def tile_backward_func(x, multiplies):
    return ops.grad(tile_forward_func, (0,))(x, multiplies)


@test_utils.run_with_cell
def tile_infer_value():
    x = Tensor(np.array([[2, 2], [3, 3]]))
    return tile_func(x, (1, 2))


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_tile_infer_value():
    """
    Feature: Ops.
    Description: test op tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.context.GRAPH_MODE)
    out = tile_infer_value()
    np_x = np.array([[2, 2], [3, 3]])
    mul = (1, 2)
    expect = np.tile(np_x, mul)
    assert np.allclose(out.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_tile_forward(mode):
    """
    Feature: Ops.
    Description: test op tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    x = Tensor(np_x)
    mul = (1, 1, 2, 2)
    out = tile_forward_func(x, mul)
    expect = np.tile(np_x, mul)
    assert np.allclose(out.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_tile_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    mul = (1, 1, 2, 2)
    grads = tile_backward_func(x, mul)
    expect = np.ones((2, 3, 4, 5)).astype(np.float32) * 4.0
    assert np.allclose(grads.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_tile_vmap(mode):
    """
    Feature: test vmap function.
    Description: test tile op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (-1, None)
    np_x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    mul = (1, 1, 2, 2)
    x = Tensor(np.tile(np_x.reshape((2, 3, 4, 5, 1, 1)), (1, 1, 1, 1, 2, 2)))
    nest_vmap = ops.vmap(ops.vmap(tile_forward_func, in_axes=in_axes), in_axes=in_axes)
    out = nest_vmap(x, mul)
    expect = np.tile(np.tile(np_x, mul).reshape(1, 1, 2, 3, 8, 10), (2, 2, 1, 1, 1, 1))
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_tile_forward_dyn(mode):
    """
    Feature: Ops.
    Description: test op tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    mul = (1, 1, 2, 2)
    dyn_x = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    fwd_cell = test_utils.to_cell_obj(tile_func)
    fwd_cell.set_inputs(dyn_x, ms.mutable(mul))

    np_x1 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    x1 = Tensor(np_x1)
    out1 = fwd_cell(x1, mul)
    expect1 = np.tile(np_x1, mul)
    assert np.allclose(out1.asnumpy(), expect1)

    np_x2 = np.random.rand(3, 4, 5, 6).astype(np.float32)
    x2 = Tensor(np_x2)
    out2 = fwd_cell(x2, mul)
    expect2 = np.tile(np_x2, mul)
    assert np.allclose(out2.asnumpy(), expect2)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE])  # ms.context.GRAPH_MODE has runtime heterogeneous bug.
def test_tile_backward_dyn(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    mul = (1, 1, 2, 2)
    dyn_x = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    bwd_cell = test_utils.to_cell_obj(tile_backward_func)
    bwd_cell.set_inputs(dyn_x, ms.mutable(mul))

    x1 = Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    grads1 = bwd_cell(x1, mul)
    expect1 = np.ones((2, 3, 4, 5)).astype(np.float32) * 4.0
    assert np.allclose(grads1.asnumpy(), expect1)

    x2 = Tensor(np.random.rand(3, 4, 5, 6).astype(np.float32))
    grads2 = bwd_cell(x2, mul)
    expect2 = np.ones((3, 4, 5, 6)).astype(np.float32) * 4.0
    assert np.allclose(grads2.asnumpy(), expect2)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@pytest.mark.parametrize('is_fwd', [True, False])
def test_tile_dynamic_len(mode, is_fwd):
    """
    Feature: test dynamic len.
    Description: test op tile.
    Expectation: expect correct result.
    """
    if mode == ms.context.GRAPH_MODE and not is_fwd:
        # ms.context.GRAPH_MODE and backward has runtime heterogeneous bug.
        return
    ms.context.set_context(mode=mode)
    np_x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    x = Tensor(np_x)
    mul = (1, 1, 2, 2)

    if is_fwd:
        # Forward.
        forward_cell = test_utils.to_cell_obj(tile_func)
        forward_cell.set_inputs(x, ms.mutable(mul, True))
        out = forward_cell(x, mul)
        fwd_expect = np.tile(np_x, mul)
        assert np.allclose(out.asnumpy(), fwd_expect)
    else:
        # Backward.
        backward_cell = test_utils.to_cell_obj(tile_backward_func)
        backward_cell.set_inputs(x, ms.mutable(mul, True))
        grads = backward_cell(x, mul)
        bwd_expect = np.ones((2, 3, 4, 5)).astype(np.float32) * 4.0
        assert np.allclose(grads.asnumpy(), bwd_expect)
