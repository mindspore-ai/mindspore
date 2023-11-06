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

""" Test concat. """

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops
import mindspore.ops.functional as F
import test_utils


def concat_func(x1, x2):
    return F.concat((x1, x2), axis=0)


def concat_bwd_func(x1, x2):
    return ops.grad(concat_func, (0, 1))(x1, x2)


@test_utils.run_with_cell
def concat_forward_func(x1, x2):
    return concat_func(x1, x2)


@test_utils.run_with_cell
def concat_backward_func(x1, x2):
    return concat_bwd_func(x1, x2)


def concat_dyn_seq_fwd_func(seq):
    return F.concat(seq, axis=0)


def concat_dyn_seq_bwd_func(seq):
    return ops.grad(concat_dyn_seq_fwd_func, (0,))(seq)


def forward_datas_prepare(shape, num=2, axis=0, diff_shapes=False, need_expect=True):
    np_inpus = []
    tensor_inputs = []
    for i in range(num):
        np_input = np.random.rand(*(shape[i] if diff_shapes else shape)).astype(np.float32)
        np_inpus.append(np_input)
        tensor_inputs.append(ms.Tensor(np_input))
    np_expect = np.concatenate(np_inpus, axis) if need_expect else None
    return tuple(tensor_inputs), np_expect


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_concat_forward(mode):
    """
    Feature: Ops.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    tensor_inputs, expect = forward_datas_prepare((2, 4))
    out = concat_forward_func(tensor_inputs[0], tensor_inputs[1])
    assert np.allclose(out.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_concat_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    x1 = ms.Tensor(np.array([[0, 1, 2, 1], [1, 1, 3, 5]]).astype(np.float32))
    x2 = ms.Tensor(np.array([[4, 6, 2, 2], [0, 6, 2, 6]]).astype(np.float32))
    grads = concat_backward_func(x1, x2)
    expect_grad1 = np.ones((2, 4)).astype(np.float32)
    expect_grad2 = np.ones((2, 4)).astype(np.float32)
    expect_grad = (expect_grad1, expect_grad2)
    for out, expect in zip(grads, expect_grad):
        assert np.allclose(out.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
#@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_concat_vmap(mode):
    """
    Feature: test vmap function.
    Description: test concat op vmap.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    in_axes = (-1, -1)
    x1_np = np.array([[0, 1, 2, 1], [1, 1, 3, 5]]).astype(np.float32)
    x1 = ms.Tensor(np.tile(x1_np.reshape((2, 4, 1, 1)), (1, 1, 2, 2)))
    x2_np = np.array([[4, 6, 2, 2], [0, 6, 2, 6]]).astype(np.float32)
    x2 = ms.Tensor(np.tile(x2_np.reshape((2, 4, 1, 1)), (1, 1, 2, 2)))
    nest_vmap = ops.vmap(ops.vmap(concat_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x1, x2)
    expect_np = np.array([[0, 1, 2, 1], [1, 1, 3, 5], [4, 6, 2, 2], [0, 6, 2, 6]]).astype(np.float32)
    expect = np.tile(expect_np.reshape((1, 1, 4, 4)), (2, 2, 1, 1))
    assert np.allclose(out.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_concat_forward_dynamic(mode, dyn_mode):
    """
    Feature: test dynamic.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    dyn_tensor_shape = [None, None] if dyn_mode == "dyn_shape" else None
    x1_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    fwd_cell = test_utils.to_cell_obj(concat_func)
    fwd_cell.set_inputs(x1_dyn, x2_dyn)

    shape1 = (2, 4)
    tensor_inputs1, expect1 = forward_datas_prepare(shape1)
    out1 = fwd_cell(*tensor_inputs1)
    assert np.allclose(out1.asnumpy(), expect1)

    shape2 = (3, 3) if dyn_mode == "dyn_shape" else (2, 2, 2)
    tensor_inputs2, expect2 = forward_datas_prepare(shape2)
    out2 = fwd_cell(*tensor_inputs2)
    assert np.allclose(out2.asnumpy(), expect2)

    shapes3 = ((3, 3), (2, 3)) if dyn_mode == "dyn_shape" else ((2, 2, 2), (3, 2, 2))
    tensor_inputs3, expect3 = forward_datas_prepare(shapes3, diff_shapes=True)
    out3 = fwd_cell(*tensor_inputs3)
    assert np.allclose(out3.asnumpy(), expect3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_concat_backward_dynamic(mode, dyn_mode):
    """
    Feature: test dynamic.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    dyn_tensor_shape = [None, None] if dyn_mode == "dyn_shape" else None
    x1_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    bwd_cell = test_utils.to_cell_obj(concat_bwd_func)
    bwd_cell.set_inputs(x1_dyn, x2_dyn)

    shape1 = (2, 4)
    (x1_1, x1_2), _ = forward_datas_prepare(shape1, need_expect=False)
    grads1 = bwd_cell(x1_1, x1_2)
    expect_grad1 = (np.ones(shape1).astype(np.float32),) * 2
    for out, expect in zip(grads1, expect_grad1):
        assert np.allclose(out.asnumpy(), expect)

    shape2 = (3, 3) if dyn_mode == "dyn_shape" else (2, 2, 2)
    (x2_1, x2_2), _ = forward_datas_prepare(shape2, need_expect=False)
    grads2 = bwd_cell(x2_1, x2_2)
    expect_grad2 = (np.ones(shape2).astype(np.float32),) * 2
    for out, expect in zip(grads2, expect_grad2):
        assert np.allclose(out.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_concat_forward_dyn_seq(mode, dyn_mode):
    """
    Feature: test forward dynamic sequence.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    dyn_tensor_shape = [None, None] if dyn_mode == "dyn_shape" else None
    x1_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    fwd_seq_cell = test_utils.to_cell_obj(concat_dyn_seq_fwd_func)
    fwd_seq_cell.set_inputs(ms.mutable((x1_dyn, x2_dyn), True))

    shape1 = (2, 4)
    num1 = 2
    tensor_inputs1, expect1 = forward_datas_prepare(shape1, num=num1)
    out1 = fwd_seq_cell(tensor_inputs1)
    assert np.allclose(out1.asnumpy(), expect1)

    # Dynamic sequence only support same shape inner now.
    shape2 = (3, 3) if dyn_mode == "dyn_shape" else (2, 2, 2)
    num2 = 2  # Should be different, set 2 here for mutable bug.
    tensor_inputs2, expect2 = forward_datas_prepare(shape2, num=num2)
    out2 = fwd_seq_cell(tensor_inputs2)
    assert np.allclose(out2.asnumpy(), expect2)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dyn_mode", ["dyn_shape", "dyn_rank"])
def test_concat_backward_dyn_seq(mode, dyn_mode):
    """
    Feature: test backward dynamic sequence.
    Description: test op concat.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    dyn_tensor_shape = [None, None] if dyn_mode == "dyn_shape" else None
    x1_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=dyn_tensor_shape, dtype=ms.float32)
    bwd_seq_cell = test_utils.to_cell_obj(concat_dyn_seq_bwd_func)
    bwd_seq_cell.set_inputs(ms.mutable((x1_dyn, x2_dyn), True))

    shape1 = (2, 4)
    num1 = 2
    input_seq1, _ = forward_datas_prepare(shape1, num=num1, need_expect=False)
    grads1 = bwd_seq_cell(input_seq1)
    expect_grad1 = (np.ones(shape1).astype(np.float32),) * num1
    for out, expect in zip(grads1, expect_grad1):
        assert np.allclose(out.asnumpy(), expect)

    # Dynamic sequence only support same shape inner now.
    shape2 = (3, 3) if dyn_mode == "dyn_shape" else (2, 2, 2)
    num2 = 2  # Should be different, set 2 here for mutable bug.
    input_seq2, _ = forward_datas_prepare(shape2, num=num2, need_expect=False)
    grad2 = bwd_seq_cell(input_seq2)
    expect_grad2 = (np.ones(shape2).astype(np.float32),) * num2
    for out, expect in zip(grad2, expect_grad2):
        assert np.allclose(out.asnumpy(), expect)
