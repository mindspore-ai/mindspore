# Copyright 2024 Huawei Technologies Co., Ltd
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
from functools import reduce
import numpy as np
import pytest

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.ops.auto_generate import dropout_ext_op
from mindspore.nn import DropoutExt
from mindspore.ops.function import dropout_ext
from mindspore.nn import Cell
from tests.st.utils import test_utils


def generate_random_input(shape, dtype):
    return np.ones(shape).astype(dtype)


@test_utils.run_with_cell
def dropout_forward_func(x, p=0.4):
    return dropout_ext(x, p)


@test_utils.run_with_cell
def dropout_backward_func(x, p=0.4):
    return ops.grad(dropout_forward_func, (0))(x, p)


def compare_output(x, p, output):
    # check output
    keep_prob = 1 - p
    if output.dtype == mstype.bfloat16:
        output_np = output.astype(mstype.float32).asnumpy()
    else:
        output_np = output.asnumpy()
    elem_count = x.size
    nonzero_count = np.count_nonzero(output_np)
    assert (elem_count * (keep_prob - 0.02)) < nonzero_count < (elem_count * (keep_prob + 0.02))

    expect_sum = np.array(nonzero_count / (1 - p), dtype=np.float64)
    output_sum = np.sum(output_np.astype(np.float64))

    if output.dtype == mstype.float32:
        np.testing.assert_allclose(output_sum, expect_sum, rtol=1e-3)
    else:
        np.testing.assert_allclose(output_sum, expect_sum, rtol=1e-2)


def compare_grad(x, p, grad):
    # check grad
    keep_prob = 1 - p
    if grad.dtype == mstype.bfloat16:
        grad_np = grad.astype(mstype.float32).asnumpy()
    else:
        grad_np = grad.asnumpy()
    elem_count = x.size
    nonzero_count = np.count_nonzero(grad_np)
    assert (elem_count * (keep_prob - 0.02)) < nonzero_count < (elem_count * (keep_prob + 0.02))


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_func_dropout_normal(context_mode, dtype):
    """
    Feature: pyboost function.
    Description: test function dropout normal.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    ms.set_context(jit_level='O0')
    x = generate_random_input((1280, 77, 77), dtype)
    p = 0.1
    output = dropout_forward_func(ms.Tensor(x), p)
    compare_output(x, p, output)

    x1 = generate_random_input((3, 4096, 1280), dtype)
    p1 = 0.1
    grad = dropout_backward_func(ms.Tensor(x1), p1)
    compare_grad(x1, p1, grad)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_func_dropout_bfloat16(context_mode):
    """
    Feature: pyboost function.
    Description: test function dropout normal.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    ms.set_context(jit_level='O0')
    x = generate_random_input((128, 128), np.float32)
    p = 0.4
    output = dropout_forward_func(ms.Tensor(x).astype(mstype.bfloat16), p)
    compare_output(x, p, output)

    x1 = generate_random_input((256, 256), np.float32)
    p1 = 0.3
    grad = dropout_backward_func(ms.Tensor(x1).astype(mstype.bfloat16), p1)
    compare_grad(x1, p1, grad)


def compare_func(x, p, output, mask=None):
    device_target = ms.context.get_context("device_target")
    keep_prob = 1 - p
    if device_target != "Ascend":
        # check output
        output_np = output.asnumpy()
        elem_count = x.size
        nonzero_count = np.count_nonzero(output_np)
        assert (elem_count * (keep_prob - 0.1)) < nonzero_count < (elem_count * (keep_prob + 0.1))
        output_sum = np.sum(output_np)
        x_sum = np.sum(x)
        assert abs(output_sum - x_sum) / x_sum < 0.1
        # check mask
        if mask is not None:
            mask_np = mask.asnumpy()
            mask_sum = np.sum(mask_np)
            assert np.count_nonzero(mask_np) == nonzero_count
            assert abs(mask_sum - nonzero_count) / nonzero_count < 0.1
    else:
        # check output
        output_np = output.asnumpy()
        elem_count = x.size
        nonzero_count = np.count_nonzero(output_np)
        assert (elem_count * (keep_prob - 0.1)) < nonzero_count < (elem_count * (keep_prob + 0.1))
        output_sum = np.sum(output_np)
        x_sum = np.sum(x)
        assert abs(output_sum - x_sum) / x_sum < 0.1
        # check mask
        if mask is not None:
            assert len(mask.shape) == 1
            assert np.ceil(reduce(lambda a, b: a * b, x.shape) / 128) * 16 == mask.shape[0]


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nn_DropoutExt_normal(context_mode):
    """
    Feature: nn.DropoutExt
    Description: forward
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    x = np.array(np.random.random((16, 16, 16, 16)), np.float32)
    p = 0.4

    net = DropoutExt(p)
    net.set_train()

    output = net(ms.tensor(x))
    compare_func(x, p, output)



@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nn_DropoutExt_bf16(context_mode):
    """
    Feature: nn.DropoutExt
    Description: bf16
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    x = np.array(np.random.random((128, 128)), np.float32)
    p = 0.4

    net = DropoutExt(p)
    net.set_train()

    output = net(ms.tensor(x, mstype.bfloat16))
    compare_func(x, p, output.float())



class DropoutExtCell(Cell):
    def __init__(self):
        super().__init__()
        self.dropout_ext = dropout_ext_op

    def construct(self, x, p):
        return self.dropout_ext(x, p)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_DropoutExt_normal(context_mode):
    """
    Feature: ops.DropoutExt
    Description: forward
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    dropout_cell = DropoutExtCell()

    x = np.array(np.random.random((128, 128)), np.float32)
    p = 0.4

    output, mask = dropout_cell(ms.tensor(x), p)
    compare_func(x, p, output, mask)

    dropout_cell.set_inputs(ms.tensor(shape=[None, None], dtype=ms.float32), ms.mutable(p))

    x = np.array(np.random.random((256, 128)), np.float32)

    output, mask = dropout_cell(ms.tensor(x), ms.mutable(p))
    compare_func(x, p, output, mask)

    x = np.array(np.random.random((128, 256)), np.float32)

    output, mask = dropout_cell(ms.tensor(x), ms.mutable(p))
    compare_func(x, p, output, mask)

    dropout_cell.set_inputs(ms.tensor(shape=None, dtype=ms.float32), ms.mutable(p))

    x = np.array(np.random.random((128, 128, 128)), np.float32)

    output, mask = dropout_cell(ms.tensor(x), ms.mutable(p))
    compare_func(x, p, output, mask)

    x = np.array(np.random.random((16, 16, 16, 16)), np.float32)

    output, mask = dropout_cell(ms.tensor(x), ms.mutable(p))
    compare_func(x, p, output, mask)
