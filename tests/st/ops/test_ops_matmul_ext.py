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

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import matmul
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.pynative.utils import allclose_nparray

rtol = 1e-3


class MatMulCell(Cell):
    def __init__(self):
        super().__init__()
        self.matmul = matmul

    def construct(self, x, y):
        return self.matmul(x, y)


class MatMulGradCell(Cell):
    def __init__(self):
        super().__init__()
        self.matmul_grad = ops.grad(matmul, (0, 1))

    def construct(self, x, y):
        return self.matmul_grad(x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize("shape1, shape2, output_shape1, output_shape2", [
    [[9], [9], (), (9,)],
    [[5, 6], [6], (5,), (5, 6)],
    [[2, 2], [2, 2], (2, 2), (2, 2)],
    [[3, 1, 2, 5, 6], [3, 4, 2, 6, 5], (3, 4, 2, 5, 5), (3, 1, 2, 5, 6)],
    [[4, 2, 5, 6], [3, 4, 2, 6, 5], (3, 4, 2, 5, 5), (4, 2, 5, 6)],
    [[4, 1, 5, 6], [4, 2, 6, 5], (4, 2, 5, 5), (4, 1, 5, 6)],
    [[4, 2, 5, 6], [4, 2, 6, 5], (4, 2, 5, 5), (4, 2, 5, 6)],
])
def test_ops(context_mode, shape1, shape2, output_shape1, output_shape2):
    """
    Feature: ops.matmul
    Description: static
    Expectation: expect correct shape result.
    """
    ms.set_context(mode=context_mode)

    matmul_cell = MatMulCell()
    # 2 x 2
    x = random_input(shape1)
    y = random_input(shape2)

    output = matmul_cell(ms.tensor(x), ms.tensor(y))
    assert output.shape == output_shape1

    output1, _ = ops.grad(matmul_cell, (0, 1))(ms.tensor(x), ms.tensor(y))
    assert output1.shape == output_shape2


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize("shape1, shape2", [
    [[231, 768], [768, 2304]],
    [[2048, 2048], [2048, 768]],
])
def test_ops_f16(context_mode, shape1, shape2):
    """
    Feature: ops.matmul
    Description: static f16
    Expectation: expect correct shape result.
    """
    ms.set_context(mode=context_mode)

    matmul_cell = MatMulCell()
    x = random_input(shape1, np.float16)
    y = random_input(shape2, np.float16)

    output = matmul_cell(ms.tensor(x), ms.tensor(y))

    allclose_nparray(output.asnumpy(), np.matmul(x, y), 1e-3, 1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@pytest.mark.parametrize("shape1, shape2", [
    [[12288, 320], [320, 2560]],
    [[3, 1280], [1280, 320]],
])
def test_ops_bf16(context_mode, shape1, shape2):
    """
    Feature: ops.matmul
    Description: static bf16
    Expectation: expect correct shape result.
    """
    ms.set_context(mode=context_mode)

    matmul_cell = MatMulCell()
    x = ms.ops.randn(shape1, dtype=ms.bfloat16).float().asnumpy()
    y = ms.ops.randn(shape2, dtype=ms.bfloat16).float().asnumpy()

    output = matmul_cell(ms.tensor(x, ms.bfloat16), ms.tensor(y, ms.bfloat16))
    allclose_nparray(output.float().asnumpy(), np.matmul(x, y), 4e-3, 4e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
def test_ops_dynamic():
    """
    Feature: ops.matmul
    Description: dynamic shape and rank
    Expectation: success
    """
    x1 = ms.Tensor(random_input([9]))
    y1 = ms.Tensor(random_input([9]))
    x2 = ms.Tensor(random_input([4, 2, 5, 6]))
    y2 = ms.Tensor(random_input([3, 4, 2, 6, 5]))

    TEST_OP(matmul, [[x1, y1], [x2, y2]], '', disable_yaml_check=True)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
def test_mix_dynamic_shape():
    """
    Feature: ops.matmul
    Description: mix dynamic shape
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    shape1, shape2 = [1, 10, 20, 64], [1, 10, 64, 77]
    ms.set_context(mode=ms.GRAPH_MODE)

    matmul_cell = MatMulCell()
    # 2 x 2
    x = random_input(shape1)
    y = random_input(shape2)

    _ = matmul_cell(ms.tensor(x), ms.tensor(y))

    grad_matmul_cell = MatMulGradCell()
    grad_matmul_cell.set_inputs(
        ms.tensor(shape=[1, 10, None, None], dtype=ms.float32),
        ms.tensor(shape=[1, 10, None, 77], dtype=ms.float32),
    )
    _, _ = grad_matmul_cell(ms.tensor(x), ms.tensor(y))


def random_input(shape, dtype=np.float32):
    return np.random.randn(*shape).astype(dtype)
