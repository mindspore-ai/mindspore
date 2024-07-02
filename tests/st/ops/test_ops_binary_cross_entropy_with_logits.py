# Copyright 2024 Huawei Technocasties Co., Ltd
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
from mindspore import ops
from mindspore.mint.nn.functional import binary_cross_entropy_with_logits
from mindspore.mint.nn import BCEWithLogitsLoss
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

@test_utils.run_with_cell
def binary_cross_entropy_with_logits_forward_func(inputx, target, weight=None, posWeight=None, reduction="mean"):
    return binary_cross_entropy_with_logits(inputx, target, weight, reduction, posWeight)

@test_utils.run_with_cell
def binary_cross_entropy_with_logits_backward_func(inputx, target, weight=None, posWeight=None, reduction="mean"):
    grad_op = ops.grad(binary_cross_entropy_with_logits_forward_func, (0, 1, 2, 3, 4))
    return grad_op(inputx, target, weight, posWeight, reduction)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", ["pynative", "KBK", "graph"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_binary_cross_entropy_with_logits_forward(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function binary_cross_entropy_with_logits forward.
    Expectation: expect correct result.
    """
    inputx = ms.Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), ms.float32)
    target = ms.Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), ms.float32)
    weight = ms.Tensor(np.array([1.0, 1.0, 1.0]), ms.float32)
    pos_weight = ms.Tensor(np.array([1.0, 1.0, 1.0]), ms.float32)

    if reduction == "sum":
        expect_fw = np.array([2.078167])
        expect_bw = np.array([[0.0100254714, -0.031475246, -0.53181231], [1.07502079, 0.301312357, -1.53181231]])
    elif reduction == "mean":
        expect_fw = np.array([0.3463612])
        expect_bw = np.array([[0.00167091191, -0.00524587464, -0.0886353850],
                              [0.179170132, 0.0502187274, -0.255302072]])
    else:
        expect_fw = np.array([[0.6111006, 0.5032824, 0.26318604], [0.58439666, 0.55301523, -0.43681395]])
        expect_bw = np.array([[0.010025471, -0.031475246, -0.53181231], [1.07502079, 0.3013123578, -1.53181231]])

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        out_fw = binary_cross_entropy_with_logits_forward_func(inputx, target, weight, pos_weight, reduction)
        out_bw = binary_cross_entropy_with_logits_backward_func(inputx, target, weight, pos_weight, reduction)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op = ms.jit(binary_cross_entropy_with_logits_forward_func, jit_config=ms.JitConfig(jit_level="O0"))
        out_fw = op(inputx, target, weight, pos_weight, reduction)
        op2 = ms.jit(binary_cross_entropy_with_logits_backward_func, jit_config=ms.JitConfig(jit_level="O0"))
        out_bw = op2(inputx, target, weight, pos_weight, reduction)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        out_fw = binary_cross_entropy_with_logits_forward_func(inputx, target, weight, pos_weight, reduction)
        out_bw = binary_cross_entropy_with_logits_backward_func(inputx, target, weight, pos_weight, reduction)

    np.testing.assert_allclose(out_fw.asnumpy(), expect_fw, rtol=1e-3)
    np.testing.assert_allclose(out_bw[0].asnumpy(), expect_bw, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_ops_binary_cross_entropy_with_logits_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function binary_cross_entropy_with_logits forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    target1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    weight1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    pos_weight1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    x2 = ms.Tensor(generate_random_input((7, 8), np.float32))
    target2 = ms.Tensor(generate_random_input((7, 8), np.float32))
    weight2 = ms.Tensor(generate_random_input((7, 8), np.float32))
    pos_weight2 = ms.Tensor(generate_random_input((7, 8), np.float32))

    # disable_yaml_check=true: reduction cannot be mutable now
    test_cell = test_utils.to_cell_obj(binary_cross_entropy_with_logits_forward_func)
    TEST_OP(test_cell, [[x1, target1, weight1, pos_weight1], [x2, target2, weight2, pos_weight2]],
            "binary_cross_entropy_with_logits", disable_mode=['GRAPH_MODE'], disable_yaml_check=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", ["pynative", "KBK", "graph"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_bce_with_logits_loss_forward(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function binary_cross_entropy_with_logits forward.
    Expectation: expect correct result.
    """
    inputx = ms.Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), ms.float32)
    target = ms.Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), ms.float32)
    weight = ms.Tensor(np.array([1.0, 1.0, 1.0]), ms.float32)
    pos_weight = ms.Tensor(np.array([1.0, 1.0, 1.0]), ms.float32)

    if reduction == "sum":
        expect_fw = np.array([2.078167])
        expect_bw = np.array([[0.0100254714, -0.031475246, -0.53181231], [1.07502079, 0.301312357, -1.53181231]])
    elif reduction == "mean":
        expect_fw = np.array([0.3463612])
        expect_bw = np.array([[0.00167091191, -0.00524587464, -0.0886353850],
                              [0.179170132, 0.0502187274, -0.255302072]])
    else:
        expect_fw = np.array([[0.6111006, 0.5032824, 0.26318604], [0.58439666, 0.55301523, -0.43681395]])
        expect_bw = np.array([[0.010025471, -0.031475246, -0.53181231], [1.07502079, 0.3013123578, -1.53181231]])

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        op = BCEWithLogitsLoss(weight, reduction, pos_weight)
        out_fw = op(inputx, target)
        grad_op = ops.grad(op, (0, 1))
        out_bw = grad_op(inputx, target)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        net = BCEWithLogitsLoss(weight, reduction, pos_weight)
        def func(inputx, target):
            return net(inputx, target)
        op = ms.jit(func, jit_config=ms.JitConfig(jit_level="O0"))
        out_fw = op(inputx, target)
        opx = BCEWithLogitsLoss(weight, reduction, pos_weight)
        def func2(inputx, target):
            return ops.grad(opx, (0, 1))(inputx, target)
        op2 = ms.jit(func2, jit_config=ms.JitConfig(jit_level="O0"))
        out_bw = op2(inputx, target)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op = BCEWithLogitsLoss(weight, reduction, pos_weight)
        out_fw = op(inputx, target)
        grad_op = ops.grad(op, (0, 1))
        out_bw = grad_op(inputx, target)

    np.testing.assert_allclose(out_fw.asnumpy(), expect_fw, rtol=1e-3)
    np.testing.assert_allclose(out_bw[0].asnumpy(), expect_bw, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_ops_bce_with_logits_loss_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function BCEWithLogitsLoss forward with dynamic shape.
    Expectation: expect correct result.
    """

    weight = ms.Tensor(generate_random_input((8, 9), np.float32))
    pos_weight = ms.Tensor(generate_random_input((8, 9), np.float32))
    reduction = 'mean'

    x1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    target1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    x2 = ms.Tensor(generate_random_input((8, 9), np.float32))
    target2 = ms.Tensor(generate_random_input((8, 9), np.float32))

    op = BCEWithLogitsLoss(weight, reduction, pos_weight)

    # disable_yaml_check=true: BCEWithLogitsLoss not same as yaml
    test_cell = test_utils.to_cell_obj(op)
    TEST_OP(test_cell, [[x1, target1], [x2, target2]],
            "binary_cross_entropy_with_logits", disable_mode=['GRAPH_MODE'], disable_yaml_check=True)
