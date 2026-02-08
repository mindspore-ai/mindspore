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
from mindspore import grad
from mindspore.mint.nn.functional import cross_entropy
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def get_input1():
    inputx = ms.Tensor(np.array([[1.81770931, 0.31360114, 0.30931599, 0.18100776, 0.40601558],
                                 [-1.23061385, -0.88784365, 0.68879239, -0.07928044, 0.9060898],
                                 [-0.57779883, -0.54754656, 0.84325396, -1.59585763, 0.61647592]]), ms.float32)
    target = ms.Tensor(np.array([[1.05003095, -0.53359773, 0.51047731, 1.31185591, -0.52938761],
                                 [-0.67310591, -0.36507688, 1.42758111, 0.89142645, 2.8884019],
                                 [-2.09405564, 0.15800516, -0.79772699, -1.04437548, -0.24338463]]), ms.float32)
    weight = ms.Tensor(np.array([-0.75326879, 1.45076741, 1.19889046, 0.09059219, -1.53917865]), ms.float32)
    ignore_index = -100
    label_smoothing = 0.5
    return inputx, target, weight, ignore_index, label_smoothing

def get_input2():
    inputx = ms.Tensor(np.array([[1.81770931, 0.31360114, 0.30931599, 0.18100776, 0.40601558],
                                 [-1.23061385, -0.88784365, 0.68879239, -0.07928044, 0.9060898],
                                 [-0.57779883, -0.54754656, 0.84325396, -1.59585763, 0.61647592]]), ms.float32)
    target = ms.Tensor(np.array([4, 3, 2]), ms.int64)
    weight = ms.Tensor(np.array([-0.75326879, 1.45076741, 1.19889046, 0.09059219, -1.53917865]), ms.float32)
    ignore_index = 3
    label_smoothing = 0.5
    return inputx, target, weight, ignore_index, label_smoothing


def get_output_forward1(reduction):
    output_mean = np.array([0.6161587])
    output_sum = np.array([1.8484762])
    output_none = np.array([0.77053377, -0.73874366, 1.8166661])
    output = {"mean": output_mean, "sum": output_sum, "none": output_none}
    return output[reduction]


def get_output_backward1(reduction):
    input_grad_sum = np.array([[0.4894, 0.2461, -0.4218, -0.0649, -0.2490],
                               [-0.2402, 0.0324, -1.3983, -0.2455, 1.8516],
                               [-0.6514, -0.1958, 0.6151, 0.0607, 0.1713]])
    target_grad_sum = np.array([[-0.2381, 1.5497, 1.2832, 0.1028, -1.5730],
                                [-1.1441, 1.9549, 0.6704, 0.0855, -0.6935],
                                [-0.8610, 1.6362, 0.5184, 0.1497, -0.8401]])
    weight_grad_sum = np.array([-2.4883, -0.1750, 1.4121, 1.3497, 1.0312])

    input_grad_mean = np.array([[0.1631, 0.0820, -0.1406, -0.0216, -0.0830],
                                [-0.0801, 0.0108, -0.4661, -0.0818, 0.6172],
                                [-0.2171, -0.0653, 0.2050, 0.0202, 0.0571]])
    target_grad_mean = np.array([[-0.079375, 0.516559, 0.427732, 0.034258, -0.524331],
                                 [-0.381382, 0.651647, 0.22347, 0.028483, -0.231162],
                                 [-0.286984, 0.545406, 0.172812, 0.049886, -0.280037]])
    weight_grad_mean = np.array([-0.8294, -0.0583, 0.4707, 0.4499, 0.3437])

    input_grad_none = np.array([[0.4894, 0.2461, -0.4218, -0.0649, -0.2490],
                                [-0.2402, 0.0324, -1.3983, -0.2455, 1.8516],
                                [-0.6514, -0.1958, 0.6151, 0.0607, 0.1713]])
    target_grad_none = np.array([[-0.2381, 1.5497, 1.2832, 0.1028, -1.5730],
                                 [-1.1441, 1.9549, 0.6704, 0.0855, -0.6935],
                                 [-0.8610, 1.6362, 0.5184, 0.1497, -0.8401]])
    weight_grad_none = np.array([-2.4883, -0.1750, 1.4121, 1.3497, 1.0312])
    output = {"mean": [input_grad_mean, target_grad_mean, weight_grad_mean],
              "sum": [input_grad_sum, target_grad_sum, weight_grad_sum],
              "none": [input_grad_none, target_grad_none, weight_grad_none]}
    return output[reduction]


def get_output_forward2(reduction):
    output_mean = np.array([2.0835])
    output_sum = np.array([-0.7090])
    output_none = np.array([-1.3481, 0.0000, 0.6391])
    output = {"mean": output_mean, "sum": output_sum, "none": output_none}
    return output[reduction]


def get_output_backward2(reduction):
    input_grad_mean = np.array([[0.9105, 0.6779, 0.6028, 0.2469, -2.4380],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [-0.4139, 0.2279, 1.3167, -0.0429, -1.0878]])
    input_grad_sum = np.array([[-0.3098, -0.2307, -0.2051, -0.0840, 0.8296],
                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.1408, -0.0776, -0.4480, 0.0146, 0.3702]])
    input_grad_none = np.array([[-0.3098, -0.2307, -0.2051, -0.0840, 0.8296],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.1408, -0.0776, -0.4480, 0.0146, 0.3702]])
    output = {"mean": [input_grad_mean],
              "sum": [input_grad_sum],
              "none": [input_grad_none]}
    return output[reduction]


@test_utils.run_with_cell
def cross_entropy_forward_func(inputx, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0):
    return cross_entropy(inputx, target, weight, ignore_index, reduction, label_smoothing)


@test_utils.run_with_cell
def cross_entropy_backward_func(inputx, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0):
    grad_op = grad(cross_entropy_forward_func, (0, 1, 2))
    return grad_op(inputx, target, weight, ignore_index, reduction, label_smoothing)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_cross_entropy_pro(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function cross_entropy backward.
    Expectation: expect correct result.
    """
    inputx, target, weight, ignore_index, label_smoothing = get_input1()
    expect_forward = get_output_forward1(reduction)
    expect_backward = get_output_backward1(reduction)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O2"})
    output_forward = cross_entropy_forward_func(inputx, target, weight, ignore_index, reduction, label_smoothing)
    output_backward = cross_entropy_backward_func(inputx, target, weight, ignore_index, reduction, label_smoothing)
    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, rtol=1e-3)
    np.testing.assert_allclose(output_backward[0].asnumpy(), expect_backward[0], rtol=1e-3)
    np.testing.assert_allclose(output_backward[1].asnumpy(), expect_backward[1], rtol=1e-3)
    np.testing.assert_allclose(output_backward[2].asnumpy(), expect_backward[2], rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_cross_entropy_class(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function cross_entropy backward.
    Expectation: expect correct result.
    """
    inputx, target, weight, ignore_index, label_smoothing = get_input2()
    expect_forward = get_output_forward2(reduction)
    expect_backward = get_output_backward2(reduction)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O2"})
    output_forward = cross_entropy_forward_func(inputx, target, weight, ignore_index, reduction, label_smoothing)
    output_backward = cross_entropy_backward_func(inputx, target, weight, ignore_index, reduction, label_smoothing)
    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, rtol=1e-3)
    np.testing.assert_allclose(output_backward[0].asnumpy(), expect_backward[0], rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_cross_entropy_dynamic_shape(reduction):
    """
    Feature: pyboost function.
    Description: test function cross_entropy forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(np.random.rand(7, 8, 9).astype(np.float32))
    target1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    weight1 = ms.Tensor(generate_random_input((8,), np.float32))
    ignore_index1 = 3
    label_smoothing1 = 0.0

    x2 = ms.Tensor(np.random.rand(9, 8).astype(np.float32))
    target2 = ms.Tensor(generate_random_input((9, 8), np.float32))
    weight2 = ms.Tensor(generate_random_input((8,), np.float32))
    ignore_index2 = 2
    label_smoothing2 = 0.4


    test_cell = test_utils.to_cell_obj(cross_entropy_forward_func)
    TEST_OP(test_cell, [[x1, target1, weight1, ignore_index1, reduction, label_smoothing1],
                        [x2, target2, weight2, ignore_index2, reduction, label_smoothing2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True})
