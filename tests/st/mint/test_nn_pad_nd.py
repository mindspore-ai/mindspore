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
import pytest
import numpy as np
import mindspore as ms
from mindspore.mint.nn import (
    ConstantPad1d, ConstantPad2d, ConstantPad3d,
    ZeroPad1d, ZeroPad2d, ZeroPad3d,
    ReflectionPad1d, ReflectionPad2d, ReflectionPad3d,
    ReplicationPad1d, ReplicationPad2d, ReplicationPad3d
)
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def expect_forward_output_constant(x, padding, value=None):
    return np.pad(x, padding, "constant", constant_values=value)


def expect_forward_output_other_modes(x, padding, mode):
    return np.pad(x, padding, mode)


def get_pad_layer(mode, padding, value=None):
    pad_layers = {
        "constant_1d": ConstantPad1d,
        "constant_2d": ConstantPad2d,
        "constant_3d": ConstantPad3d,
        "zero_1d": ZeroPad1d,
        "zero_2d": ZeroPad2d,
        "zero_3d": ZeroPad3d,
        "reflection_1d": ReflectionPad1d,
        "reflection_2d": ReflectionPad2d,
        "reflection_3d": ReflectionPad3d,
        "replication_1d": ReplicationPad1d,
        "replication_2d": ReplicationPad2d,
        "replication_3d": ReplicationPad3d,
    }
    if "constant" in mode:
        return pad_layers[mode](padding, value)
    return pad_layers[mode](padding)


@test_utils.run_with_cell
def pad_nd_forward(input_x, padding, value, mode):
    net = get_pad_layer(mode, padding, value)
    return net(input_x)


@test_utils.run_with_cell
def pad_nd_forward_for_dyn(input_x, mode):
    padding_map = {
        "constant_1d": (2, 3),
        "constant_2d": (1, 1, 2, 2),
        "constant_3d": (1, 1, 2, 2, 3, 3),
        "zero_1d": (2, 3),
        "zero_2d": (1, 1, 2, 2),
        "zero_3d": (1, 1, 2, 2, 3, 3),
        "reflection_1d": (1, 1),
        "reflection_2d": (1, 1, 2, 2),
        "reflection_3d": (1, 1, 1, 1, 1, 1),
        "replication_1d": (2, 3),
        "replication_2d": (1, 1, 2, 2),
        "replication_3d": (1, 1, 2, 2, 3, 3),
    }
    padding = padding_map[mode]
    value = 2 if "constant" in mode else None
    net = get_pad_layer(mode, padding, value)
    return net(input_x)


@test_utils.run_with_cell
def pad_nd_backward(input_x, padding, value, mode):
    return ms.grad(pad_nd_forward, (0))(input_x, padding, value, mode)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nn_PadNd_normal(context_mode):
    """
    Feature: pyboost nn API ConstantPadNd, ZeroPadNd, ReflectionPadNd, ReplicationPadNd
        which are all derived from mint.nn.functional.pad.
    Description: test forward and auto grad of ConstantPadNd, ZeroPadNd, ReflectionPadNd,
        ReplicationPadNd where N = 1, 2, 3.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    ## nn.ConstantPad1d
    constant_value = 1
    x = generate_random_input((2, 3), np.float32)
    padding = (1, 1)
    padding_np = ((0, 0), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "constant_1d")
    expect = expect_forward_output_constant(x, padding_np, constant_value)
    np.testing.assert_array_equal(output.asnumpy(), expect)
    # auto grad
    out_backward = pad_nd_backward(ms.Tensor(x), padding, constant_value, "constant_1d")
    expect_backward = np.ones_like(x)
    np.testing.assert_array_equal(out_backward.asnumpy(), expect_backward)

    ## nn.ConstantPad2d
    x = generate_random_input((2, 3, 4, 5), np.float32)
    padding = (1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "constant_2d")
    expect = expect_forward_output_constant(x, padding_np, constant_value)
    np.testing.assert_array_equal(output.asnumpy(), expect)
    # auto grad
    out_backward = pad_nd_backward(ms.Tensor(x), padding, constant_value, "constant_2d")
    expect_backward = np.ones_like(x)
    np.testing.assert_array_equal(out_backward.asnumpy(), expect_backward)

    ## nn.ConstantPad3d
    x = generate_random_input((2, 3, 4, 5, 6), np.float32)
    padding = (1, 1, 1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "constant_3d")
    expect = expect_forward_output_constant(x, padding_np, constant_value)
    np.testing.assert_array_equal(output.asnumpy(), expect)
    # auto grad
    out_backward = pad_nd_backward(ms.Tensor(x), padding, constant_value, "constant_3d")
    expect_backward = np.ones_like(x)
    np.testing.assert_array_equal(out_backward.asnumpy(), expect_backward)

    ## nn.ZeroPad1d
    constant_value = 0
    x = generate_random_input((2, 3), np.float32)
    padding = (1, 1)
    padding_np = ((0, 0), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "zero_1d")
    expect = expect_forward_output_constant(x, padding_np, constant_value)
    np.testing.assert_array_equal(output.asnumpy(), expect)
    # auto grad
    out_backward = pad_nd_backward(ms.Tensor(x), padding, constant_value, "zero_1d")
    expect_backward = np.ones_like(x)
    np.testing.assert_array_equal(out_backward.asnumpy(), expect_backward)

    ## nn.ZeroPad2d
    x = generate_random_input((2, 3, 4, 5), np.float32)
    padding = (1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "zero_2d")
    expect = expect_forward_output_constant(x, padding_np, constant_value)
    np.testing.assert_array_equal(output.asnumpy(), expect)
    # auto grad
    out_backward = pad_nd_backward(ms.Tensor(x), padding, constant_value, "zero_2d")
    expect_backward = np.ones_like(x)
    np.testing.assert_array_equal(out_backward.asnumpy(), expect_backward)

    ## nn.ZeroPad3d
    x = generate_random_input((2, 3, 4, 5, 6), np.float32)
    padding = (1, 1, 1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "zero_3d")
    expect = expect_forward_output_constant(x, padding_np, constant_value)
    np.testing.assert_array_equal(output.asnumpy(), expect)
    # auto grad
    out_backward = pad_nd_backward(ms.Tensor(x), padding, constant_value, "zero_3d")
    expect_backward = np.ones_like(x)
    np.testing.assert_array_equal(out_backward.asnumpy(), expect_backward)

    ## nn.ReflectionPad1d
    constant_value = 0
    x = generate_random_input((2, 3), np.float32)
    padding = (1, 1)
    padding_np = ((0, 0), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "reflection_1d")
    expect = expect_forward_output_other_modes(x, padding_np, "reflect")
    np.testing.assert_array_equal(output.asnumpy(), expect)
    # auto grad
    out_backward = pad_nd_backward(ms.Tensor(x), padding, constant_value, "reflection_1d")
    expect_backward = np.array([[1, 3, 1], [1, 3, 1]])
    np.testing.assert_array_equal(out_backward.asnumpy(), expect_backward)

    ## nn.ReflectionPad2d
    x = generate_random_input((2, 3, 4, 5), np.float32)
    padding = (1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "reflection_2d")
    expect = expect_forward_output_other_modes(x, padding_np, "reflect")
    np.testing.assert_array_equal(output.asnumpy(), expect)
    # auto grad
    out_backward = pad_nd_backward(ms.Tensor(x), padding, constant_value, "reflection_2d")
    c = [[1, 2, 1, 2, 1], [2, 4, 2, 4, 2], [2, 4, 2, 4, 2], [1, 2, 1, 2, 1]]
    expect_backward = np.array([[c, c, c], [c, c, c]])
    np.testing.assert_array_equal(out_backward.asnumpy(), expect_backward)

    ## nn.ReflectionPad3d
    x = generate_random_input((1, 1, 2, 2, 2), np.float32)
    padding = (1, 1, 1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "reflection_3d")
    expect = expect_forward_output_other_modes(x, padding_np, "reflect")
    np.testing.assert_array_equal(output.asnumpy(), expect)
    # auto grad
    out_backward = pad_nd_backward(ms.Tensor(x), padding, constant_value, "reflection_3d")
    expect_backward = np.array([[[[[8, 8], [8, 8]], [[8, 8], [8, 8]]]]])
    np.testing.assert_array_equal(out_backward.asnumpy(), expect_backward)

    ## nn.ReplicationPad1d
    constant_value = 0
    x = generate_random_input((2, 3), np.float32)
    padding = (1, 1)
    padding_np = ((0, 0), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "replication_1d")
    expect = expect_forward_output_other_modes(x, padding_np, "edge")
    np.testing.assert_array_equal(output.asnumpy(), expect)
    # auto grad
    out_backward = pad_nd_backward(ms.Tensor(x), padding, constant_value, "replication_1d")
    expect_backward = np.array([[2, 1, 2], [2, 1, 2]])
    np.testing.assert_array_equal(out_backward.asnumpy(), expect_backward)

    ## nn.ReplicationPad2d
    x = generate_random_input((1, 1, 2, 2), np.float32)
    padding = (1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "replication_2d")
    expect = expect_forward_output_other_modes(x, padding_np, "edge")
    np.testing.assert_array_equal(output.asnumpy(), expect)
    # auto grad
    out_backward = pad_nd_backward(ms.Tensor(x), padding, constant_value, "replication_2d")
    expect_backward = np.array([[[[4, 4], [4, 4]]]])
    np.testing.assert_array_equal(out_backward.asnumpy(), expect_backward)

    ## nn.ReplicationPad3d
    x = generate_random_input((1, 1, 3, 3, 3), np.float32)
    padding = (1, 1, 1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1))
    output = pad_nd_forward(ms.Tensor(x), padding, constant_value, "replication_3d")
    expect = expect_forward_output_other_modes(x, padding_np, "edge")
    np.testing.assert_array_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_nn_PadNd_dynamic():
    """
    Feature: pyboost dynamic shape feature of nn API ConstantPadNd, ZeroPadNd, ReflectionPadNd, ReplicationPadNd
        which are all derived from mint.nn.functional.pad.
    Description:pyboost dynamic shape feature of nn API ConstantPadNd, ZeroPadNd, ReflectionPadNd,
        ReplicationPadNd where N = 1, 2, 3.
    Expectation: expect correct result.
    """
    input1d_1 = generate_random_input((2, 3), np.float32)
    input1d_2 = generate_random_input((2, 3, 4), np.float32)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input1d_1), "constant_1d"], [ms.Tensor(input1d_2), "constant_1d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True})

    input2d_1 = generate_random_input((2, 3, 4, 5), np.float32)
    input2d_2 = generate_random_input((2, 3, 4), np.float32)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input2d_1), "constant_2d"], [ms.Tensor(input2d_2), "constant_2d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True})

    input3d_1 = generate_random_input((2, 3, 4, 5), np.float32)
    input3d_2 = generate_random_input((2, 3, 4), np.float32)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input3d_1), "constant_3d"], [ms.Tensor(input3d_2), "constant_3d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True})

    input_zero_1d_1 = generate_random_input((2, 3, 4, 5), np.float32)
    input_zero_1d_2 = generate_random_input((2, 3, 4), np.float32)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input_zero_1d_1), "zero_1d"], [ms.Tensor(input_zero_1d_2), "zero_1d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True})

    input_zero_2d_1 = generate_random_input((2, 3, 4, 5), np.float32)
    input_zero_2d_2 = generate_random_input((2, 3, 4), np.float32)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input_zero_2d_1), "zero_2d"], [ms.Tensor(input_zero_2d_2), "zero_2d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True})

    input_zero_3d_1 = generate_random_input((2, 3, 4, 5), np.float32)
    input_zero_3d_2 = generate_random_input((2, 3, 4), np.float32)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input_zero_3d_1), "zero_3d"], [ms.Tensor(input_zero_3d_2), "zero_3d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True})

    input_reflection_1d_1 = generate_random_input((2, 3), np.float32)
    input_reflection_1d_2 = generate_random_input((2, 3, 4), np.float32)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input_reflection_1d_1), "reflection_1d"], \
                                     [ms.Tensor(input_reflection_1d_2), "reflection_1d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True})

    input_reflection_2d_1 = generate_random_input((2, 3, 4, 5), np.float64)
    input_reflection_2d_2 = generate_random_input((2, 3, 4), np.float64)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input_reflection_2d_1), "reflection_2d"], \
                                     [ms.Tensor(input_reflection_2d_2), "reflection_2d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True})

    input_reflection_3d_1 = generate_random_input((2, 3, 4, 5), np.float32)
    input_reflection_3d_2 = generate_random_input((2, 3, 4, 5, 6), np.float32)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input_reflection_3d_1), "reflection_3d"], \
                                     [ms.Tensor(input_reflection_3d_2), "reflection_3d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True})

    input_replication_1d_1 = generate_random_input((4, 5), np.float32)
    input_replication_1d_2 = generate_random_input((2, 3, 4), np.float32)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input_replication_1d_1), "replication_1d"], \
                                     [ms.Tensor(input_replication_1d_2), "replication_1d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True})

    input_replication_2d_1 = generate_random_input((2, 3, 4, 5), np.float64)
    input_replication_2d_2 = generate_random_input((2, 3, 4), np.float64)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input_replication_2d_1), "replication_2d"], \
                                     [ms.Tensor(input_replication_2d_2), "replication_2d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'deterministic_use_origin_inputs': True})

    input_replication_3d_1 = generate_random_input((2, 3, 4, 5), np.float32)
    input_replication_3d_2 = generate_random_input((2, 3, 4, 6, 9), np.float32)
    TEST_OP(pad_nd_forward_for_dyn, [[ms.Tensor(input_replication_3d_1), "replication_3d"], \
                                     [ms.Tensor(input_replication_3d_2), "replication_3d"]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True})
