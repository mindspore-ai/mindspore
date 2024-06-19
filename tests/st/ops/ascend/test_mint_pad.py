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
from tests.st.utils import test_utils
import mindspore as ms
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.mint.nn.functional import pad
from grad import GradOfFirstInput

def generate_random_input(shape, dtype):
    return np.random.randint(1, 10, size=shape).astype(dtype)

def expect_forward_output_constant(x, padding, value=None):
    return np.pad(x, padding, "constant", constant_values=value)

def expect_forward_output(x, padding, mode):
    return np.pad(x, padding, mode)

@test_utils.run_with_cell
def pad_forward_func(x, padding, mode="constant", value=None):
    return pad(x, padding, mode, value)

class Pad(Cell):
    def __init__(self, padding, mode, value):
        super().__init__()
        self.mode = mode
        self.padding = padding
        self.value = value

    def construct(self, x):
        output = pad(x, self.padding, self.mode, self.value)
        return output


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_forward_constantND(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward. mode = "constant". ND
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    constant_value = 1
    x = generate_random_input((2, 3), np.float32)
    padding = (1, 1)
    padding_np = ((0, 0), (1, 1))
    output = pad_forward_func(ms.Tensor(x), padding, "constant", constant_value)
    expect = expect_forward_output_constant(x, padding_np, constant_value)
    np.testing.assert_array_equal(output.asnumpy(), expect)

    x = generate_random_input((2, 3, 4, 5), np.float32)
    padding = (1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1))
    output = pad_forward_func(ms.Tensor(x), padding, "constant", constant_value)
    expect = expect_forward_output_constant(x, padding_np, constant_value)
    np.testing.assert_array_equal(output.asnumpy(), expect)

    x = generate_random_input((2, 3, 4, 5, 6), np.float32)
    padding = (1, 1, 1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1))
    output = pad_forward_func(ms.Tensor(x), padding, "constant", constant_value)
    expect = expect_forward_output_constant(x, padding_np, constant_value)
    np.testing.assert_array_equal(output.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_forward_reflect1d(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward. reflect1d
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3), np.float32)
    padding = (1, 1)
    padding_np = ((0, 0), (1, 1))
    output = pad_forward_func(ms.Tensor(x), padding, "reflect")
    expect = expect_forward_output(x, padding_np, "reflect")
    np.testing.assert_array_equal(output.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_forward_reflect2d(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward. reflect2d
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    padding = (1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1))
    output = pad_forward_func(ms.Tensor(x), padding, "reflect")
    expect = expect_forward_output(x, padding_np, "reflect")
    np.testing.assert_array_equal(output.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_forward_reflect3d(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward. reflect3d
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5, 6), np.float32)
    padding = (1, 1, 1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1))
    output = pad_forward_func(ms.Tensor(x), padding, "reflect")
    expect = expect_forward_output(x, padding_np, "reflect")
    np.testing.assert_array_equal(output.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_forward_replicate1d(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3), np.float32)
    padding = (1, 1)
    padding_np = ((0, 0), (1, 1))
    output = pad_forward_func(ms.Tensor(x), padding, "replicate")
    expect = expect_forward_output(x, padding_np, "edge")
    np.testing.assert_array_equal(output.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_forward_replicate2d(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    padding = (1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1))
    output = pad_forward_func(ms.Tensor(x), padding, "replicate")
    expect = expect_forward_output(x, padding_np, "edge")
    np.testing.assert_array_equal(output.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_forward_replicate3d(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5, 6), np.float32)
    padding = (1, 1, 1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1))
    output = pad_forward_func(ms.Tensor(x), padding, "replicate")
    expect = expect_forward_output(x, padding_np, "edge")
    np.testing.assert_array_equal(output.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_constantNd_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    padding = (1, 1, 1, 1)
    test_cell.set_inputs(x_dyn, padding, "constant", 1)

    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1))
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), padding, "constant", 1)
    expect = expect_forward_output_constant(x1, padding_np, 1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), padding, "constant", 1)
    expect = expect_forward_output_constant(x2, padding_np, 1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_constantNd_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    padding = (1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1))
    test_cell.set_inputs(x_dyn, padding, "constant", 1)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), padding, "constant", 1)
    expect = expect_forward_output_constant(x1, padding_np, 1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), padding, "constant", 1)
    expect = expect_forward_output_constant(x2, padding_np, 1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_reflect1d_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    padding = (1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1))
    test_cell.set_inputs(x_dyn, padding, "reflect")

    x1 = generate_random_input((2, 3, 4), np.float32)
    output = test_cell(ms.Tensor(x1), padding, "reflect")
    expect = expect_forward_output(x1, padding_np, "reflect")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 6), np.float32)
    output = test_cell(ms.Tensor(x2), padding, "reflect")
    expect = expect_forward_output(x2, padding_np, "reflect")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_reflect2d_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    padding = (1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1))
    test_cell.set_inputs(x_dyn, padding, "reflect")

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), padding, "reflect")
    expect = expect_forward_output(x1, padding_np, "reflect")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((2, 3, 4, 6), np.float32)
    output = test_cell(ms.Tensor(x2), padding, "reflect")
    expect = expect_forward_output(x2, padding_np, "reflect")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_reflect3d_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    padding = (1, 1, 1, 1, 1, 1)
    padding_np = ((0, 0), (1, 1), (1, 1), (1, 1))
    test_cell.set_inputs(x_dyn, padding, "reflect")

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), padding, "reflect")
    expect = expect_forward_output(x1, padding_np, "reflect")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((2, 3, 4, 6), np.float32)
    output = test_cell(ms.Tensor(x2), padding, "reflect")
    expect = expect_forward_output(x2, padding_np, "reflect")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_reflect1d_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    test_cell.set_inputs(x_dyn, (1, 1), "reflect")

    x1 = generate_random_input((2, 3, 4), np.float32)
    output = test_cell(ms.Tensor(x1), (1, 1), "reflect")
    expect = expect_forward_output(x1, ((0, 0), (0, 0), (1, 1)), "reflect")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_reflect2d_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    test_cell.set_inputs(x_dyn, (1, 1, 1, 1), "reflect")

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), (1, 1, 1, 1), "reflect")
    expect = expect_forward_output(x2, ((0, 0), (0, 0), (1, 1), (1, 1)), "reflect")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_reflect3d_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    test_cell.set_inputs(x_dyn, (1, 1, 1, 1, 1, 1), "reflect")

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), (1, 1, 1, 1, 1, 1), "reflect")
    expect = expect_forward_output(x2, ((0, 0), (1, 1), (1, 1), (1, 1)), "reflect")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_replicate1d_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    padding = (1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1))
    test_cell.set_inputs(x_dyn, padding, "replicate")

    x1 = generate_random_input((2, 3, 4), np.float32)
    output = test_cell(ms.Tensor(x1), padding, "replicate")
    expect = expect_forward_output(x1, padding_np, "edge")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 6), np.float32)
    output = test_cell(ms.Tensor(x2), padding, "replicate")
    expect = expect_forward_output(x2, padding_np, "edge")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_replicate2d_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    padding = (1, 1, 1, 1)
    padding_np = ((0, 0), (0, 0), (1, 1), (1, 1))
    test_cell.set_inputs(x_dyn, padding, "replicate")

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), padding, "replicate")
    expect = expect_forward_output(x1, padding_np, "edge")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((2, 3, 4, 6), np.float32)
    output = test_cell(ms.Tensor(x2), padding, "replicate")
    expect = expect_forward_output(x2, padding_np, "edge")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_replicate3d_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    padding = (1, 1, 1, 1, 1, 1)
    padding_np = ((0, 0), (1, 1), (1, 1), (1, 1))
    test_cell.set_inputs(x_dyn, padding, "replicate")

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), padding, "replicate")
    expect = expect_forward_output(x1, padding_np, "edge")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((2, 3, 4, 6), np.float32)
    output = test_cell(ms.Tensor(x2), padding, "replicate")
    expect = expect_forward_output(x2, padding_np, "edge")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_replicate1d_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    test_cell.set_inputs(x_dyn, (1, 1), "replicate")

    x1 = generate_random_input((2, 3, 4), np.float32)
    output = test_cell(ms.Tensor(x1), (1, 1), "replicate")
    expect = expect_forward_output(x1, ((0, 0), (0, 0), (1, 1)), "edge")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_replicate2d_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    test_cell.set_inputs(x_dyn, (1, 1, 1, 1), "replicate")

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), (1, 1, 1, 1), "replicate")
    expect = expect_forward_output(x2, ((0, 0), (0, 0), (1, 1), (1, 1)), "edge")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_pad_replicate3d_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(pad_forward_func)
    test_cell.set_inputs(x_dyn, (1, 1, 1, 1, 1, 1), "replicate")

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), (1, 1, 1, 1, 1, 1), "replicate")
    expect = expect_forward_output(x2, ((0, 0), (1, 1), (1, 1), (1, 1)), "edge")
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_pad_backward_constantND(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad backward. mode = "constant". ND
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O0')
    x_value = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    x = Tensor(x_value, dtype=ms.float32)
    padding = (1, 2, 2, 1)
    expect = np.array([[[[1., 2., 3.],
                         [4., 5., 6.],
                         [7., 8., 9.]]]])
    net = Pad(padding, "constant", 2)
    out_grad = net(x)
    grad_net = GradOfFirstInput(net)
    input_grad = grad_net(x, out_grad)
    np.testing.assert_array_equal(input_grad.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_pad_backward_reflect1d(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad backward. mode = "reflect". 1D
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O0')
    x_value = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    x = Tensor(x_value, dtype=ms.float32)
    padding = (1, 2)
    expect = np.array([[[2., 6., 3.],
                        [8., 15., 6.],
                        [14., 24., 9.]]])
    net = Pad(padding, "reflect", None)
    out_grad = net(x)
    grad_net = GradOfFirstInput(net)
    input_grad = grad_net(x, out_grad)
    np.testing.assert_array_equal(input_grad.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_pad_backward_reflect2d(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad backward. mode = "reflect". 2D
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O0')
    x_value = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    x = Tensor(x_value, dtype=ms.float32)
    padding = (1, 2, 1, 2)
    expect = np.array([[[[4., 12., 6.],
                         [24., 45., 18.],
                         [14., 24., 9.]]]])
    net = Pad(padding, "reflect", None)
    out_grad = net(x)
    grad_net = GradOfFirstInput(net)
    input_grad = grad_net(x, out_grad)
    np.testing.assert_array_equal(input_grad.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_pad_backward_replicate1d(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad backward. mode = "replicate". 1D
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O0')
    x_value = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    x = Tensor(x_value, dtype=ms.float32)
    padding = (1, 2)
    expect = np.array([[[2., 2., 9.],
                        [8., 5., 18.],
                        [14., 8., 27.]]])
    net = Pad(padding, "replicate", None)
    out_grad = net(x)
    grad_net = GradOfFirstInput(net)
    input_grad = grad_net(x, out_grad)
    np.testing.assert_array_equal(input_grad.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_pad_backward_replicate2d(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad backward. mode = "replicate". 2D
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O0')
    x_value = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    x = Tensor(x_value, dtype=ms.float32)
    padding = (1, 2, 1, 2)
    expect = np.array([[[[4., 4., 18.],
                         [8., 5., 18.],
                         [42., 24., 81.]]]])
    net = Pad(padding, "replicate", None)
    out_grad = net(x)
    grad_net = GradOfFirstInput(net)
    input_grad = grad_net(x, out_grad)
    np.testing.assert_array_equal(input_grad.asnumpy(), expect)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_pad_backward_replicate3d(context_mode):
    """
    Feature: pyboost function.
    Description: test function pad backward. mode = "replicate". 3D
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O0')
    x_value = [[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]]
    x = Tensor(x_value, dtype=ms.float32)
    padding = (1, 2, 1, 2, 1, 2)
    expect = np.array([[[[[16., 16., 72.],
                          [32., 20., 72.],
                          [168., 96., 324.]]]]])
    net = Pad(padding, "replicate", None)
    out_grad = net(x)
    grad_net = GradOfFirstInput(net)
    input_grad = grad_net(x, out_grad)
    np.testing.assert_array_equal(input_grad.asnumpy(), expect)
