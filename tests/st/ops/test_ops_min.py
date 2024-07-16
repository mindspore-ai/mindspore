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
from mindspore import Tensor, context
from mindspore.ops.function.array_func import min_ext as min_
from mindspore import ops
from mindspore.common import dtype as mstype

from tests.st.utils.test_utils import to_cell_obj, compare
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def np_backward_func(x, out, dout):
    mask = np.not_equal(x, x) if np.isnan(out) else np.equal(x, out)
    dx = np.zeros_like(x)
    filled_value = np.divide(dout, np.sum(mask))
    dx[np.where(mask)] = filled_value
    return dx


def min_max_case(op_func, np_func, data_dtype=np.float32, has_nan=False, save_graphs=False):
    np_dtype = data_dtype
    is_bfloat16 = data_dtype == "bfloat16"
    if is_bfloat16:
        np_dtype = np.float32
    x_np = np.array([[3, 20, 5], [67, 8, 130], [3, 15, 130]], np_dtype)
    if has_nan:
        x_np = np.array([[3, 20, np.nan, 5], [67, 8, np.nan, 130], [3, np.nan, 15, 130]], np_dtype)
    x = Tensor(x_np)
    if is_bfloat16:
        x = Tensor(x_np, mstype.bfloat16)
    net = to_cell_obj(op_func)
    input_args = (x,)
    # forward:
    if save_graphs:
        context.set_context(save_graphs=True, save_graphs_path="graph_forward")
    output = net(*input_args)
    expect = np_func(x_np)
    if is_bfloat16:
        output = output.float()
    compare(output, expect)
    if not data_dtype in [np.float16, np.float32, np.float64, "bfloat16"]:
        return
    # backward:
    if save_graphs:
        context.set_context(save_graphs=True, save_graphs_path="graph_backward")
    output_grad = ops.grad(net)(*input_args)
    if is_bfloat16:
        output_grad = output_grad.float()
    expect_grad = np_backward_func(x_np, expect, np.ones_like(expect))
    assert np.allclose(output_grad.asnumpy(), expect_grad)


def min_max_case_vmap(op_func):
    def func_vmap_case(x):
        return op_func(x)

    x_batched = np.array([[5., 3., 4.], [2., 4., 3.], [3., 1., 4.]], dtype=np.float32)
    output_vmap = ops.vmap(func_vmap_case, in_axes=0)(Tensor(x_batched))
    value_batched = []
    for x in x_batched:
        value = func_vmap_case(Tensor(x))
        value_batched.append(value.asnumpy())
    expect = np.stack(value_batched)
    compare(output_vmap, expect)


def min_max_case_all_dyn(op_func, data_dtype=np.float32):
    is_bfloat16 = data_dtype == "bfloat16"
    if is_bfloat16:
        data_dtype = mstype.bfloat16
    t1 = Tensor(np.array([[1, 20], [67, 8]], dtype=data_dtype))
    input_case1 = [t1]
    t2 = Tensor(np.array([[[1, 20, 5], [67, 8, 9]], [[130, 24, 15], [16, 64, 32]]], dtype=data_dtype))
    input_case2 = [t2]
    disable_grad = False
    if not data_dtype in [np.float16, np.float32, np.float64, "bfloat16"]:
        disable_grad = True
    TEST_OP(op_func, [input_case1, input_case2], '', disable_yaml_check=True, disable_grad=disable_grad)


def np_min(input_x):
    return np.min(input_x)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_dtype', [np.float32])
def test_min(mode, data_dtype):
    """
    Feature: Test min op.
    Description: Test min.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_case(min_, np_min, data_dtype=data_dtype)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_min_nan(mode):
    """
    Feature: Test min op.
    Description: Test min.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_case(min_, np_min, has_nan=True)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_min_vmap(mode):
    """
    Feature: Test min op.
    Description: Test min vmap.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_case_vmap(min_)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_dtype', [np.float32])
def test_min_all_dynamic(data_dtype):
    """
    Feature: Test min op.
    Description: Test min with input is dynamic.
    Expectation: the result match with expected result.
    """
    min_max_case_all_dyn(min_, data_dtype=data_dtype)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_dtype', [np.float32])
def test_min_tensor(mode, data_dtype):
    """
    Feature: Test min op.
    Description: Test min.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    x_np = np.array([[3, 20, 5], [67, 8, 130], [3, 15, 130]], data_dtype)
    x = Tensor(x_np)
    output = x.min()
    expect = np_min(x_np)
    compare(output, expect)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
@pytest.mark.parametrize('data_dtype', ["bfloat16"])
def test_min_control_flow(mode, data_dtype):
    """
    Feature: Test min op.
    Description: Test min and control flow under pynative grad.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_case(min_, np_min, data_dtype=data_dtype)
