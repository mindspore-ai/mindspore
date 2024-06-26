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
from mindspore import ops, Tensor
from mindspore import mint
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, dim=None, keepdim=False):
    return np.argmax(x, axis=dim)


def generate_expect_backward_output(x, dim=None, keepdim=False):
    return 0


@test_utils.run_with_cell
def argmax_ext_forward_func(x, dim=None, keepdim=False):
    return mint.argmax(x, dim=dim, keepdim=keepdim)


@test_utils.run_with_cell
def argmax_ext_backward_func(x, dim=None, keepdim=False):
    return ops.grad(argmax_ext_forward_func)(x, dim, keepdim)

def GenInputData(np_data_type, shape=(3, 4, 5)):
    """GenInputData"""
    size = 1
    for s in shape:
        size *= s
    data = np.arange(size).reshape(*shape).astype(np_data_type)
    return Tensor(data)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_argmax_ext_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function argmax forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    dim = 0
    keepdim = False
    output = argmax_ext_forward_func(ms.Tensor(x), dim, keepdim)
    expect = generate_expect_forward_output(x, dim, keepdim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_argmax_ext_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function argmax backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = GenInputData(np.float32, (2, 3, 4, 5))
    dim = 0
    keepdim = False
    output = argmax_ext_backward_func(ms.Tensor(x), dim, keepdim)
    expect = generate_expect_backward_output(x, dim, keepdim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_argmax_ext_dynamic_shape():
    """
    Feature: Test argmax with dynamic shape in graph mode.
    Description: call mint.argmax with valid input and dim, keepdim is False.
    Expectation: return the correct value.
    """
    ms_data1 = GenInputData(np.float32, (2, 3, 4, 5))
    dim1 = 0

    ms_data2 = GenInputData(np.float32, (5, 8, 7))
    dim2 = 1
    TEST_OP(argmax_ext_forward_func, [[ms_data1, dim1], [ms_data2, dim2]], '', disable_yaml_check=True)
