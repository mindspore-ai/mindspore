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
import pytest
import numpy as np
from mindspore import Tensor, context
import mindspore.common.dtype as mstype
from mindspore import ops
from tests.st.utils.test_utils import to_cell_obj
from tests.mark_utils import arg_mark

def argmin_(input_x, axis, output_type):
    return ops.Argmin(axis, output_type)(input_x)

def argmin_argmax_case(op_func, np_func):
    x_np = np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]], np.float32)
    x = Tensor(x_np)
    net = to_cell_obj(op_func)
    axis = -1
    input_args = (x, axis, mstype.int32)
    # forward:
    output = net(*input_args)
    expect_output = np_func(x_np, axis).astype(np.int32)
    assert np.allclose(output.asnumpy(), expect_output)
    # backward:
    output_grad = ops.grad(net)(*input_args)
    expect_grad = np.zeros_like(x_np)
    assert np.allclose(output_grad.asnumpy(), expect_grad)


def argmin_argmax_case_vmap(op_func):
    def func_vmap_case(x):
        return op_func(x, -1, mstype.int32)

    x_batched = np.array([[5., 3., 4.], [2., 4., 3.], [3., 1., 4.]], dtype=np.float16)
    output_vmap = ops.vmap(func_vmap_case, in_axes=0)(Tensor(x_batched))
    output_batched = []
    for x in x_batched:
        output_batched.append(func_vmap_case(Tensor(x)).asnumpy())
    expect = np.stack(output_batched)
    assert np.allclose(output_vmap.asnumpy(), expect)


def argmin_argmax_case_dyn(op_func, np_func):
    def func_dyn_case(x):
        # Currently, only test the dynamics of x,
        # the axis has some framework problems in graph_mode
        return op_func(x, -1, mstype.int32)

    t1 = Tensor([[1, 20], [67, 8]], mstype.float32)
    t2 = Tensor([[1, 20, 5], [67, 8, 9], [130, 24, 15]], mstype.float32)
    test_cell = to_cell_obj(func_dyn_case)
    test_cell.set_inputs(Tensor(shape=[None, None], dtype=mstype.float32))
    expect1 = np_func(t1.asnumpy(), -1)
    output1 = test_cell(t1)
    assert np.allclose(output1.asnumpy(), expect1)
    expect2 = np_func(t2.asnumpy(), -1)
    output2 = test_cell(t2)
    assert np.allclose(output2.asnumpy(), expect2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmin(mode):
    """
    Feature: Test argmin op.
    Description: Test argmin.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_argmax_case(argmin_, np.argmin)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmin_vmap(mode):
    """
    Feature: Test argmin op.
    Description: Test argmin vmap.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_argmax_case_vmap(argmin_)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_argmin_dyn(mode):
    """
    Feature: Test argmin op.
    Description: Test argmin dynamic shape.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    argmin_argmax_case_dyn(argmin_, np.argmin)
