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
from mindspore.graph.api import _pynative_executor
from mindspore.mint.nn.functional import l1_loss
from mindspore.mint.nn import L1Loss
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def get_input():
    inputx = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]]), ms.float32)
    target = ms.Tensor(np.array([[6, 5, 4], [3, 2, 1]]), ms.float32)
    return inputx, target

def get_output_forward(reduction):
    output_mean = np.array([3.0])
    output_sum = np.array([18.0])
    output_none = np.array([[5.0, 3.0, 1.0], [1.0, 3.0, 5.0]])
    output = {"mean": output_mean, "sum": output_sum, "none": output_none}
    return output[reduction]

def get_output_backward(reduction):
    output_mean = np.array([[-0.16667, -0.16667, -0.16667], [0.16667, 0.16667, 0.16667]])
    output_sum = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    output_none = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    output = {"mean": output_mean, "sum": output_sum, "none": output_none}
    return output[reduction]

@test_utils.run_with_cell
def l1_loss_forward_func(inputx, target, reduction="mean"):
    return l1_loss(inputx, target, reduction)


@test_utils.run_with_cell
def l1_loss_backward_func(inputx, target, reduction="mean"):
    grad_op = ms.grad(l1_loss_forward_func, (0, 1, 2))
    return grad_op(inputx, target, reduction)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_l1_loss_forward(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function l1_loss forward.
    Expectation: expect correct result.
    """
    inputx, target = get_input()
    expect_forward_value = get_output_forward(reduction)
    expect_backward_value = get_output_backward(reduction)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output_forward_value = l1_loss_forward_func(inputx, target, reduction)
        output_backward_value = l1_loss_backward_func(inputx, target, reduction)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        forward_op = ms.jit(l1_loss_forward_func, backend="ms_backend", jit_level="O0")
        backward_op = ms.jit(l1_loss_backward_func, backend="ms_backend", jit_level="O0")
        output_forward_value = forward_op(inputx, target, reduction)
        output_backward_value = backward_op(inputx, target, reduction)

    np.testing.assert_allclose(output_forward_value.asnumpy(), expect_forward_value, rtol=1e-3)
    np.testing.assert_allclose(output_backward_value[0].asnumpy(), expect_backward_value, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_nn_L1Loss(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function l1_loss forward.
    Expectation: expect correct result.
    """
    inputx, target = get_input()
    expect_forward_value = get_output_forward(reduction)
    expect_backward_value = get_output_backward(reduction)
    net = L1Loss(reduction)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output_forward_value = net(inputx, target)
        output_backward_value = l1_loss_backward_func(inputx, target, reduction)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
        output_forward_value = net(inputx, target)
        output_backward_value = l1_loss_backward_func(inputx, target, reduction)

    np.testing.assert_allclose(output_forward_value.asnumpy(), expect_forward_value, rtol=1e-3)
    np.testing.assert_allclose(output_backward_value[0].asnumpy(), expect_backward_value, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_l1_loss_dynamic_shape(reduction):
    """
    Feature: pyboost function.
    Description: test function l1_loss forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    target1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    x2 = ms.Tensor(generate_random_input((8, 9), np.float32))
    target2 = ms.Tensor(generate_random_input((8, 9), np.float32))

    test_cell = test_utils.to_cell_obj(l1_loss_forward_func)
    TEST_OP(test_cell, [[x1, target1, reduction], [x2, target2, reduction]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True})


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_ops_l1_loss_promote_dtype(mode):
    """
    Feature: pyboost function.
    Description: test function l1_loss forward with promote dtype.
    Expectation: expect correct result.
    """
    ms_dtype_groups = {
        'bool_': [0, ms.bool_],
        'int8_': [1, ms.int8],
        'int16': [2, ms.int16],
        'int32': [3, ms.int32],
        'int64': [4, ms.int64],
        'uint8': [5, ms.uint8],
        'fp16_': [6, ms.float16],
        'fp32_': [7, ms.float32],
        'fp64_': [8, ms.float64],
        'cx64_': [9, ms.complex64],
        'cx128': [10, ms.complex128],
        'bf16_': [11, ms.bfloat16],
    }

    out_dtype = [
        #bool     int8     int16    int32    int64    uint8    fp16     fp32     fp64     cx64     cx128    bf16 target/input
        ['undef', 'undef', 'undef', 'undef', 'int64', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #bool
        ['undef', 'undef', 'undef', 'undef', 'int64', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int8
        ['undef', 'undef', 'undef', 'undef', 'int64', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int16
        ['undef', 'undef', 'undef', 'undef', 'int64', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int32
        ['int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int64
        ['undef', 'undef', 'undef', 'undef', 'int64', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #uint8
        ['fp16_', 'fp16_', 'fp16_', 'fp16_', 'fp16_', 'fp16_', 'fp16_', 'fp32_', 'undef', 'undef', 'undef', 'fp32_'],  #fp16
        ['fp32_', 'fp32_', 'fp32_', 'fp32_', 'fp32_', 'fp32_', 'fp32_', 'fp32_', 'undef', 'undef', 'undef', 'fp32_'],  #fp32
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #fp64
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #cx64
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #cx128
        ['bf16_', 'bf16_', 'bf16_', 'bf16_', 'bf16_', 'bf16_', 'fp32_', 'fp32_', 'undef', 'undef', 'undef', 'bf16_'],  #bf16
    ]

    grad_input_dtype = [
        #bool     int8     int16    int32    int64    uint8    fp16     fp32     fp64     cx64     cx128    bf16 target/input
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #bool
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int8
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int16
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int32
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int64
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #uint8
        ['fp16_', 'fp16_', 'fp16_', 'fp16_', 'fp16_', 'fp16_', 'fp16_', 'fp16_', 'undef', 'undef', 'undef', 'fp16_'],  #fp16
        ['fp32_', 'fp32_', 'fp32_', 'fp32_', 'fp32_', 'fp32_', 'fp32_', 'fp32_', 'undef', 'undef', 'undef', 'fp32_'],  #fp32
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #fp64
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #cx64
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #cx128
        ['bf16_', 'bf16_', 'bf16_', 'bf16_', 'bf16_', 'bf16_', 'bf16_', 'bf16_', 'undef', 'undef', 'undef', 'bf16_'],  #bf16
    ]

    grad_target_dtype = [
        #bool     int8     int16    int32    int64    uint8    fp16     fp32     fp64     cx64     cx128    bf16 target/input
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #bool
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int8
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int16
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int32
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #int64
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #uint8
        ['bool_', 'int8_', 'int16', 'int32', 'int64', 'uint8', 'fp16_', 'fp32_', 'undef', 'undef', 'undef', 'bf16_'],  #fp16
        ['bool_', 'int8_', 'int16', 'int32', 'int64', 'uint8', 'fp16_', 'fp32_', 'undef', 'undef', 'undef', 'bf16_'],  #fp32
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #fp64
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #cx64
        ['undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef', 'undef'],  #cx128
        ['bool_', 'int8_', 'int16', 'int32', 'int64', 'uint8', 'fp16_', 'fp32_', 'undef', 'undef', 'undef', 'bf16_'],  #bf16
    ]

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    for _, input_dtype in ms_dtype_groups.items():
        for _, target_dtype in ms_dtype_groups.items():
            inputx = ms.Tensor(generate_random_input((2, 2), np.float32), dtype=input_dtype[1])
            target = ms.Tensor(generate_random_input((2, 2), np.float32), dtype=target_dtype[1])
            if out_dtype[input_dtype[0]][target_dtype[0]] == 'undef':
                with pytest.raises(Exception):
                    output = l1_loss_forward_func(inputx, target, reduction='none')
                    _pynative_executor.sync()
            else:
                try:
                    output = l1_loss_forward_func(inputx, target, reduction='none')
                    assert output.dtype == ms_dtype_groups[out_dtype[input_dtype[0]][target_dtype[0]]][1]
                except Exception as e:
                    print(f'inputx.dtype {inputx.dtype}, target.dtype {target.dtype} -> output.dtype {output.dtype}')
                    print(f'expect output.dtype {ms_dtype_groups[out_dtype[input_dtype[0]][target_dtype[0]]][1]}')
                    raise e

            if mode == "KBK":
                continue
            if grad_input_dtype[input_dtype[0]][target_dtype[0]] == 'undef':
                with pytest.raises(Exception):
                    grad_input, grad_target = l1_loss_backward_func(inputx, target, reduction='none')
                    _pynative_executor.sync()
            else:
                try:
                    grad_input, grad_target = l1_loss_backward_func(inputx, target, reduction='none')
                    expect_grad_input_dtype = ms_dtype_groups[grad_input_dtype[input_dtype[0]][target_dtype[0]]][1]
                    expect_grad_target_dtype = ms_dtype_groups[grad_target_dtype[input_dtype[0]][target_dtype[0]]][1]
                    assert grad_input.dtype == expect_grad_input_dtype
                    assert grad_target.dtype == expect_grad_target_dtype
                except Exception as e:
                    print(f'inputx.dtype {inputx.dtype}, target.dtype {target.dtype} -> '
                          f'grad_input.dtype {grad_input.dtype}, grad_target.dtype {grad_target.dtype}')
                    print(f'expect grad_input.dtype {expect_grad_input_dtype}, '
                          f'expect grad_target.dtype {expect_grad_target_dtype}')
                    raise e
