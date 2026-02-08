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
from mindspore.ops import GradOperation
from mindspore import Tensor, nn
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def _assert_equals(result: Tensor, slf: Tensor, exp: Tensor, ms_dtype):
    assert result.dtype == ms_dtype
    assert np.array_equal(result.asnumpy(), exp.asnumpy())
    # inplace operation: self and returns should be the same tensor
    assert np.array_equal(slf.asnumpy(), exp.asnumpy())
    result[0, 0, 0] += np.random.rand() * 8
    assert result[0, 0, 0] == slf[0, 0, 0]


@test_utils.run_with_cell
def scatter_src(x, dim, index, src, reduce):
    return x.scatter_(dim=dim, index=index, src=src, reduce=reduce)


@test_utils.run_with_cell
def scatter_src_with_grad(x, dim, index, src, reduce='none'):
    return (x * Tensor(1, dtype=x.dtype)).scatter_(
        dim=dim, index=index, src=src,
        **(dict(reduce=reduce) if reduce != 'none' else {})
    )


@test_utils.run_with_cell
def scatter_val(x, dim, index, value, reduce):
    return x.scatter_(dim=dim, index=index, value=value, reduce=reduce)


@test_utils.run_with_cell
def scatter_val_with_grad(x, dim, index, value, reduce='none'):
    return (x * Tensor(1, dtype=x.dtype)).scatter_(
        dim=dim, index=index, value=value,
        **(dict(reduce=reduce) if reduce != 'none' else {})
    )


class ScatterGrad(nn.Cell):
    def __init__(self, net: nn.Cell, sens: Tensor):
        super().__init__()
        self.net = net
        self.grad_op = GradOperation(get_all=True, sens_param=True)
        self.grad_wrt_output = sens

    def construct(self, x, dim, index, src_or_val, reduce):
        return self.grad_op(self.net)(x, dim, index, src_or_val, reduce, self.grad_wrt_output)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_scatter_src(mode):
    """
    Feature: Tensor ops.
    Description: test op scatter_ with tensor input.
    Expectation: expect correct result.
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode, device_target="Ascend")

    _test_inplace_scatter_src_main(ms.float32)
    _test_forbid_manual_none("Src")

    _test_inplace_scatter_src_backward1(ms.float32)
    _test_inplace_scatter_src_backward2(ms.float32)
    _test_inplace_scatter_reduce_src_backward()


def _test_inplace_scatter_src_main(input_type):
    ## forward with multiply
    slf = Tensor([[[2] * 5] * 4] * 3, dtype=input_type)
    src = Tensor(np.random.rand(3, 4, 5), dtype=input_type)
    index = Tensor(np.array([[list(range(5))] * 4] * 3, dtype=np.int64))

    result = Tensor.scatter_(slf, dim=2, index=index, src=src, reduce="multiply")
    _assert_equals(result, slf, src * 2, input_type)

    ### directly use a tensor with `shape == []` should be run as using src, not as value
    try:
        Tensor([[[2] * 5] * 4] * 3, dtype=input_type).scatter_(2, index, Tensor(1, dtype=input_type))
    except RuntimeError:
        pass
    else:
        assert False, "Should reject this case because rank(index) != rank(src)"

    src = Tensor(1, dtype=input_type)
    result = Tensor.scatter_(src, 0, Tensor(0), Tensor(-1, dtype=input_type))
    assert src == Tensor(-1, dtype=input_type)
    assert result == src
    result.scatter_(0, Tensor(0), Tensor(1, dtype=input_type), reduce="add")
    assert src == Tensor(0, dtype=input_type)


def _test_inplace_scatter_src_backward1(input_type):
    ## inplace backward
    slf = Tensor([[2] * 4] * 3, dtype=input_type)
    src = Tensor(np.random.rand(3, 4), dtype=input_type)
    index = Tensor(np.array([list(range(4))] * 3, dtype=np.int64))
    grad = Tensor(np.random.rand(3, 4), dtype=input_type)
    grads = ScatterGrad(scatter_src_with_grad, grad)(slf, 1, index, src, 'none')
    assert np.allclose(grads[0].asnumpy(), np.zeros((3, 4), dtype=np.float32))  # self
    # grads[1] is index which has no grad
    assert np.allclose(grads[2].asnumpy().astype(np.float32), grad.asnumpy().astype(np.float32))  # src


def _test_inplace_scatter_src_backward2(input_type):
    ## inplace backward
    slf = Tensor([[2] * 4] * 3, dtype=input_type)
    src = Tensor(np.random.rand(3, 4), dtype=input_type)
    index = Tensor(np.array([list(range(3)) + [2]] * 3, dtype=np.int64))  # slf[:, 3] is reserved
    grad = Tensor(np.random.rand(3, 4), dtype=input_type)
    grad_np = grad.asnumpy().copy().astype(np.float32)
    grads = ScatterGrad(scatter_src_with_grad, grad)(slf, 1, index, src, 'none')
    grad_np[:, :3] = 0
    # grad of src[:, 3] is uncertain, only assert self
    assert np.allclose(grads[0].asnumpy().astype(np.float32), grad_np)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_scatter_value(mode):
    """
    Feature: Tensor ops.
    Description: test op scatter_ with value (scalar/imm) input.
    Expectation: expect correct result.
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode, device_target="Ascend")

    _test_inplace_scatter_value_main(ms.float32)
    _test_forbid_manual_none("Value")

    _test_inplace_scatter_value_backward(ms.float32)
    _test_inplace_scatter_reduce_value_backward()


def _test_inplace_scatter_value_main(input_type):
    ## forward with no_reduce/add/multiply
    np_slf = np.random.rand(3, 4, 5)
    origin_self = Tensor(np_slf.copy(), dtype=input_type)
    index = Tensor(np.array([[list(range(5))] * 4] * 3, dtype=np.int64))

    ### run without `value=` is ok if val is an imm
    slf = Tensor(np_slf.copy(), dtype=input_type)
    result = Tensor.scatter_(slf, 2, index, 2, reduce="add")
    _assert_equals(result, slf, origin_self + 2, input_type)

    ### run with `value=` is ok if val is a Tensor with shape == [], and allows self.dtype != value.dtype
    result = origin_self.scatter_(2, index, value=Tensor(2, dtype=ms.float64), reduce="multiply")
    _assert_equals(result, origin_self, Tensor(np_slf * 2, dtype=input_type), input_type)


def _test_inplace_scatter_value_backward(input_type):
    ## inplace backward
    slf = Tensor([[2] * 4] * 3, dtype=input_type)
    value = np.random.rand() * 10
    index = Tensor(np.array([list(range(3)) + [2]] * 3, dtype=np.int64))  # slf[:, 3] is reserved
    grad = Tensor(np.random.rand(3, 4), dtype=input_type)
    grad_np = grad.asnumpy().copy().astype(np.float32)
    grads = ScatterGrad(scatter_val_with_grad, grad)(slf, 1, index, value, 'none')
    # only self has grad
    grad_np[:, :3] = 0
    assert np.allclose(grads[0].asnumpy().astype(np.float32), grad_np)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scatter_bfloat16(mode):
    """
    Feature: Tensor ops.
    Description: test op scatter_.
    Expectation: expect correct result.
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode, device_target="Ascend")

    _test_inplace_scatter_src_main(ms.bfloat16)
    _test_inplace_scatter_value_main(ms.bfloat16)

    _test_inplace_scatter_src_backward1(ms.bfloat16)
    _test_inplace_scatter_value_backward(ms.bfloat16)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_inplace_scatter_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test ops.scatter_ dynamic shape feature.
    Expectation: expect correct result.
    """
    x1 = Tensor(np.array([[[[0.6777, -3.8882, 1.4999, 2.4321]]]], dtype=np.float32))
    dim1 = 3
    index1 = Tensor(np.array([[[[1, 3]]]], dtype=np.int64))
    src1 = Tensor(np.array([[[[-2, -3]]]], dtype=np.float32))

    x2 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    dim2 = 0
    index2 = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]], dtype=np.int64))
    src2 = Tensor(np.array([[1.2333, 2.6667, 3], [4.8, 5.12, -6.5536], [3.59, 7.87, -0.919]], dtype=np.float32))
    TEST_OP(
        scatter_src,
        [
            [x1, dim1, index1, src1, "add"],
            [x2, dim2, index2, src2, "add"],
        ],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['ScalarTensor'],
        case_config={'disable_input_check': True,
                     'disable_grad': True},
        inplace_update=True
    )
    TEST_OP(
        scatter_src_with_grad,
        [
            [x1, dim1, index1, src1],
            [x2, dim2, index2, src2],
        ],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['ScalarTensor'],
        inplace_update=True,
    )
    TEST_OP(
        scatter_val,
        [
            [x1, dim1, index1, 1, "multiply"],
            [x2, dim2, index2, 2, "multiply"],
        ],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['ScalarTensor'],
        case_config={'disable_input_check': True,
                     'disable_grad': True},
        inplace_update=True
    )
    TEST_OP(
        scatter_val_with_grad,
        [
            [x1, dim1, index1, 1],
            [x2, dim2, index2, 2],
        ],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['ScalarTensor'],
        inplace_update=True,
    )


def _test_forbid_manual_none(key: str):
    err = f"For InplaceScatter{key}Reduce, reduce must be either 'add' or 'multiply', but got: 'none'."
    with pytest.raises(ValueError, match=err):
        Tensor(1).scatter_(0, Tensor(0), reduce='none', **{key.lower(): Tensor(0)})


def _test_inplace_scatter_reduce_src_backward():
    slf = Tensor([[2] * 4] * 3, dtype=ms.float32)
    src = Tensor(np.random.rand(3, 4), dtype=ms.float32)
    index = Tensor(np.array([list(range(4))] * 3, dtype=np.int64))
    grad = Tensor(np.random.rand(3, 4), dtype=ms.float32)
    grads = ScatterGrad(scatter_src_with_grad, grad)(slf, 1, index, src, 'add')
    assert np.allclose(grads[0].asnumpy(), np.zeros((3, 4), dtype=np.float32))  # self
    assert np.allclose(grads[2].asnumpy().astype(np.float32), np.zeros((3, 4)))  # src


def _test_inplace_scatter_reduce_value_backward():
    slf = Tensor([[2] * 4] * 3, dtype=ms.float32)
    index = Tensor(np.array([list(range(4))] * 3, dtype=np.int64))
    grad = Tensor(np.random.rand(3, 4), dtype=ms.float32)
    grads = ScatterGrad(scatter_val_with_grad, grad)(slf, 1, index, 3, 'add')
    assert np.allclose(grads[0].asnumpy(), np.zeros((3, 4), dtype=np.float32))  # self
