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


# If there's a triple in the comment, it indicates the value of conditions in
# the grad function when x means not affected (for white box test). They means:
# 0. If the forward output is an empty Tensor.
# 1. If len(repeats) > self.rank (output tensor has broadcasted dimensions);
# 2. if not all values in repeats[-self.rank:] are 1 (should reduce the grad in
# the existing dimensions, this is impossible for a scalar input tensor);
# Only 2 cases test a scalar as self Tensor because others are impossible:
# [1, x, x]: scalar -> empty tensor
# [0, 1, 0]: scalar -> nonempty tensor
#
# A `(max)` in comment means this case reaches the maximum number of supported
# dimensions after a reshape during calculating (when using Ascend).



@test_utils.run_with_cell
def repeat_positional(x: Tensor, repeats: tuple):
    return x.repeat(repeats)


@test_utils.run_with_cell
def repeat_named(x: Tensor, repeats: tuple):
    return x.repeat(repeats=repeats)


@test_utils.run_with_cell
def repeat_args(x: Tensor, repeats):
    return x.repeat(*repeats)


@test_utils.run_with_cell
def repeat_args_for_dyn(x: Tensor, *repeats):
    return x.repeat(*repeats)


class RepeatGrad(nn.Cell):
    def __init__(self, net: nn.Cell, sens: Tensor):
        super().__init__()
        self.net = net
        self.grad_op = GradOperation(sens_param=True)
        self.grad_wrt_output = sens

    def construct(self, *args):
        return self.grad_op(self.net)(*args, self.grad_wrt_output)


def _test_container(ms_type, container, op):
    x = Tensor([[0, 1], [2, 3]], dtype=ms_type)
    repeats = container([2, 3, 4])
    y: Tensor = op(x, repeats)
    expect = np.array([[[0, 1] * 4, [2, 3] * 4] * 3] * 2, dtype=np.float32)
    assert np.allclose(y.numpy().astype(np.float32), expect)


def _test_repeat_forward_main(ms_type):
    # list + positional
    _test_container(ms_type, list, repeat_positional)
    # tuple + named
    _test_container(ms_type, tuple, repeat_named)
    # by *args
    _test_container(ms_type, tuple, repeat_args)
    # one int
    x0 = np.random.rand()
    x = Tensor([x0], dtype=ms_type)
    y: Tensor = repeat_positional(x, 3)
    expect = np.array([x0] * 3, dtype=np.float32)
    assert np.allclose(y.numpy().astype(np.float32), expect)


def _mul(vals):
    ret = 1
    for v in vals:
        ret *= v
    return ret


def _test_repeat_backward_main(ms_type,
                               x_shape=(2, 3),
                               repeats=(2, 1, 3),
                               grad_shape=(2, 2, 9),
                               grad_reshape=(2, 1, 1, 2, 3, 3),
                               grad_reduce=(0, 2, 4)):
    # the default args provide a grad test for [0, 1, 1]
    x = Tensor(np.random.rand(*x_shape), dtype=ms_type)
    grad = Tensor(np.arange(_mul(grad_shape)).reshape(*grad_shape), dtype=ms_type)
    grad_x = RepeatGrad(repeat_positional, grad)(x, repeats)
    expected = np.arange(_mul(grad_shape)).reshape(grad_reshape).astype(np.float32)
    if grad_reduce:
        expected = expected.sum(grad_reduce).astype(np.float32)
    assert np.allclose(grad_x.asnumpy().astype(np.float32), expected)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize(
    'mode, level',
    [
        (ms.GRAPH_MODE, 'O0'),
        (ms.GRAPH_MODE, 'O2'),
        (ms.PYNATIVE_MODE, 'O0'),
    ]
)
def test_repeat(mode, level):
    """
    Feature: Tensor ops.
    Description: test op repeat which is like numpy.tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode, jit_level=level, device_target="Ascend")

    _test_repeat_forward_main(ms.float32)
    _test_repeat_backward_main(ms.float32)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_repeat_bfloat16(mode):
    """
    Feature: Tensor ops.
    Description: test op repeat supports bfloat16 in forward/backward.
    Expectation: expect correct result.
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode, device_target="Ascend")

    _test_container(ms.bfloat16, list, repeat_positional)
    # other tests are at test_repeat
    # one int
    x0 = np.random.rand()
    x = Tensor([x0], dtype=ms.bfloat16)
    y: Tensor = repeat_positional(x, 3)
    expect = Tensor([x0] * 3, dtype=ms.bfloat16).numpy().astype(np.float32)
    assert np.allclose(y.numpy().astype(np.float32), expect)
    # backward
    _test_repeat_backward_main(ms.bfloat16)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize(
    'mode, level',
    [
        (ms.GRAPH_MODE, 'O0'),
        (ms.GRAPH_MODE, 'O2'),
        (ms.PYNATIVE_MODE, 'O0'),
    ]
)
def test_repeat_grad_extra(mode, level):
    """
    Feature: Tensor ops.
    Description: test grad calculation of op repeat which is like numpy.tile.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode, jit_level=level, device_target="Ascend")

    # no extended dims, no reshape/reduce, x.grad == grad  [0, 0, 0]
    _test_repeat_backward_main(ms.float32, (2,) * 8, (1,) * 8, (2,) * 8, (2,) * 8, ())

    # no extended dims, reshape to rank=8, and 4 reduce dims after reshape  [0, 0, 1]  (max)
    _test_repeat_backward_main(ms.float32, (2,) * 4, (3,) * 4, (6,) * 4, (3, 2) * 4, (0, 2, 4, 6))

    # scalar in, extends/reduce all  [0, 1, 0]
    _test_repeat_backward_main(ms.float32, (), [2, 3, 4], [2, 3, 4], [24], (0,))
    # 4 extended dims, no reshape/reduce after reshape  [0, 1, 0]
    _test_repeat_backward_main(ms.float32, (2,) * 4, (2,) * 4 + (1,) * 4, (2,) * 8, (2,) * 8, (0, 1, 2, 3))

    # 7 extended dims, reshape to rank=2, and 1 reduce dim after reshape  [0, 1, 1]
    _test_repeat_backward_main(ms.float32, (3,), (2,) * 8, (2,) * 7 + (6,), (2,) * 8 + (3,), tuple(range(8)))
    # 4 extended dims, reshape to rank=8, and 4 reduce dims (all existing dims) after reshape  [0, 1, 1]  (max)
    _test_repeat_backward_main(ms.float32, (3,) * 4, (2,) * 8, (2,) * 4 + (6,) * 4,
                               (2,) * 4 + (2, 3) * 4,
                               (0, 1, 2, 3, 4, 6, 8, 10))
    # 4 extended dims, reshape to rank=6, and 2 reduce dims after reshape  [0, 1, 1]
    _test_repeat_backward_main(ms.float32, (3,) * 4, (2,) * 4 + (2, 1, 2, 1),
                               (2,) * 4 + (6, 3, 6, 3),
                               (2,) * 4 + (2, 3, 3) * 2,
                               (0, 1, 2, 3, 4, 7))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_repeat_dynamic_classic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test Tensor.repeat dynamic shape feature.
    Expectation: expect correct result.
    """
    x1 = Tensor(np.array([[[[0.6777, -3.8882, 1.4999, 2.4321]]]], dtype=np.float32))
    repeats1 = tuple(range(2, 8))  # len = 6
    # x1 + repeats1: 2 extended dims, 4 existing dims and reduce 4  [0, 1, 1]  (max)
    x2 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    repeats2 = tuple(range(3, 7))  # len = 4
    # x2 + repeats2: 2 extended dims, 2 existing dims and reduce 2  [0, 1, 1]
    TEST_OP(
        repeat_positional,
        [
            [x1, repeats1],
            [x2, repeats2],
        ],
        disable_mode=['GRAPH_MODE_GE'],   # not support yet
    )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.00)
def test_repeat_dynamic_bprop():
    """
    Feature: test dynamic backward by TEST_OP.
    Description: test Tensor.repeat dynamic shape bprop function (white box).
    Expectation: expect correct result.
    """
    # grad enhanced check
    # cases only records self.shape and repeats. Yaml check is skipped.
    # Each triple in the comment indicates the value of conditions in the grad function when x means not affected.
    # Only 2 cases test a scalar as self Tensor because others are impossible.
    cases = [
        (
            repeat_positional,
            [[], [2, 3, 0, 4, 5]],  # scalar to empty tensor  [1, x, x]  (scalar)
            [[2, 3, 4, 5], [1, 1, 1, 1]],  # no extended dim, no reshape/reduce  [0, 0, 0]
        ),
        (
            repeat_named,
            [[1, 1, 4], [2, 3, 4, 5, 0]],  # output is empty tensor (by repeats have 0)  [1, x, x]
            [[2, 3, 4, 5], [2, 2, 2, 2]],  # no extended dim, reshape and reduce for all dim  [0, 0, 1]  (max)
        ),
        (
            repeat_positional,
            [[3, 3, 0], list(range(2, 6))],  # output is empty tensor (by self.shape have 0)  [1, x, x]
            [[2, 3, 4, 5], [2, 3, 1, 1, 1, 1]],  # 2 extended dims, no reshape/reduce  [0, 1, 0]
        ),
        (
            repeat_named,
            [[], [2, 3, 4]],  # scalar and should reduce all dims  [0, 1, 0]  (scalar)
            [[2, 3, 4, 5], [2, 3, 1, 2, 1, 2]],  # 2 extended dims, 4 existing dims and reduce 2  [0, 1, 1]
        ),
    ]
    for index, testcase in enumerate(cases):
        func = testcase[0]
        x3_size, repeats3 = testcase[1]
        x4_size, repeats4 = testcase[2]
        x3 = Tensor(np.random.rand(*x3_size), dtype=ms.float32)
        x4 = Tensor(np.random.rand(*x4_size), dtype=ms.float32)

        disable_generalize = False
        if index == 2:
            disable_generalize = True

        TEST_OP(func, [[x3, repeats3], [x4, repeats4]], disable_generalize_test=disable_generalize,
                disable_mode=['GRAPH_MODE_GE'])
