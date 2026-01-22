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
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import Tensor
from mindspore import context, ops
from mindspore.graph.api import _pynative_executor
from mindspore.device_context.cpu.op_tuning import threads_num


def set_mode(mode):
    if mode == "GRAPH_MODE_O0":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    elif mode == "GRAPH_MODE":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O2"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@test_utils.run_with_cell
def upsample_nearest_forward_func(x, size=None, scale_factor=None):
    return ops.function.nn_func.interpolate_ext(x, size, scale_factor, "nearest")


@test_utils.run_with_cell
def upsample_nearest_backward_func(x, size=None, scale_factor=None):
    return ms.grad(upsample_nearest_forward_func, (0,))(x, size, scale_factor)


@test_utils.run_with_cell
def upsample_nearest3d_grad(gradOut, input_size, output_size, scale_factor):
    op = ops.auto_generate.UpsampleNearest3DGrad()
    return op(gradOut, input_size, output_size, scale_factor)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ["GRAPH_MODE_O0", "PYNATIVE_MODE"])
def test_upsample_nearest(mode):
    """
    Feature: UpsampleNearest1D/2D/3D
    Description: Test cases for UpsampleNearest1D/2D/3D with output_size.
    Expectation: The result match expected output.
    """
    set_mode(mode)

    # nearest1d
    input_tensor = Tensor(
        np.array([[[0.1, 0.3, 0.5, 0.7], [0.9, 1.1, 1.3, 1.5]]]).astype(np.float32)
    )
    expected = np.array(
        [
            [
                [0.1000, 0.1000, 0.3000, 0.3000, 0.5000, 0.5000, 0.7000, 0.7000],
                [0.9000, 0.9000, 1.1000, 1.1000, 1.3000, 1.3000, 1.5000, 1.5000],
            ]
        ]
    ).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, (8,), None)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array([[[0.1000, 0.1000, 0.3000, 0.3000,
                           0.5000, 0.5000, 0.7000, 0.7000],
                          [0.9000, 0.9000, 1.1000, 1.1000,
                           1.3000, 1.3000, 1.5000, 1.5000]]]).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, None, (2.,))
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array([[[3.0, 3.0, 3.0, 3.0],
                          [3.0, 3.0, 3.0, 3.0]]]).astype(np.float32)
    out = upsample_nearest_backward_func(input_tensor, None, [3.,])
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    # nearest2d
    input_tensor = Tensor(
        np.array(
            [[[[0.1, 0.3, 0.5], [0.7, 0.9, 1.1]], [[1.3, 1.5, 1.7], [1.9, 2.1, 2.3]]]]
        ).astype(np.float32)
    )

    expected = np.array(
        [[[[0.1000, 0.1000, 0.3000, 0.3000, 0.5000],
           [0.1000, 0.1000, 0.3000, 0.3000, 0.5000],
           [0.7000, 0.7000, 0.9000, 0.9000, 1.1000],
           [0.7000, 0.7000, 0.9000, 0.9000, 1.1000]],
          [[1.3000, 1.3000, 1.5000, 1.5000, 1.7000],
           [1.3000, 1.3000, 1.5000, 1.5000, 1.7000],
           [1.9000, 1.9000, 2.1000, 2.1000, 2.3000],
           [1.9000, 1.9000, 2.1000, 2.1000, 2.3000]]]]).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, (4, 5), None)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array(
        [[[[0.1000, 0.1000, 0.1000, 0.3000, 0.3000, 0.3000, 0.5000, 0.5000, 0.5000],
           [0.1000, 0.1000, 0.1000, 0.3000, 0.3000, 0.3000, 0.5000, 0.5000, 0.5000],
           [0.7000, 0.7000, 0.7000, 0.9000, 0.9000, 0.9000, 1.1000, 1.1000, 1.1000],
           [0.7000, 0.7000, 0.7000, 0.9000, 0.9000, 0.9000, 1.1000, 1.1000, 1.1000]],
          [[1.3000, 1.3000, 1.3000, 1.5000, 1.5000, 1.5000, 1.7000, 1.7000, 1.7000],
           [1.3000, 1.3000, 1.3000, 1.5000, 1.5000, 1.5000, 1.7000, 1.7000, 1.7000],
           [1.9000, 1.9000, 1.9000, 2.1000, 2.1000, 2.1000, 2.3000, 2.3000, 2.3000],
           [1.9000, 1.9000, 1.9000, 2.1000, 2.1000, 2.1000, 2.3000, 2.3000, 2.3000]]]]
    ).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, None, (2., 3.))
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array(
        [[[[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]], [[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]]]]
    ).astype(np.float32)
    out = upsample_nearest_backward_func(input_tensor, None, [3., 2.])
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    input_tensor = Tensor(
        np.array(
            [[[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]]]
        ).astype(np.float32)
    )
    expected = np.array(
        [
            [
                [
                    [
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000],
                    ],
                    [
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000],
                    ],
                    [
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000],
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000],
                    ],
                ]
            ]
        ]
    ).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, [3, 4, 5], None)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array(
        [
            [
                [
                    [
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000],
                    ],
                    [
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000],
                    ],
                    [
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000, 0.9000],
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000, 0.9000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000, 1.2000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000, 1.2000],
                    ],
                    [
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000, 0.9000],
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000, 0.9000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000, 1.2000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000, 1.2000],
                    ],
                ]
            ]
        ]
    ).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, None, [2.0, 2.0, 2.0])
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array(
        [[[[[12.0, 8.0, 8.0], [12.0, 8.0, 8.0]], [[6.0, 4.0, 4.0], [6.0, 4.0, 4.0]]]]]
    ).astype(np.float32)
    out = upsample_nearest_backward_func(input_tensor, None, [1.5, 2.0, 2.5])
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_gpu'],
          level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", ["GRAPH_MODE_O0", "PYNATIVE_MODE"])
def test_upsample_nearest_gpu(mode):
    """
    Feature: UpsampleNearest1D/2D/3D
    Description: Test cases for UpsampleNearest1D/2D/3D with output_size.
    Expectation: The result match expected output.
    """
    set_mode(mode)
    input_tensor = Tensor(
        np.array([[[0.1, 0.3, 0.5, 0.7], [0.9, 1.1, 1.3, 1.5]]]).astype(np.float32)
    )
    expected = np.array(
        [
            [
                [0.1000, 0.1000, 0.3000, 0.3000, 0.5000, 0.5000, 0.7000, 0.7000],
                [0.9000, 0.9000, 1.1000, 1.1000, 1.3000, 1.3000, 1.5000, 1.5000],
            ]
        ]
    ).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, (8,), None)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array([[[0.1000, 0.1000, 0.1000, 0.3000,
                           0.3000, 0.5000, 0.5000, 0.7000, 0.7000],
                          [0.9000, 0.9000, 0.9000, 1.1000,
                           1.1000, 1.3000, 1.3000, 1.5000, 1.5000]]]).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, None, (2.3,))
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array([[[3.0, 2.0, 2.0, 2.0],
                          [3.0, 2.0, 2.0, 2.0]]]).astype(np.float32)
    out = upsample_nearest_backward_func(input_tensor, None, [2.3,])
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    input_tensor = Tensor(
        np.array(
            [[[[0.1, 0.3, 0.5], [0.7, 0.9, 1.1]], [[1.3, 1.5, 1.7], [1.9, 2.1, 2.3]]]]
        ).astype(np.float32)
    )
    expected = np.array(
        [[[[0.1000, 0.1000, 0.3000, 0.3000, 0.5000],
           [0.1000, 0.1000, 0.3000, 0.3000, 0.5000],
           [0.7000, 0.7000, 0.9000, 0.9000, 1.1000],
           [0.7000, 0.7000, 0.9000, 0.9000, 1.1000]],
          [[1.3000, 1.3000, 1.5000, 1.5000, 1.7000],
           [1.3000, 1.3000, 1.5000, 1.5000, 1.7000],
           [1.9000, 1.9000, 2.1000, 2.1000, 2.3000],
           [1.9000, 1.9000, 2.1000, 2.1000, 2.3000]]]]).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, (4, 5), None)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array(
        [[[[0.1000, 0.1000, 0.1000, 0.3000, 0.3000, 0.5000],
           [0.1000, 0.1000, 0.1000, 0.3000, 0.3000, 0.5000],
           [0.7000, 0.7000, 0.7000, 0.9000, 0.9000, 1.1000]],
          [[1.3000, 1.3000, 1.3000, 1.5000, 1.5000, 1.7000],
           [1.3000, 1.3000, 1.3000, 1.5000, 1.5000, 1.7000],
           [1.9000, 1.9000, 1.9000, 2.1000, 2.1000, 2.3000]]]]).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, None, (1.7, 2.3))
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array(
        [[[[6.0, 4.0, 2.0], [3.0, 2.0, 1.0]], [[6.0, 4.0, 2.0], [3.0, 2.0, 1.0]]]]
    ).astype(np.float32)
    out = upsample_nearest_backward_func(input_tensor, None, [1.7, 2.3])
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    input_tensor = Tensor(
        np.array(
            [[[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]]]
        ).astype(np.float32)
    )
    expected = np.array(
        [
            [
                [
                    [
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000],
                    ],
                    [
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000],
                    ],
                    [
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000],
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000],
                    ],
                ]
            ]
        ]
    ).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, [3, 4, 5], None)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array(
        [
            [
                [
                    [
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000],
                    ],
                    [
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
                        [0.1000, 0.1000, 0.2000, 0.2000, 0.3000, 0.3000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000],
                        [0.4000, 0.4000, 0.5000, 0.5000, 0.6000, 0.6000],
                    ],
                    [
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000, 0.9000],
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000, 0.9000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000, 1.2000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000, 1.2000],
                    ],
                    [
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000, 0.9000],
                        [0.7000, 0.7000, 0.8000, 0.8000, 0.9000, 0.9000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000, 1.2000],
                        [1.0000, 1.0000, 1.1000, 1.1000, 1.2000, 1.2000],
                    ],
                ]
            ]
        ]
    ).astype(np.float32)
    out = upsample_nearest_forward_func(input_tensor, None, [2.0, 2.0, 2.0])
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array(
        [[[[[12.0, 8.0, 8.0], [12.0, 8.0, 8.0]], [[6.0, 4.0, 4.0], [6.0, 4.0, 4.0]]]]]
    ).astype(np.float32)
    out = upsample_nearest_backward_func(input_tensor, None, [1.5, 2.0, 2.5])
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux'],
          level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_upsample_nearest_1d_size_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleNearest1D and UpsampleNearest1DGrad.
    Expectation: expect UpsampleNearest1D and UpsampleNearest1DGrad result.
    """
    threads_num(1)  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15), dtype=ms.float32)
    TEST_OP(
        upsample_nearest_forward_func,
        [
            [input_case1, (100,), None],
            [input_case2, (40,), None],
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True}
    )


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux'],
          level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_upsample_nearest_1d_scale_factor_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleNearest1D and UpsampleNearest1DGrad.
    Expectation: expect UpsampleNearest1D and UpsampleNearest1DGrad result.
    """
    threads_num(1)  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15), dtype=ms.float32)
    TEST_OP(
        upsample_nearest_forward_func,
        [
            [input_case1, None, (1.5,)],
            [input_case2, None, (2.1,)],
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True}
    )


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux'],
          level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_upsample_nearest_2d_size_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleNearest2D and UpsampleNearest2DGrad.
    Expectation: expect UpsampleNearest2D and UpsampleNearest2DGrad result.
    """
    threads_num(1)  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60, 40), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15, 30), dtype=ms.float32)
    TEST_OP(
        upsample_nearest_forward_func,
        [
            [input_case1, (100, 80), None],
            [input_case2, (40, 60), None],
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True}
    )


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux'],
          level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_upsample_nearest_2d_scale_factor_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleNearest2D and UpsampleNearest2DGrad.
    Expectation: expect UpsampleNearest2D and UpsampleNearest2DGrad result.
    """
    threads_num(1)  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60, 80), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15, 30), dtype=ms.float32)
    TEST_OP(
        upsample_nearest_forward_func,
        [
            [input_case1, None, (1.5, 1.7)],
            [input_case2, None, (2.1, 2.8)],
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True}
    )


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_upsample_nearest_3d_size_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleNearest3D and UpsampleNearest3DGrad.
    Expectation: expect UpsampleNearest3D and UpsampleNearest3DGrad result.
    """
    threads_num(1)  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60, 30, 128), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15, 10, 64), dtype=ms.float32)
    TEST_OP(
        upsample_nearest_forward_func,
        [
            [input_case1, (100, 200, 300), None],
            [input_case2, (40, 80, 80), None],
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True}
    )


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_upsample_nearest_3d_scale_factor_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleNearest3D and UpsampleNearest3DGrad.
    Expectation: expect UpsampleNearest3D and UpsampleNearest3DGrad result.
    """
    threads_num(1)  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60, 30, 128), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15, 10, 64), dtype=ms.float32)
    TEST_OP(
        upsample_nearest_forward_func,
        [
            [input_case1, None, (1.5, 1.6, 1.7)],
            [input_case2, None, (2.1, 2.5, 3.5)],
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True}
    )


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_upsample_nearest_3d_error(mode):
    """
    Feature: UpsampleNearest3D
    Description: Test cases for UpsampleNearest3D operator with errors.
    Expectation: Raise expected error type.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = upsample_nearest_forward_func

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, [3, 4, 5], None)
        _pynative_executor.sync()

    with pytest.raises(TypeError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.int32))
        net(input_tensor, [3, 4, 5], None)
        _pynative_executor.sync()

    with pytest.raises(TypeError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, None, [1, 2, 3])
        _pynative_executor.sync()

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, [3, 4], None)
        _pynative_executor.sync()

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, None, [1.0, 2.0, 3.0, 4.0])
        _pynative_executor.sync()

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, [3, 4, 5], [1.0, 2.0, 3.0])
        _pynative_executor.sync()

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, None, None)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", ["GRAPH_MODE", "GRAPH_MODE_O0", "PYNATIVE_MODE"])
def test_vmap_upsample_nearest3d(mode):
    """
    Feature:  UpsampleNearest3D GPU op vmap feature.
    Description: test the vmap feature of UpsampleNearest3D.
    Expectation: success.
    """
    set_mode(mode)
    # 3 batches
    input_tensor = Tensor(
        np.arange(0, 4.8, 0.1).reshape([3, 1, 1, 2, 2, 4]).astype(np.float32)
    )
    expect = np.array(
        [
            [
                [
                    [
                        [[0.0, 0.2], [0.4, 0.6]],
                        [[0.0, 0.2], [0.4, 0.6]],
                        [[0.8, 1.0], [1.2, 1.4]],
                    ]
                ]
            ],
            [
                [
                    [
                        [[1.6, 1.8], [2.0, 2.2]],
                        [[1.6, 1.8], [2.0, 2.2]],
                        [[2.4, 2.6], [2.8, 3.0]],
                    ]
                ]
            ],
            [
                [
                    [
                        [[3.2, 3.4], [3.6, 3.8]],
                        [[3.2, 3.4], [3.6, 3.8]],
                        [[4.0, 4.2], [4.4, 4.6]],
                    ]
                ]
            ],
        ]
    )
    out_vmap = ops.vmap(upsample_nearest_forward_func, in_axes=(0, None, None))(
        input_tensor, [3, 2, 2], None
    )
    error = np.ones(shape=expect.shape) * 1.0e-4
    assert np.all(abs(out_vmap.asnumpy() - expect) < error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", ["GRAPH_MODE", "GRAPH_MODE_O0", "PYNATIVE_MODE"])
def test_vmap_upsample_nearest3d_grad(mode):
    """
    Feature:  UpsampleNearest3DGrad GPU op vmap feature.
    Description: test the vmap feature of UpsampleNearest3DGrad.
    Expectation: success.
    """
    set_mode(mode)
    # 3 batches
    input_size = (1, 1, 2, 2, 2)
    gradOut = Tensor(
        np.arange(0, 7.2, 0.1).reshape((3, 1, 1, 2, 3, 4)).astype(np.float32)
    )
    net = upsample_nearest3d_grad
    expect = np.array(
        [
            [[[[[1.0, 1.8], [1.7, 2.1]], [[5.8, 6.6], [4.1, 4.5]]]]],
            [[[[[10.6, 11.4], [6.5, 6.9]], [[15.4, 16.2], [8.9, 9.299999]]]]],
            [
                [
                    [
                        [[20.2, 21.0], [11.299999, 11.700001]],
                        [[25.0, 25.8], [13.700001, 14.1]],
                    ]
                ]
            ],
        ]
    ).astype(np.float32)
    out_vmap = ops.vmap(net, in_axes=(0, None, None, None))(
        gradOut, input_size, [2, 3, 4], None
    )
    error = np.ones(shape=expect.shape) * 1.0e-4
    assert np.all(abs(out_vmap.asnumpy() - expect) < error)
