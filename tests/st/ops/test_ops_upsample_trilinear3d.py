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
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import Tensor
from mindspore import ops, context, mint


@test_utils.run_with_cell
def upsample_trilinear3d_forward_func(x, size=None, scale_factor=None, align_corners=False):
    return mint.nn.functional.interpolate(x, size, scale_factor, "trilinear", align_corners)


@test_utils.run_with_cell
def upsample_trilinear3d_backward_func(x, size=None, scale_factor=None, align_corners=False):
    return ops.grad(upsample_trilinear3d_forward_func, (0,))(x, size, scale_factor, align_corners)


@test_utils.run_with_cell
def upsample_trilinear3d_grad(gradOut, input_size, output_size, scale_factor):
    op = ops.auto_generate.UpsampleTrilinear3DGrad(False)
    return op(gradOut, input_size, output_size, scale_factor)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_upsample_trilinear_3d(mode):
    """
    Feature: test ops.
    Description: test op UpsampleTrillinear3D.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_tensor = Tensor(
        np.array([[[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]]]).astype(np.float32))
    expected = np.array(
        [[[[[0.1, 0.11, 0.15, 0.19, 0.23000002, 0.27, 0.3],
            [0.175, 0.185, 0.225, 0.265, 0.305, 0.34500003, 0.375],
            [0.32500002, 0.335, 0.37499997, 0.41500002, 0.45500004, 0.49500003, 0.52500004],
            [0.4, 0.41, 0.45, 0.49, 0.53000003, 0.57000005, 0.6]],
           [[0.4, 0.41, 0.45, 0.49, 0.53000003, 0.57, 0.6],
            [0.475, 0.48499998, 0.525, 0.565, 0.605, 0.64500004, 0.675],
            [0.625, 0.635, 0.67499995, 0.71500003, 0.755, 0.795, 0.82500005],
            [0.7, 0.71, 0.75, 0.79, 0.83000004, 0.87000006, 0.90000004]],
           [[0.7, 0.71, 0.75, 0.79, 0.83000004, 0.87, 0.9],
            [0.775, 0.78499997, 0.825, 0.865, 0.90500003, 0.94500005, 0.975],
            [0.925, 0.935, 0.97499996, 1.015, 1.055, 1.095, 1.125],
            [1., 1.01, 1.05, 1.09, 1.13, 1.1700001, 1.2]]]]]).astype(np.float32)
    out = upsample_trilinear3d_forward_func(input_tensor, None, [1.5, 2.0, 2.5], False)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array([[[[[7.5, 7.5, 6.], [7.5, 7.5, 6.]],
                           [[7.5, 7.5, 6.], [7.5, 7.5, 6.]]]]]).astype(np.float32)
    out = upsample_trilinear3d_backward_func(input_tensor, None, [1.5, 2.0, 2.5], False)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_upsample_trilinear_3d_size_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleTrillinear3D and UpsampleTrillinear3DGrad.
    Expectation: expect UpsampleTrillinear3D and UpsampleTrillinear3DGrad result.
    """
    ms.context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60, 30, 128), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15, 10, 64), dtype=ms.float32)
    TEST_OP(
        upsample_trilinear3d_forward_func,
        [
            [input_case1, (100, 200, 300), None, True],
            [input_case2, (40, 80, 80), None, False],
        ],
        'upsample_trilinear3d', disable_input_check=True
    )


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_upsample_trilinear_3d_scale_factor_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleTrillinear3D and UpsampleTrillinear3DGrad.
    Expectation: expect UpsampleTrillinear3D and UpsampleTrillinear3DGrad result.
    """
    ms.context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60, 30, 128), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15, 10, 64), dtype=ms.float32)
    TEST_OP(
        upsample_trilinear3d_forward_func,
        [
            [input_case1, None, 1.7, True],
            [input_case2, None, 3.1, False],
        ],
        'upsample_trilinear3d', disable_input_check=True
    )


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_vmap_upsample_trilinear_3d(mode):
    """
    Feature:  UpsampleTrilinear3D vmap feature.
    Description: test the vmap feature of UpsampleTrilinear3D.
    Expectation: success.
    """
    # 3 batches
    context.set_context(mode=mode)
    net = upsample_trilinear3d_forward_func
    expect = np.array([[[[[[0.0, 0.3], [0.4, 0.7]],
                          [[0.4, 0.70000005], [0.8, 1.1]],
                          [[0.8, 1.1], [1.2, 1.5]]]]],
                       [[[[[1.6, 1.9], [2.0, 2.3]],
                          [[2.0, 2.3], [2.4, 2.6999998]],
                          [[2.4, 2.7], [2.8, 3.1]]]]],
                       [[[[[3.2, 3.5], [3.6, 3.9]],
                          [[3.6, 3.9], [4.0, 4.3]],
                          [[4.0, 4.3], [4.4, 4.7]]]]]])
    x = Tensor(np.arange(0, 4.8, 0.1).reshape([3, 1, 1, 2, 2, 4]).astype(np.float32))
    out_vmap = ops.vmap(net, in_axes=(0, None, None, None))(x, [3, 2, 2], None, True)
    error = np.ones(shape=expect.shape) * 1.0e-4
    assert np.all(abs(out_vmap.asnumpy() - expect) < error)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_vmap_upsample_trilinear_3d_grad(mode):
    """
    Feature:  UpsampleTrilinear3DGrad vmap feature.
    Description: test the vmap feature of UpsampleTrilinear3DGrad.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_size = (1, 1, 2, 2, 2)
    output_size = (2, 3, 3)
    gradOut = Tensor(np.arange(0, 5.4, 0.1).reshape((3, 1, 1, 2, 3, 3)).astype(np.float32))
    net = upsample_trilinear3d_grad
    expect = np.array([[[[[[0.3, 0.6], [1.2, 1.5]],
                          [[2.325, 2.6250002], [3.2250001, 3.5250003]]]]],
                       [[[[[4.35, 4.65], [5.25, 5.5499997]],
                          [[6.375, 6.675], [7.275, 7.575]]]]],
                       [[[[[8.4, 8.700001], [9.299999, 9.599999]],
                          [[10.425001, 10.725],
                           [11.325, 11.625001]]]]]]).astype(np.float32)
    out_vmap = ops.vmap(net, in_axes=(0, None, None, None))(gradOut, input_size, output_size, None)
    error = np.ones(shape=expect.shape) * 1.0e-4
    assert np.all(abs(out_vmap.asnumpy() - expect) < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_upsample_trilinear_3d_error(mode):
    """
    Feature: UpsampleTrilinear3D
    Description: Test cases for UpsampleTrilinear3D operator with errors.
    Expectation: Raise expected error type.
    """
    context.set_context(mode=mode, device_target="CPU")
    net = upsample_trilinear3d_forward_func

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, [3, 4, 5], None)

    with pytest.raises(TypeError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.int32))
        net(input_tensor, [3, 4, 5], None)

    with pytest.raises(TypeError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, None, [1, 2, 3])

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, [3, 4], None)

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, None, [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, [3, 4, 5], [1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        input_tensor = Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
        net(input_tensor, None, None)
