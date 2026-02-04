# Copyright 2025 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, context
from mindspore import mint
from mindspore.device_context.cpu.op_tuning import threads_num


@test_utils.run_with_cell
def pixel_shuffle_forward_func(input_x, upscale_factor):
    return mint.nn.functional.pixel_shuffle(input_x, upscale_factor)


@test_utils.run_with_cell
def pixel_shuffle_backward_func(input_x, upscale_factor):
    return ms.grad(pixel_shuffle_forward_func, (0,))(input_x, upscale_factor)


def set_mode(mode):
    """
    set mode
    """
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    elif mode == "GE":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O2"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


def gloden_forward(input_tensor, upscale_factor):
    input_x = input_tensor.asnumpy()
    input_x_shape = input_x.shape
    c = input_x_shape[-3]
    h = input_x_shape[-2]
    w = input_x_shape[-1]
    oc = c // (upscale_factor ** 2)
    oh = h * upscale_factor
    ow = w * upscale_factor

    added_dims_shape = input_x_shape[:-3] + (oc, upscale_factor, upscale_factor, h, w)
    input_reshaped = np.reshape(input_x, added_dims_shape)

    permutation = np.arange(len(input_x_shape) - 3)
    permutation = tuple(permutation) + (-5, -2, -4, -1, -3)
    input_permuted = np.transpose(input_reshaped, permutation)

    final_shape = input_x_shape[:-3] + (oc, oh, ow)
    out = np.reshape(input_permuted, final_shape)
    return out


def generate_inputs(input_shape, upscale_factor):
    input_tensor = Tensor(np.random.randn(*input_shape).astype(np.float32))
    return [input_tensor, upscale_factor]


@arg_mark(plat_marks=['cpu_linux', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize("mode", ["GE", "PYBOOST", "KBK"])
def test_pixel_shuffle_static(mode):
    """
    Feature: PixelShuffle
    Description: test mint.nn.functional.pixel_shuffle
    Expectation: expect correct result.
    """
    set_mode(mode)
    input_tensor, upscale_factor = generate_inputs((2, 3, 4, 9, 3, 3), 3)
    forward_out = pixel_shuffle_forward_func(input_tensor, upscale_factor)
    backward_out = pixel_shuffle_backward_func(input_tensor, upscale_factor)

    expect_forward = gloden_forward(input_tensor, upscale_factor)
    expect_backward = np.ones(input_tensor.shape).astype(np.float32)

    loss = 1e-5
    assert np.allclose(forward_out.asnumpy(), expect_forward, loss, loss)
    assert np.allclose(backward_out.asnumpy(), expect_backward, loss, loss)


@arg_mark(
    plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=1.00
)
def test_pixel_shuffle_dynamic():
    """
    Feature: Ops
    Description: test op pixel_shuffle dynamic shape
    Expectation: expect correct result.
    """
    threads_num(1)
    TEST_OP(
        pixel_shuffle_forward_func,
        [generate_inputs((1, 9, 2, 2), 3),
         generate_inputs((2, 2, 20, 5, 5), 2)],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['ScalarTensor']
    )
