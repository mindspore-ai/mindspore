# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest

from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore import nn


def get_data(dtype):
    input_data = np.array(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]).astype(dtype)
    size = (9, 9)
    expected_output = np.array([[[[0.1, 0.1333, 0.1666, 0.2, 0.2333, 0.2666, 0.3, 0.3, 0.3],
                                  [0.2, 0.2333, 0.2666, 0.2998, 0.3333, 0.3667, 0.4, 0.4, 0.4],
                                  [0.2998, 0.3333, 0.3667, 0.4, 0.433, 0.4666, 0.5, 0.5, 0.5],
                                  [0.4, 0.433, 0.4666, 0.5, 0.533, 0.5664, 0.6, 0.6, 0.6],
                                  [0.5, 0.533, 0.5664, 0.5996, 0.6333, 0.6665, 0.6997, 0.6997, 0.6997],
                                  [0.6, 0.6333, 0.6665, 0.6997, 0.733, 0.766, 0.8, 0.7993, 0.8],
                                  [0.7, 0.7334, 0.7666, 0.8, 0.833, 0.866, 0.9, 0.8994, 0.8994],
                                  [0.7, 0.7334, 0.7666, 0.8, 0.833, 0.866, 0.8994, 0.8994, 0.8994],
                                  [0.7, 0.7334, 0.7666, 0.8, 0.8325, 0.866,
                                   0.8994, 0.8994, 0.8994]]]]).astype(dtype)
    return input_data, size, expected_output


class NetResizeBilinear(nn.Cell):
    def construct(self, inputs, size, indices_input, axis):
        unique_input_index, _ = ops.unique(indices_input)
        inputs_dyn = ops.gather(inputs, unique_input_index, axis)
        return ops.ResizeBilinearV2(align_corners=False, half_pixel_centers=False)(inputs_dyn, size)


def case_input_dyn(mode, device_target, dtype="float32"):
    context.set_context(mode=mode, device_target=device_target)
    input_data, size, expected = get_data(dtype)

    resize_nn = NetResizeBilinear()
    axis_input = 3
    indices_input = np.array([i for i in range(input_data.shape[axis_input])])
    output = resize_nn(Tensor(input_data), size, Tensor(indices_input), axis_input)
    assert np.allclose(output.asnumpy(), expected, 1e-3, 1e-3)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_resize_bilinear_ascend():
    """
    Feature: Test resize_bilinear on ascend.
    Description:  The shape of input is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    case_input_dyn(context.GRAPH_MODE, "Ascend")
    case_input_dyn(context.PYNATIVE_MODE, "Ascend")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resize_bilinear_gpu():
    """
    Feature: Test resize_bilinear on GPU.
    Description:  The shape of input is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    case_input_dyn(context.GRAPH_MODE, "GPU")
    case_input_dyn(context.PYNATIVE_MODE, "GPU")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_resize_bilinear_cpu():
    """
    Feature: Test resize_bilinear on CPU.
    Description:  The shape of input is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    case_input_dyn(context.GRAPH_MODE, "CPU")
    case_input_dyn(context.PYNATIVE_MODE, "CPU")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resize_bilinear_gpu_fp64():
    """
    Feature: Test resize_bilinear on GPU (fp64).
    Description:  The shape of input is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    case_input_dyn(context.GRAPH_MODE, "GPU", "float64")
    case_input_dyn(context.PYNATIVE_MODE, "GPU", "float64")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_resize_bilinear_cpu_fp64():
    """
    Feature: Test resize_bilinear on CPU (fp64).
    Description:  The shape of input is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    case_input_dyn(context.GRAPH_MODE, "CPU", "float64")
    case_input_dyn(context.PYNATIVE_MODE, "CPU", "float64")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resize_bilinear_gpu_fp16():
    """
    Feature: Test resize_bilinear on GPU (fp16).
    Description:  The shape of input is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    case_input_dyn(context.GRAPH_MODE, "GPU", "float16")
    case_input_dyn(context.PYNATIVE_MODE, "GPU", "float16")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_resize_bilinear_cpu_fp16():
    """
    Feature: Test resize_bilinear on CPU (fp16).
    Description:  The shape of input is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    case_input_dyn(context.GRAPH_MODE, "CPU", "float16")
    case_input_dyn(context.PYNATIVE_MODE, "CPU", "float16")
