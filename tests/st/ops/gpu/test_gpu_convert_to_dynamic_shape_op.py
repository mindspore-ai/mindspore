# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

from mindspore import Tensor
from mindspore.ops.operations import _inner_ops as inner
import mindspore.nn as nn
import mindspore.context as context

# test to make sure this op actually generates a dynamically shaped output
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dyanamic_shape_confirm_dynamic():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    class AssertDynamicShapeNet(nn.Cell):
        def __init__(self):
            super(AssertDynamicShapeNet, self).__init__()
            self.gpu_convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()
            self.error_on_dynamic_shape_input = inner.ErrorOnDynamicShapeInput()

        def construct(self, x):
            output = self.gpu_convert_to_dynamic_shape(x)
            self.error_on_dynamic_shape_input(output)
            return output

    assert_dynamic_shape_net = AssertDynamicShapeNet()
    x = Tensor(np.array([0, 0, 0, 0]).astype(np.float32))

    with pytest.raises(ValueError) as info:
        assert_dynamic_shape_net(x)
    assert "Input is dynamically shaped" in str(info.value)

def gpu_convert_to_dynamic_shape(x):
    class GpuConvertToDynamicShapeNet(nn.Cell):
        def __init__(self):
            super(GpuConvertToDynamicShapeNet, self).__init__()
            self.gpu_convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()

        def construct(self, x):
            return self.gpu_convert_to_dynamic_shape(x)

    gpu_convert_to_dynamic_shape_net = GpuConvertToDynamicShapeNet()
    return gpu_convert_to_dynamic_shape_net(Tensor(x)).asnumpy()

def gpu_convert_to_dynamic_shape_float(dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    np.random.seed(0)
    finfo = np.finfo(dtype)

    # np.random.uniform will overflow if we use min/max for float64, so we use
    # the finfo for float32, but still test the operator with float64 input.
    if dtype == np.float64:
        finfo = np.finfo(np.float32)

    float_min = finfo.min
    float_max = finfo.max
    x = np.random.uniform(low=float_min, high=float_max, size=12).astype(dtype)
    ms_out = gpu_convert_to_dynamic_shape(x)
    np.testing.assert_array_equal(x, ms_out)

def gpu_convert_to_dynamic_shape_int(dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    np.random.seed(0)
    iinfo = np.iinfo(dtype)
    int_min = iinfo.min
    int_max = iinfo.max
    x = np.random.uniform(low=int_min, high=int_max, size=12).astype(dtype)
    ms_out = gpu_convert_to_dynamic_shape(x)
    np.testing.assert_array_equal(x, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_bool():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    np.random.seed(0)
    x = np.random.choice([False, True], 12)
    ms_out = gpu_convert_to_dynamic_shape(x)
    np.testing.assert_array_equal(x, ms_out)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_float16():
    gpu_convert_to_dynamic_shape_float(np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_float32():
    gpu_convert_to_dynamic_shape_float(np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_float64():
    gpu_convert_to_dynamic_shape_float(np.float64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_int8():
    gpu_convert_to_dynamic_shape_int(np.int8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_int16():
    gpu_convert_to_dynamic_shape_int(np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_int32():
    gpu_convert_to_dynamic_shape_int(np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_int64():
    gpu_convert_to_dynamic_shape_int(np.int64)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_uint8():
    gpu_convert_to_dynamic_shape_int(np.uint8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_uint16():
    gpu_convert_to_dynamic_shape_int(np.uint16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_uint32():
    gpu_convert_to_dynamic_shape_int(np.uint32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_convert_to_dynamic_shape_uint64():
    gpu_convert_to_dynamic_shape_int(np.uint64)
