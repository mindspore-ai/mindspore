# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
import mindspore.nn as nn
import mindspore.context as context


class DTypeNet(nn.Cell):
    def __init__(self):
        super(DTypeNet, self).__init__()
        self.dtype = P.DType()

    def construct(self, x):
        return self.dtype(x)


class DTypeDynamicNet(nn.Cell):
    def __init__(self):
        super(DTypeDynamicNet, self).__init__()
        self.d = inner.GpuConvertToDynamicShape()
        self.dtype = P.DType()

    def construct(self, x):
        x = self.d(x)
        return self.dtype(x)


def dtype_with_testcase(mstype):
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = Tensor(np.arange(34).reshape(2, 17), dtype=mstype)
    net = DTypeNet()
    assert mstype == net(x)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    assert mstype == net(x)


def dtype_dynamic_with_testcase(mstype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = Tensor(np.arange(34).reshape(2, 17), dtype=mstype)
    net = DTypeDynamicNet()
    assert mstype == net(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dtype_bool():
    dtype_with_testcase(ms.bool_)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dtype_int8():
    dtype_with_testcase(ms.int8)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dtype_uint8():
    dtype_with_testcase(ms.uint8)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dtype_int16():
    dtype_with_testcase(ms.int16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dtype_uint16():
    dtype_with_testcase(ms.uint16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dtype_int32():
    dtype_with_testcase(ms.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dtype_int64():
    dtype_with_testcase(ms.int64)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dtype_float16():
    dtype_with_testcase(ms.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dtype_float32():
    dtype_with_testcase(ms.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dtype_float64():
    dtype_with_testcase(ms.float64)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_dtype_bool():
    dtype_dynamic_with_testcase(ms.bool_)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_dtype_int8():
    dtype_dynamic_with_testcase(ms.int8)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_dtype_uint8():
    dtype_dynamic_with_testcase(ms.uint8)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_dtype_int16():
    dtype_dynamic_with_testcase(ms.int16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_dtype_uint16():
    dtype_dynamic_with_testcase(ms.uint16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_dtype_int32():
    dtype_dynamic_with_testcase(ms.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_dtype_int64():
    dtype_dynamic_with_testcase(ms.int64)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_dtype_float16():
    dtype_dynamic_with_testcase(ms.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_dtype_float32():
    dtype_dynamic_with_testcase(ms.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_dtype_float64():
    dtype_dynamic_with_testcase(ms.float64)
