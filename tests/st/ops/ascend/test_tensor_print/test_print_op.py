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

import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, ops
from tests.st.utils import test_utils


class PrintNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.print = ops.Print()

    def construct(self, x):
        self.print("scalar int:", 2, "scalar float:", 1.2, "scalar bool:", False, "Tensor type:", x)
        return x


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                   np.uint32, np.uint64, np.bool, np.float64, np.float32, np.float16])
def test_print_op_dtype(mode, dtype):
    """
    Feature: cpu Print ops.
    Description: test print with the different types.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="Ascend")

    net = PrintNet()
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]).astype(dtype))
    net(x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_print_op_dynamic_shape(mode):
    """
    Feature: cpu Print op.
    Description: test Print with dynamic shape.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="Ascend")

    net = PrintNet()
    x = Tensor(np.random.randn(3, 4, 5).astype(np.float32))
    x_dyn = Tensor(shape=[None, None, None], dtype=ms.float32)
    net.set_inputs(x_dyn)
    net(x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_print_tensor_dtype(mode):
    """
    Feature: Print op.
    Description: test Print with tensor dtype.
    Expectation: success.
    """
    class PrintDtypeNet(nn.Cell):
        def construct(self, x, y):
            print(f"Tensor x type: {x.dtype}")
            print("Tensor y type:", y.dtype)
            return x, y

    context.set_context(mode=mode, device_target="Ascend")
    x = Tensor(np.random.randn(3, 4, 5).astype(np.float32))
    y = Tensor([1, 2], dtype=ms.int32)
    net = PrintDtypeNet()
    net(x, y)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_print_tensor_dtype_in_nested_tuple(mode):
    """
    Feature: Print op.
    Description: test Print with tensor dtype in nested tuple.
    Expectation: success.
    """
    class PrintDtypeNet(nn.Cell):
        def construct(self, x, y):
            dtype_tuple = (x.dtype, y)
            dtype_tuple_tuple = (x, dtype_tuple)
            print("Tensor type in tuple:", dtype_tuple_tuple)
            return x + y

    context.set_context(mode=mode, device_target="Ascend")
    x = Tensor([3, 4], dtype=ms.int32)
    y = Tensor([1, 2], dtype=ms.int32)
    net = PrintDtypeNet()
    net(x, y)
