# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import os
import stat
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class OpNetWrapper(nn.Cell):
    def __init__(self, op):
        super(OpNetWrapper, self).__init__()
        self.op = op

    def construct(self, *inputs):
        return self.op(*inputs)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_logicaland():
    op = P.LogicalAnd()
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.array([True, False, False]))
    input_y = Tensor(np.array([True, True, False]))
    outputs = op_wrapper(input_x, input_y)

    assert np.allclose(outputs.asnumpy(), (True, False, False))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_logicalor():
    op = P.LogicalOr()
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.array([True, False, False]))
    input_y = Tensor(np.array([True, True, False]))
    outputs = op_wrapper(input_x, input_y)

    assert np.allclose(outputs.asnumpy(), (True, True, False))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_logicalor_onnx():
    """
    Feature: Test the function of exporting op LogicalOr to ONNX.
    Description: Test the function of exporting LogicalOr to ONNX.
    Expectation: The result match to the expect value.
    """
    op = P.LogicalOr()
    op_wrapper = OpNetWrapper(op)

    x_np = np.array([True, False, False])
    y_np = np.array([True, True, False])
    input_x = Tensor(x_np)
    input_y = Tensor(y_np)
    outputs = op_wrapper(input_x, input_y).asnumpy()

    file_name = 'logical_or.onnx'
    export(op_wrapper, input_x, input_y,
           file_name=file_name, file_format='ONNX')
    assert os.path.exists(file_name)

    import onnxruntime
    sess = onnxruntime.InferenceSession(file_name)
    onnx_input_x = sess.get_inputs()[0].name
    onnx_input_y = sess.get_inputs()[1].name
    result = sess.run([], {onnx_input_x: x_np, onnx_input_y: y_np})[0]
    assert np.all(outputs == result)

    os.chmod(file_name, stat.S_IWRITE)
    os.remove(file_name)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_logicalnot():
    op = P.LogicalNot()
    op_wrapper = OpNetWrapper(op)

    input_x = Tensor(np.array([True, False, False]))
    outputs = op_wrapper(input_x)

    assert np.allclose(outputs.asnumpy(), (False, True, True))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_xor_functional_api_modes(mode):
    """
    Feature: Test logical_xor functional api.
    Description: Test logical_xor functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([True, False, True], mstype.bool_)
    y = Tensor([True, True, False], mstype.bool_)
    output = F.logical_xor(x, y)
    expected = np.array([False, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_and_tensor_api_modes(mode):
    """
    Feature: Test logical_and tensor api.
    Description: Test logical_and tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    input_x = Tensor([True, False, True], mstype.bool_)
    other = Tensor([True, True, False], mstype.bool_)
    output = input_x.logical_and(other)
    expected = np.array([True, False, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_not_tensor_api_modes(mode):
    """
    Feature: Test logical_not tensor api.
    Description: Test logical_not tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    input_x = Tensor([True, False, True], mstype.bool_)
    output = input_x.logical_not()
    expected = np.array([False, True, False])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_or_tensor_api_modes(mode):
    """
    Feature: Test logical_or tensor api.
    Description: Test logical_or tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    input_x = Tensor([True, False, True], mstype.bool_)
    other = Tensor([True, True, False], mstype.bool_)
    output = input_x.logical_or(other)
    expected = np.array([True, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logical_xor_tensor_api_modes(mode):
    """
    Feature: Test logical_xor tensor api.
    Description: Test logical_xor tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    input_x = Tensor([True, False, True], mstype.bool_)
    other = Tensor([True, True, False], mstype.bool_)
    output = input_x.logical_xor(other)
    expected = np.array([False, True, True])
    np.testing.assert_array_equal(output.asnumpy(), expected)


if __name__ == '__main__':
    test_logicaland()
    test_logicalor()
    test_logicalnot()
