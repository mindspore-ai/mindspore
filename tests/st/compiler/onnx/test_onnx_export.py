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
""" Test onnx export"""
import os
import numpy as np
import onnxruntime as ort
import mindspore as ms
from mindspore import nn
from mindspore import Tensor, ops, Parameter
from mindspore.onnx import export
from tests.mark_utils import arg_mark


ms.set_context(mode=ms.GRAPH_MODE)


class NetGeLU(nn.Cell):
    def __init__(self, mul_size):
        super().__init__()
        self.op = ops.operations.GeLU()
        mul_np = np.full(mul_size, 2.0, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")

    def construct(self, inputs):
        out = ops.operations.Mul()(inputs, self.mul_weight)
        out = self.op(out)
        return out


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_ms_onnx_export_gelu():
    """
    Feature: Onnx
    Description: Test onnx export
    Expectation: The exported ONNX file meets expectations.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    _cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(_cur_dir, "test_ms_onnx_export_gelu")
    onnx_file_name = file_name + '.onnx'

    mul_size = (3,)
    input_data = [1.0, 2.0, 3.0]
    ms_input_data = Tensor(input_data, dtype=ms.float32)
    np_input_data = np.array(input_data, dtype=np.float32)

    try:
        net = NetGeLU(mul_size=mul_size)
        ms_output = net(ms_input_data)
        ms_output_numpy = ms_output.numpy()
        export(net, ms_input_data, file_name=onnx_file_name)
        assert os.path.exists(onnx_file_name)

        session = ort.InferenceSession(onnx_file_name)
        onnx_output = session.run(None, {'inputs': np_input_data})
        assert np.allclose(ms_output_numpy, onnx_output[0], 0.001, 0.001)
    finally:
        if os.path.exists(onnx_file_name):
            os.remove(onnx_file_name)


class SiLUNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()

    def construct(self, x):
        output = self.silu(x)
        return output


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_onnx_001():
    """
    Feature: Onnx
    Description: Test onnx export with silu
    Expectation: The exported ONNX file meets expectations.
    """
    input_np = np.random.randint(low=-25, high=25, size=(5, 7, 7)).astype(np.float16)
    x = Tensor(input_np)
    net = SiLUNet()
    output = net(x)
    onnx_file = './silu_onnx_001.onnx'
    try:
        export(net, x, file_name=onnx_file)
        session = ort.InferenceSession(onnx_file)
        output_onnx = session.run(None, {"x": input_np})
        assert np.allclose(output, output_onnx[0], rtol=1e-3, atol=1e-3), "silu not equal, please check"
    finally:
        if os.path.exists(onnx_file):
            os.remove(onnx_file)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_onnx_002():
    """
    Feature: Onnx
    Description: Test onnx export with silu
    Expectation: The exported ONNX file meets expectations.
    """
    input_np = np.random.randn(9, 8, 9, 7, 7, 3, 4).astype(np.float32)
    x = Tensor(input_np)
    net = SiLUNet()
    output = net(x)
    onnx_file = './silu_onnx_002.onnx'
    try:
        export(net, x, file_name=onnx_file)
        session = ort.InferenceSession(onnx_file)
        output_onnx = session.run(None, {"x": input_np})
        assert np.allclose(output, output_onnx[0], rtol=1e-4, atol=1e-4), "silu not equal, please check"
    finally:
        if os.path.exists(onnx_file):
            os.remove(onnx_file)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_onnx_003():
    """
    Feature: Onnx
    Description: Test onnx export with silu
    Expectation: The exported ONNX file meets expectations.
    """
    input_np = np.random.randn(9, ).astype(np.float32)
    x = Tensor(input_np)
    net = SiLUNet()
    output = net(x)
    onnx_file = './silu_onnx_003.onnx'
    try:
        export(net, x, file_name=onnx_file)
        session = ort.InferenceSession(onnx_file)
        output_onnx = session.run(None, {"x": input_np})
        assert np.allclose(output, output_onnx[0], rtol=1e-4, atol=1e-4), "silu not equal, please check"
    finally:
        if os.path.exists(onnx_file):
            os.remove(onnx_file)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_onnx_004():
    """
    Feature: Onnx
    Description: Test onnx export with softmax
    Expectation: The exported ONNX file meets expectations.
    """
    input_np = np.random.randn(64, 12, 128, 128).astype(np.float32)
    x = Tensor(input_np)
    net = SoftmaxNet(-1)
    output = net(x)
    onnx_file = './softmax_onnx_004.onnx'
    try:
        export(net, x, file_name=onnx_file)
        session = ort.InferenceSession(onnx_file)
        output_onnx = session.run(None, {"x": input_np})
        assert np.allclose(output, output_onnx[0], rtol=1e-4, atol=1e-4), "softmax not equal, please check"
    finally:
        if os.path.exists(onnx_file):
            os.remove(onnx_file)


class SoftmaxNet(nn.Cell):
    def __init__(self, axis=1):
        super().__init__()
        self.softmax = nn.Softmax(axis)

    def construct(self, x):
        return self.softmax(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_onnx_005():
    """
    Feature: Onnx
    Description: Test onnx export with softmax
    Expectation: The exported ONNX file meets expectations.
    """
    input_np = np.random.randn(2, 32).astype(np.float32)
    x = Tensor(input_np)
    net = SoftmaxNet(1)
    output = net(x)
    onnx_file = './softmax_onnx_005.onnx'
    try:
        export(net, x, file_name=onnx_file)
        session = ort.InferenceSession(onnx_file)
        output_onnx = session.run(None, {"x": input_np})
        assert np.allclose(output, output_onnx[0], rtol=1e-4, atol=1e-4), "softmax not equal, please check"
    finally:
        if os.path.exists(onnx_file):
            os.remove(onnx_file)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_onnx_006():
    """
    Feature: Onnx
    Description: Test onnx export with softmax
    Expectation: The exported ONNX file meets expectations.
    """
    input_np = np.random.randn(200, 3, 128).astype(np.float32)
    x = Tensor(input_np)
    net = SoftmaxNet(2)
    output = net(x)
    onnx_file = './softmax_onnx_006.onnx'
    try:
        export(net, x, file_name=onnx_file)
        session = ort.InferenceSession(onnx_file)
        output_onnx = session.run(None, {"x": input_np})
        assert np.allclose(output, output_onnx[0], rtol=1e-4, atol=1e-4), "softmax not equal, please check"
    finally:
        if os.path.exists(onnx_file):
            os.remove(onnx_file)


class MulsNet(nn.Cell):
    def __init__(self, other):
        super().__init__()
        self.y = other

    def construct(self, x):
        z = x * self.y
        return z


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_onnx_007():
    """
    Feature: Onnx
    Description: Test onnx export with mul operator.
    Expectation: The exported ONNX file meets expectations.
    """
    np_x = np.random.randn(1, ).astype(np.float32)
    other_x = 2
    x = Tensor(np_x)
    other = Tensor(other_x)
    net = MulsNet(other)
    ms_output = net(x)
    onnx_file = './muls_onnx_007.onnx'
    try:
        export(net, x, file_name=onnx_file)
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), "muls not equal, please check"
    finally:
        if os.path.exists(onnx_file):
            os.remove(onnx_file)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_onnx_008():
    """
    Feature: Onnx
    Description: Test onnx export with mul operator.
    Expectation: The exported ONNX file meets expectations.
    """
    np_x = np.random.randint(-128, 127, (9, 7, 4, 9, 5)).astype(np.int8)
    other_x = -6
    x = Tensor(np_x)
    other = Tensor([other_x])
    net = MulsNet(other)
    ms_output = net(x)
    onnx_file = './muls_onnx_008.onnx'
    try:
        export(net, x, file_name=onnx_file)
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), "muls not equal, please check"
    finally:
        if os.path.exists(onnx_file):
            os.remove(onnx_file)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_onnx_009():
    """
    Feature: Onnx
    Description: Test onnx export with mul operator.
    Expectation: The exported ONNX file meets expectations.
    """
    np_x = np.random.randn(4718592).astype(np.float16)
    other = 0.7898204683379066
    x = Tensor(np_x)
    net = MulsNet(other)
    ms_output = net(x)
    onnx_file = './muls_onnx_009.onnx'
    try:
        export(net, x, file_name=onnx_file)
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), "muls not equal, please check"
    finally:
        if os.path.exists(onnx_file):
            os.remove(onnx_file)
