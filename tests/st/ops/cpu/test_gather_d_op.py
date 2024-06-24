# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.train.serialization import export

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetGatherD(nn.Cell):
    def __init__(self, dim=1):
        super(NetGatherD, self).__init__()
        self.gatherd = P.GatherD()
        self.dim = int(dim)

    def construct(self, x, index):
        return self.gatherd(x, self.dim, index)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_gatherd_fp32():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(5, 5, 5).astype(np.float32) * prop
    index = np.random.randint(0, 5, (5, 3, 5)).astype(np.int32)
    dim = 1

    gatherd = NetGatherD(dim)
    output = gatherd(Tensor(x), Tensor(index))

    expect = np.zeros(index.shape).astype(np.float32)
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            for k in range(index.shape[2]):
                expect[i, j, k] = x[i, index[i, j, k], k]
    error = np.ones(shape=expect.shape) * 1.0e-6
    assert np.all(np.abs(output.asnumpy() - expect) < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_gatherd_fp16():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(5, 5, 5).astype(np.float16) * prop
    index = np.random.randint(0, 5, (3, 5, 5)).astype(np.int64)
    dim = 0

    gatherd = NetGatherD(dim)
    output = gatherd(Tensor(x), Tensor(index))

    expect = np.zeros(index.shape).astype(np.float16)
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            for k in range(index.shape[2]):
                expect[i, j, k] = x[index[i, j, k], j, k]
    error = np.ones(shape=expect.shape) * 1.0e-6
    assert np.all(np.abs(output.asnumpy() - expect) < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_gatherd_int32():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(5, 5, 5).astype(np.int32) * prop
    index = np.random.randint(0, 5, (5, 5, 8)).astype(np.int32)
    dim = -1

    gatherd = NetGatherD(dim)
    output = gatherd(Tensor(x), Tensor(index))

    expect = np.zeros(index.shape).astype(np.int32)
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            for k in range(index.shape[2]):
                expect[i, j, k] = x[i, j, index[i, j, k]]
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_gatherd_bool():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(5, 5, 5).astype(np.int32) * prop
    x = (x >= 0).astype(np.bool)
    index = np.random.randint(0, 5, (5, 5, 8)).astype(np.int32)
    dim = -1

    gatherd = NetGatherD(dim)
    output = gatherd(Tensor(x), Tensor(index))

    expect = np.zeros(index.shape).astype(np.bool)
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            for k in range(index.shape[2]):
                expect[i, j, k] = x[i, j, index[i, j, k]]
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_gatherd_cpu_dynamic_shape():
    """
    Feature: test GatherD op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    dim = -1
    gatherd = NetGatherD(dim)
    x_dyn = Tensor(shape=[None, 5, 5], dtype=mindspore.float32)
    index_dyn = Tensor(shape=[5, 5, None], dtype=mindspore.int32)
    gatherd.set_inputs(x_dyn, index_dyn)
    x = np.random.randn(5, 5, 5)
    y = np.random.randn(5, 5, 8)
    output = gatherd(Tensor(x, mindspore.float32), Tensor(y, mindspore.int32))
    expect_shape = (5, 5, 8)
    assert output.asnumpy().shape == expect_shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gatherd_cpu_onnx():
    """
    Feature: test GatherD op in cpu.
    Description: test the ops export onnx.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    dim = 1
    net = NetGatherD(dim)
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 0]], dtype=np.int32)
    out_ms = net(Tensor(data), Tensor(indices)).asnumpy()
    file = 'gatherd.onnx'
    export(net, Tensor(data), Tensor(indices), file_name=file, file_format="ONNX")
    assert os.path.exists(file)

    import onnxruntime
    sess = onnxruntime.InferenceSession(file)
    input_x = sess.get_inputs()[0].name
    input_indices = sess.get_inputs()[1].name
    result = sess.run([], {input_x: data, input_indices: indices})[0]
    assert np.all(out_ms == result)

    os.chmod(file, stat.S_IWRITE)
    os.remove(file)
