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

import os
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from mindspore.train.serialization import export

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetSin(nn.Cell):
    def __init__(self):
        super(NetSin, self).__init__()
        self.sin = P.Sin()

    def construct(self, x):
        return self.sin(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sin():
    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype('float32')
    input_x = Tensor(np_array)
    net = NetSin()
    output = net(input_x)
    print(output)
    expect = np.sin(np_array)
    assert np.allclose(output.asnumpy(), expect)

    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype('float64')
    input_x = Tensor(np_array)
    net = NetSin()
    output = net(input_x)
    print(output)
    expect = np.sin(np_array)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sin_onnx():
    """
    Feature: test sin op in cpu
    Description: test the ops onnx export
    Expectation: expect correct result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype('float32')
    input_x = Tensor(np_array)
    net = NetSin()
    ms_output = net(input_x)
    file = 'sin.onnx'
    export(net, input_x, file_name=file, file_format="ONNX")
    assert os.path.exists(file)

    import onnxruntime as ort
    import onnx
    onnx_model = onnx.load_model(file)
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    result = sess.run([], {input_name: np_array})
    assert np.allclose(ms_output.asnumpy(), result[0])
